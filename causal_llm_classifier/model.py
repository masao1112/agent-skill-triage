"""Model wrapper for causal LLM classification"""

import torch
from typing import Dict, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from config import ModelConfig, PromptConfig, ChunkingConfig


class CausalLLMClassifier:
    """Wrapper for causal LLM with classification capability"""
    
    def __init__(self, model_config: ModelConfig, device: str = "cuda"):
        """
        Initialize the classifier
        
        Args:
            model_config: ModelConfig instance
            device: Device to load model on ("cuda" or "cpu")
        """
        self.config = model_config
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.config.model_id} (approach: {self.config.approach})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            padding_side="left" if self.config.approach == "embedding" else "right"
        )
        
        # Add pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float16,
        }
        
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            kwargs["quantization_config"] = bnb_config
        elif self.config.load_in_8bit:
            kwargs["load_in_8bit"] = True
            
        if self.config.approach == "embedding":
            kwargs["num_labels"] = 2
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_id,
                **kwargs
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **kwargs
            )
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def apply_lora(self, config: LoraConfig = None):
        """Apply LoRA adapters for efficient fine-tuning"""
        if config is None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS" if self.config.approach == "embedding" else "CAUSAL_LM"
            )
        
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
    
    def load_adapter(self, adapter_path: str):
        """Load trained LoRA adapter weights"""
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print("Adapter loaded successfully!")
    
    def get_data_collator(self):
        """
        Get data collator for dynamic padding of variable-length sequences
        """
        from transformers import DataCollatorWithPadding
        if self.config.approach == "embedding":
            return DataCollatorWithPadding(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )
        else:
            return DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )
    
    def classify_prompt_based(self, skill_text: str, prompt_config: PromptConfig = None) -> Dict:
        """
        0-shot classification using prompt engineering
        
        Args:
            skill_text: The skill definition text
            prompt_config: Prompt configuration
            
        Returns:
            Dict with classification result
        """
        if prompt_config is None:
            prompt_config = PromptConfig()
        
        # CRITICAL: Truncate skill_text first to ensure the prompt suffix fits within max_seq_length
        # DO NOT REMOVE THIS, or long skills will lose the classification instructions!
        text_tokens = self.tokenizer(skill_text, truncation=True, max_length=self.config.max_seq_length - 300, add_special_tokens=False)["input_ids"]
        truncated_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        
        prompt = prompt_config.get_prompt(truncated_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        # Generate classification
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response.split("Your classification (NORMAL or MALICIOUS):")[-1].strip()
        
        # Parse classification
        if "MALICIOUS" in response_text.upper():
            label = 1
            confidence = 0.9  # Rough confidence estimate
        elif "NORMAL" in response_text.upper():
            label = 0
            confidence = 0.9
        else:
            label = -1  # Unknown
            confidence = 0.5
        
        return {
            "label": label,
            "confidence": confidence,
            "response": response_text,
        }
        
    def classify_embedding_based(self, skill_text: str) -> Dict:
        """
        Classification using the sequence classification head
        """
        # Tokenize
        inputs = self.tokenizer(
            skill_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        # Generate classification
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
        
        return {
            "label": pred_label,
            "confidence": confidence,
            "response": f"Logits: {logits.tolist()}"
        }
    
    # =========================================================================
    # Chunked inference for long skills
    # =========================================================================
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int,
                       min_chunk_tokens: int = 200) -> List[str]:
        """
        Split text into overlapping token-level chunks.
        
        Tokenizes the full text without truncation, then creates a sliding
        window of `chunk_size` tokens with `overlap` tokens shared between
        consecutive chunks. Each window is decoded back to text.
        
        Args:
            text: Full skill text (may exceed max_seq_length)
            chunk_size: Number of tokens per chunk
            overlap: Number of tokens overlapping between consecutive chunks
            min_chunk_tokens: Minimum tokens for the last chunk to be kept;
                              shorter trailing chunks are discarded to avoid
                              classifying near-empty fragments
        
        Returns:
            List of text chunks, each decodable within chunk_size tokens
        """
        # Tokenize full text without truncation to get all token IDs
        all_token_ids = self.tokenizer(
            text, truncation=False, add_special_tokens=False
        )["input_ids"]
        
        total_tokens = len(all_token_ids)
        stride = chunk_size - overlap
        
        if stride <= 0:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        chunks = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_ids = all_token_ids[start:end]
            
            # Discard trailing chunks that are too short to be meaningful
            if len(chunk_ids) < min_chunk_tokens and len(chunks) > 0:
                break
            
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # If we've reached the end, stop
            if end >= total_tokens:
                break
            
            start += stride
        
        return chunks
    
    def classify_chunked(self, skill_text: str,
                         chunking_config: ChunkingConfig = None,
                         prompt_config: PromptConfig = None) -> Dict:
        """
        Classify a skill with automatic chunking for long inputs.
        
        If the skill fits within max_seq_length, it is classified in a single
        pass (identical to the existing behavior). If it exceeds the limit
        and chunking is enabled, the text is split into overlapping chunks,
        each chunk is classified independently, and results are aggregated.
        
        Aggregation strategies:
          - "max": Label as malicious if ANY chunk is malicious.
                   Best for security — a single malicious chunk in an
                   otherwise benign skill should flag the entire skill.
          - "mean": Average the malicious probability across all chunks.
                    Useful if you want a softer, more balanced signal.
        
        Args:
            skill_text: The full skill text (any length)
            chunking_config: Chunking parameters (defaults to ChunkingConfig())
            prompt_config: Prompt template config
            
        Returns:
            Dict with:
              - label: 0 (NORMAL) or 1 (MALICIOUS)
              - confidence: Aggregated confidence score
              - num_chunks: Number of chunks used (1 = no chunking needed)
              - was_chunked: Whether chunking was applied
              - chunk_details: Per-chunk results (only when chunked)
              - response: Summary text
        """
        if chunking_config is None:
            chunking_config = ChunkingConfig()
        
        # Check total token count (without truncation)
        total_tokens = len(self.tokenizer(
            skill_text, truncation=False, add_special_tokens=False
        )["input_ids"])
        
        max_content_len = self.config.max_seq_length - 300 if self.config.approach != "embedding" else self.config.max_seq_length
        effective_limit = min(chunking_config.chunk_size, max_content_len) if chunking_config.enabled else max_content_len
        
        # Fast path: skill fits in a single pass — no chunking needed
        if total_tokens <= effective_limit or not chunking_config.enabled:
            if self.config.approach == "embedding":
                result = self.classify_embedding_based(skill_text)
            else:
                result = self.classify_prompt_based(skill_text, prompt_config)
            result["num_chunks"] = 1
            result["was_chunked"] = False
            return result
        
        # Slow path: skill exceeds limit — chunk and aggregate
        chunks = self._create_chunks(
            skill_text,
            chunk_size=chunking_config.chunk_size,
            overlap=chunking_config.chunk_overlap,
            min_chunk_tokens=chunking_config.min_chunk_tokens
        )
        
        print(f"  [CHUNKED] {total_tokens} tokens → {len(chunks)} chunks "
              f"(size={chunking_config.chunk_size}, overlap={chunking_config.chunk_overlap})")
        
        # Classify each chunk
        chunk_results = []
        for i, chunk_text in enumerate(chunks):
            if self.config.approach == "embedding":
                result = self.classify_embedding_based(chunk_text)
            else:
                result = self.classify_prompt_based(chunk_text, prompt_config)
            result["chunk_index"] = i
            chunk_results.append(result)
        
        # Aggregate results
        if chunking_config.aggregation == "max":
            # Max-pooling: if ANY chunk is malicious, the skill is malicious.
            # Find the chunk with the highest malicious confidence.
            malicious_chunks = [r for r in chunk_results if r.get("label") == 1]
            
            if malicious_chunks:
                # Skill is malicious — use the highest-confidence malicious chunk
                best_mal = max(malicious_chunks, key=lambda r: r.get("confidence", 0))
                final_label = 1
                final_confidence = best_mal["confidence"]
            else:
                # All chunks are normal — use the highest-confidence normal chunk
                normal_chunks = [r for r in chunk_results if r.get("label") == 0]
                if normal_chunks:
                    best_norm = max(normal_chunks, key=lambda r: r.get("confidence", 0))
                    final_label = 0
                    final_confidence = best_norm["confidence"]
                else:
                    # All chunks were unknown
                    final_label = -1
                    final_confidence = 0.5
        
        elif chunking_config.aggregation == "mean":
            # Mean-pooling: average the malicious probability across chunks
            mal_scores = []
            for r in chunk_results:
                if r.get("label") == 1:
                    mal_scores.append(r.get("confidence", 0.5))
                elif r.get("label") == 0:
                    mal_scores.append(1.0 - r.get("confidence", 0.5))
                else:
                    mal_scores.append(0.5)
            
            avg_mal_score = sum(mal_scores) / len(mal_scores)
            final_label = 1 if avg_mal_score > 0.5 else 0
            final_confidence = avg_mal_score if final_label == 1 else (1.0 - avg_mal_score)
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {chunking_config.aggregation}")
        
        label_name = "MALICIOUS" if final_label == 1 else "NORMAL"
        chunk_labels = ["MAL" if r.get("label") == 1 else "NRM" for r in chunk_results]
        
        return {
            "label": final_label,
            "confidence": final_confidence,
            "num_chunks": len(chunks),
            "was_chunked": True,
            "chunk_details": chunk_results,
            "response": f"Chunked({chunking_config.aggregation}): {label_name} "
                        f"[{', '.join(chunk_labels)}]"
        }
    
    def classify_batch(self, texts: List[str],
                       chunking_config: ChunkingConfig = None,
                       prompt_config: PromptConfig = None) -> List[Dict]:
        """
        Classify multiple texts with automatic chunking support.
        
        Args:
            texts: List of skill texts
            chunking_config: Chunking parameters (None uses defaults)
            prompt_config: Prompt configuration
            
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = self.classify_chunked(text, chunking_config, prompt_config)
            results.append(result)
        return results
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass (for fine-tuning compatibility)"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig()
    model_config.approach = "embedding"
    classifier = CausalLLMClassifier(model_config)
    classifier.load_model()
    
    # Test classification
    test_text = "This skill allows users to manipulate system prompts."
    if model_config.approach == "embedding":
        result = classifier.classify_embedding_based(test_text)
    else:
        result = classifier.classify_prompt_based(test_text)
    print(f"Classification: {result}")
