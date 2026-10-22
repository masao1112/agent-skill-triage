"""Training script with LoRA fine-tuning for causal LLM classifier"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model import CausalLLMClassifier
from dataset import SkillDataset, SkillTokenizedDataset, SkillEmbeddingDataset, load_all_data, load_data_from_csv
from config import ModelConfig, TrainingConfig, DataConfig


def preprocess_logits_for_metrics(logits, labels):
    """Preprocess logits to save memory during evaluation"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    """Compute classification metrics from causal LM outputs"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import numpy as np
    
    predictions, labels = eval_pred
    
    # Check if this is the embedding approach (predictions shape will be 1D after argmax)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    if len(predictions.shape) == 1 or len(labels.shape) == 1:
        # Embedding approach: predictions are 1D (batch_size,)
        valid_preds = predictions
        valid_labels = labels
    else:
        # Instruction approach: predictions are aligned with input_ids, but causal LM loss shifts them.
        # To match predictions to targets correctly, we shift them by 1.
        preds = predictions[:, :-1]
        labels = labels[:, 1:]
        
        # Mask out ignored labels (-100)
        mask = labels != -100
        valid_preds = preds[mask]
        valid_labels = labels[mask]
        
    if len(valid_labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    acc = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }



class SkillClassificationTrainer:
    """Trainer for fine-tuning causal LLM on skill classification"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.classifier = None
        self.trainer = None
    
    def setup(self):
        """Initialize model and training setup"""
        self.classifier = CausalLLMClassifier(self.model_config)
        self.classifier.load_model()
        
        # Prepare model for k-bit training (casts layernorms and modules_to_save to fp32 for stable training)
        if getattr(self.classifier.model, "is_loaded_in_4bit", False) or getattr(self.classifier.model, "is_loaded_in_8bit", False) or self.model_config.load_in_4bit:
            self.classifier.model = prepare_model_for_kbit_training(self.classifier.model)
        
        # Apply LoRA adapters
        lora_config = LoraConfig(
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            target_modules=self.training_config.lora_target_modules,
            lora_dropout=self.training_config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS" if self.model_config.approach == "embedding" else "CAUSAL_LM"
        )
        
        self.classifier.model = get_peft_model(self.classifier.model, lora_config)
        self.classifier.model.print_trainable_parameters()
    
    def prepare_data(self, data_dir: str = None, csv_path: str = None):
        """Prepare training data (same logic as ChunkBERT)"""
        print("Loading dataset...")
        
        if data_dir and os.path.exists(data_dir):
            samples = load_all_data(data_dir)
        elif csv_path and os.path.exists(csv_path):
            samples = load_data_from_csv(csv_path)
        else:
            raise ValueError(f"Must provide valid data_dir or csv_path. Got: data_dir={data_dir}, csv_path={csv_path}")
        
        if not samples:
            raise ValueError(f"No samples loaded from {data_dir or csv_path}")
        
        print(f"Total samples loaded: {len(samples)}")
        
        dataset = SkillDataset(data=samples)
        stats = dataset.get_statistics()
        print(f"Statistics: {stats}")
        
        # Split data
        train_samples, val_samples, test_samples = dataset.split_data(
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def train(self, checkpoint_dir: str = "./checkpoints", data_dir: str = None, csv_path: str = None):
        """Train the model with LoRA and dynamic padding via data collator"""
        
        # Setup
        self.setup()
        
        # Prepare data
        train_samples, val_samples, test_samples = self.prepare_data(data_dir=data_dir, csv_path=csv_path)
        
        print(f"Creating tokenized datasets for approach: {self.model_config.approach}")
        if self.model_config.approach == "embedding":
            train_dataset = SkillEmbeddingDataset(
                train_samples,
                tokenizer=self.classifier.tokenizer,
                max_seq_length=self.model_config.max_seq_length
            )
            val_dataset = SkillEmbeddingDataset(
                val_samples,
                tokenizer=self.classifier.tokenizer,
                max_seq_length=self.model_config.max_seq_length
            )
        else:
            train_dataset = SkillTokenizedDataset(
                train_samples,
                tokenizer=self.classifier.tokenizer,
                max_seq_length=self.model_config.max_seq_length
            )
            val_dataset = SkillTokenizedDataset(
                val_samples,
                tokenizer=self.classifier.tokenizer,
                max_seq_length=self.model_config.max_seq_length
            )
        
        # Get data collator for dynamic padding
        data_collator = self.classifier.get_data_collator()
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps,
            weight_decay=self.training_config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=5,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            gradient_checkpointing=True,
        )
        
        print("="*50)
        print("Training configuration:")
        print(f"Output dir: {checkpoint_dir}")
        print(f"Epochs: {self.training_config.num_epochs}")
        print(f"Batch size: {self.training_config.batch_size}")
        print(f"Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.training_config.learning_rate}")
        print(f"Max sequence length: {self.model_config.max_seq_length}")
        print(f"Dynamic padding: ENABLED (pads to longest in batch)")
        print("="*50)
        
        # Create Trainer with data collator
        trainer = Trainer(
            model=self.classifier.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,  # Dynamic padding collator
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(os.path.join(checkpoint_dir, "final"))
        print(f"Model saved to {os.path.join(checkpoint_dir, 'final')}")


def main():
    parser = argparse.ArgumentParser(description="Train causal LLM for skill classification")
    parser.add_argument("--model", type=str, default="llama3.2",
                       choices=list(ModelConfig.MODEL_OPTIONS.keys()),
                       help="Model to use")
    parser.add_argument("--data_dir", type=str, default="../raw_samples",
                       help="Directory containing training data")
    parser.add_argument("--csv_path", type=str, default="../processed_samples/processed_skills.csv",
                       help="CSV file path (fallback if data_dir not available)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--disable_4bit", action="store_true",
                       help="Disable 4-bit quantization (4-bit is enabled by default to prevent OOM)")
    parser.add_argument("--approach", type=str, default="instruction",
                       choices=["embedding", "instruction"],
                       help="Classification approach: embedding (Approach 1) or instruction (Approach 2)")
    args = parser.parse_args()
    
    # Setup configs
    model_config = ModelConfig()
    model_config.model_name = args.model
    model_config.model_id = ModelConfig.MODEL_OPTIONS[args.model]
    model_config.load_in_4bit = not args.disable_4bit
    model_config.approach = args.approach
    
    training_config = TrainingConfig()
    training_config.num_epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    
    # Train
    trainer = SkillClassificationTrainer(model_config, training_config)
    trainer.train(checkpoint_dir=args.checkpoint_dir, data_dir=args.data_dir, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
