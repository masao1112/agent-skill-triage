"""Data loading and preprocessing for causal LLM classifier"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from config import PromptConfig


def parse_skill_sample(skill_dir: Path) -> dict:
    """Parses a single skill directory structure (same as ChunkBERT)."""
    skill_md = skill_dir / "SKILL.md"
    meta_json = skill_dir / "sample_meta.json"
    
    if not skill_md.exists():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")
    
    content = skill_md.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    frontmatter = yaml.safe_load(parts[1]) if len(parts) >= 3 else {}
    body = parts[2] if len(parts) >= 3 else content
    
    # Read ALL files recursively, matching the notebook's format exactly:
    #   ### FILE: {rel_path} ###\n{text}
    # This is critical for train/inference consistency since the model was
    # trained on data produced by process_skills.ipynb using this format.
    all_texts = []
    for root, dirs, files in os.walk(skill_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in sorted(files):
            if fname.startswith('.'):
                continue
            fpath = Path(root) / fname
            rel_path = fpath.relative_to(skill_dir)
            try:
                text_content = fpath.read_text(encoding="utf-8", errors="ignore")
                all_texts.append(f"### FILE: {rel_path} ###\n{text_content}")
            except Exception:
                all_texts.append(f"### FILE: {rel_path} ### [BINARY/UNREADABLE]")
    
    full_text = "\n\n".join(all_texts)
    
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}
    
    # Binary labels: 0 for normal, 1 for malicious (injection, exfil, jailbreak)
    original_label = meta.get("label", skill_dir.parent.name.split('_')[1] if '_' in skill_dir.parent.name else "unknown")
    if original_label == "normal":
        label_id = 0
    elif original_label in ["prompt_injection", "data_exfiltration", "jailbreak", "data_exfil"]:
        label_id = 1
    else:
        # Default based on directory name
        if "normal" in skill_dir.parent.name.lower():
            label_id = 0
        else:
            label_id = 1
    
    return {
        "id": meta.get("id", skill_dir.name),
        "original_label": original_label,
        "label_id": label_id,
        "description": frontmatter.get("description", ""),
        "body": body.strip(),
        "full_text": full_text,
    }


def load_all_data(raw_samples_dir: str) -> List[Dict]:
    """Loads all samples from the raw_samples directory (same as ChunkBERT)."""
    raw_path = Path(raw_samples_dir)
    samples = []
    
    if not raw_path.exists():
        print(f"[WARN] Data directory {raw_samples_dir} does not exist.")
        return samples
    
    for group_dir in raw_path.iterdir():
        if group_dir.is_dir():
            for skill_dir in group_dir.iterdir():
                if (skill_dir / "SKILL.md").exists():
                    try:
                        samples.append(parse_skill_sample(skill_dir))
                    except Exception as e:
                        print(f"[WARN] Skip {skill_dir.name}: {e}")
    
    return samples


def load_data_from_csv(csv_path: str) -> List[Dict]:
    """Load from processed CSV (same as ChunkBERT)."""
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    samples = []
    for idx, row in df.iterrows():
        # Binary classification mapping directly from 'label' column
        label_id = int(row.get("label", 0))
        
        full_text = row.get("full_text", "")
        if pd.isna(full_text):
            full_text = ""
        else:
            full_text = str(full_text)
        
        samples.append({
            "id": row.get("skill_name", str(idx)),
            "original_label": "normal" if label_id == 0 else "malicious",
            "label_id": label_id,
            "full_text": full_text
        })
    return samples


class SkillDataset:
    """Load and preprocess skill SKILL.md files"""
    
    def __init__(self, data: List[Dict] = None, max_seq_length: int = 10000):
        """
        Args:
            data: List of sample dictionaries (from load_all_data or load_data_from_csv)
            max_seq_length: Maximum sequence length (will be handled by data collator)
        """
        self.data = data if data is not None else []
        self.max_seq_length = max_seq_length
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List, List, List]:
        """
        Split data into train/val/test sets
        
        Returns:
            (train_samples, val_samples, test_samples)
        """
        test_ratio = 1.0 - train_ratio - val_ratio
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            self.data, test_size=test_ratio, 
            stratify=[s["label_id"] for s in self.data],
            random_state=42
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_size,
            stratify=[s["label_id"] for s in train_val],
            random_state=42
        )
        
        return train, val, test
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.data:
            return {}
        
        labels = [s["label_id"] for s in self.data]
        texts_len = [len(str(s.get("full_text", ""))) for s in self.data]
        
        return {
            "total_samples": len(self.data),
            "normal_count": sum(1 for l in labels if l == 0),
            "malicious_count": sum(1 for l in labels if l == 1),
            "avg_text_length": sum(texts_len) / len(texts_len) if texts_len else 0,
            "max_text_length": max(texts_len) if texts_len else 0,
            "min_text_length": min(texts_len) if texts_len else 0,
        }


class SkillTokenizedDataset(Dataset):
    """PyTorch Dataset for tokenized skill samples with dynamic padding via collator"""
    
    def __init__(self, samples: List[Dict], tokenizer, max_seq_length: int = 10000):
        """
        Args:
            samples: List of sample dictionaries with 'full_text' and 'label_id'
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length (for truncation)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_config = PromptConfig()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("full_text", "")
        label = sample.get("label_id", 0)
        
        target_text = "NORMAL" if label == 0 else "MALICIOUS"
        
        # Truncate text first to leave room for the prompt template and target label
        text_tokens = self.tokenizer(text, truncation=True, max_length=self.max_seq_length - 300, add_special_tokens=False)["input_ids"]
        truncated_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        
        prompt = self.prompt_config.get_prompt(truncated_text)
        
        full_text = prompt + " " + target_text
        if self.tokenizer.eos_token:
            full_text += self.tokenizer.eos_token
            
        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize prompt to find where target starts
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        prompt_len = len(prompt_encoding["input_ids"])
        labels = encoding["input_ids"].copy()
        
        # Mask out prompt in labels with -100 so loss is only computed on target
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
            
        encoding["labels"] = labels
        return encoding


class SkillEmbeddingDataset(Dataset):
    """PyTorch Dataset for causal LLM classification head (Approach 1).
    Tokenizes the text and returns an integer label, with no generation prompts.
    """
    
    def __init__(self, samples: List[Dict], tokenizer, max_seq_length: int = 10000):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("full_text", "")
        label = int(sample.get("label_id", 0))
        
        # We just tokenize the text itself. DataCollatorWithPadding will pad it.
        # Note: padding_side="left" should ideally be used on tokenizer before this,
        # or DataCollator will pad on the right (which AutoModelForSequenceClassification 
        # handles correctly if pad_token_id is set).
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        # AutoModelForSequenceClassification expects "labels" as an integer/float for classification
        encoding["labels"] = label
        return encoding


if __name__ == "__main__":
    # Example usage
    samples = load_all_data("../raw_samples")
    print(f"Loaded {len(samples)} samples from directories")
    
    dataset = SkillDataset(samples)
    print(dataset.get_statistics())
    
    train, val, test = dataset.split_data()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
