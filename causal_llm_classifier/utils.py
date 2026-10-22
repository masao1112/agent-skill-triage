"""Utility functions for causal LLM classifier"""

import os
import torch
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def evaluate_classification(predictions: List[int], ground_truth: List[int]) -> Dict:
    """
    Evaluate classification results
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions, zero_division=0),
        "recall": recall_score(ground_truth, predictions, zero_division=0),
        "f1": f1_score(ground_truth, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(ground_truth, predictions).tolist(),
    }


def get_device() -> str:
    """Get appropriate device (cuda/cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_model_size(model) -> Dict:
    """Print model size statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "ratio": 100 * trainable_params / total_params,
    }


def save_predictions(predictions: List[Dict], output_file: str):
    """Save predictions to file"""
    import json
    
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_file}")


def load_predictions(input_file: str) -> List[Dict]:
    """Load predictions from file"""
    import json
    
    with open(input_file, "r") as f:
        predictions = json.load(f)
    
    return predictions
