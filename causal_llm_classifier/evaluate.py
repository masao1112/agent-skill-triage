"""
Evaluation Script for Causal LLM Classifier (with Reasoning Token & Token Efficiency Metrics & Chunking Support)

Computes:
  1. Classification Metrics: Accuracy, Precision, Recall, Macro F1, Micro F1, Confusion Matrix
  2. Consumed Token Statistics: prompt_tokens, output_tokens, reasoning_tokens, answer_tokens, total_tokens (min, max, mean)
  3. Compute Statistics: GFLOPs / TFLOPs based on parameter count and total tokens
  4. Latency Statistics: min, max, mean inference latency per sample
  5. Dynamic Chunking: Sliding window chunking for long skill texts
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import ModelConfig, PromptConfig, DataConfig, ChunkingConfig
from model import CausalLLMClassifier
from dataset import SkillDataset, load_all_data, load_data_from_csv


def compute_metrics_and_token_stats(results_detail: List[Dict]) -> Dict:
    """
    Compute classification metrics, token statistics (prompt, output, reasoning, answer),
    GFLOPs, and latency metrics across valid evaluation samples.
    """
    valid_results = [r for r in results_detail if r.get("pred_label") != -1]
    
    if not valid_results:
        print("[WARN] No valid results found for evaluation.")
        return {
            "total_samples": len(results_detail),
            "valid_samples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "confusion_matrix": [],
            "token_stats": {},
            "gflop_stats": {},
            "latency_stats": {}
        }
    
    y_true = [r["true_label"] for r in valid_results]
    y_pred = [r["pred_label"] for r in valid_results]

    # Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Consumed Token Statistics
    output_tokens = [r.get("output_tokens", 0) for r in valid_results]
    reasoning_tokens = [r.get("reasoning_tokens", 0) for r in valid_results]
    answer_tokens = [r.get("answer_tokens", r.get("output_tokens", 0)) for r in valid_results]
    prompt_tokens = [r.get("prompt_tokens", 0) for r in valid_results]
    total_tokens = [r.get("total_tokens", prompt + out) for r, prompt, out in zip(valid_results, prompt_tokens, output_tokens)]
    latencies = [r.get("latency_seconds", 0.0) for r in valid_results]

    # GFLOPs Statistics (FLOPs ≈ 2 * Num_Params_B * Total_Tokens)
    gflops_list = []
    for r, tot in zip(valid_results, total_tokens):
        gflops = r.get("gflops")
        if gflops is None:
            params_b = float(r.get("params_b", 1.0))
            gflops = 2.0 * params_b * tot
        gflops_list.append(float(gflops))

    token_stats = {
        "output_tokens_consumed": {
            "min": int(np.min(output_tokens)) if output_tokens else 0,
            "max": int(np.max(output_tokens)) if output_tokens else 0,
            "mean": float(np.mean(output_tokens)) if output_tokens else 0.0,
        },
        "reasoning_tokens_consumed": {
            "min": int(np.min(reasoning_tokens)) if reasoning_tokens else 0,
            "max": int(np.max(reasoning_tokens)) if reasoning_tokens else 0,
            "mean": float(np.mean(reasoning_tokens)) if reasoning_tokens else 0.0,
        },
        "answer_tokens_consumed": {
            "min": int(np.min(answer_tokens)) if answer_tokens else 0,
            "max": int(np.max(answer_tokens)) if answer_tokens else 0,
            "mean": float(np.mean(answer_tokens)) if answer_tokens else 0.0,
        },
        "prompt_tokens_consumed": {
            "min": int(np.min(prompt_tokens)) if prompt_tokens else 0,
            "max": int(np.max(prompt_tokens)) if prompt_tokens else 0,
            "mean": float(np.mean(prompt_tokens)) if prompt_tokens else 0.0,
        },
        "total_tokens_consumed": {
            "min": int(np.min(total_tokens)) if total_tokens else 0,
            "max": int(np.max(total_tokens)) if total_tokens else 0,
            "mean": float(np.mean(total_tokens)) if total_tokens else 0.0,
        }
    }

    gflop_stats = {
        "min_gflops": float(np.min(gflops_list)) if gflops_list else 0.0,
        "max_gflops": float(np.max(gflops_list)) if gflops_list else 0.0,
        "mean_gflops": float(np.mean(gflops_list)) if gflops_list else 0.0,
        "total_gflops": float(np.sum(gflops_list)) if gflops_list else 0.0,
        "total_tflops": float(np.sum(gflops_list) / 1000.0) if gflops_list else 0.0
    }

    latency_stats = {
        "min_latency_sec": float(np.min(latencies)) if latencies else 0.0,
        "max_latency_sec": float(np.max(latencies)) if latencies else 0.0,
        "mean_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
    }

    chunked_samples = [r for r in results_detail if r.get("was_chunked", False)]
    total_chunks = sum(r.get("num_chunks", 1) for r in results_detail)
    chunking_stats = {
        "chunked_samples_count": len(chunked_samples),
        "total_chunks_processed": total_chunks
    }

    return {
        "total_samples": len(results_detail),
        "valid_samples": len(valid_results),
        "accuracy": float(acc),
        "precision": float(prec_macro),
        "recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "micro_f1": float(f1_micro),
        "confusion_matrix": cm,
        "token_stats": token_stats,
        "gflop_stats": gflop_stats,
        "latency_stats": latency_stats,
        "chunking_stats": chunking_stats
    }


def evaluate_sample_single(classifier: CausalLLMClassifier, skill_text: str,
                           num_params_b: float = 1.0, max_new_tokens: int = 128,
                           prompt_config: Optional[PromptConfig] = None) -> Dict:
    """Runs inference on a single text chunk and extracts token metrics including reasoning tokens."""
    if prompt_config is None:
        prompt_config = PromptConfig()
    
    start_t = time.time()
    
    if classifier.config.approach == "embedding":
        inputs = classifier.tokenizer(
            skill_text,
            return_tensors="pt",
            truncation=True,
            max_length=classifier.config.max_seq_length
        ).to(classifier.device)
        
        prompt_tokens = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = classifier.model(**inputs)
            
        latency = time.time() - start_t
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
        
        output_tokens = 0
        reasoning_tokens = 0
        answer_tokens = 0
        total_tokens = prompt_tokens
        gflops = 2.0 * num_params_b * total_tokens
        raw_output = f"Logits: {logits.tolist()}"
    else:
        text_tokens = classifier.tokenizer(
            skill_text, truncation=True,
            max_length=classifier.config.max_seq_length - 300,
            add_special_tokens=False
        )["input_ids"]
        truncated_text = classifier.tokenizer.decode(text_tokens, skip_special_tokens=True)
        prompt = prompt_config.get_prompt(truncated_text)
        
        inputs = classifier.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=classifier.config.max_seq_length
        ).to(classifier.device)
        
        prompt_tokens = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            output_ids = classifier.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=classifier.tokenizer.pad_token_id,
                eos_token_id=classifier.tokenizer.eos_token_id
            )
        latency = time.time() - start_t
        
        generated_ids = output_ids[0][prompt_tokens:]
        output_tokens = len(generated_ids)
        total_tokens = prompt_tokens + output_tokens
        gflops = 2.0 * num_params_b * total_tokens
        
        raw_output = classifier.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        reasoning_tokens = 0
        answer_tokens = output_tokens
        if "<think>" in raw_output and "</think>" in raw_output:
            try:
                think_content = raw_output.split("</think>")[0] + "</think>"
                reasoning_tokens = len(classifier.tokenizer.encode(think_content, add_special_tokens=False))
                answer_tokens = max(0, output_tokens - reasoning_tokens)
            except Exception:
                reasoning_tokens = 0
                answer_tokens = output_tokens

        upper_out = raw_output.upper()
        if "MALICIOUS" in upper_out:
            pred_label = 1
            confidence = 0.9
        elif "NORMAL" in upper_out:
            pred_label = 0
            confidence = 0.9
        else:
            pred_label = -1
            confidence = 0.5

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "prediction_text": "MALICIOUS" if pred_label == 1 else ("NORMAL" if pred_label == 0 else "UNKNOWN"),
        "raw_output": raw_output,
        "latency_seconds": latency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": total_tokens,
        "gflops": gflops,
        "params_b": num_params_b
    }


def evaluate_sample(classifier: CausalLLMClassifier, skill_text: str,
                    num_params_b: float = 1.0, max_new_tokens: int = 128,
                    chunking_config: Optional[ChunkingConfig] = None) -> Dict:
    """
    Runs evaluation on a skill sample, using sliding-window chunking if the text
    exceeds context limits / chunk_size and chunking is enabled.
    """
    if chunking_config is None:
        chunking_config = ChunkingConfig()

    prompt_config = PromptConfig()
    
    total_tokens = len(classifier.tokenizer(
        skill_text, truncation=False, add_special_tokens=False
    )["input_ids"])
    
    max_content_len = classifier.config.max_seq_length - 300 if classifier.config.approach != "embedding" else classifier.config.max_seq_length
    effective_limit = min(chunking_config.chunk_size, max_content_len) if chunking_config.enabled else max_content_len

    if total_tokens <= effective_limit or not chunking_config.enabled:
        res = evaluate_sample_single(classifier, skill_text, num_params_b, max_new_tokens, prompt_config)
        res["num_chunks"] = 1
        res["was_chunked"] = False
        return res

    chunks = classifier._create_chunks(
        skill_text,
        chunk_size=chunking_config.chunk_size,
        overlap=chunking_config.chunk_overlap,
        min_chunk_tokens=chunking_config.min_chunk_tokens
    )
    chunk_results = []
    for chunk_text in chunks:
        c_res = evaluate_sample_single(classifier, chunk_text, num_params_b, max_new_tokens, prompt_config)
        chunk_results.append(c_res)

    if chunking_config.aggregation == "max":
        mal_chunks = [c for c in chunk_results if c["pred_label"] == 1]
        if mal_chunks:
            best_mal = max(mal_chunks, key=lambda c: c.get("confidence", 0.5))
            final_label = 1
            final_confidence = best_mal["confidence"]
        else:
            norm_chunks = [c for c in chunk_results if c["pred_label"] == 0]
            if norm_chunks:
                best_norm = max(norm_chunks, key=lambda c: c.get("confidence", 0.5))
                final_label = 0
                final_confidence = best_norm["confidence"]
            else:
                final_label = -1
                final_confidence = 0.5
    elif chunking_config.aggregation == "mean":
        mal_scores = []
        for c in chunk_results:
            if c.get("pred_label") == 1:
                mal_scores.append(c.get("confidence", 0.5))
            elif c.get("pred_label") == 0:
                mal_scores.append(1.0 - c.get("confidence", 0.5))
            else:
                mal_scores.append(0.5)
        
        avg_mal_score = sum(mal_scores) / len(mal_scores)
        final_label = 1 if avg_mal_score > 0.5 else 0
        final_confidence = avg_mal_score if final_label == 1 else (1.0 - avg_mal_score)
    else:
        raise ValueError(f"Unknown aggregation strategy: {chunking_config.aggregation}")

    total_prompt_tokens = sum(c["prompt_tokens"] for c in chunk_results)
    total_output_tokens = sum(c["output_tokens"] for c in chunk_results)
    total_reasoning_tokens = sum(c["reasoning_tokens"] for c in chunk_results)
    total_answer_tokens = sum(c["answer_tokens"] for c in chunk_results)
    total_sample_tokens = sum(c["total_tokens"] for c in chunk_results)
    total_latency = sum(c["latency_seconds"] for c in chunk_results)
    total_gflops = sum(c["gflops"] for c in chunk_results)
    raw_outputs = " | ".join(c["raw_output"] for c in chunk_results)

    return {
        "pred_label": final_label,
        "confidence": final_confidence,
        "prediction_text": "MALICIOUS" if final_label == 1 else ("NORMAL" if final_label == 0 else "UNKNOWN"),
        "raw_output": raw_outputs,
        "latency_seconds": total_latency,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "reasoning_tokens": total_reasoning_tokens,
        "answer_tokens": total_answer_tokens,
        "total_tokens": total_sample_tokens,
        "gflops": total_gflops,
        "params_b": num_params_b,
        "num_chunks": len(chunks),
        "was_chunked": True,
        "chunk_details": chunk_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Causal LLM Classifier on Malicious Skills Dataset")
    parser.add_argument("--model_name", type=str, default="llama3.2", choices=["llama3.2", "qwen3"],
                        help="Model key choice from ModelConfig")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Custom HuggingFace Model ID (overrides --model_name)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to trained LoRA adapter directory (e.g. ./checkpoints)")
    parser.add_argument("--data_dir", type=str, default="../raw_samples",
                        help="Directory containing raw skill samples")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Optional CSV file path for dataset")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"],
                        help="Dataset split to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens allowed during generation")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization")
    parser.add_argument("--approach", type=str, default="instruction", choices=["instruction", "embedding"],
                        help="Classification approach: 'instruction' (prompt-based) or 'embedding' (sequence classification head)")
    # Chunking options
    parser.add_argument("--chunk_size", type=int, default=8000,
                        help="Max tokens per chunk for long skills (default: 8000)")
    parser.add_argument("--chunk_overlap", type=int, default=1000,
                        help="Token overlap between consecutive chunks (default: 1000)")
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "mean"],
                        help="Chunk aggregation strategy: 'max' (any malicious=malicious) or 'mean'")
    parser.add_argument("--no_chunking", action="store_true", default=False,
                        help="Disable chunking and truncate long skills instead")
    args = parser.parse_args()

    # Model Configuration
    model_config = ModelConfig()
    model_config.model_name = args.model_name
    if args.model_id:
        model_config.model_id = args.model_id
    elif args.model_name in model_config.MODEL_OPTIONS:
        model_config.model_id = model_config.MODEL_OPTIONS[args.model_name]

    model_config.load_in_4bit = args.load_in_4bit
    model_config.load_in_8bit = args.load_in_8bit
    model_config.approach = args.approach
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunking_config = ChunkingConfig(
        enabled=not args.no_chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        aggregation=args.aggregation
    )

    print(f"=== Initializing Model Evaluation ===")
    print(f"Model ID: {model_config.model_id}")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Chunking Enabled: {chunking_config.enabled} (chunk_size={chunking_config.chunk_size}, overlap={chunking_config.chunk_overlap}, agg={chunking_config.aggregation})")

    classifier = CausalLLMClassifier(model_config, device=device)
    classifier.load_model()

    if args.adapter_path and os.path.exists(args.adapter_path):
        classifier.load_adapter(args.adapter_path)

    # Estimate parameters in Billions
    try:
        total_params = sum(p.numel() for p in classifier.model.parameters())
        num_params_b = total_params / 1e9
    except Exception:
        num_params_b = 1.0

    # Load Dataset
    if args.csv_path and os.path.exists(args.csv_path):
        samples = load_data_from_csv(args.csv_path)
    else:
        samples = load_all_data(args.data_dir)

    if not samples:
        print(f"[ERROR] No samples loaded from {args.csv_path or args.data_dir}")
        sys.exit(1)

    print(f"Loaded {len(samples)} total samples.")

    # Split dataset
    dataset_handler = SkillDataset(samples)
    train_s, val_s, test_s = dataset_handler.split_data()

    if args.split == "test":
        eval_samples = test_s
    elif args.split == "val":
        eval_samples = val_s
    elif args.split == "train":
        eval_samples = train_s
    else:
        eval_samples = samples

    print(f"Evaluating {len(eval_samples)} samples on split='{args.split}'...")

    results_detail = []
    for idx, sample in enumerate(eval_samples, 1):
        true_label = sample["label_id"]
        skill_text = sample.get("full_text", sample.get("body", ""))
        sample_id = sample.get("id", f"sample_{idx}")

        pred_res = evaluate_sample(
            classifier,
            skill_text,
            num_params_b=num_params_b,
            max_new_tokens=args.max_new_tokens,
            chunking_config=chunking_config
        )

        record = {
            "sample_id": sample_id,
            "true_label": true_label,
            "true_text": "MALICIOUS" if true_label == 1 else "NORMAL",
            "pred_label": pred_res["pred_label"],
            "prediction_text": pred_res["prediction_text"],
            "raw_output": pred_res["raw_output"],
            "latency_seconds": pred_res["latency_seconds"],
            "prompt_tokens": pred_res["prompt_tokens"],
            "output_tokens": pred_res["output_tokens"],
            "reasoning_tokens": pred_res["reasoning_tokens"],
            "answer_tokens": pred_res["answer_tokens"],
            "total_tokens": pred_res["total_tokens"],
            "gflops": pred_res["gflops"],
            "params_b": num_params_b,
            "num_chunks": pred_res.get("num_chunks", 1),
            "was_chunked": pred_res.get("was_chunked", False)
        }
        results_detail.append(record)

        chunk_tag = f" (chunked: {record['num_chunks']} chunks)" if record["was_chunked"] else ""
        print(f"[{idx}/{len(eval_samples)}] Sample: {sample_id} | True: {record['true_text']} | Pred: {record['prediction_text']} | Out Tokens: {record['output_tokens']} (Reasoning: {record['reasoning_tokens']}){chunk_tag}")

    # Compute Summary Statistics
    summary = compute_metrics_and_token_stats(results_detail)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Precision (Macro): {summary['precision']:.4f}")
    print(f"Recall (Macro): {summary['recall']:.4f}")
    print(f"Macro F1 Score: {summary['macro_f1']:.4f}")
    print(f"Micro F1 Score: {summary['micro_f1']:.4f}")
    print(f"Confusion Matrix: {summary['confusion_matrix']}")
    print("-" * 50)
    print(f"Chunked Samples Count: {summary['chunking_stats']['chunked_samples_count']}")
    print(f"Total Chunks Processed: {summary['chunking_stats']['total_chunks_processed']}")
    print(f"Output Tokens (Mean): {summary['token_stats']['output_tokens_consumed']['mean']:.2f}")
    print(f"Reasoning Tokens (Mean): {summary['token_stats']['reasoning_tokens_consumed']['mean']:.2f}")
    print(f"Answer Tokens (Mean): {summary['token_stats']['answer_tokens_consumed']['mean']:.2f}")
    print(f"Prompt Tokens (Mean): {summary['token_stats']['prompt_tokens_consumed']['mean']:.2f}")
    print(f"Total Tokens (Mean): {summary['token_stats']['total_tokens_consumed']['mean']:.2f}")
    print(f"Mean Latency: {summary['latency_stats']['mean_latency_sec']:.3f} s")
    print(f"Total GFLOPs: {summary['gflop_stats']['total_gflops']:.2f}")
    print("=" * 50)

    # Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    detail_path = os.path.join(args.output_dir, "results_detail.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results_detail, f, indent=2)

    print(f"Saved summary to {summary_path}")
    print(f"Saved detail to {detail_path}")


if __name__ == "__main__":
    main()
