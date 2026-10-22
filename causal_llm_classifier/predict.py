"""Inference script for causal LLM classifier"""

import argparse
import os
import glob
from pathlib import Path
from typing import List, Dict
import torch
from model import CausalLLMClassifier
from config import ModelConfig, PromptConfig, ChunkingConfig


def predict_single_skill(classifier: CausalLLMClassifier, skill_dir: str,
                         chunking_config: ChunkingConfig = None) -> Dict:
    """
    Predict for a single skill directory
    
    Args:
        classifier: CausalLLMClassifier instance
        skill_dir: Path to skill directory containing SKILL.md
        chunking_config: Chunking parameters for long skills
        
    Returns:
        Prediction result
    """
    skill_file = os.path.join(skill_dir, "SKILL.md")
    
    if not os.path.exists(skill_file):
        return {"error": f"SKILL.md not found in {skill_dir}"}
    
    try:
        # Read ALL files recursively, matching the notebook's format exactly:
        #   ### FILE: {rel_path} ###\n{text}
        # This ensures train/inference consistency since the model was
        # trained on data produced by process_skills.ipynb using this format.
        skill_path = Path(skill_dir)
        all_texts = []
        for root, dirs, files in os.walk(skill_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in sorted(files):
                if fname.startswith('.'):
                    continue
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(skill_path)
                try:
                    text_content = fpath.read_text(encoding="utf-8", errors="ignore")
                    all_texts.append(f"### FILE: {rel_path} ###\n{text_content}")
                except Exception:
                    all_texts.append(f"### FILE: {rel_path} ### [BINARY/UNREADABLE]")
        
        full_text = "\n\n".join(all_texts)
        
        # Use chunked classification — handles both short and long skills
        result = classifier.classify_chunked(full_text, chunking_config)
        result["skill_dir"] = skill_dir
        
        label_name = "MALICIOUS" if result["label"] == 1 else "NORMAL"
        chunked_tag = f" (chunked: {result['num_chunks']} chunks)" if result.get("was_chunked") else ""
        print(f"[{label_name}] {skill_dir}: confidence={result['confidence']:.2f}{chunked_tag}")
        
        return result
    
    except Exception as e:
        return {"error": str(e), "skill_dir": skill_dir}


def predict_batch_skills(classifier: CausalLLMClassifier, batch_dir: str,
                         chunking_config: ChunkingConfig = None) -> List[Dict]:
    """
    Predict for a batch of skills
    
    Args:
        classifier: CausalLLMClassifier instance
        batch_dir: Directory containing skill subdirectories
        chunking_config: Chunking parameters for long skills
        
    Returns:
        List of prediction results
    """
    results = []
    
    # Find all SKILL.md files
    skill_dirs = []
    for root, dirs, files in os.walk(batch_dir):
        if "SKILL.md" in files:
            skill_dirs.append(root)
    
    print(f"Found {len(skill_dirs)} skills to classify")
    
    for i, skill_dir in enumerate(skill_dirs, 1):
        result = predict_single_skill(classifier, skill_dir, chunking_config)
        results.append(result)
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(skill_dirs)}")
    
    return results


def evaluate_results(results: List[Dict]) -> Dict:
    """Evaluate classification results"""
    malicious_count = sum(1 for r in results if r.get("label") == 1)
    normal_count = sum(1 for r in results if r.get("label") == 0)
    unknown_count = sum(1 for r in results if r.get("label") == -1)
    error_count = sum(1 for r in results if "error" in r)
    chunked_count = sum(1 for r in results if r.get("was_chunked", False))
    total_chunks = sum(r.get("num_chunks", 1) for r in results if "error" not in r)
    
    valid_results = [r for r in results if "error" not in r]
    avg_confidence = sum(r.get("confidence", 0.5) for r in valid_results) / len(valid_results) if valid_results else 0
    
    return {
        "total": len(results),
        "malicious": malicious_count,
        "normal": normal_count,
        "unknown": unknown_count,
        "errors": error_count,
        "avg_confidence": avg_confidence,
        "chunked_skills": chunked_count,
        "total_chunks_processed": total_chunks,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify skills using causal LLM")
    parser.add_argument("--model", type=str, default="llama3.2", 
                       choices=list(ModelConfig.MODEL_OPTIONS.keys()),
                       help="Model to use")
    parser.add_argument("--skill_dir", type=str, default=None,
                       help="Single skill directory to classify")
    parser.add_argument("--batch_dir", type=str, default=None,
                       help="Batch directory containing skills")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--approach", type=str, default="instruction", choices=["embedding", "instruction"],
                       help="Approach to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (JSON)")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="Path to trained LoRA adapter (e.g. /content/drive/MyDrive/causal_llm_checkpoints/final)")
    
    # Chunking arguments
    parser.add_argument("--chunk_size", type=int, default=8000,
                       help="Max tokens per chunk for long skills (default: 8000)")
    parser.add_argument("--chunk_overlap", type=int, default=1000,
                       help="Token overlap between consecutive chunks (default: 1000)")
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "mean"],
                       help="Chunk aggregation strategy: 'max' (any malicious=malicious) or 'mean' (average)")
    parser.add_argument("--no_chunking", action="store_true",
                       help="Disable chunking (truncate long skills as before)")
    args = parser.parse_args()
    
    # Setup model
    model_config = ModelConfig()
    model_config.model_name = args.model
    model_config.model_id = ModelConfig.MODEL_OPTIONS[args.model]
    model_config.device = args.device
    model_config.approach = args.approach
    
    classifier = CausalLLMClassifier(model_config, device=args.device)
    classifier.load_model()
    
    if args.adapter_path:
        classifier.load_adapter(args.adapter_path)
    
    # Setup chunking config
    chunking_config = ChunkingConfig(
        enabled=not args.no_chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        aggregation=args.aggregation,
    )
    
    if chunking_config.enabled:
        print(f"Chunking: ENABLED (size={chunking_config.chunk_size}, "
              f"overlap={chunking_config.chunk_overlap}, agg={chunking_config.aggregation})")
    else:
        print("Chunking: DISABLED (long skills will be truncated)")
    
    # Run predictions
    if args.skill_dir:
        print(f"Classifying single skill: {args.skill_dir}")
        result = predict_single_skill(classifier, args.skill_dir, chunking_config)
        results = [result]
    
    elif args.batch_dir:
        print(f"Classifying batch: {args.batch_dir}")
        results = predict_batch_skills(classifier, args.batch_dir, chunking_config)
    
    else:
        print("Please provide --skill_dir or --batch_dir")
        return
    
    # Evaluate and print results
    evaluation = evaluate_results(results)
    print("\n" + "="*50)
    print("Classification Results:")
    print(f"Total: {evaluation['total']}")
    print(f"Malicious: {evaluation['malicious']}")
    print(f"Normal: {evaluation['normal']}")
    print(f"Unknown: {evaluation['unknown']}")
    print(f"Errors: {evaluation['errors']}")
    print(f"Average Confidence: {evaluation['avg_confidence']:.3f}")
    print(f"Skills chunked: {evaluation['chunked_skills']}")
    print(f"Total chunks processed: {evaluation['total_chunks_processed']}")
    print("="*50)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump({
                "evaluation": evaluation,
                "chunking_config": {
                    "enabled": chunking_config.enabled,
                    "chunk_size": chunking_config.chunk_size,
                    "chunk_overlap": chunking_config.chunk_overlap,
                    "aggregation": chunking_config.aggregation,
                },
                "results": results
            }, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
