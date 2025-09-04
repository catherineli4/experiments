#!/usr/bin/env python3
"""
Script to compute perplexity scores for Alpaca dataset using OLMo2 1B HF model
and split the dataset into 4 quartiles based on those scores.
Uses lm-evaluation-harness for efficient perplexity computation.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import os
import sys 

sys.path.append("/home/catheri4")
from utils.lm_metrics.lm_metrics import log_likelihood_rolling
# Import HFLM from lm-evaluation-harness
try:
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
except ImportError:
    raise ImportError(
        "Could not import lm_eval.models.huggingface. Please install lm-evaluation-harness: pip install lm-eval"
    )

import torch  # For device check


def load_alpaca_dataset(file_path: str) -> List[Dict]:
    """Load the Alpaca dataset from JSON file."""
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def format_alpaca_prompt(sample: Dict) -> str:
    """Format an Alpaca sample into a text string for perplexity computation."""
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    
    # Format as instruction-following text
    if input_text:
        formatted_text = f"{instruction}\n{input_text}\n"
    else:
        formatted_text = f"{instruction}\n"
    
    return formatted_text


#MEAN LL
def compute_mean_ll_lm_eval(model, dataset: List[Dict]) -> List[Tuple[int, float]]:
    """Compute mean LL scores for all samples in the dataset using lm-eval-harness HFLM."""
    prompts = [format_alpaca_prompt(sample) for sample in dataset]
    continuations = [sample.get('output', '') for sample in dataset]

    print("Computing mean LL (mll) scores with lm_metrics...")
    log_likelihoods = log_likelihood_rolling(model, model.tokenizer, prompts, continuations, "mean")
    
    return enumerate(log_likelihoods)


def split_dataset_by_quartiles(dataset: List[Dict], mean_ll: List[Tuple[int, float]]) -> Dict[str, List[Dict]]:
    """Split the dataset into 4 quartiles based on perplexity scores."""
    sorted_results = sorted(mean_ll, key=lambda x: x[1], reverse=True)
    
    # Calculate quartile boundaries
    n_samples = len(sorted_results)
    q1_boundary = n_samples // 4
    q2_boundary = n_samples // 2
    q3_boundary = 3 * n_samples // 4
    
    # Split into quartiles
    quartiles = {
        'top_25': [],      # Top 25% (highest mean LL)
        'q25_50': [],      # 25-50%
        'q50_75': [],      # 50-75%
        'bottom_25': []    # Bottom 25% (lowest mean LL)
    }
    
    # Assign samples to quartiles
    for i, (idx, mean_ll) in enumerate(sorted_results):
        sample = dataset[idx]
        sample_with_score = {
            **sample,
            'mean loglikelihood': mean_ll,
            'rank': i + 1
        }
        
        if i < q1_boundary:
            quartiles['top_25'].append(sample_with_score)
        elif i < q2_boundary:
            quartiles['q25_50'].append(sample_with_score)
        elif i < q3_boundary:
            quartiles['q50_75'].append(sample_with_score)
        else:
            quartiles['bottom_25'].append(sample_with_score)
    
    return quartiles


def save_quartiles(quartiles: Dict[str, List[Dict]], output_dir: str):
    """Save each quartile to a separate JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    for quartile_name, samples in quartiles.items():
        output_file = os.path.join(output_dir, f"{quartile_name}.json")
        
        # Sort by perplexity within each quartile (ascending - lower is better)
        sorted_samples = sorted(samples, key=lambda x: x['mean loglikelihood'], reverse=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(sorted_samples)} samples to {output_file}")
        print(f"  Mean loglikelihood range: {sorted_samples[0]['mean loglikelihood']:.2f} to {sorted_samples[-1]['mean loglikelihood']:.2f}")


def print_statistics(quartiles: Dict[str, List[Dict]]):
    """Print statistics about the quartiles."""
    print("\n" + "="*50)
    print("QUARTILE STATISTICS")
    print("="*50)
    
    for quartile_name, samples in quartiles.items():
        if not samples:
            continue
        
        mean_loglikelihood = [s['mean loglikelihood'] for s in samples]
        mean_mll = np.mean(mean_loglikelihood)
        std_mll = np.std(mean_loglikelihood)
        min_mll = np.min(mean_loglikelihood)
        max_mll = np.max(mean_loglikelihood)
        
        print(f"\n{quartile_name.upper()}:")
        print(f"  Number of samples: {len(samples)}")
        print(f"  Mean MLL: {mean_mll:.2f}")
        print(f"  Std MLL: {std_mll:.2f}")
        print(f"  Min MLL: {min_mll:.2f}")
        print(f"  Max MLL: {max_mll:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Split Alpaca dataset by mean LL quartiles (fast, using lm-eval-harness)")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="datasets/alpaca_data_cleaned.json",
        help="Path to the Alpaca dataset JSON file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="allenai/OLMo-2-0425-1B",
        help="HuggingFace model name for OLMo2 1B"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="datasets/alpaca_quartiles",
        help="Output directory for quartile files"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference (increase for faster GPU inference)"
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_alpaca_dataset(args.dataset_path)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"Limited to {len(dataset)} samples for testing")

    # Load HFLM model (from lm-eval-harness)
    print(f"Loading OLMo model with lm-eval-harness HFLM: {args.model_name}")
    model = HFLM(
        pretrained=args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        use_fast_tokenizer=True,
        trust_remote_code=True,
    )

    # Compute perplexities (fast, robust)
    mll = compute_mean_ll_lm_eval(model, dataset)

    # Split into quartiles
    quartiles = split_dataset_by_quartiles(dataset, mll)

    # Print statistics
    print_statistics(quartiles)

    # Save quartiles
    save_quartiles(quartiles, args.output_dir)

    print(f"\nProcessing complete! Quartiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main() 