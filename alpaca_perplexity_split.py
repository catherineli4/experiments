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


def format_alpaca_sample(sample: Dict) -> str:
    """Format an Alpaca sample into a text string for perplexity computation."""
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    # Format as instruction-following text
    if input_text:
        formatted_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
    else:
        formatted_text = f"Instruction: {instruction}\nOutput: {output}"
    
    return formatted_text


def get_answer(sample: Dict) -> str:
    answer = sample.get('output', '')
    return answer

def truncate_continutation(continuation, model, max_tokens) -> str:
    encoded_cont = model.tokenizer.encode(continuation)
    num_tokens = len(encoded_cont)
    if num_tokens > max_tokens:
        encoded_cont_truncated = encoded_cont[:max_tokens]
        return model.tokenizer.decode(encoded_cont_truncated)
    else:
        return continuation


def compute_perplexities_lm_eval(model, dataset: List[Dict]) -> List[Tuple[int, float]]:
    """Compute perplexity scores for all samples in the dataset using lm-eval-harness HFLM."""
    max_tokens = 2048
    answers = [get_answer(sample) for sample in dataset]
    requests = [
        Instance(
            arguments = (f"Instruction: {sample.get('instruction', '')}\nInput: {sample.get('input', '')}", f"Output: {truncate_continutation(sample.get('output', ''), model, max_tokens-8)}"),
            request_type= "loglikelihood",
            doc={},
            idx=i,
        )
        for i, sample in enumerate(dataset)
    ]
    print("Computing perplexity scores with lm-eval-harness HFLM...")

    log_likelihoods = [ll for ll, _ in model.loglikelihood(requests)]
    
    # Convert log likelihoods to perplexities
    perplexities = []
    for idx, (log_likelihood, text) in enumerate(zip(log_likelihoods, answers)):
        # Count tokens in the text
        tokens = model.tokenizer.encode(text)

        num_tokens = min(max_tokens, len(tokens))
        
        # Compute average log likelihood per token
        avg_log_likelihood = log_likelihood / num_tokens
        
        # Perplexity = exp(-avg_log_likelihood)
        perplexity = np.exp(-avg_log_likelihood)

        # print(f"Perplexity: {perplexity}")

        perplexities.append((idx, perplexity))
    
    return perplexities


def split_dataset_by_quartiles(dataset: List[Dict], perplexities: List[Tuple[int, float]]) -> Dict[str, List[Dict]]:
    """Split the dataset into 4 quartiles based on perplexity scores."""
    # Sort by perplexity (ascending order - lower is better)
    sorted_results = sorted(perplexities, key=lambda x: x[1], reverse=False)
    
    # Calculate quartile boundaries
    n_samples = len(sorted_results)
    q1_boundary = n_samples // 4
    q2_boundary = n_samples // 2
    q3_boundary = 3 * n_samples // 4
    
    # Split into quartiles
    quartiles = {
        'top_25': [],      # Top 25% (lowest perplexity)
        'q25_50': [],      # 25-50%
        'q50_75': [],      # 50-75%
        'bottom_25': []    # Bottom 25% (highest perplexity)
    }
    
    # Assign samples to quartiles
    for i, (idx, perplexity) in enumerate(sorted_results):
        sample = dataset[idx]
        sample_with_score = {
            **sample,
            'perplexity': perplexity,
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
        sorted_samples = sorted(samples, key=lambda x: x['perplexity'], reverse=False)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(sorted_samples)} samples to {output_file}")
        print(f"  Perplexity range: {sorted_samples[0]['perplexity']:.2f} to {sorted_samples[-1]['perplexity']:.2f}")


def print_statistics(quartiles: Dict[str, List[Dict]]):
    """Print statistics about the quartiles."""
    print("\n" + "="*50)
    print("QUARTILE STATISTICS")
    print("="*50)
    
    for quartile_name, samples in quartiles.items():
        if not samples:
            continue
        
        perplexities = [s['perplexity'] for s in samples]
        mean_perp = np.mean(perplexities)
        std_perp = np.std(perplexities)
        min_perp = np.min(perplexities)
        max_perp = np.max(perplexities)
        
        print(f"\n{quartile_name.upper()}:")
        print(f"  Number of samples: {len(samples)}")
        print(f"  Mean perplexity: {mean_perp:.2f}")
        print(f"  Std perplexity: {std_perp:.2f}")
        print(f"  Min perplexity: {min_perp:.2f}")
        print(f"  Max perplexity: {max_perp:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Split Alpaca dataset by perplexity quartiles (fast, using lm-eval-harness)")
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
        max_length=2048,
        use_fast_tokenizer=True,
        trust_remote_code=True,
    )

    # Compute perplexities (fast, robust)
    perplexities = compute_perplexities_lm_eval(model, dataset)

    # Split into quartiles
    quartiles = split_dataset_by_quartiles(dataset, perplexities)

    # Print statistics
    print_statistics(quartiles)

    # Save quartiles
    save_quartiles(quartiles, args.output_dir)

    print(f"\nProcessing complete! Quartiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main() 