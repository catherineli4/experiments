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



def unformat_alpaca_sample(sample: Dict) -> str:
    """Format an Alpaca sample into a text string for perplexity computation."""
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    # Format as instruction-following text
    if input_text:
        formatted_text = f"{instruction}\n{input_text}\n{output}"
    else:
        formatted_text = f"{instruction}\n{output}"
    
    return formatted_text

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

def compute_ll_lm_metrics_format(model, dataset: List[Dict], mode: str) -> List[Tuple[int, float]]:
    """Compute perplexity scores for all samples in the dataset using lm-eval-harness HFLM."""

    formatted_dataset = [format_alpaca_sample(sample) for sample in dataset]

    print("Computing LL scores with lm_metrics...")

    log_likelihoods = log_likelihood_rolling(model, model.tokenizer, formatted_dataset, mode)
    
    return enumerate(log_likelihoods)

def compute_ll_lm_metrics_no_format(model, dataset: List[Dict], mode: str) -> List[Tuple[int, float]]:
    """Compute perplexity scores for all samples in the dataset using lm-eval-harness HFLM."""

    formatted_dataset = [unformat_alpaca_sample(sample) for sample in dataset]

    print("Computing LL scores with lm_metrics...")

    log_likelihoods = log_likelihood_rolling(model, model.tokenizer, formatted_dataset, mode)
    
    return enumerate(log_likelihoods)

def compute_ll_lm_eval(model, dataset: List[Dict]) -> List[Tuple[int, float]]:
    """Compute perplexity scores for all samples in the dataset using lm-eval-harness HFLM."""
    answers = [get_answer(sample) for sample in dataset]
    requests = [
        Instance(
            arguments = format_alpaca_sample(sample),
            request_type= "loglikelihood_rolling",
            doc={},
            idx=i,
        )
        for i, sample in enumerate(dataset)
    ]
    print("Computing LL scores with lm-eval-harness HFLM...")

    log_likelihoods = [ll for ll in model.loglikelihood_rolling(requests)]
    
    return enumerate(log_likelihoods)

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
def compute_mean_ll_lm_eval_masking(model, dataset: List[Dict]) -> List[Tuple[int, float]]:
    """Compute mean LL scores for all samples in the dataset using lm-eval-harness HFLM."""
    prompts = [format_alpaca_prompt(sample) for sample in dataset]
    continuations = [sample.get('output', '') for sample in dataset]

    print("Computing mean LL (mll) scores with lm_metrics...")
    log_likelihoods = log_likelihood_rolling(model, model.tokenizer, prompts, continuations, "mean")
    
    return enumerate(log_likelihoods)

def compute_token_lengths(model, dataset):

    formatted_dataset = [unformat_alpaca_sample(sample) for sample in dataset]

    lengths = [len(model.tokenizer.encode(sample)) for sample in formatted_dataset]

    return enumerate(lengths)

def print_statistics(scores: List[Tuple[int, float]], name: str):
    """Print statistics about the quartiles."""
    print("\n" + "="*50)
    print(f"{name} STATISTICS")
    print("="*50)
    
    for i, sample_ll in scores:
        print(f"  Sample {i} : {sample_ll}")



def main():
    parser = argparse.ArgumentParser(description="Split Alpaca dataset by perplexity quartiles (fast, using lm-eval-harness)")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/home/catheri4/experiments/alpaca_partition/alpaca_split_perplexity/datasets/alpaca_data_cleaned.json",
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

    ll_lm_metrics = compute_mean_ll_lm_eval_masking(model, dataset)

    print_statistics(ll_lm_metrics, "MEAN LL FOR SAMPLE CONTINUATION")




if __name__ == "__main__":
    main() 