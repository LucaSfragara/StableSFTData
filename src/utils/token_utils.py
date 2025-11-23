"""
Utility functions for counting tokens in datasets.
"""

import numpy as np
from typing import Dict, Any, Callable
import datasets
from transformers import PreTrainedTokenizer


def count_dataset_tokens(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizer,
    format_fn: Callable[[Any], Dict[str, str]],
) -> Dict[str, Any]:
    """
    Count total tokens in dataset after formatting.

    Args:
        dataset: The dataset to count tokens for
        tokenizer: The tokenizer to use for counting
        format_fn: Function that takes an example and returns {"text": formatted_string}

    Returns:
        Dictionary with token statistics:
            - total_tokens: Total tokens across all samples
            - mean_tokens: Mean tokens per sample
            - median_tokens: Median tokens per sample
            - min_tokens: Minimum tokens in a sample
            - max_tokens: Maximum tokens in a sample
            - num_samples: Number of samples
    """
    print(f"\nCounting tokens in dataset ({len(dataset)} samples)...")

    total_tokens = 0
    token_counts = []

    for example in dataset:
        formatted = format_fn(example)
        tokens = tokenizer(formatted["text"], return_tensors=None)
        num_tokens = len(tokens["input_ids"])
        token_counts.append(num_tokens)
        total_tokens += num_tokens

    token_stats = {
        "total_tokens": total_tokens,
        "mean_tokens": float(np.mean(token_counts)),
        "median_tokens": float(np.median(token_counts)),
        "min_tokens": int(min(token_counts)),
        "max_tokens": int(max(token_counts)),
        "num_samples": len(dataset),
    }

    print(f"Token statistics:")
    print(f"  Total tokens: {token_stats['total_tokens']:,}")
    print(f"  Mean tokens/sample: {token_stats['mean_tokens']:.1f}")
    print(f"  Median tokens/sample: {token_stats['median_tokens']:.1f}")
    print(f"  Min tokens: {token_stats['min_tokens']}")
    print(f"  Max tokens: {token_stats['max_tokens']}")

    return token_stats
