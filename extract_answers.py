import os
import re
import argparse
from typing import Dict, Any, List, Union
from datasets import load_dataset, DatasetDict
from datasets.config import HF_DATASETS_CACHE
from src.utils.parser import Parser

# Extract final answers from GSM8K and save a processed dataset in the HF cache.

def extract_final_answer(text: str) -> float:
    answer = Parser.extract_true_number(text)
    return answer

def map_example(example: Dict[str, Any], source_col: str, target_col: str) -> Dict[str, Any]:
    src = example.get(source_col, "")

    return {target_col: extract_final_answer(src)}


def process_dataset(
    dataset_name: str = "gsm8k",
    config_name: str = "main",
    source_col: str = "answer",
    target_col: str = "answer_num",
    num_proc: int = 1,
) -> DatasetDict:
    
    ds = load_dataset(dataset_name, config_name)
    # Replace/insert target_col with extracted answer
    mapped = ds.map(
        lambda ex: map_example(ex, source_col, target_col),
        desc=f"Extracting final answers into '{target_col}'",
        num_proc=num_proc if num_proc and num_proc > 1 else None,
    )
    return mapped


def save_to_hf_cache(ds: DatasetDict, dataset_name: str, config_name: str, suffix: str = "answers") -> str:
    out_dir = os.path.join(
        HF_DATASETS_CACHE,
        "processed",
        f"{dataset_name}_{config_name}_{suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Extract final answers from GSM8K and save in HF cache.")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="HF dataset name")
    parser.add_argument("--config", type=str, default="main", help="HF dataset config (e.g., main or socratic)")
    parser.add_argument("--source-col", type=str, default="answer", help="Source column containing rationale/solution")
    parser.add_argument("--target-col", type=str, default="answer_num", help="Target column name for extracted answer")
    parser.add_argument("--num-proc", type=int, default=0, help="Parallel workers for map (0/1 = no parallelism)")
    parser.add_argument("--suffix", type=str, default="with_num_answer", help="Suffix for saved dataset directory")
    args = parser.parse_args()

    num_proc = args.num_proc if args.num_proc and args.num_proc > 1 else None
    ds = process_dataset(
        dataset_name=args.dataset,
        config_name=args.config,
        source_col=args.source_col,
        target_col=args.target_col,
        num_proc=num_proc,
    )
    out_dir = save_to_hf_cache(ds, args.dataset, args.config, args.suffix)
    print(f"Saved processed dataset to: {out_dir}")
    for split, d in ds.items():
        print(f"{split}: {len(d)} rows; columns={d.column_names}")

if __name__ == "__main__":
    main()