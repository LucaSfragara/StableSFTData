# Force transformers to not use TensorFlow to avoid slow imports/conflicts
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse

import sys
from typing import Optional, Tuple
from datasets import load_from_disk, Dataset, DatasetDict
import matplotlib.pyplot as plt
from pprint import pprint
from src.scoring.scorer import Scorer
import numpy as np
from src.train.data_selector import FullDataSelector, RandomDataSelector, ThresholdDataSelector, TopKDataSelector


def is_hf_dataset_dir(path: str) -> Optional[bool]:
    if not os.path.isdir(path):
        return None
    # A saved Dataset has dataset_info.json; a DatasetDict has dataset_dict.json
    has_info = os.path.isfile(os.path.join(path, "dataset_info.json"))
    has_dict = os.path.isfile(os.path.join(path, "dataset_dict.json"))
    return has_info or has_dict


def find_first_dataset_dir(base_path: str) -> Optional[str]:
    if is_hf_dataset_dir(base_path):
        return base_path
    # Search direct subdirectories for a dataset dir
    try:
        for name in sorted(os.listdir(base_path)):
            sub = os.path.join(base_path, name)
            if is_hf_dataset_dir(sub):
                return sub
    except FileNotFoundError:
        return None
    return None


def load_local_dataset(path: str):
    try:
        ds = load_from_disk(path)
        return ds
    except Exception as e:
        print(f"Failed to load dataset at: {path}\n{e}", file=sys.stderr)
        return None


def summarize_dataset(ds: Dataset) -> None:
    if isinstance(ds, DatasetDict):
        print("Loaded DatasetDict")
        print(f"Splits: {', '.join([f'{k}({len(v)})' for k, v in ds.items()])}")
    elif isinstance(ds, Dataset):
        print("Loaded Dataset")
        print(f"Rows: {len(ds)}")
    else:
        print(f"Unknown dataset type: {type(ds)}")
    
    if "accuracy" in ds.features:
        print("Mean Accuracy: ", np.mean(ds["accuracy"]))
        
    print("Average length of answers: ", np.mean([len(str(a).split()) for a in ds["answer"]]))
   
def print_features(ds: Dataset) -> None:
    try:
        feats = ds.features
        print("Features:")
        for name, f in feats.items():
            dtype = getattr(f, "dtype", str(f))
            print(f"  - {name}: {dtype}")
    except Exception:
        print(f"Columns: {list(ds.column_names)}")


def show_rows(ds: Dataset, n: int) -> None:
    
    n = max(0, min(n, len(ds)))
    if n == 0:
        print("No rows to display.")
        return
    print(f"First {n} row(s):")
    
    for i in range(n):
        row = ds[i]
        
        pprint(row, compact=True, sort_dicts=False)


def main():
    parser = argparse.ArgumentParser(description="Explore a Hugging Face dataset saved to disk.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the saved dataset directory (defaults to ./results or the first dataset inside it).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Which split to display (e.g., train/test/validation). For DatasetDict only.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to show from the selected split or dataset.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(script_dir, "results")

    base_path = args.path if args.path else default_results
    dataset_dir = "/orcd/home/002/lsfragar/orcd/pool/gsm8k_scored_20251119_001339"

    if dataset_dir is None:
        print(f"Could not find a saved HF dataset under: {base_path}")
        print("Tip: pass --path /full/path/to/your/saved_dataset")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_dir}")
    ds = load_local_dataset(dataset_dir)
    if ds is None:
        sys.exit(2)

    # Save histogram of accuracy scores
    if "accuracy" in ds.features:
        plt.figure()
        plt.hist(ds["accuracy"], bins=20)
        plt.title("Accuracy Score Distribution")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.grid()
        plt.savefig("accuracy_histogram.png")
        plt.close()

    data_selector = TopKDataSelector(
        score_column="accuracy",
        k=7000,
        ascending=False,
        seed=42,
    )

    ds_selected = data_selector.select_data(ds) #type: ignore
    #ds_selected = ds
    summarize_dataset(ds_selected)

    if isinstance(ds_selected, DatasetDict):
        # Pick split
        split_to_show = None
        if args.split:
            if args.split in ds_selected:
                split_to_show = args.split
            else:
                print(f"Requested split '{args.split}' not found. Available: {list(ds_selected.keys())}")
                # Fallback to 'train' or first
        if split_to_show is None:
            split_to_show = "train" if "train" in ds_selected else next(iter(ds_selected.keys()))
        print(f"Showing split: {split_to_show} (rows={len(ds_selected[split_to_show])})")
        print_features(ds_selected[split_to_show])
        show_rows(ds_selected[split_to_show], args.rows)
    else:
        print_features(ds_selected)
        #show_rows(ds_selected, args.rows)


if __name__ == "__main__":
    main()