import datasets
import re

def _extract_label(answer:str) -> str:
    
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer.strip())
    return m.group(1) if m else answer.strip()

def load_gsm8k_dataset(split):
    """
    Load the GSM8K dataset from the Hugging Face datasets library.

    Returns:
        DatasetDict: A dictionary-like object containing the GSM8K dataset splits.
    """
    dataset = datasets.load_dataset("gsm8k", "main")[split]
    print(f"Loaded GSM8K dataset with {len(dataset)} examples in the '{split}' split.")
    
    
    
    return dataset