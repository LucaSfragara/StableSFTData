import datasets
import re
from src.core.types import Sample, ScoreRecord
from typing import List

def _extract_label(answer:str) -> str:
    
    """
    Extract the final answer label from the answer string.
    Args:
        answer (str): Full answer string including reasoning and final answer
    Returns:
        str: The extracted final answer label   
    """
    
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer.strip())
    return m.group(1) if m else answer.strip()

def _extract_reasoning(answer: str) -> str:
    """
    Extract the reasoning part from the answer (everything before ####).
    
    Args:
        answer (str): Full answer string including reasoning and final answer
    
    Returns:
        str: The reasoning part of the answer
    """
    m = re.split(r"####", answer.strip())[0].strip()
    return m


def load_gsm8k_dataset(split) -> List[Sample]:
    """
    Load the GSM8K dataset from the Hugging Face datasets library.

    Returns:
        DatasetDict: A dictionary-like object containing the GSM8K dataset splits.
    """
    dataset_raw = datasets.load_dataset("gsm8k", "main")[split]
    print(f"Loaded GSM8K dataset with {len(dataset_raw)} examples in the '{split}' split.")

    dataset = [
        Sample(
            id=i,
            question=example["question"],
            reasoning=_extract_reasoning(example["answer"]),
            answer=_extract_label(example["answer"])
        )
        for i, example in enumerate(dataset_raw)
    ]

    return dataset

if __name__ == "__main__":
    train_data = load_gsm8k_dataset("train")
    print(f"First training example:\n{train_data[0]}")