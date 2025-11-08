from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
import datasets 

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
BENCHMARKS = "gsm8k"

dataset = datasets.load_dataset("gsm8k", "main")

# Print basic dataset info
print("\nDataset Structure:")
print(dataset)

# Show available splits
print("\nAvailable Splits:")
for split in dataset.keys():
    print(f"- {split}: {len(dataset[split])} examples")

# Display column names
print("\nColumn Names:")
print(dataset["train"].column_names)

# Show first example
print("\nFirst Example:")
example = dataset["train"][0]
for key, value in example.items():
    print(f"\n{key}:")
    print(value)

# Dataset statistics
print("\nDataset Features:")
print(dataset["train"].features)

# Sample random examples
print("\nRandom Example:")
import random
random_idx = random.randint(0, len(dataset["train"]) - 1)
random_example = dataset["train"][random_idx]
print(f"Example #{random_idx}:")
print(f"Question: {random_example['question']}")
print(f"Answer: {random_example['answer']}")