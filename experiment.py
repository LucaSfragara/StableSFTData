"""
Multi-dataset training experiment with automatic token budget calculation.

This script:
1. Defines multiple dataset configurations (e.g., easy vs hard subsets)
2. Counts tokens for each dataset
3. Uses the minimum token count as the training budget
4. Trains all datasets with the same token budget for fair comparison
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.model.lm import HFModel
from src.train import (
    Trainer,
    TrainingConfig,
    TopKDataSelector,
    RandomDataSelector,
    DataSelector,
)
from datasets import load_from_disk
from src.prompts import GSM8K_FINE_TUNE
from datasets.config import HF_DATASETS_CACHE
from src.train.callbacks import GenerationEvaluationCallback
from src.utils.token_utils import count_dataset_tokens
from src.utils.prompt_builder import build_prompt

from typing import List, Dict, Any
import wandb

os.environ["WANDB_DIR"] = "/tmp"


def count_tokens_for_dataset(
    tokenizer,
    train_dataset,
    data_selector: DataSelector,
    n_samples: int,
    system_prompt: str,
    use_custom_chat_template: bool = True
) -> Dict[str, Any]:
    """
    Count tokens for a dataset configuration without training.

    Args:
        tokenizer: The tokenizer to use
        train_dataset: The training dataset
        data_selector: Data selector to apply
        n_samples: Number of samples to select
        system_prompt: System prompt for formatting
        use_custom_chat_template: Whether to use custom chat template

    Returns:
        Dictionary with token statistics
    """
    print(f"Selecting data using {data_selector.__class__.__name__}...")

    # Apply data selector
    selected_data = data_selector.select_data(train_dataset, n_samples=n_samples)
    print(f" --> selected {len(selected_data)} samples from the training dataset.")

    # Define formatting function
    def format_sample(example):
        if use_custom_chat_template:
            text = build_prompt(
                system_prompt=system_prompt,
                question=example["question"],
                answer=example["answer"] if "answer" in example else None,
            )
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend([
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]}
            ])
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        return {"text": text}

    # Count tokens using utility function
    token_stats = count_dataset_tokens(selected_data, tokenizer, format_sample)

    return token_stats


def main():
    # Load model and datasets
    print("="*60)
    print("MULTI-DATASET TOKEN-BASED TRAINING EXPERIMENT")
    print("="*60)

    model = HFModel("allenai/open-instruct-pythia-6.9b-tulu")
    eval_dataset = load_from_disk(os.path.join(HF_DATASETS_CACHE, "gsm8k_processed"))['test']
    train_dataset = load_from_disk(os.path.join("/orcd/home/002/lsfragar/orcd/pool/", "gsm8k_scored_20251119_001339"))

    # Define experiment configurations
    # Each experiment is a different subset of the data
    experiments = [
        {
            "name": "hard_subset_long",
            "description": "Hard problems (low accuracy, ascending=True)",
            "selector": TopKDataSelector(
                score_column="accuracy",
                k=2000,
                ascending=True,  # Select lowest accuracy = hardest
                seed=42,
            ),
            "n_samples": None,  # TopK determines the size
        },
        {
            "name": "easy_subset_long",
            "description": "Easy problems (high accuracy, ascending=False)",
            "selector": TopKDataSelector(
                score_column="accuracy",
                k=2000,
                ascending=False,  # Select highest accuracy = easiest
                seed=42,
            ),
            "n_samples": None,
        },
        {
             "name": "random_subset_long",
             "description": "Random sample",
             "selector": RandomDataSelector(),
             "n_samples": 2000,
         },
    ]

    # Base training configuration (shared across all experiments)
    base_config = {
        "output_dir": "/orcd/home/002/lsfragar/orcd/pool/checkpoints",
        "learning_rate": 2e-5,
        "batch_size": 64,
        "gradient_accumulation_steps": 1,
        "use_lora": True,
        "lora_r": 32,
        "lora_alpha": 64,
        "logging_steps": 1,
        "lora_dropout": 0.15,
        "eval_steps": 20,
        "gradient_checkpointing": True,
        "save_every_n_steps": 1,
        "num_epochs": 13,
    }

    # Step 1: Count tokens for all experiments
    print("\n" + "="*60)
    print("STEP 1: Counting tokens for all experiments")
    print("="*60 + "\n")

    token_counts = {}
    for exp in experiments:
        print(f"\nCounting tokens for: {exp['name']}")
        print(f"Description: {exp['description']}")

        token_stats = count_tokens_for_dataset(
            tokenizer=model.tokenizer,
            train_dataset=train_dataset,
            data_selector=exp["selector"],
            n_samples=exp["n_samples"],
            system_prompt=GSM8K_FINE_TUNE,
            use_custom_chat_template=True
        )

        token_counts[exp["name"]] = token_stats
        print(f"  → Total tokens for {exp['name']}: {token_stats['total_tokens']:,}")

    # Step 2: Calculate token budget for multiple epochs
    min_tokens_per_epoch = min(stats["total_tokens"] for stats in token_counts.values())
    num_epochs = base_config["num_epochs"]
    min_tokens = min_tokens_per_epoch * num_epochs

    print("\n" + "="*60)
    print("TOKEN BUDGET CALCULATION")
    print("="*60)
    print("\nToken counts per dataset (1 epoch):")
    for name, stats in token_counts.items():
        print(f"  {name}: {stats['total_tokens']:,} tokens")
    print(f"\n✓ Minimum tokens per epoch: {min_tokens_per_epoch:,}")
    print(f"✓ Number of epochs: {num_epochs}")
    print(f"✓ Total token budget: {min_tokens:,} tokens ({min_tokens_per_epoch:,} × {num_epochs})")
    print(f"\nAll experiments will train until they see {min_tokens:,} tokens total.")
    print("="*60 + "\n")

    # Step 3: Train all experiments with the same token budget
    print("\n" + "="*60)
    print("STEP 2: Training all experiments with fixed token budget")
    print("="*60 + "\n")

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Training: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Total token budget: {min_tokens:,} ({min_tokens_per_epoch:,} per epoch × {num_epochs} epochs)")
        print(f"{'='*60}\n")

        # Create training config with token-based training
        config = TrainingConfig(
            **base_config,
            run_name=f"{exp['name']}_token_controlled",
            training_mode="tokens",
            max_training_tokens=min_tokens,
            selector=exp['selector'].__class__.__name__,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset, #type: ignore
            eval_dataset=eval_dataset, #type: ignore
            data_selector=exp["selector"],
            config=config,
            use_custom_chat_template=True
        )
        
        # Add evaluation callback
        eval_callback = GenerationEvaluationCallback(
            trainer_instance=trainer,
            eval_dataset=eval_dataset,
            num_eval_samples=100,
            enable_thinking=False,
            max_new_tokens=256,
        )
        trainer.callbacks.append(eval_callback)

        # Train
        results = trainer.train(
            n_samples=exp["n_samples"],
            system_prompt=GSM8K_FINE_TUNE,
        )

        print(f"\n✓ Completed training for {exp['name']}")
        print(f"  Train loss: {results.get('train_loss', 'N/A')}")
        print(f"  Output dir: {results['output_dir']}\n")

        #Load best model
        
        #trainer.load_pretrained_model(trainer.best_model_name) #type: ignore
        trainer.load_pretrained_model(trainer.best_model_name)
        # Final evaluation
        print(f"\nRunning final evaluation for {exp['name']}...")
        eval_metrics = trainer.evaluate_generation_quality(
            eval_dataset=eval_dataset,
            num_samples=500,
            use_cache=True,
            max_new_tokens=256,
        )

        print(f"\nFinal metrics for {exp['name']}:")
        for key, value in eval_metrics.items():
            if key != "train/global_step":
                print(f"  {key}: {value}")

        # Close W&B run for this experiment
        wandb.finish()

        print(f"\n{'='*60}")
        print(f"✓ Completed experiment: {exp['name']}")
        print(f"{'='*60}\n")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\nSummary:")
    print(f"  Token budget used: {min_tokens:,} tokens")
    print(f"  Experiments run: {len(experiments)}")
    print("\nCheck W&B for detailed results and comparisons.")


if __name__ == "__main__":
    main()
