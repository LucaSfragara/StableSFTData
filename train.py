from src.model.lm import HFModel
from src.train import (
    Trainer,
    TrainingConfig,
    FullDataSelector,
    RandomDataSelector,
    ThresholdDataSelector
)
from datasets import load_dataset, Dataset, DatasetDict
from src.prompts import GSM8K_FINE_TUNE, GSM8K

def main():
  
    model = HFModel("Qwen/Qwen3-0.6B")

    # Load GSM8K dataset
    dataset_dict = load_dataset("gsm8k", "main")
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    # 3. Configure training
    config = TrainingConfig(
        output_dir="checkpoints",
        run_name="gsm8k_top1000",
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        use_lora=True,  # Use LoRA for efficient fine-tuning
        lora_r=16,
    )
    
    # 4. Select training strategy
    # Option A: Train on top-scoring examples
    #selector = ThresholdDataSelector(score_column="score", ascending=False)
    
    # Option B: Train on full dataset
    selector = FullDataSelector()

    # Option C: Random selection
    # selector = RandomDataSelector(seed=42)

    # Option D: Threshold-based
    # 5. Initialize trainer with proper splits and optional subsampling
    trainer = Trainer(model,
                      train_dataset=train_dataset, #type: ignore
                      eval_dataset=eval_dataset, #type: ignore
                      data_selector=selector,
                      config=config)
    # 6. Train!
    results = trainer.train(
        n_samples=1000,
        system_prompt_train=GSM8K_FINE_TUNE,
        system_prompt_eval=GSM8K
    )
    
    print("\nTraining Results:")
    print(results)

if __name__ == "__main__":
    main()