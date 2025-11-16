
from src.model.lm import HFModel
from src.train import (
    Trainer,
    TrainingConfig,
    FullDataSelector,
    RandomDataSelector,
    ThresholdDataSelector
)
from datasets import load_dataset, load_from_disk
from src.prompts import GSM8K_FINE_TUNE
from datasets.config import HF_DATASETS_CACHE
from src.train.callbacks import GenerationEvaluationCallback
import os
from src.utils.prompt_builder import build_prompt

def main():
  
    #model = HFModel("Qwen/Qwen3-0.6B")
    model = HFModel("allenai/open-instruct-pythia-6.9b-tulu")
    # Load GSM8K dataset
    dataset_dict = load_from_disk(os.path.join(HF_DATASETS_CACHE, "gsm8k_processed"))
    
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    # 3. Configure training
    config = TrainingConfig(
        output_dir="checkpoints",
        run_name="GSM8K_0.15dropout",
        learning_rate=2e-5,
        num_epochs=4,
        batch_size=64,
        gradient_accumulation_steps=1,
        use_lora=True,  # Use LoRA for efficient fine-tuning
        lora_r=32,
        lora_alpha=64,
        logging_steps=20,
        lora_dropout=0.15,
        eval_steps=20,
        gradient_checkpointing=True,
        save_every_n_steps=1,
    )
    
    # 4. Select training strategy
    # Option A: Train on top-scoring examples
    #selector = ThresholdDataSelector(score_column="score", ascending=False)
    
    # Option B: Train on full dataset
    selector = FullDataSelector(seed=42)

    # Option C: Random selection
    # selector = RandomDataSelector(seed=42)

    
    # Option D: Threshold-based
    # 5. Initialize trainer with proper splits and optional subsampling
    trainer = Trainer(model,
                      train_dataset=train_dataset, #type: ignore
                      eval_dataset=eval_dataset, #type: ignore
                      data_selector=selector,
                      config=config, 
                      use_custom_chat_template=True)
    
    
    eval_callback = GenerationEvaluationCallback(
        trainer_instance=trainer,
        eval_dataset=eval_dataset, # type: ignore
        num_eval_samples=100,
        enable_thinking=False,
        max_new_tokens=256
    )
    
    trainer.callbacks.append(eval_callback)
    # 6. Train!
    results = trainer.train(
        n_samples=None, 
        system_prompt=GSM8K_FINE_TUNE,
    )
    
    #print("\nTraining Results:")
    print(results)
    
    # 7. Evaluate generation quality on a held-out set
    print("\n" + "="*60)
    print("Running custom evaluation...")
    print("="*60)
    
    eval_metrics = trainer.evaluate_generation_quality(
        eval_dataset=eval_dataset,  # type: ignore
        num_samples=500,  # Evaluate on 500 random samples
        use_cache = True, 
        max_new_tokens=256,
    )
    
    print("\nFinal Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()