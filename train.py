
from src.model.lm import HFModel
from src.train import (
    Trainer,
    TrainingConfig,
    FullDataSelector,
    RandomDataSelector,
    ThresholdDataSelector, 
    TopKDataSelector,
)
from datasets import load_dataset, load_from_disk
from src.prompts import GSM8K_FINE_TUNE
from datasets.config import HF_DATASETS_CACHE
from src.train.callbacks import GenerationEvaluationCallback
import os
from src.utils.prompt_builder import build_prompt
os.environ["WANDB_DIR"] = "/tmp"

def main():
  
    #model = HFModel("Qwen/Qwen3-0.6B")
    model = HFModel("allenai/open-instruct-pythia-6.9b-tulu")
    # Load GSM8K dataset
    eval_dataset = load_from_disk(os.path.join(HF_DATASETS_CACHE, "gsm8k_processed"))['test']
    train_dataset = load_from_disk(os.path.join("/orcd/home/002/lsfragar/orcd/pool/", "gsm8k_scored_20251119_001339"))
    
    
    #train_dataset = dataset_dict["train"]
    #eval_dataset = dataset_dict["test"]
    
    # 3. Configure training
    config = TrainingConfig(
        output_dir="/orcd/home/002/lsfragar/orcd/pool/checkpoints",
        run_name="GSM8K_Random2500",
        learning_rate=2e-5,
        num_epochs=6,
        batch_size=64,
        gradient_accumulation_steps=1,
        use_lora=True,  # Use LoRA for efficient fine-tuning
        lora_r=32,
        lora_alpha=64,
        logging_steps=1,
        lora_dropout=0.15,
        eval_steps=20,
        gradient_checkpointing=True,
        save_every_n_steps=1,
        selector = "Random",
        k = 2500, 
        minimum_score =0
    )
    
    #Initialize data selector based on config
    score_column="accuracy"
    
    if config.selector == "Full":
        selector = FullDataSelector()
    elif config.selector == "Random":
        selector = RandomDataSelector()
    elif config.selector == "Threshold":
        selector = ThresholdDataSelector(
            score_column=score_column, 
            minimum_score=config.minimum_score
        )
    elif config.selector == "TopK":
        selector = TopKDataSelector(
            score_column="accuracy",
            k=config.k,
            ascending=True,
            seed=42,
        )

    # 5. Initialize trainer with proper splits and optional subsampling
    trainer = Trainer(model,
                      train_dataset=train_dataset, #type: ignore
                      eval_dataset=eval_dataset, #type: ignore
                      data_selector=selector,
                      config=config, 
                      use_custom_chat_template=True)
    
    #trainer.load_pretrained_model("checkpoint-260")
    
    eval_callback = GenerationEvaluationCallback(
        trainer_instance=trainer,
        eval_dataset=eval_dataset, # type: ignore
        num_eval_samples=100,
        enable_thinking=False,
        max_new_tokens=256, 
    )
    
    trainer.callbacks.append(eval_callback)
    # 6. Train!
    results = trainer.train(
        n_samples=None if config.selector == "TopK" else config.k, 
        system_prompt=GSM8K_FINE_TUNE,
    )
    
    print("\nTraining Results:")
    print(results)
    
    # 7. Evaluate generation quality on a held-out set
    
    print("\n" + "="*60)
    print("Running custom evaluation...")
    print("="*60)
    
    eval_metrics = trainer.evaluate_generation_quality(
        eval_dataset=eval_dataset,  # type: ignore
        num_samples=500,  # Evaluate on 1000 random samples
        use_cache = True, 
        max_new_tokens=256,
    )
    
    print("\nFinal Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()