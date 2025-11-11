import torch
from src.model.lm import HFModel
import datasets
from src.train.train_config import TrainingConfig
import os
from trl import SFTTrainer # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.train.data_selector import RandomDataSelector, DataSelector
from typing import Dict, Any
from transformers import TrainingArguments

class Trainer: 
    
    def __init__(self, 
                 model: HFModel,
                 train_dataset: datasets.Dataset,
                 eval_dataset: datasets.Dataset,
                 data_selector: DataSelector,
                 config: TrainingConfig):
        
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_selector = data_selector
        self.config = config
    
        self.device = self.model.model.device
        
        self.output_dir = os.path.join(config.output_dir, config.run_name or "default")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def _setup_lora(self):
        """Configure model for LoRA fine-tuning if enabled in config."""
        if not self.config.use_lora:
            return self.model.model
        
        model = prepare_model_for_kbit_training(self.model.model)
        
        lora_config = LoraConfig(
            r = self.config.lora_r,
            lora_alpha = self.config.lora_alpha,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout = self.config.lora_dropout,
            bias = "none",
            task_type = "CAUSAL_LM"
            )

        model = get_peft_model(model, lora_config)
        print("LoRA configuration applied to the model.")
        model.print_trainable_parameters() 
        
        return model

    
    # Format as conversations
    def _format_sample(self, example, system_prompt: str) -> Dict[str, str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend([
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ])
        
        # Apply chat template
        text = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}


    def prepare_train_dataset(self, 
                              dataset: datasets.Dataset,
                              n_samples: int,
                              system_prompt: str) -> datasets.Dataset:
        """Prepare dataset for training by applying data selection strategy
            and any necessary preprocessing.
        """
        
        print(f"Selecting data using {self.data_selector.__class__.__name__}...")
        
        selected_data = self.data_selector.select_data(dataset, n_samples=n_samples)


        print(f"Selected {len(selected_data)} samples from the training dataset.")
        
        selected_data = selected_data.map(
            lambda example: self._format_sample(example, system_prompt),
            desc="Formatting examples",
        )
        
        return selected_data

    def train(self, n_samples: int, system_prompt_train: str, system_prompt_eval: str) -> Dict[str, Any]:

        """
        Fine-tune the model using the provided datasets and configuration.

        Returns:
            Training Metrics and Statistics
        """
        
        model = self._setup_lora() #setup LoRA if enabled in config, otherwise use base model

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            run_name=self.config.run_name,
            
            # Training hyperparameters
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            
            # Logging and saving
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Evaluation
            eval_strategy="steps" if self.eval_dataset else "no",
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,

            # Hardware optimization
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # Misc
            report_to=["tensorboard"],
            remove_unused_columns=False,
        )
        
        selected_train_dataset = self.prepare_train_dataset(
            self.train_dataset,
            n_samples=n_samples,
            system_prompt=system_prompt_train
        )
      
        prepared_eval_dataset = self.eval_dataset.map(
            lambda example: self._format_sample(example, system_prompt_eval),
            desc="Formatting eval examples",
        )
                
        SFT_trainer = SFTTrainer(
            model=model, #type: ignore
            args=training_args,
            train_dataset=selected_train_dataset,
            eval_dataset=prepared_eval_dataset
        )
        
        print(f"\nStarting training...")
        print(f"Output directory: {self.output_dir}")
        print(f"Training samples: {len(selected_train_dataset)}")
        if prepared_eval_dataset:
            print(f"Evaluation samples: {len(prepared_eval_dataset)}")

        train_result = SFT_trainer.train()

        # Save final model
        #SFT_trainer.save_model()
        #self.model.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\nTraining complete! Model saved to: {self.output_dir}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_samples": len(selected_train_dataset),
            "output_dir": self.output_dir,
        }
        