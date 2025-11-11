import torch
from src.model.lm import HFModel
import datasets
from src.train.train_config import TrainingConfig
import os
from trl import SFTTrainer # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any
from transformers import TrainingArguments

class Trainer: 
    
    def __init__(self, model: HFModel, train_dataset: datasets.Dataset, eval_dataset: datasets.Dataset, config: TrainingConfig):
        
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
    
        self.device = self.model.model.device
        
        self.output_dir = os.path.join(config.output_dir, config.run_name or "default")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def setup_lora(self):
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
    
    def train(self) -> Dict[str, Any]:
        
        """
        Fine-tune the model using the provided datasets and configuration.

        Returns:
            Training Metrics and Statistics
        """
        
        model = self.setup_lora() #setup LoRA if enabled in config, otherwise use base model
        
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
            eval_steps=self.config.eval_steps if eval_data else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Evaluation
            eval_strategy="steps" if eval_data else "no",
            load_best_model_at_end=True if eval_data else False,
            metric_for_best_model="eval_loss" if eval_data else None,
            
            # Hardware optimization
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # Misc
            report_to=["tensorboard"],
            remove_unused_columns=False,
        )
        