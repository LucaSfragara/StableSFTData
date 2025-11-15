import torch
from src.model.lm import HFModel
import datasets
from src.train.train_config import TrainingConfig
import os
from trl import SFTTrainer # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.train.data_selector import RandomDataSelector, DataSelector
from typing import Dict, Any, Optional
from transformers import TrainingArguments, EvalPrediction
import re
import numpy as np
from src.prompts import GSM8K_FINE_TUNE
from src.utils.parser import Parser
import json
from trl import DataCollatorForCompletionOnlyLM
from src.utils.prompt_builder import build_prompt

class Trainer: 
    
    def __init__(self, 
                 model: HFModel,
                 train_dataset: datasets.Dataset,
                 eval_dataset: datasets.Dataset,
                 data_selector: DataSelector,
                 config: TrainingConfig, 
                 callbacks: Optional[list] = None, 
                 use_custom_chat_template: bool = False):
        
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_selector = data_selector
        self.config = config
    
        self.device = self.model.model.device
        
        self.output_dir = os.path.join(config.output_dir, config.run_name or "default")
        self.callbacks = callbacks or []
        self.use_custom_chat_template = use_custom_chat_template

        
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def _setup_lora(self):
        if not self.config.use_lora:
            return self.model.model

        base_model = self.model.model
        base_model = prepare_model_for_kbit_training(base_model)
        base_model.config.use_cache = False

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        return model

    def _extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the numerical answer from generated text.
        Looks for patterns like '#### 42' or 'The answer is 42'
        """
        # Try to find #### pattern (GSM8K format)
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1)
        
        # Try to find "answer is X" pattern
        match = re.search(r'answer is\s*(-?\d+(?:\.\d+)?)', text.lower())
        if match:
            return match.group(1)
        
        # Try to find last number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def evaluate_generation_quality(
        self, 
        eval_dataset: datasets.Dataset,
        num_samples: int, 
        max_new_tokens: int = 512,
        enable_thinking: bool = False, 
        step: int = 0,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model's generation quality by generating answers
        and comparing them to ground truth.
        
        This is separate from the training loop evaluation.
        Call this manually after training to assess performance.
        """
        print(f"\nEvaluating generation quality on {num_samples} samples...")
        
        # Sample from eval dataset
        if num_samples < len(eval_dataset):
            eval_subset = eval_dataset.shuffle(seed=42).select(range(num_samples))
        else:
            eval_subset = eval_dataset
        
        correct = 0
        total = 0
        
        batch_size = getattr(self.config, "batch_size", 1)
        
        for start in range(0, len(eval_subset), batch_size):
            
            indices = list(range(start, min(start + batch_size, len(eval_subset))))
            batch = eval_subset.select(indices)
            
        
            conversations = [
                [
                    {"role": "system", "content": GSM8K_FINE_TUNE},
                    {"role": "user", "content": ex["question"]} #type: ignore
                ]
                for ex in batch
            ]

            try:
                generated_batch = self.model.chat(
                    conversations,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    temperature=0.0, 
                    use_custom_chat_template=self.use_custom_chat_template, 
                    use_cache=use_cache
                )
            except Exception as e:
                print(f"Batch generation error at indices {indices}: {e}")
                continue

            for ex, generated in zip(batch, generated_batch):
                question = ex["question"] #type: ignore
                true_answer = ex["answer_num"] #type: ignore

                # Persist generated text for monitoring
                pred_answer = Parser.extract_generated_number(generated)
                
                try:
                    log_record = {
                        "question": question,
                        "true_answer": true_answer,
                        "predicted_answer": pred_answer,
                        "generated_text": generated,
                    }

                    log_path = os.path.join(self.output_dir, f"generation_monitor_{step}.jsonl")

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Warning: failed to write generation log: {e}")

                if pred_answer is not None and pred_answer == true_answer:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
           
        metrics = {
            "generation_accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        print(f"Generation Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return metrics

    
    # Format as conversations
    def _format_sample(self, example, system_prompt: str) -> Dict[str, str]:
        messages = []
        
        if not self.use_custom_chat_template:
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
        else:
            text = build_prompt(
                system_prompt=system_prompt,
                question=example["question"],
                answer=example["answer"] if "answer" in example else None,
            )
        
        return {"text": text}


    def prepare_train_dataset(self, 
                              dataset: datasets.Dataset,
                              n_samples: Optional[int],
                              system_prompt: str) -> datasets.Dataset:
        """Prepare dataset for training by applying data selection strategy
            and any necessary preprocessing.
        """
        
        print(f"Selecting data using {self.data_selector.__class__.__name__}...")
        
        selected_data = self.data_selector.select_data(dataset, n_samples=n_samples)


        print(f" --> selected {len(selected_data)} samples from the training dataset.")
        
        selected_data = selected_data.map(
            lambda example: self._format_sample(example, system_prompt),
            desc="Formatting train samples with chat template",
         
        )
        
        return selected_data

    def train(self, n_samples: Optional[int], system_prompt: str) -> Dict[str, Any]:

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
            save_steps=self.config.save_every_n_steps * self.config.eval_steps if self.eval_dataset else None,
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
            remove_unused_columns=True,
            
        )
        
        selected_train_dataset = self.prepare_train_dataset(
            self.train_dataset,
            n_samples=n_samples,
            system_prompt=system_prompt
        )
      
        prepared_eval_dataset = self.eval_dataset.map(
            lambda example: self._format_sample(example, system_prompt),
            desc="Formatting eval examples",
        )
        
        response_template =  "<|assistant|>\n"
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.model.tokenizer,
            response_template=response_template,
            return_tensors="pt",
        )
        
        SFT_trainer = SFTTrainer(
            model=model, #type: ignore
            args=training_args,
            train_dataset=selected_train_dataset,
            eval_dataset=prepared_eval_dataset,
            callbacks=self.callbacks, 
            data_collator=data_collator
        )
        
        print(f"\nStarting training...")
        print(f"Output directory: {self.output_dir}")
        print(f"Training samples: {len(selected_train_dataset)}")
        if prepared_eval_dataset:
            print(f"Evaluation samples: {len(prepared_eval_dataset)}")

        train_result = SFT_trainer.train()

        # Save final model
        SFT_trainer.save_model()
        self.model.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\nTraining complete! Model saved to: {self.output_dir}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_samples": len(selected_train_dataset),
            "output_dir": self.output_dir,
        }
        