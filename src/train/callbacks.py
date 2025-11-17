from math import e
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Optional
import datasets
from src.train.trainer import Trainer

class GenerationEvaluationCallback(TrainerCallback):
    """
    Custom callback to evaluate generation quality during training.
    Runs every `eval_steps` and computes accuracy on actual generated answers.
    """
    
    def __init__(
        self,
        trainer_instance: Trainer,
        eval_dataset: datasets.Dataset,
        num_eval_samples: int = 100,
        eval_every_n_steps: Optional[int] = None,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
    ):
        """
        Args:
            trainer_instance: Your Trainer class instance (has evaluate_generation_quality method)
            eval_dataset: Dataset to evaluate on
            num_eval_samples: Number of samples to evaluate
            eval_every_n_steps: Evaluate every N steps (if None, uses args.eval_steps)
        """
        self.trainer_instance = trainer_instance
        self.eval_dataset = eval_dataset
        self.num_eval_samples = num_eval_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.best_accuracy = 0.0
        self.eval_history = []
        self.enable_thinking = enable_thinking
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, **kwargs):
        """
        Called after standard evaluation (eval_loss computation).
        This is where we add our custom generation-based evaluation.
        """
        # Only run if this is an evaluation step (not just logging)
        print(state.global_step)
        if state.global_step == 0:
            return
        
        if state.global_step < 100:
            return
        
        if state.global_step > 300:
            return
            
        # Check if we should run this step
        if self.eval_every_n_steps is not None:
            if state.global_step % self.eval_every_n_steps != 0:
                return
            
           
        print(f"\n{'='*60}")
        print(f"Custom Generation Evaluation at Step {state.global_step}")
        print(f"{'='*60}")
        
        # Run your custom evaluation
        eval_metrics = self.trainer_instance.evaluate_generation_quality(
            eval_dataset=self.eval_dataset,
            num_samples=self.num_eval_samples,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=self.enable_thinking, 
            step = state.global_step,
        )
        
        # Track best accuracy
        current_accuracy = eval_metrics.get('generation_accuracy', 0.0)
        if current_accuracy >= self.best_accuracy:
            self.best_accuracy = current_accuracy
            #save model
            self.trainer_instance.save_pretrained_model(f"best@{current_accuracy:.4f}-step{state.global_step}")
            print(f"🎉 New best generation accuracy: {current_accuracy:.2%}")
        
        # Store in history
        self.eval_history.append({
            'step': state.global_step,
            'epoch': state.epoch,
            **eval_metrics
        })
        
        # Log to tensorboard if available
        if state.is_world_process_zero:
            for key, value in eval_metrics.items():
                kwargs.get('logs', {})[f'eval_{key}'] = value
        
        print(f"{'='*60}\n")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Print summary at the end of training."""
        print(f"\n{'='*60}")
        print("Generation Evaluation Summary")
        print(f"{'='*60}")
        print(f"Best Generation Accuracy: {self.best_accuracy:.2%}")
        print(f"\nHistory:")
        for entry in self.eval_history:
            print(f"  Step {entry['step']:4d} (Epoch {entry['epoch']:.2f}): "
                  f"Accuracy = {entry['generation_accuracy']:.2%}")
        print(f"{'='*60}\n")