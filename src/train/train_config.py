from dataclasses import dataclass
from typing import Optional
import datetime

@dataclass
class TrainingConfig:
    """Configuration for supervised fine-tuning."""
    
    # Model and data
    output_dir: str = "checkpoints"
    run_name: Optional[str] = None
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512

    # Token-based training control
    training_mode: str = "epochs"  # "epochs" or "tokens"
    max_training_tokens: Optional[int] = None  # Total tokens to train on (used when training_mode="tokens")
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100

    save_total_limit: int = 3
    save_every_n_steps: int = 10
    # Advanced options
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Hardware
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    #data selector
    selector: str = "Full"
    k: int = 1000
    
    minimum_score: float = 0.0
    
    def __post_init__(self):
        if self.run_name is None:
           
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"sft_{timestamp}"
            
    #check selector is valid
        valid_selectors = ["FullDataSelector", "RandomDataSelector", "ThresholdDataSelector", "TopKDataSelector"]
        if self.selector not in valid_selectors:
            raise ValueError(f"Invalid selector '{self.selector}'. Must be one of {valid_selectors}.")

        # Check training_mode is valid
        valid_training_modes = ["epochs", "tokens"]
        if self.training_mode not in valid_training_modes:
            raise ValueError(f"Invalid training_mode '{self.training_mode}'. Must be one of {valid_training_modes}.")

        # Check token-based training has max_training_tokens set
        if self.training_mode == "tokens" and self.max_training_tokens is None:
            raise ValueError("training_mode='tokens' requires max_training_tokens to be set.")