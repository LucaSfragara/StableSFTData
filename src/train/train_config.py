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
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100

    save_total_limit: int = 20
    save_every_n_steps: int = 10
    # Advanced options
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Hardware
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        if self.run_name is None:
           
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"sft_{timestamp}"
            