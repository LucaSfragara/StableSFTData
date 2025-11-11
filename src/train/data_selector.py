from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import random
import datasets

class DataSelector(ABC):
    """Abstract base class for data selection strategies."""

    def __init__(self, seed:int):
        self.seed = seed

    @abstractmethod
    def select_data(self, dataset: Any,  n_samples: int) -> datasets.Dataset:
        """Select and return a subset of the dataset based on the strategy.

        Args:
            dataset: The input dataset from which to select data.
            n_samples: The number of samples to select.

        Returns:
            A subset of the input dataset.
        """
        pass

class RandomDataSelector(DataSelector):
    """Randomly selects a subset of the dataset."""

    def __init__(self, seed: int = 42): 
        super().__init__(seed)
        
    def select_data(self, dataset: datasets.Dataset, n_samples: int) -> Any:
       
        """Randomly selects n_samples from the dataset"""
        random.seed(self.seed)
        
        if n_samples > len(dataset):
            raise ValueError(f"n_samples ({n_samples}) cannot be greater than dataset size ({len(dataset)})")
        
        shuffled_dataset = dataset.shuffle(seed=self.seed)
        
        return shuffled_dataset.select(range(n_samples))
    
class FullDataSelector(DataSelector):
    """Selects the full dataset without any filtering."""

    def __init__(self):
        super().__init__(seed=0)  # Seed is irrelevant for full selection

    def select_data(self, dataset: datasets.Dataset, n_samples: int) -> Any:
        """Returns the full dataset regardless of n_samples."""
        return dataset
    
class ThresholdDataSelector(DataSelector):
    pass
    """Selects samples from the dataset based on a score threshold."""
    
class TopKDataSelector(DataSelector):
    pass
    """Selects the top K samples from the dataset based on a score column."""