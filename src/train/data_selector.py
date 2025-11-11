from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class DataSelector(ABC):
    """Abstract base class for data selection strategies."""

    @abstractmethod
    def select_data(self, dataset: Any, config: Dict[str, Any]) -> Any:
        """Select and return a subset of the dataset based on the strategy.

        Args:
            dataset: The input dataset from which to select data.
            config: A dictionary of configuration parameters for data selection.

        Returns:
            A subset of the input dataset.
        """
        pass

class RandomDataSelector(DataSelector):
    """Randomly selects a subset of the dataset."""

    def __init__(self, seed: int = 42): 
        self.seed = seed
    
    def select_data(self, dataset: Any, config: Dict[str, Any]) -> Any:
        import random

        selection_size = config.get("selection_size", 100)
        selected_indices = random.sample(range(len(dataset)), selection_size)
        return [dataset[i] for i in selected_indices]