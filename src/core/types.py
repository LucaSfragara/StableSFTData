from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class Sample: 
    id: int
    question: str
    reasoning: Optional[str]
    answer: str

    def __repr__(self) -> str:
        return f"""
        Sample(
            id = {self.id}, 
            question = {self.question}, 
            reasoning = {self.reasoning},
            answer = {self.answer}
        )"""

@dataclass
class Dataset:
    samples: List[Sample]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]
    
    def __iter__(self):
        return iter(self.samples)

@dataclass
class ScoreRecord: 
    id: int
    score: float
