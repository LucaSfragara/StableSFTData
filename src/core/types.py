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
class ScoreRecord: 
    id: int
    score: float
