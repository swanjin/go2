from dataclasses import dataclass

@dataclass
class Round:
    round: int
    assistant: str
    feedback: str

