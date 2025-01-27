from dataclasses import dataclass
from ai_client_base import ResponseMsg

@dataclass
class Round:
    round_number: int
    detected_objects: list
    chat: list
    assistant: ResponseMsg