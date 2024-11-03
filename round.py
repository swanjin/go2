from dataclasses import dataclass

@dataclass
class Round:
    def __init__(self, round, assistant, feedback):
        self.round = round
        self.assistant = assistant
        self.feedback = feedback

