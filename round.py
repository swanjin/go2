from dataclasses import dataclass

@dataclass
class Round:
    def __init__(self, round, assistant, feedback, feedback_factor):
        self.round = round
        self.assistant = assistant
        self.feedback = feedback
        self.feedback_factor = feedback_factor

