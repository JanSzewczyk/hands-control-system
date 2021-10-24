from dataclasses import dataclass, field

from models.HandType import HandType


@dataclass
class GestureClassificationResult:
    gesture_type: HandType = field(init=False)
    score: float = field(init=False)

