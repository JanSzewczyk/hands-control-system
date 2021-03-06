from dataclasses import dataclass, field

from hcs.models.gesture_type import GestureType


@dataclass
class GestureClassificationResult:
    gesture_type: GestureType = field(init=False)
    score: float = field(init=False)

