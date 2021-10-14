from dataclasses import dataclass, field
from typing import List, Tuple

from models.HandType import HandType


@dataclass
class Hand:
    landmarks: List[List[int]] = field(init=False)
    border_box: Tuple[int, int, int, int] = field(init=False)
    center: Tuple[int] = field(init=False)
    type: HandType = field(init=False)
