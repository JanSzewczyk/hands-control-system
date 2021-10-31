from dataclasses import dataclass, field
from typing import List, Tuple, NamedTuple

from hcs.models.hand_type import HandType


class BorderBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int


@dataclass
class Hand:
    landmarks: List[List[float]] = field(init=False)
    border_box: BorderBox = field(init=False)
    center: Tuple[int, int] = field(init=False)
    score: float = field(init=False)
    type: HandType = field(init=False)
