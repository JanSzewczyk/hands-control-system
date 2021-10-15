from typing import List
from models import Hand
import numpy as np
import matplotlib.pyplot as plt


def prepare_hand_data(hand: Hand) -> List[List[int]]:
    """
    Function that calculates hand landmarks based on the frame area.
    Scales Hand landmarks to ranges [0, 1].

    Args:
        hand: Hand information.

    Returns:
        List of processed hand landmarks.
    """

    x_min, y_min, box_width, box_height = hand.border_box
    landmarks = []

    for landmark in hand.landmarks:
        x = np.interp(landmark[0] - x_min, (0, box_width), (0, 1))
        y = np.interp(landmark[1] - y_min, (0, box_height), (0, 1))

        landmarks.append([x, y, landmark[2]])

    return landmarks


def draw_hand_gesture(landmarks) -> None:
    """
    A function that draws landmarks on a graph.

    Args:
        landmarks: Hand landmarks.

    Returns:
    """
    landmarks_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]

    x = list(map(lambda landmark: landmark[0], landmarks))
    y = list(map(lambda landmark: landmark[1], landmarks))

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    for p_1, p_2 in landmarks_connections:
        plt.plot([x[p_1], x[p_2]], [y[p_1], y[p_2]], 'r')

    plt.scatter(x, y)
    for index, coord in enumerate(zip(x, y)):
        plt.text(coord[0], coord[1], index, color='blue')

    plt.show()
