import cv2

from typing import Any, Tuple, Optional

from hcs.models import Hand, GestureClassificationResult


def draw_bounding_box(img: Any, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> Any:
    """
    Draw bounding box in image.

    Args:
        img (Any): Image to draw the border box.
        pt1 (Tuple[int, int]): Vertex of the box.
        pt2 (Tuple[int, int]): Vertex of the box opposite to pt1.

    Returns:
        Any: An image with a bounding box.
    """
    cv2.rectangle(img, pt1, pt2, (255, 0, 255), 1)

    return img


def draw_border_box(img: Any, border_box: Tuple[int, int, int, int]) -> Any:
    """
    Draw hand border in image.

    Args:
        img (Any): Image to draw the border box.
        border_box (Tuple[int, int, int, int]): Dimensions and position of the border box.

    Returns:
        Any: An image with a border box.
    """

    length = 30
    thickness = 6
    rectangle_thickness = 1

    x, y, w, h = border_box
    x1, y1 = x + w, y + h

    cv2.rectangle(img, border_box, (255, 163, 51), rectangle_thickness)
    # Top Left x,y
    cv2.line(img, (x, y), (x + length, y), (240, 130, 0), thickness)
    cv2.line(img, (x, y), (x, y + length), (240, 130, 0), thickness)

    # Top Right x1,y
    cv2.line(img, (x1, y), (x1 - length, y), (240, 130, 0), thickness)
    cv2.line(img, (x1, y), (x1, y + length), (240, 130, 0), thickness)

    # Bottom left x,y1
    cv2.line(img, (x, y1), (x + length, y1), (240, 130, 0), thickness)
    cv2.line(img, (x, y1), (x, y1 - length), (240, 130, 0), thickness)

    # Bottom Right x1,y1
    cv2.line(img, (x1, y1), (x1 - length, y1), (240, 130, 0), thickness)
    cv2.line(img, (x1, y1), (x1, y1 - length), (240, 130, 0), thickness)

    return img


def draw_hand_info(img: Any, hand: Hand) -> Any:
    """
    List the hand information in the picture in the lower left corner of the hand border box.
    Information such as:
        * hand type LEFT/RIGHT
        * probability

    Args:
        img (Any): Image to list hand information.
        hand (Hand): Hand information.

    Returns:
        Any: An image with listed hand information.
    """
    x, y, w, h = hand.border_box

    # Draw hand type
    cv2.putText(img, 'TYPE', (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 130, 0), 2, cv2.LINE_AA)
    cv2.putText(img, hand.type.name, (x + 70, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 130, 0), 2,
                cv2.LINE_AA)

    # Draw hand probability
    cv2.putText(img, 'PROB', (x + 10, y + h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 130, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(round(hand.score, 2)), (x + 70, y + h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 130, 0),
                2, cv2.LINE_AA)

    return img


def draw_gesture_info(img: Any, gesture_clf: Optional[GestureClassificationResult],
                      border_box: Tuple[int, int, int, int]) -> Any:
    """
    List the hand gesture predicted information in the picture in the higher left corner of the hand border box.
    Information such as:
        * gesture type name
        * probability

    Args:
        img (Any): Image to list gesture information.
        gesture_clf (Optional[GestureClassificationResult]): Result of gesture classification.
        border_box (Tuple[int, int, int, int]): Dimensions and position of the border box

    Returns:
        Any: An image with or without listed gesture information.
    """

    x, y, w, h = border_box

    if gesture_clf:
        # Draw classified gesture type
        cv2.putText(img, 'TYPE', (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 130, 0), 2, cv2.LINE_AA)  # 12px
        cv2.putText(img, gesture_clf.gesture_type.name, (x + 70, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 130, 0), 2, cv2.LINE_AA)  # 15px

        # Draw gesture probability
        cv2.putText(img, 'PROB', (x + 10, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 130, 0), 2, cv2.LINE_AA)
        cv2.putText(img, str(round(gesture_clf.score, 2)), (x + 70, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 130, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, 'GESTURE UNCLASSIFIED', (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 130, 0), 2,
                    cv2.LINE_AA)

    return img
