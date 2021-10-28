import cv2
import mediapipe as mp
from typing import List, Any, Tuple, Union, NamedTuple, Optional

import utils.draw_utils as du

from models import Hand, HandType


class HandDetector:
    """
    Finds hands using mediapipe library. Exports the landmarks in pixel format. Adds extra functionalities like
    finding how many fingers are up. Also provides bounding box info of the hand found.

    Attributes:
        static_image_mode (bool): In static mode, detection is done on each image: slower.
        max_num_hands (int): Maximum number of hands detected.
        min_detection_confidence (float): Minimum Detection Confidence Threshold.
        min_tracking_confidence (float): Minimum Tracking Confidence Threshold.
        mp_hands (mediapipe.python.solutions.hands): MediaPipe Hands tools.
        hands (mediapipe.python.solutions.hands.Hands): Instance attribute hands of hand_detector.HandDetector.
        mp_draw (mediapipe.python.solutions.drawing_utils): MediaPipe solution drawing utils.
        tip_ids (List[int]): List of tips id.
        results (Optional[NamedTuple]): Instance attribute results of hand_detector.HandDetector.
    """

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Constructor.

        Args:
            static_image_mode (bool): Defaults to False. In static mode, detection is done on each image: slower.
            max_num_hands (int): Defaults to 2. Maximum number of hands detected.
            min_detection_confidence (float): Defaults to 0.5. Minimum Detection Confidence Threshold.
            min_tracking_confidence (float): Defaults to 0.5. Minimum Tracking Confidence Threshold.
        """

        self.static_image_mode: bool = static_image_mode
        self.max_num_hands: int = max_num_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands: mp.solutions.hands.Hands = self.mp_hands.Hands(self.static_image_mode, self.max_num_hands,
                                                                   self.min_detection_confidence,
                                                                   self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids: List[int] = [4, 8, 12, 16, 20]
        self.results: Optional[NamedTuple] = None

    def find_hands(self, img, draw=True, flip_type=True) -> Union[Tuple[List[Hand], Any], List[Hand]]:
        """
        Find hands in a BGR image.

        Args:
            img (Any): Image to find the hands in.
            draw (bool): Defaults to True. Flag to draw the output on the image.
            flip_type (bool): Defaults to True. Flag to flip hands type.

        Returns:
            Union[Tuple[List[Hand], Any], List[Hand]]: Hands info with or without Image with drawings.
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results: NamedTuple = self.hands.process(img_rgb)

        all_hands: List[Hand] = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for hand_info, hand_landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                hand = Hand()
                hand_landmarks_list = []

                x_list = []
                y_list = []

                for _, landmark in enumerate(hand_landmarks.landmark):
                    px, py, pz = int(landmark.x * w), int(landmark.y * h), landmark.z
                    hand_landmarks_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                # Border box
                x_min, x_max = min(x_list) - 20, max(x_list) + 20
                y_min, y_max = min(y_list) - 20, max(y_list) + 20
                box_width, box_height = x_max - x_min, y_max - y_min
                border_box = x_min, y_min, box_width, box_height
                center_x, center_y = border_box[0] + (border_box[2] // 2), border_box[1] + (border_box[3] // 2)

                hand.landmarks = hand_landmarks_list
                hand.border_box = border_box
                hand.center = (center_x, center_y)
                hand.score = hand_info.classification[0].score

                # set hand type
                if hand_info.classification[0].label == "Right":
                    hand.type = HandType.LEFT if flip_type else HandType.RIGHT
                else:
                    hand.type = HandType.RIGHT if flip_type else HandType.LEFT

                all_hands.append(hand)

                # Draw border box
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    img = du.draw_border_box(img, border_box)
                    img = du.draw_hand_info(img, hand)

        if draw:
            return all_hands, img
        else:
            return all_hands

    def get_fingers_up(self, hand: Hand) -> List[int]:
        """
        Finds how many fingers are open and returns in a list.

        Args:
            hand (Hand): Hand information.

        Returns:
            List[int]: List of which fingers are up.
        """

        hand_type = hand.type
        hand_landmarks = hand.landmarks

        fingers = []

        # Thumb
        if hand_type == HandType.RIGHT:
            if hand_landmarks[self.tip_ids[0]][0] > hand_landmarks[self.tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks[self.tip_ids[0]][0] < hand_landmarks[self.tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for finger_id in range(1, 5):
            if hand_landmarks[self.tip_ids[finger_id]][1] < hand_landmarks[self.tip_ids[finger_id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
