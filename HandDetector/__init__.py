import cv2
import mediapipe as mp
from typing import List, Any, Tuple, Union

from models import Hand, HandType


class HandDetector:
    """
    Finds hands using mediapipe library. Exports the landmarks in pixel format. Adds extra functionalities like
    finding how many fingers are up or the distance between two fingers. Also provides bounding box info of
    the hand found.
    """

    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        :param static_image_mode: In static mode, detection is done on each image: slower
        :param max_num_hands: Maximum number of hands detected
        :param min_detection_confidence: Minimum Detection Confidence Threshold
        :param min_tracking_confidence: Minimum Tracking Confidence Threshold
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode, self.max_num_hands, self.min_detection_confidence,
                                         self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids = [4, 8, 12, 16, 20]
        self.landmarks_list = []
        self.results = None

    def find_hands(self, img, draw=True, flip_type=True) -> Union[Tuple[List[Hand], Any], List[Hand]]:
        """
        Find hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :param flip_type: Flag to flip hands type.
        :return: Hands info with or without Image with drawings.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
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

                # set hand type
                if hand_info.classification[0].label == "Right":
                    hand.type = HandType.LEFT if flip_type else HandType.RIGHT
                else:
                    hand.type = HandType.RIGHT if flip_type else HandType.LEFT

                all_hands.append(hand)

                # Draw border box
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    img = self.draw_border_box(img, border_box)

        if draw:
            return all_hands, img
        else:
            return all_hands

    # def find_position(self, img, hand_no=0, draw=True):
    #
    #     x_list = []
    #     y_list = []
    #     border_box = []
    #     self.landmarks_list = []
    #     if self.results.multi_hand_landmarks:
    #         my_hand = self.results.multi_hand_landmarks[hand_no]
    #         for landmark_id, lm in enumerate(my_hand.landmark):
    #             h, w, c = img.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #
    #             x_list.append(cx)
    #             y_list.append(cy)
    #
    #             self.landmarks_list.append([landmark_id, cx, cy])
    #
    #             if draw:
    #                 cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
    #
    #         x_min, x_max = min(x_list), max(x_list)
    #         y_min, y_max = min(y_list), max(y_list)
    #         border_box = x_min, y_min, x_max, y_max
    #
    #         if draw:
    #             cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
    #
    #     return self.landmarks_list, border_box

    def get_fingers_up(self, hand: Hand) -> List[int]:
        """
        Finds how many fingers are open and returns in a list.
        :param hand: Hand information.
        :return: List of which fingers are up.
        """
        hand_type = hand.type
        hand_landmarks = hand.landmarks

        fingers = []
        if self.results.multi_hand_landmarks:
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

    # def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
    #     x1, y1 = self.landmarks_list[p1][1:]
    #     x2, y2 = self.landmarks_list[p2][1:]
    #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #
    #     if draw:
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
    #         cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
    #         cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
    #         cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
    #
    #     length = math.hypot(x1 - x2, y1 - y2)
    #
    #     return length, img, [x1, y1, x2, y2, cx, cy]

    @staticmethod
    def draw_border_box(img: Any, border_box: Tuple) -> Any:
        """
        Draw border in image.
        :param img: Image to draw the border box.
        :param border_box: Dimensions and position of the border box.
        :return: An image with a border box.
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
