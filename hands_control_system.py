import cv2
import numpy as np
from typing import List, Tuple, Any

from hand_gesture_detector import HandGestureDetector
from camera_video_capture import CameraVideoCapture
from hand_detector import HandDetector
from mouse_controller import MouseController
from fps import FPS

import utils.draw_utils as du

from models import HandType, Hand, GestureType


class HandsControlSystem:

    def __init__(self):
        self.cap = CameraVideoCapture()
        self.detector = HandDetector(max_num_hands=2, min_detection_confidence=0.8)
        self.gesture_detector = HandGestureDetector()
        self.mouse_control = MouseController()
        self.fps_reader = FPS()

        self.index_of_pointer_landmark: int = 5
        self.frame_reduction: int = 160

    def run(self):
        while self.cap.is_opened():
            success, img = self.cap.read()
            all_hands, img = self.detector.find_hands(img)

            for hand in all_hands:

                if hand.type == HandType.RIGHT:
                    # Show boundary box
                    du.draw_bounding_box(img, (self.frame_reduction, self.frame_reduction), (
                        self.cap.cam_width - self.frame_reduction, self.cap.cam_height - self.frame_reduction))

                    self.__right_hand_control(img, hand)

                if hand.type == HandType.LEFT:
                    self.__left_hand_control(hand)

            # Shop FPS
            _, img = self.fps_reader.update(img)

            cv2.imshow("HCS - preview", img)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def __right_hand_control(self, img: Any, right_hand: Hand) -> None:
        # Move pointer
        x, y = self.__calculate_pointer_position(right_hand.landmarks)
        self.mouse_control.move(x, y)

        detection_result = self.gesture_detector.predict(right_hand)

        # Draw right hand detection info
        du.draw_gesture_info(img, detection_result, right_hand.border_box)

        # Check if the hand gesture has been classified
        if detection_result:
            # Left button mouse click
            if detection_result.gesture_type == GestureType.CLICK:
                self.mouse_control.click()

            # Grab action
            if detection_result.gesture_type == GestureType.GRAB:
                self.mouse_control.grab()

            # Go back action
            if detection_result.gesture_type == GestureType.GO_BACK:
                self.mouse_control.go_back()

    def __calculate_pointer_position(self, landmarks: List[List[float]]) -> Tuple[float, float]:
        x1, y1, _ = landmarks[self.index_of_pointer_landmark]

        x2 = np.interp(x1, (self.frame_reduction, self.cap.cam_width - self.frame_reduction),
                       (0, self.mouse_control.screen_width))
        y2 = np.interp(y1, (self.frame_reduction, self.cap.cam_height - self.frame_reduction),
                       (0, self.mouse_control.screen_height))

        return x2, y2

    def __left_hand_control(self, left_hand: Hand) -> None:
        # TODO to implement
        pass
