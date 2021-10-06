import cv2
import pyautogui
import numpy as np
import mouse

import HandDetector as hd
from FPS import FPS
from models.HandType import HandType

############################
cam_width, cam_height = 1280, 720
screen_width, screen_height = pyautogui.size()
frame_reduction = 150
smoothening = 5


############################

def main():
    detector = hd.HandDetector(max_num_hands=2)

    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    fps_reader = FPS()

    while True:
        success, img = cap.read()
        all_hands, img = detector.find_hands(img)

        for hand in all_hands:
            """
            MOUSE CONTROL:
            RIGHT hand 
            5th landmark on hand 
            """
            print(hand.landmarks)

            if hand.type == HandType.RIGHT:
                x1, y1, _ = hand.landmarks[5]
                right_fingers_up = detector.get_fingers_up(hand)

                x3 = np.interp(x1, (frame_reduction, cam_width - frame_reduction), (0, screen_width))
                y3 = np.interp(y1, (frame_reduction, cam_height - frame_reduction), (0, screen_height))

                # mouse move
                if right_fingers_up.count(1) == 5:
                    mouse.move(screen_width - x3, y3)

                # click if all fingers are down
                if right_fingers_up.count(1) == 0:
                    mouse.click()

            if hand.type == HandType.LEFT:
                left_fingers_up = detector.get_fingers_up(hand)

                # click if all fingers are down
                if left_fingers_up.count(1) == 0:
                    pyautogui.hotkey('alt', 'left')

        _, img = fps_reader.update(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
