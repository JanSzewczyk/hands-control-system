import cv2
import numpy as np
import autopy
from autopy.key import Code, Modifier
import time

from hand_detector import HandDetector
from camera_video_capture import CameraVideoCapture
from fps import FPS
from models import HandType, ActionType

############################
screen_width, screen_height = autopy.screen.size()
frame_reduction = 150
smoothening = 7
print(f"SCREEN SIZE: {screen_width} X {screen_height} ")  # 1920.0 X 1080.0


############################


def main():
    prev_location_x, prev_location_y = 0, 0
    curr_location_x, curr_location_y = 0, 0
    active_grab = False

    last_action = ActionType.RESET
    detector = HandDetector(max_num_hands=2)

    cap = CameraVideoCapture()

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
            # print(last_action)

            if hand.type == HandType.RIGHT:
                right_fingers_up = detector.get_fingers_up(hand)
                cv2.rectangle(img, (frame_reduction, frame_reduction),
                              (cap.cam_width - frame_reduction, cap.cam_height - frame_reduction), (255, 0, 255), 1)

                # mouse move
                # if right_fingers_up.count(1) == 5:
                x1, y1, _ = hand.landmarks[5]

                x3 = np.interp(x1, (frame_reduction, cap.cam_width - frame_reduction), (0, screen_width))
                y3 = np.interp(y1, (frame_reduction, cap.cam_height - frame_reduction), (0, screen_height))

                # Smoothen values
                curr_location_x = prev_location_x + (x3 - prev_location_x) / smoothening
                curr_location_y = prev_location_y + (y3 - prev_location_y) / smoothening

                # using int remove error in 'autopy.mouse.move()'
                autopy.mouse.move(int(screen_width - curr_location_x), int(curr_location_y))
                last_action = ActionType.RESET

                prev_location_x, prev_location_y = curr_location_x, curr_location_y

                # click if all fingers are down
                if right_fingers_up == [0, 1, 1, 0, 0]:
                    autopy.mouse.click()
                    time.sleep(0.2)
                    print('Action : ', ActionType.CLICK)

                if right_fingers_up == [0, 0, 0, 0, 0]:
                    if active_grab:
                        autopy.mouse.toggle(down=False)
                    else:
                        autopy.mouse.toggle(down=True)

                    time.sleep(0.3)
                    active_grab = not active_grab
                    print('Action : ', ActionType.GRAB, active_grab)

            if hand.type == HandType.LEFT:
                left_fingers_up = detector.get_fingers_up(hand)

                # click if all fingers are down
                if left_fingers_up.count(1) == 0:
                    autopy.key.tap(Code.LEFT_ARROW, [Modifier.ALT])
                    time.sleep(0.3)
                    # Najman's problem
                    # pyautogui.hotkey('alt', 'left')

        _, img = fps_reader.update(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
