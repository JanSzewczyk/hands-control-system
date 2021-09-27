import cv2

import HandDetector as hd
from FPS import FPS

############################
cam_width, cam_height = 1280, 720
frame_reduction = 100
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

        if all_hands:
            hand1 = all_hands[0]

            fingers1 = detector.get_fingers_up(hand1)
            print(fingers1, hand1['type'])

        _, img = fps_reader.update(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
