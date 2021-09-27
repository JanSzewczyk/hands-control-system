import time
import cv2


class FPS:
    """
    FPS helps in computing Frames Per Second and displaying on an OpenCV Image
    """

    def __init__(self):
        self.prev_time = time.time()

    def update(self, img=None, position=(20, 50), color=(255, 0, 0), scale=3, thickness=3) -> [float, any]:
        """
        Update the frame rate.
        :param img: Image to display on, can be left blank if only fps value required.
        :param position: Position on the FPS on the image.
        :param color: Color of the FPS Value displayed.
        :param scale: Scale of the FPS Value displayed.
        :param thickness: Thickness of the FPS Value displayed.
        :return: Frames per second with or without Image.
        """

        current_time = time.time()
        try:
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            if img is None:
                return fps
            else:
                cv2.putText(img, f"FPS: {int(fps)}", position, cv2.FONT_HERSHEY_PLAIN,
                            scale, color, thickness)
                return fps, img
        except:
            return 0
