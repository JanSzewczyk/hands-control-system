import time
import cv2

from typing import Tuple, Any


class FPS:
    """
    FPS helps in computing Frames Per Second and displaying on an OpenCV Image.

    Attributes:
        prev_time (float): Previous frame display time.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.prev_time: float = time.time()

    def update(self, img: Any, position: Tuple[int, int] = (20, 50), color: Tuple[int, int, int] = (255, 0, 0),
               scale: int = 3, thickness: int = 3) -> Tuple[float, Any]:
        """
        Update the frame rate.

        Args:
            img (Any): Image to display on, can be left blank if only fps value required.
            position (Tuple[int, int]): Defaults to (20, 50). Position on the fps on the image.
            color (Tuple[int, int, int]): Defaults to (255, 0, 0). Color of the fps Value displayed.
            scale (int): Defaults to 3. Scale of the fps Value displayed.
            thickness: Defaults to 3. Thickness of the fps Value displayed.

        Returns:
            Tuple[float, Any]: Frames per second with Image.
        """

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Draw fps on the image
        cv2.putText(img, f"fps: {int(fps)}", position, cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)

        return fps, img
