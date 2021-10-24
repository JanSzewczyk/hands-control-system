import cv2
from typing import Any, Tuple


class CameraVideoCapture:
    """
    Creating an instance of the VideoCapture class from the cv2 library.
    Configures and stores information about the webcam.

    Attributes:
        cap: VideoCapture instance.
        cam_width: Camera width.
        cam_height: Camera height.
    """

    def __init__(self, device_num: int = 0, cam_width: int = 1280, cam_height: int = 720):
        """
        Constructor.

        Args:
            device_num: Device id number.
            cam_width: Camera width.
            cam_height: Camera height.
        """
        self.cap = cv2.VideoCapture(device_num)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)

        self.cam_width = cam_width
        self.cam_height = cam_height

    def read(self, image=None) -> Tuple[bool, Any]:
        return self.cap.read(image)

    def release(self) -> None:
        self.cap.release()

    def is_opened(self) -> bool:
        return self.cap.isOpened()
