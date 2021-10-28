import cv2
from typing import Any, Tuple


class CameraVideoCapture:
    """
    Creating an instance of the VideoCapture class from the cv2 library.
    Configures and stores information about the webcam.

    Attributes:
        cap (cv2.cv2.VideoCapture.VideoCapture): VideoCapture instance.
        cam_width (int): Camera width.
        cam_height (int): Camera height.
    """

    def __init__(self, device_num: int = 0, cam_width: int = 1280, cam_height: int = 720):
        """
        Constructor.

        Args:
            device_num (int): Defaults to 0. Device id number.
            cam_width (int): Defaults to 1280. Camera width.
            cam_height (int): Defaults to 720. Camera height.
        """

        self.cap: cv2.VideoCapture = cv2.VideoCapture(device_num)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)

        self.cam_width = cam_width
        self.cam_height = cam_height

    def read(self, image: Any = None) -> Tuple[bool, Any]:
        """
        Grabs, decodes and returns the next video frame.

        Args:
            image (Any): Defaults to None. Image the video frame is returned here. If no frames has been grabbed the
                image will be empty.

        Returns:
            Tuple[bool, Any]: return value which is 'False' no frames has been grabbed, and grabbed frame
        """

        return self.cap.read(image)

    def release(self) -> None:
        """
         The method is automatically called by subsequent VideoCapture::open and by VideoCapture destructor.
        """

        self.cap.release()

    def is_opened(self) -> bool:
        """
        The method check if the previous call to VideoCapture constructor or VideoCapture::open() succeeded

        Returns:
            bool: Returns true if video capturing has been initialized already.
        """

        return self.cap.isOpened()
