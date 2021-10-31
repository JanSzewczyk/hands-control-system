import time

import autopy
from autopy.key import Code, Modifier


class MouseController:
    """
    Perform a mouse action using autopy library. Allows you to change the position of the mouse pointer.
    Offers actions that can be performed with a computer mouse such as:
        * click - left mouse button click
        * grab - grab and moving elements
        * go_back - an action imitating a keyboard shortcut LEFT_ARROW + ALT

    Attributes:
        screen_width (float): Device screen width.
        screen_height (float): Device screen height.
        _smoothing_factor (float): Mouse marker movement smoothing factor.
        _prev_location_x (float): X previous mouse marker location.
        _prev_location_y (float): Y previous mouse marker location.
        _curr_location_x (float): X current mouse marker location.
        _curr_location_y (float): Y current mouse marker location.
        _active_grab (bool): Flag active grab action.
    """

    def __init__(self, smoothing_factor: float = 7.0):
        """
        Constructor.

        Args:
            smoothing_factor (float): Defaults to 7.0 . Mouse marker movement smoothing factor.
        """

        self._smoothing_factor: float = smoothing_factor

        self.screen_width, self.screen_height = autopy.screen.size()
        self._prev_location_x, self._prev_location_y = 0, 0
        self._curr_location_x, self._curr_location_y = 0, 0
        self._active_grab: bool = False

    def move(self, x: float, y: float) -> None:
        """
        Move mouse pointer action.

        Args:
            x (float): X mouse marker location.
            y (float): Y mouse marker location.
        """

        # Smoothen values
        curr_location_x = self._prev_location_x + (x - self._prev_location_x) / self._smoothing_factor
        curr_location_y = self._prev_location_y + (y - self._prev_location_y) / self._smoothing_factor

        # using int remove error in 'autopy.mouse.move()'
        autopy.mouse.move(int(self.screen_width - curr_location_x), int(curr_location_y))

        self._prev_location_x, self._prev_location_y = curr_location_x, curr_location_y

    def click(self) -> None:
        """
        Mouse left button click action.
        """

        autopy.mouse.click()

        # Reset grab flag
        self._reset_grab_action()

        time.sleep(0.2)

    def grab(self) -> None:
        """
        The action of grabbing items with the mouse.
        """

        if self._active_grab:
            autopy.mouse.toggle(down=False)
        else:
            autopy.mouse.toggle(down=True)

        time.sleep(0.3)

        self._active_grab = not self._active_grab

    def go_back(self) -> None:
        """
        An action imitating a keyboard shortcut LEFT_ARROW + ALT.
        """

        autopy.key.tap(Code.LEFT_ARROW, [Modifier.ALT])
        time.sleep(0.3)

    def _reset_grab_action(self) -> None:
        """
        Reset grab flag if is active.
        """

        if self._active_grab:
            self._active_grab = False
