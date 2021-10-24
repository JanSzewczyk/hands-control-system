import time

import autopy
from autopy.key import Code, Modifier


class MouseController:

    def __init__(self, smoothing_factor: float = 7.0):
        """
        Constructor.
        """
        self._smoothing_factor = smoothing_factor

        self.screen_width, self.screen_height = autopy.screen.size()
        self._prev_location_x, self._prev_location_y = 0, 0
        self._curr_location_x, self._curr_location_y = 0, 0
        self._active_grab = False

    def move(self, x: float, y: float) -> None:
        # Smoothen values
        curr_location_x = self._prev_location_x + (x - self._prev_location_x) / self._smoothing_factor
        curr_location_y = self._prev_location_y + (y - self._prev_location_y) / self._smoothing_factor

        # using int remove error in 'autopy.mouse.move()'
        autopy.mouse.move(int(self.screen_width - curr_location_x), int(curr_location_y))

        self._prev_location_x, self._prev_location_y = curr_location_x, curr_location_y

    def click(self) -> None:
        autopy.mouse.click()
        time.sleep(0.2)

    def grab(self) -> None:
        if self._active_grab:
            autopy.mouse.toggle(down=False)
        else:
            autopy.mouse.toggle(down=True)

        time.sleep(0.3)

        self._active_grab = not self._active_grab

    def go_back(self):
        autopy.key.tap(Code.LEFT_ARROW, [Modifier.ALT])
        time.sleep(0.3)
