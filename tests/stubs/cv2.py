"""Minimal cv2 stub for unit-test environments without OpenCV GUI libs."""

CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4


class VideoCapture:
    def __init__(self, *args, **kwargs):
        self._opened = True

    def set(self, *args, **kwargs):
        return True

    def release(self):
        self._opened = False
        return None


def imread(*args, **kwargs):
    return None


def imwrite(*args, **kwargs):
    return True
