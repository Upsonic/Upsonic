import base64
from ..gui.signal import signal_handler
import pyautogui


from ..utils.db import just_screenshot_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def take_screenshot():
        screenshot = pyautogui.screenshot()
        screenshot.save(just_screenshot_path)
        signal_handler.assistant_thinking.emit()
