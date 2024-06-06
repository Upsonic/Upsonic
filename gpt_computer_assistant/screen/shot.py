import base64
import pyautogui

try:  
    from ..gui.signal import signal_handler
    from ..utils.db import just_screenshot_path
except ImportError:
    from gui.signal import signal_handler
    from utils.db import just_screenshot_path


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        return None


def take_screenshot():
    try:
        screenshot_path = get_screenshot_path()
        screenshot = pyautogui.screenshot()
        screenshot.save(just_screenshot_path)
        signal_handler.assistant_thinking.emit()
    except Exception as e:
        print(f"An error occurred while taking the screenshot: {e}")
