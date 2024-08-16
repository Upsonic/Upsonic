import base64
import pyautogui

try:
    from ..gui.signal import signal_handler
    from ..utils.db import just_screenshot_path
except ImportError:
    from gui.signal import signal_handler
    from utils.db import just_screenshot_path


def encode_image(image_path):
    """
    Encode an image file to base64 format.

    Parameters:
    - image_path (str): The path to the image file to encode.

    Returns:
    - str or None: The base64 encoded string of the image, or None if an error occurs.
    """
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
    """
    Take a screenshot using pyautogui and save it.

    This function takes a screenshot of the entire screen using pyautogui,
    saves it to the specified path, and emits a signal indicating that
    the assistant is thinking.

    Returns:
    - None
    """
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(just_screenshot_path)
        signal_handler.assistant_thinking.emit()
    except Exception as e:
        print(f"An error occurred while taking the screenshot: {e}")
