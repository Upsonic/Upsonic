try:
    from ..utils.db import *
    from ..llm import get_model
    from ..top_bar_wrapper import wrapper
    from ..llm_settings import llm_settings
except ImportError:
    from utils.db import *
    from top_bar_wrapper import wrapper
    from llm_settings import llm_settings

from langchain.tools import tool


import base64
import math
import os
import platform
import shlex
import shutil
import tempfile
import time
from strenum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
import pyautogui
from anthropic.types.beta import BetaToolComputerUse20241022Param
from PIL import Image

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),
    "WXGA": Resolution(width=1280, height=800),
    "FWXGA": Resolution(width=1366, height=768),
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def smooth_move_to(x, y, duration=1.2):
    start_x, start_y = pyautogui.position()
    dx = x - start_x
    dy = y - start_y
    distance = math.hypot(dx, dy)

    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if (elapsed_time > duration):
            break

        t = elapsed_time / duration
        eased_t = (1 - math.cos(t * math.pi)) / 2

        target_x = start_x + dx * eased_t
        target_y = start_y + dy * eased_t
        pyautogui.moveTo(target_x, target_y)

    pyautogui.moveTo(x, y)


def key_action(text: str):
    if platform.system() == "Darwin":
        text = text.replace("super+", "command+")

    def normalize_key(key):
        key = key.lower().replace("_", "")
        key_map = {
            "pagedown": "pgdn",
            "pageup": "pgup",
            "enter": "return",
            "return": "enter",
        }
        return key_map.get(key, key)

    keys = [normalize_key(k) for k in text.split("+")]

    if len(keys) > 1:
        if "darwin" in platform.system().lower():
            keystroke, modifier = (keys[-1], "+".join(keys[:-1]))
            modifier = modifier.lower() + " down"
            if keystroke.lower() == "space":
                keystroke = " "
            elif keystroke.lower() == "enter":
                keystroke = "\n"
            script = f"""
            tell application "System Events"
                keystroke "{keystroke}" using {modifier}
            end tell
            """
            os.system("osascript -e '{}'".format(script))
        else:
            pyautogui.hotkey(*keys)
    else:
        pyautogui.press(keys[0])


def type_action(text: str):
    pyautogui.write(text, interval=TYPING_DELAY_MS / 1000)


def click_action(action: str):
    time.sleep(0.1)
    button = {
        "left_click": "left",
        "right_click": "right",
        "middle_click": "middle",
    }
    if action == "double_click":
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.click()
    else:
        pyautogui.click(button=button.get(action, "left"))


def screenshot_action():
    temp_dir = Path(tempfile.gettempdir())
    path = temp_dir / f"screenshot_{uuid4().hex}.png"

    screenshot = pyautogui.screenshot()
    screenshot.save(str(path))

    if _scaling_enabled:
        x, y = scale_coordinates(ScalingSource.COMPUTER, width, height)
        print(f"Scaling screenshot to {x}x{y}")
        with Image.open(path) as img:
            img = img.resize((x, y), Image.Resampling.LANCZOS)
            img.save(path)

    if path.exists():

        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        path.unlink()
        return ToolResult(base64_image=base64_image)
    raise ToolError(f"Failed to take screenshot")


def screenshot_action_(path):


    screenshot = pyautogui.screenshot()
    screenshot.save(str(path))

    if _scaling_enabled:
        x, y = scale_coordinates(ScalingSource.COMPUTER, width, height)
        print(f"Scaling screenshot to {x}x{y}")
        with Image.open(path) as img:
            img = img.resize((x, y), Image.Resampling.LANCZOS)
            img.save(path)




def cursor_position_action():
    x, y = pyautogui.position()
    x, y = scale_coordinates(ScalingSource.COMPUTER, x, y)
    return ToolResult(output=f"X={x},Y={y}")


def shell_action(command: str, take_screenshot=True) -> ToolResult:
    _, stdout, stderr = run(command)
    base64_image = None

    if take_screenshot:
        time.sleep(_screenshot_delay)
        base64_image = screenshot_action().base64_image

    return ToolResult(output=stdout, error=stderr, base64_image=base64_image)


def scale_coordinates(source: ScalingSource, x: int, y: int):
    if not _scaling_enabled:
        return x, y
    ratio = width / height
    target_dimension = None
    for dimension in MAX_SCALING_TARGETS.values():
        if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
            if dimension["width"] < width:
                target_dimension = dimension
            break
    if target_dimension is None:
        return x, y
    x_scaling_factor = target_dimension["width"] / width
    y_scaling_factor = target_dimension["height"] / height
    if source == ScalingSource.API:
        if x > width or y > height:
            raise ToolError(f"Coordinates {x}, {y} are out of bounds")
        return round(x / x_scaling_factor), round(y / y_scaling_factor)
    return round(x * x_scaling_factor), round(y * y_scaling_factor)


@wrapper
def computer_tool_(action: Action, text: str | None = None, coordinate: tuple[int, int] | None = None, **kwargs):
    """
    Perform the specified action using the computer.

    - action: The action to perform.
        - key: Press a key.
        - type: Type text.
        - mouse_move: Move the mouse to the specified coordinate.
        - left_click: Perform a left click.
        - right_click: Perform a right click.
        - middle_click: Perform a middle click.
        - double_click: Perform a double click.
        - left_click_drag: Perform a left click and drag.

    - text: The text to type or the key to press.
    - coordinate: The coordinate to move the mouse to or click at.

    """
    print(f"ComputerTool: action={action}, text={text}, coordinate={coordinate}")
    if action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            raise ToolError(f"coordinate is required for {action}")
        x, y = scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])

        if action == "mouse_move":
            smooth_move_to(x, y)
        elif action == "left_click_drag":
            smooth_move_to(x, y)
            pyautogui.dragTo(x, y, button="left")

    elif action in ("key", "type"):
        if text is None:
            raise ToolError(f"text is required for {action}")

        if action == "key":
            key_action(text)
        elif action == "type":
            type_action(text)

    elif action in ("left_click", "right_click", "double_click", "middle_click"):
        click_action(action)


    else:
        raise ToolError(f"Invalid action: {action}")





# Initialize global variables
width, height = pyautogui.size()

display_num = None
_screenshot_delay = 2.0
_scaling_enabled = True



computer_tool = tool(computer_tool_)