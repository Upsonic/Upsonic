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


def screenshot_action(direct_base64: bool = False) -> ToolResult:
    """
    See the screenshot of the current screen.
    """


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
        if direct_base64:
            return base64_image
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
    """
    Get the current position of the cursor as (x, y).
    """

    x, y = pyautogui.position()
    x, y = scale_coordinates(ScalingSource.COMPUTER, x, y)
    return ToolResult(output=f"X={x},Y={y}")


def shell_action(command: str, take_screenshot=True) -> ToolResult:
    """
    Run a shell command and return the output.
    """
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
def mouse_move_action(coordinate: tuple[int, int]):
    """Move the mouse to the specified coordinate."""
    if coordinate is None:
        raise ToolError("coordinate is required for mouse_move")
    x, y = scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])
    smooth_move_to(x, y)

@wrapper
def left_click_drag_action(coordinate: tuple[int, int]):
    """Perform a left click and drag to the specified coordinate."""
    if coordinate is None:
        raise ToolError("coordinate is required for left_click_drag")
    x, y = scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])
    smooth_move_to(x, y)
    pyautogui.dragTo(x, y, button="left")

@wrapper
def key_action_handler(text: str):
    """Press a specific key."""
    if text is None:
        raise ToolError("text is required for key")
    key_action(text)

@wrapper
def type_action_handler(text: str):
    """Type the specified text."""
    if text is None:
        raise ToolError("text is required for type")
    type_action(text)

@wrapper
def left_click_action():
    """Perform a left click."""
    click_action("left_click")

@wrapper
def right_click_action():
    """Perform a right click."""
    click_action("right_click")

@wrapper
def middle_click_action():
    """Perform a middle click."""
    click_action("middle_click")

@wrapper
def double_click_action():
    """Perform a double click."""
    click_action("double_click")








# Initialize global variables
width, height = pyautogui.size()

display_num = None
_screenshot_delay = 2.0
_scaling_enabled = True



computer_tool = [tool(mouse_move_action), tool(left_click_drag_action), tool(key_action_handler), tool(type_action_handler), tool(left_click_action), tool(right_click_action), tool(middle_click_action), tool(double_click_action), tool(screenshot_action), tool(cursor_position_action), tool(shell_action)]
