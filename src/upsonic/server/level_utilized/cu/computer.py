import asyncio
import base64
import math
import os
import platform
import shlex
import shutil
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

# Add import for PyAutoGUI
import pyautogui
from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool
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


class ScalingMode(StrEnum):
    AUTO = "auto"  # Automatically determine best scaling
    FIXED = "fixed"  # Use fixed target resolutions
    RELATIVE = "relative"  # Scale by percentage
    NONE = "none"  # No scaling


# Base resolutions for different device categories
DEVICE_CATEGORIES: dict[str, Resolution] = {
    "HD": Resolution(width=1280, height=720),      # 720p
}

class ScalingConfig(TypedDict):
    mode: ScalingMode
    target_resolution: Resolution | None  # Used for FIXED mode
    scale_factor: float | None  # Used for RELATIVE mode
    min_scale: float  # Minimum scaling factor
    max_scale: float  # Maximum scaling factor
    preserve_aspect_ratio: bool


DEFAULT_SCALING_CONFIG = ScalingConfig(
    mode=ScalingMode.RELATIVE,
    target_resolution=None,
    scale_factor=1.0,  # No scaling by default for HD resolution
    min_scale=0.25,
    max_scale=1.0,
    preserve_aspect_ratio=True
)

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
    distance = math.hypot(dx, dy)  # Calculate the distance in pixels

    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break

        t = elapsed_time / duration
        eased_t = (1 - math.cos(t * math.pi)) / 2  # easeInOutSine function

        target_x = start_x + dx * eased_t
        target_y = start_y + dy * eased_t
        pyautogui.moveTo(target_x, target_y)

    # Ensure the mouse ends up exactly at the target (x, y)
    pyautogui.moveTo(x, y)


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the primary monitor's screen, keyboard, and mouse.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: None
    scaling_config: ScalingConfig

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()
        self.width, self.height = pyautogui.size()
        self.display_num = None
        self.scaling_config = self._determine_optimal_scaling()

    def _determine_optimal_scaling(self) -> ScalingConfig:
        """Determine the optimal scaling configuration for HD displays (1280x720)."""
        config = DEFAULT_SCALING_CONFIG.copy()
        
        # Get current resolution
        print(f"Current resolution: {self.width}x{self.height}")
        
        # For 720p displays, use simple 1:1 scaling (no scaling)
        if abs(self.width - 1280) <= 100 and abs(self.height - 720) <= 100:
            config["mode"] = ScalingMode.RELATIVE
            config["scale_factor"] = 1.0
            print("Using 1:1 scaling for HD (720p) display")
        else:
            # For any other resolution, scale to match HD
            config["mode"] = ScalingMode.FIXED
            config["target_resolution"] = DEVICE_CATEGORIES["HD"]
            print(f"Scaling to target HD resolution: 1280x720")
        
        # Print final configuration
        print(f"Scaling config: mode={config['mode']}, scale_factor={config['scale_factor']}")
        return config

    def scale_coordinates(self, source: ScalingSource, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates based on the current scaling configuration."""
        if not self._scaling_enabled:
            return x, y

        if self.scaling_config["mode"] == ScalingMode.NONE:
            return x, y

        # Get current scaling factors
        x_factor = 1.0
        y_factor = 1.0

        if self.scaling_config["mode"] == ScalingMode.FIXED and self.scaling_config["target_resolution"]:
            # Fixed mode - scale to target resolution
            x_factor = self.scaling_config["target_resolution"]["width"] / self.width
            y_factor = self.scaling_config["target_resolution"]["height"] / self.height
            
            if self.scaling_config["preserve_aspect_ratio"]:
                # Use the same factor for both dimensions to preserve aspect ratio
                x_factor = y_factor = min(x_factor, y_factor)
        
        elif self.scaling_config["mode"] == ScalingMode.RELATIVE and self.scaling_config["scale_factor"] is not None:
            # Relative mode - use scale factor directly
            x_factor = y_factor = self.scaling_config["scale_factor"]
        
        # Apply scaling based on source
        if source == ScalingSource.API:
            # Scale up (from scaled coordinates to actual screen coordinates)
            return round(x / x_factor), round(y / y_factor)
        else:
            # Scale down (from actual screen coordinates to scaled coordinates)
            return round(x * x_factor), round(y * y_factor)

    def update_scaling_config(self, new_config: dict) -> None:
        """Update the scaling configuration with new settings."""
        self.scaling_config.update(new_config)

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print("action", action)
        print("text", text)
        print("coordinate", coordinate)
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                return {"text": f"coordinate is required for {action}"}
            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                smooth_move_to(x, y)
                return {"text": f"Mouse moved to X={x}, Y={y}"}
            elif action == "left_click_drag":
                smooth_move_to(x, y)
                pyautogui.dragTo(x, y, button="left")
                return {"text": f"Mouse dragged to X={x}, Y={y}"}

        elif action in ("key", "type"):
            if text is None:
                return {"text": f"text is required for {action}"}

            if action == "key":
                if platform.system() == "Darwin":  # Check if we're on macOS
                    text = text.replace("super+", "command+")

                # Normalize key names
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
                        # Use AppleScript for hotkey on macOS
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
                return {"text": f"Key pressed: {text}"}
            elif action == "type":
                pyautogui.write(text, interval=TYPING_DELAY_MS / 1000)
                return {"text": f"Text typed: {text}"}

        elif action in ("left_click", "right_click", "double_click", "middle_click"):
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
                return {"text": "Double click performed"}
            else:
                pyautogui.click(button=button.get(action, "left"))
                return {"text": f"{action.replace('_', ' ').title()} performed"}

        elif action == "screenshot":
            screenshot_result = self.screenshot()
            return {"type": "image", "source": screenshot_result["source"]}

        elif action == "cursor_position":
            x, y = pyautogui.position()
            x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
            return {"text": f"X={x},Y={y}"}

        else:
            return {"text": f"Invalid action: {action}"}

        # Take a screenshot after the action (except for cursor_position)
        if action != "cursor_position":
            screenshot_result = self.screenshot()
            return {
                "type": "image",
                "text": f"Action '{action}' completed",
                "source": screenshot_result["source"]
            }

    def screenshot(self, return_bytes=False):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        temp_dir = Path(tempfile.gettempdir())
        path = temp_dir / f"screenshot_{uuid4().hex}.png"

        screenshot = pyautogui.screenshot()
        
        # Save original screenshot
        screenshot.save(str(path))
        print(f"Original file size: {os.path.getsize(path)} bytes")

        # Only apply scaling if enabled and necessary
        if self._scaling_enabled and self.scaling_config["mode"] != ScalingMode.NONE:
            from PIL import Image

            # Get target dimensions
            if self.scaling_config["mode"] == ScalingMode.FIXED and self.scaling_config["target_resolution"]:
                # Fixed mode - use target resolution directly
                target_width = self.scaling_config["target_resolution"]["width"]
                target_height = self.scaling_config["target_resolution"]["height"]
            else:
                # Use relative scaling
                scale = self.scaling_config.get("scale_factor", 1.0)
                target_width = int(self.width * scale)
                target_height = int(self.height * scale)
            
            # Only resize if dimensions are different
            if target_width != self.width or target_height != self.height:
                print(f"Resizing screenshot to {target_width}x{target_height}")
                with Image.open(path) as img:
                    # Resize with high-quality downsampling
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    # Save with optimization
                    img.save(path, optimize=True, quality=90)

        if path.exists():
            print(f"Final screenshot size: {os.path.getsize(path)} bytes")
            if return_bytes:
                return path.read_bytes()
            else:
                base64_image = base64.b64encode(path.read_bytes()).decode()
                path.unlink()  # Remove the temporary file

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                }
            }
                
        return {"text": "Failed to take screenshot"}

    async def shell(self, command: str, take_screenshot=True):
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        result = {"text": stdout}
        if stderr:
            result["text"] += f"\nError: {stderr}"

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            screenshot_result = await self.screenshot()
            result = {
                "type": "image",
                "text": result["text"],
                "source": screenshot_result["source"]
            }

        return result


async def ComputerUse__type(text: str):
    """Execute a typing action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="type", text=text)

async def ComputerUse__key(text: str):
    """Execute a key press action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="key", text=text)

async def ComputerUse__mouse_move(coordinate: list[int, int]):
    """Execute a mouse move action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="mouse_move", coordinate=coordinate)

async def ComputerUse__left_click():
    """Execute a left click action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="left_click")

async def ComputerUse__right_click():
    """Execute a right click action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="right_click")

async def ComputerUse__middle_click():
    """Execute a middle click action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="middle_click")

async def ComputerUse__double_click():
    """Execute a double click action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="double_click")

async def ComputerUse__left_click_drag(coordinate: list[int, int]):
    """Execute a left click drag action using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="left_click_drag", coordinate=coordinate)

async def ComputerUse__screenshot():
    """Take a screenshot using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="screenshot")

def ComputerUse_screenshot_tool():
    """Take a screenshot using the ComputerTool and return the base64 encoded image."""
    tool = ComputerTool()
    return tool.screenshot()

def ComputerUse_screenshot_tool_bytes():
    """Take a screenshot using the ComputerTool and return the bytes."""
    tool = ComputerTool()
    return tool.screenshot(return_bytes=True)

async def ComputerUse__cursor_position():
    """Get the current cursor position using the ComputerTool."""
    tool = ComputerTool()
    return await tool(action="cursor_position")

# List of all computer use tools
ComputerUse_tools = [
    ComputerUse__type,
    ComputerUse__key,
    ComputerUse__mouse_move,
    ComputerUse__left_click,
    ComputerUse__right_click,
    ComputerUse__middle_click,
    ComputerUse__double_click,
    ComputerUse__left_click_drag,
    ComputerUse__screenshot,
    ComputerUse__cursor_position
]

