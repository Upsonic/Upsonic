import json
import re
from langchain.tools import tool
import traceback

try:
    from .utils.db import *
    from .llm import get_model
    from .top_bar_wrapper import wrapper
    from .llm_settings import llm_settings

except ImportError:
    from utils.db import *
    from top_bar_wrapper import wrapper
    from llm_settings import llm_settings












def mouse_scroll_(direction: str, amount: int = 1) -> bool:
    """
    A function to scroll the mouse wheel.

    Parameters:
    - direction (str): The direction of the scroll. Possible values are "up" and "down".
    - amount (int): The amount of scrolling to be performed. The default value is 1.

    Returns:
    - bool: True if the scrolling was performed successfully, False otherwise.
    """
    try:
        import pyautogui

        pyautogui.FAILSAFE = False

        if direction == "up":
            pyautogui.scroll(amount)
        elif direction == "down":
            pyautogui.scroll(-amount)
        return True
    except:
        traceback.print_exc()
        return False


mouse_scroll = tool(mouse_scroll_)



def extract_code_from_result(llm_output):
    """
    Extract the Python code from the LLM output.
    """
    code_match = re.search(r'```json\n(.*?)```', llm_output, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return llm_output.strip()


def click_to_text_(text:str) -> bool:
    """
    Click on the text
   
    """

    try:
        from .cu.ask_anthropic import ask_anthropic
        from .cu.computer import click_action, mouse_move_action
    except ImportError:
        from cu.ask_anthropic import ask_anthropic
        from cu.computer import click_action, mouse_move_action

    print("click_to_text")
    print("text", text)
    x_y = ask_anthropic(f"dont use tools, give me exactly location of '{text}' text as json x,y like"+ """{'x': 0, 'y': 0}"""+". Only return the json with ```json ```")
    print("result", x_y)

    x_y = extract_code_from_result(x_y)

    x_y = json.loads(x_y)

    mouse_move_action((x_y['x'], x_y['y']))
    click_action("left_click")

    return True


click_to_text = tool(click_to_text_)



def click_to_icon_(icon:str) -> bool:
    """
    Click on the icon
   
    """

    try:
        from .cu.ask_anthropic import ask_anthropic
        from .cu.computer import click_action, mouse_move_action
    except ImportError:
        from cu.ask_anthropic import ask_anthropic
        from cu.computer import click_action, mouse_move_action

    print("click_to_icon")
    print("icon", icon)
    x_y = ask_anthropic(f"dont use tools, give me exactly location of '{icon}' icon as json x,y like"+ """{'x': 0, 'y': 0}"""+". Only return the json with ```json ```")
    print("result", x_y)

    x_y = extract_code_from_result(x_y)

    x_y = json.loads(x_y)

    mouse_move_action((x_y['x'], x_y['y']))
    click_action("left_click")

    return True


click_to_icon = tool(click_to_icon_)


def click_to_area_(
    area:str
) -> bool:
    """
    Click on the area like search bar
    """

    try:
        from .cu.ask_anthropic import ask_anthropic
        from .cu.computer import click_action, mouse_move_action
    except ImportError:
        from cu.ask_anthropic import ask_anthropic
        from cu.computer import click_action, mouse_move_action

    print("click_to_area")
    print("area", area)
    x_y = ask_anthropic(f"dont use tools, give me exactly location of '{area}' area as json x,y like"+ """{'x': 0, 'y': 0}"""+". Only return the json with ```json ```")
    print("result", x_y)

    x_y = extract_code_from_result(x_y)

    x_y = json.loads(x_y)

    mouse_move_action((x_y['x'], x_y['y']))
    click_action("left_click")

    return True




click_to_area = tool(click_to_area_)
