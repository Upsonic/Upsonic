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



def click_to_text_(text:str):
    """
    Click on the text
   
    """

    try:
        from .cu.ask_anthropic import ask_anthropic
    except ImportError:
        from cu.ask_anthropic import ask_anthropic

    print("click_to_text")
    print("text", text)
    x_y = ask_anthropic(f"dont use tools, give me exactly location of '{text}' text as x,y")
    print("result", x_y)

    result = ask_anthropic(f"click on {x_y} text")

    return result


click_to_text = tool(click_to_text_)



def click_to_icon_(icon:str):
    """
    Click on the icon
   
    """

    try:
        from .cu.ask_anthropic import ask_anthropic
    except ImportError:
        from cu.ask_anthropic import ask_anthropic

    print("click_to_icon")
    print("icon", icon)
    x_y = ask_anthropic(f"dont use tools, give me exactly location of '{icon}' icon as x,y")
    print("result", x_y)

    result = ask_anthropic(f"click on {x_y} icon")

    return result   


click_to_icon = tool(click_to_icon_)



