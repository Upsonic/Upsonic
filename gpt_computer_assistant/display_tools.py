from langchain.tools import tool
import traceback

try:
    from .utils.db import load_api_key
    from .llm import get_model
except ImportError:
    from utils.db import load_api_key
    from llm import get_model


def click_on_a_text_on_the_screen_(text:str, click_type: str = "singular") -> bool:
    """
    A function to click on a text on the screen.

    Parameters:
    - text (str): The text to be clicked on.
    - click_type (str): The type of click to be performed. The default value is "singular". Possible values are "singular" and "double".

    Returns:
    - bool: True if the text was clicked on successfully, False otherwise.
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False


        from interpreter import OpenInterpreter





        interpreter = OpenInterpreter()

        interpreter.llm.api_key = load_api_key()

        screenshot = pyautogui.screenshot()

        text_locations = interpreter.computer.display.find_text(text, screenshot=screenshot)

        print(text_locations)


        x, y = text_locations[0]["coordinates"]
        x *= interpreter.computer.display.width
        y *= interpreter.computer.display.height
        x = int(x)
        y = int(y)

        if click_type == "singular":
            interpreter.computer.mouse.click(x=x, y=y, screenshot=screenshot)
        elif click_type == "double":
            interpreter.computer.mouse.double_click(x=x, y=y, screenshot=screenshot)
        return True
    except:
        traceback.print_exc()
        return False


click_on_a_text_on_the_screen = tool(click_on_a_text_on_the_screen_)




def move_on_a_text_on_the_screen_(text:str) -> bool:
    """
    A function to move on a text on the screen.

    Parameters:
    - text (str): The text to be moved on.

    Returns:
    - bool: True if the text was moved on successfully, False otherwise.
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False


        from interpreter import OpenInterpreter





        interpreter = OpenInterpreter()

        interpreter.llm.api_key = load_api_key()

        screenshot = pyautogui.screenshot()

        text_locations = interpreter.computer.display.find_text(text, screenshot=screenshot)

        print(text_locations)


        x, y = text_locations[0]["coordinates"]
        x *= interpreter.computer.display.width
        y *= interpreter.computer.display.height
        x = int(x)
        y = int(y)

        interpreter.computer.mouse.move(x=x, y=y, screenshot=screenshot)

        return True
    except:
        traceback.print_exc()
        return False


move_on_a_text_on_the_screen = tool(move_on_a_text_on_the_screen_)



def click_on_a_icon_on_the_screen_(icon_name:str, click_type: str = "singular") -> bool:
    """
    A function to click on a icon name on the screen.

    Parameters:
    - icon_name (str): The icon name to be clicked on.
    - click_type (str): The type of click to be performed. The default value is "singular". Possible values are "singular" and "double".

    Returns:
    - bool: True if the icon name was clicked on successfully, False otherwise.
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False


        from interpreter import OpenInterpreter


        screenshot = pyautogui.screenshot()


        interpreter = OpenInterpreter()

        interpreter.llm.api_key = load_api_key()



        if click_type == "singular":
            interpreter.computer.mouse.click(icon=icon_name, screenshot=screenshot)
        elif click_type == "double":
            interpreter.computer.mouse.double_click(icon=icon_name, screenshot=screenshot)
        return True

    except:
        traceback.print_exc()
        return False

click_on_a_icon_on_the_screen = tool(click_on_a_icon_on_the_screen_)




def move_on_a_icon_on_the_screen_(icon_name:str,) -> bool:
    """
    A function to move on a icon name on the screen.

    Parameters:
    - icon_name (str): The icon name to be move on.

    Returns:
    - bool: True if the icon name was moved on successfully, False otherwise.
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False


        from interpreter import OpenInterpreter


        screenshot = pyautogui.screenshot()


        interpreter = OpenInterpreter()

        interpreter.llm.api_key = load_api_key()

        interpreter.computer.mouse.move(icon=icon_name, screenshot=screenshot)
        return True

    except:
        traceback.print_exc()
        return False

move_on_a_icon_on_the_screen = tool(move_on_a_icon_on_the_screen_)




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
