from langchain.tools import tool
import traceback

try:
    from .utils.db import load_api_key
except ImportError:
    from utils.db import load_api_key


def Tool(func):
    """
    A decorator function to register a tool with the custom tools list.

    Parameters:
    - func (callable): The function to be registered as a tool.

    Returns:
    - callable: The input function `func` unchanged.
    """
    from .agent.agent import custom_tools
    global custom_tools
    custom_tools.append(tool(func))
    return func


def click_on_a_text_on_the_screen(text:str, click_type: str = "singular") -> bool:
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


def click_on_a_icon_on_the_screen(icon_name:str, click_type: str = "singular") -> bool:
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

