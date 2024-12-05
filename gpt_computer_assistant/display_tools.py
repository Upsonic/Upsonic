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






@wrapper
def ocr_test_(psm:int =11, oem:int=3, confidence:int=70) -> dict:
    """
    A function to extract possible coordinates of a text on the screen.
    """

    try:

        import pyautogui
        import pytesseract
        from PIL import ImageGrab
        import cv2
        import numpy as np





        def capture_screen():
            # Capture the entire screen
            screenshot = ImageGrab.grab()
            screenshot_np = np.array(screenshot)
            return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        def find_all_text_coordinates(confidence=confidence):
            # Capture the screen and convert it to grayscale
            img = capture_screen()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Use pytesseract to get data about the text on the screen
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=f"--psm {psm} --oem {oem}")

            texts_with_coordinates = []
            for i in range(len(data['text'])):
                if data['text'][i].strip() and int(data['conf'][i]) > confidence:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    texts_with_coordinates.append({
                        'text': data['text'][i],
                        'coordinates': (x + w // 2, y + h // 2)
                    })
            
            return texts_with_coordinates


        # return a list of possible coordinates of the text
        texts_with_coordinates = find_all_text_coordinates()
        result =  [{"name": item["text"], "coordinates": {"x":item["coordinates"][0], "y":item["coordinates"][1]}} for item in texts_with_coordinates]
        return result

    except:
        traceback.print_exc()
        exception_str = traceback.format_exc()
        return {"error": exception_str}

ocr_test = tool(ocr_test_)



@wrapper
def extract_possible_coordinates_of_text_(text:str) -> dict:
    """
    A function to extract possible coordinates of a text on the screen.
    """


    import pyautogui
    import pytesseract
    from PIL import ImageGrab
    import cv2
    import numpy as np





    def capture_screen():
        # Capture the entire screen
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    def find_all_text_coordinates(confidence=70):
        # Capture the screen and convert it to grayscale
        img = capture_screen()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use pytesseract to get data about the text on the screen
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--psm 11')

        texts_with_coordinates = []
        for i in range(len(data['text'])):
            if data['text'][i].strip() and int(data['conf'][i]) > confidence:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                texts_with_coordinates.append({
                    'text': data['text'][i],
                    'coordinates': (x + w // 2, y + h // 2)
                })
        
        return texts_with_coordinates


    # return a list of possible coordinates of the text
    texts_with_coordinates = find_all_text_coordinates()
    result =  [{"name": item["text"], "coordinates": {"x":item["coordinates"][0], "y":item["coordinates"][1]}} for item in texts_with_coordinates if item["text"].startswith(text)]
    return result


extract_possible_coordinates_of_text = tool(extract_possible_coordinates_of_text_)



@wrapper
def click_on_a_text_on_the_screen_(text: str, click_type: str = "singular") -> bool:
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
        from PIL import ImageGrab, ImageDraw

        pyautogui.FAILSAFE = False

        from interpreter import OpenInterpreter

        interpreter = OpenInterpreter()

        model = load_model_settings()
        if llm_settings[model]["provider"] == "azureai":
            interpreter.llm.model = f"azure/{model}"
            import os 
            os.environ["AZURE_API_KEY"] = load_api_key()
            os.environ["AZURE_API_BASE"] = load_openai_url()
            os.environ["AZURE_API_VERSION"] = load_api_version()

        else:
            interpreter.llm.api_key = load_api_key()


        text_locations = extract_possible_coordinates_of_text_(text)

        x, y = text_locations[0]["coordinates"]["x"], text_locations[0]["coordinates"]["y"]
 


        screen_width, screen_height = pyautogui.size()
        screenshot = ImageGrab.grab()
        screenshot_width, screenshot_height = screenshot.size

        # Calculate scaling factors
        scale_x = screen_width / screenshot_width
        scale_y = screen_height / screenshot_height

        # Apply scaling factors to coordinates
        scaled_x = x * scale_x
        scaled_y = y * scale_y

        if click_type == "singular":
            interpreter.computer.mouse.click(x=scaled_x, y=scaled_y)
        elif click_type == "double":
            interpreter.computer.mouse.double_click(x=scaled_x, y=scaled_y)
        return True
    except:
        traceback.print_exc()
        return False


click_on_a_text_on_the_screen = tool(click_on_a_text_on_the_screen_)


@wrapper
def move_on_a_text_on_the_screen_(text: str) -> bool:
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

        model = load_model_settings()
        if llm_settings[model]["provider"] == "azureai":
            interpreter.llm.model = f"azure/{model}"
            import os 
            os.environ["AZURE_API_KEY"] = load_api_key()
            os.environ["AZURE_API_BASE"] = load_openai_url()
            os.environ["AZURE_API_VERSION"] = load_api_version()

        else:
            interpreter.llm.api_key = load_api_key()

        screenshot = pyautogui.screenshot()

        text_locations = interpreter.computer.display.find_text(
            text, screenshot=screenshot
        )

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


@wrapper
def click_on_a_icon_on_the_screen_(
    icon_name: str, click_type: str = "singular"
) -> bool:
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

        model = load_model_settings()
        if llm_settings[model]["provider"] == "azureai":
            interpreter.llm.model = f"azure/{model}"
            import os 
            os.environ["AZURE_API_KEY"] = load_api_key()
            os.environ["AZURE_API_BASE"] = load_openai_url()
            os.environ["AZURE_API_VERSION"] = load_api_version()

        else:
            interpreter.llm.api_key = load_api_key()

        if click_type == "singular":
            interpreter.computer.mouse.click(icon=icon_name, screenshot=screenshot)
        elif click_type == "double":
            interpreter.computer.mouse.double_click(
                icon=icon_name, screenshot=screenshot
            )
        return True

    except:
        traceback.print_exc()
        return False


click_on_a_icon_on_the_screen = tool(click_on_a_icon_on_the_screen_)


@wrapper
def move_on_a_icon_on_the_screen_(
    icon_name: str,
) -> bool:
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

        model = load_model_settings()
        if llm_settings[model]["provider"] == "azureai":
            interpreter.llm.model = f"azure/{model}"
            import os 
            os.environ["AZURE_API_KEY"] = load_api_key()
            os.environ["AZURE_API_BASE"] = load_openai_url()
            os.environ["AZURE_API_VERSION"] = load_api_version()

        else:
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


@wrapper
def get_texts_on_the_screen_() -> str:
    """
    It returns the texts on the screen.
    """

    try:
        pass

    except:
        pass

    import pyautogui

    the_screenshot_path = "temp_screenshot.png"
    the_screenshot = pyautogui.screenshot()
    the_screenshot.save(the_screenshot_path)

    from interpreter.core.computer.utils.computer_vision import pytesseract_get_text

    return pytesseract_get_text(the_screenshot_path)


get_texts_on_the_screen = tool(get_texts_on_the_screen_)
