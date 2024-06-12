from langchain.tools import tool
import traceback

try:
    from .utils.db import load_api_key
    from .llm import get_model
except ImportError:
    from utils.db import load_api_key
    from llm import get_model


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


def search_on_internet_and_report_team_(the_subject:str, copy_to_clipboard: bool=False) -> str:
    """
    A function to search the internet generates a report. Just use in detailed searches

    Parameters:
    - the_subject (str): The subject to search the internet for.
    - copy_to_clipboard (bool): A flag to indicate whether to copy the report to the clipboard. The default value is False.

    Returns:
    - str: The report of the search.
    """



    from crewai import Task, Crew, Agent

    from .agent.agent import tools

    the_tool_list = []
    for each in tools:
        if "team" not in each.name:
            the_tool_list.append(each)

    # Create the agents


    search_engine_master = Agent(
        role="search_engine_master",
        goal="Search the internet",
        backstory="I am the search engine master",
        max_iter=15
    )


    report_generator = Agent(
        role="report_generator",
        goal="Generate a report",
        backstory="I am the report generator",
        max_iter=15
    )

    agents = [search_engine_master, report_generator]


    print("Tools:", the_tool_list)

    task = Task(
        description=f"Make a search about {the_subject} in the search engines and get the websites", expected_output="Website list", agent=search_engine_master, tools=the_tool_list
    )

    task_2 = Task(
        description="Read the websites and summarize the information", expected_output="Summary", agent=report_generator, tools=the_tool_list, context=[task]
    )


    task_3 = Task(
        description="Generate a report", expected_output="Report", agent=report_generator, tools=the_tool_list, context=[task, task_2]
    )


    the_tasks = []

    if copy_to_clipboard:
        task_4 = Task(
            description="Copy the report to the clipboard", expected_output="Success", agent=report_generator, tools=the_tool_list, context=[task_2]
        )
        the_tasks = [task, task_2, task_3, task_4]

    else:
        the_tasks = [task, task_2, task_3]


    the_crew = Crew(
        llm=get_model(high_context=True),
        agents=agents,
        tasks=the_tasks,
        full_output=True,
        verbose=True,
    )

    result = the_crew.kickoff()["final_output"]


search_on_internet_and_report_team = tool(search_on_internet_and_report_team_)