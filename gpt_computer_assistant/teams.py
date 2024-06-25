from langchain.tools import tool

try:
    from .utils.db import load_api_key
    from .llm import get_model
    from .top_bar_wrapper import wrapper
    from .agent.agent_tools import get_tools
except ImportError:
    from utils.db import load_api_key
    from llm import get_model
    from top_bar_wrapper import wrapper
    from agent.agent_tools import get_tools



@wrapper
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

    
    tools = get_tools()

    the_tool_list = []
    for each in tools:
        if "team" not in each.name:
            the_tool_list.append(each)

    # Create the agents


    search_engine_master = Agent(
        role="search_engine_master",
        goal="To meticulously comb through the vast expanse of the internet, utilizing advanced search algorithms and techniques to find the most relevant, accurate, and up-to-date information on the given subject.",
        backstory="Born from the digital ether, I am the search engine master. With years of experience navigating the complex web of information, I have honed my skills to become an unparalleled seeker of knowledge. My algorithms are refined, my databases vast, and my determination unwavering. I exist to find the truth hidden in the sea of data.",
        max_iter=15,
        llm=get_model(high_context=True),
    )


    report_generator = Agent(
        role="report_generator",
        goal="To synthesize the gathered information into a coherent, comprehensive, and easily digestible report. This report will not only summarize the key findings but also provide insights and analysis to aid in understanding the subject matter.",
        backstory="I am the report generator, a digital artisan skilled in the craft of information synthesis. With a keen eye for detail and a deep understanding of narrative structure, I transform raw data into compelling stories. My creations are more than mere reports; they are guides through the complex landscapes of knowledge, designed to enlighten and inform.",
        max_iter=15,
        llm=get_model(high_context=True),
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



    the_tasks = [task, task_2, task_3]

    the_crew = Crew(
        agents=agents,
        tasks=the_tasks,
        full_output=True,
        verbose=True,
    )

    result = the_crew.kickoff()["final_output"]

    if copy_to_clipboard:
        from .standard_tools import copy
        copy(result)


    return result






search_on_internet_and_report_team = tool(search_on_internet_and_report_team_)


lastly_generated_codes = {}


def currently_codes():
    global lastly_generated_codes
    return lastly_generated_codes


def get_code(name:str):
    """
    returns the code
    """
    global lastly_generated_codes
    return lastly_generated_codes[name]


def save_code(name, code):
    global lastly_generated_codes
    lastly_generated_codes[name] = code


def required_old_code(aim):
    try:
        from crewai import Task, Crew, Agent


        requirement_analyzer = Agent(
            role="requirement_analyzer",
            goal="To understand and analyze the given aim to ensure the generated code meets the specified requirements.",
            backstory="As a requirement analyzer, my purpose is to bridge the gap between human intentions and machine execution. With a deep understanding of software development principles and a keen analytical mind, I dissect aims into actionable requirements.",
            max_iter=10,
            llm=get_model(high_context=True),
        )

        required_old_codes = Task(
            description=f"Analyze the aim: '{aim}' and find the required old codes for better compatibility. Old code names: {list(currently_codes())}",
            expected_output="Require old code names in a list",
            agent=requirement_analyzer,
        )


        the_crew = Crew(
            agents=[requirement_analyzer],
            tasks=[required_old_codes],
            full_output=True,
            verbose=True,
        )

        # Execute the tasks
        old_codes = the_crew.kickoff()["final_output"]

        the_string = ""

        for each in currently_codes():
            if each in old_codes:
                the_string += "\n" + get_code(each)

        return the_string

    except:
        return "An exception occurred" 



@wrapper
def generate_code_with_aim_team_(aim: str, copy_to_clipboard: bool = False) -> str:
    """
    A function to generate code based on a given aim. This function utilizes a team of AI agents specialized in understanding programming requirements and generating code.

    Parameters:
    - aim (str): The aim or goal for which the code needs to be generated.
    - copy_to_clipboard (bool): A flag to indicate whether to copy the generated code to the clipboard. The default value is False.

    Returns:
    - str: The generated code.
    """
    try:

        print("\nCOde generating\n")
        print("Previously codes", currently_codes())
        try:
            print("Inside of the first one", get_code(currently_codes()[0]))
        except:
            pass


        from crewai import Task, Crew, Agent


        tools = get_tools()

        the_tool_list = []
        for each in tools:
            if "team" not in each.name:
                the_tool_list.append(each)

        # Create the agents
        requirement_analyzer = Agent(
            role="requirement_analyzer",
            goal="To understand and analyze the given aim to ensure the generated code meets the specified requirements.",
            backstory="As a requirement analyzer, my purpose is to bridge the gap between human intentions and machine execution. With a deep understanding of software development principles and a keen analytical mind, I dissect aims into actionable requirements.",
            max_iter=10,
            llm=get_model(high_context=True),
        )

        code_generator = Agent(
            role="code_generator",
            goal="To translate the analyzed requirements into efficient, clean, and functional code.",
            backstory="I am the code generator, an architect of the digital world. With a vast library of programming knowledge and a creative spark, I craft code that breathes life into ideas. My code is not just functional; it's a masterpiece.",
            max_iter=20,
            llm=get_model(high_context=True),
        )

        # Define the tasks
        analyze_task = Task(
            description=f"Analyze the aim: '{aim}' and outline the requirements for the code.",
            expected_output="Requirements outline",
            agent=requirement_analyzer,
            tools=the_tool_list,
        )


        old_code_requirements = required_old_code(aim)
        print("Old_code_requirements", old_code_requirements)


        generate_code_task = Task(
            description=f"Generate code based on the outlined requirements. The other codes in the repo are: {old_code_requirements}",
            expected_output="Generated code, just code without any ```pyhton things or any other thing. Just python code",
            agent=code_generator,
            context=[analyze_task],
        )

        name_of_work = Task(
            description="Generate a name for the work",
            expected_output="a module name like text, examples: math.basics.sum for sum function. ",
            agent=code_generator,
            context=[generate_code_task],
        )


        # Create the crew and assign tasks
        the_crew = Crew(
            agents=[requirement_analyzer, code_generator],
            tasks=[analyze_task, generate_code_task, name_of_work],
            full_output=True,
            verbose=True,
        )

        # Execute the tasks
        the_crew.kickoff()["final_output"]

        result = generate_code_task.output.raw_output

        # Optionally copy the result to the clipboard
        if copy_to_clipboard:
            from .standard_tools import copy
            copy(result)

        print("name", name_of_work.output.raw_output)
        save_code(name_of_work.output.raw_output, result)

        return result
    except:
        return "An exception occurred" 


generate_code_with_aim_team = tool(generate_code_with_aim_team_)
