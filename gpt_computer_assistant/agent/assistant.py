import random


from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .chat_history import *
from .agent import *


try:
    from ..screen.shot import *
    from ..utils.db import load_model_settings, agents
    from ..llm import get_model
    from ..llm_settings import llm_settings
    from ..utils.chat_history import ChatHistory
except ImportError:
    from screen.shot import *
    from utils.db import load_model_settings, agents
    from utils.chat_history import ChatHistory
    from llm import get_model
    from llm_settings import llm_settings

config = {"configurable": {"thread_id": "abc123"}}



def random_charachter(length=10):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))



def agentic(
    llm_input, llm_history, client, screenshot_path=None, dont_save_image=False
):
    global agents
    from crewai import Task, Crew

    from crewai import Agent as crewai_Agent

    the_agents = []

    for each in agents:
        the_agents.append(
            crewai_Agent(
                role=each["role"],
                goal=each["goal"],
                backstory=each["backstory"],
                llm=get_model(high_context=True),
            )
        )

    agents = the_agents

    print("LLM INPUT", llm_input)

    def image_explaination():
        the_message = [
            {"type": "text", "text": "Explain the image"},
        ]

        if screenshot_path:
            base64_image = encode_image(screenshot_path)
            the_message.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            )
            print("LEN OF İMAGE", len(base64_image))

        the_message = HumanMessage(content=the_message)
        get_chat_message_history().add_message(the_message)

        the_model = load_model_settings()

        if (
            llm_settings[the_model]["provider"] == "openai"
            or llm_settings[the_model]["provider"] == "ollama"
            or llm_settings[the_model]["provider"] == "azureai"
            or llm_settings[the_model]["provider"] == "anthropic"
        ):
            msg = get_agent_executor().invoke(
                {"messages": llm_history + [the_message]}, config=config
            )

        if llm_settings[the_model]["provider"] == "google":
            msg = get_agent_executor().invoke(
                {"messages": llm_history + [the_message]}, config=config
            )

        the_last_messages = msg["messages"]

        return the_last_messages[-1].content

    if screenshot_path:
        image_explain = image_explaination()
        llm_input += "User Sent Image and image content is: " + image_explain

    llm_input = llm_input 

    task = Task(
        description=llm_input,
        expected_output="Answer",
        agent=agents[0],
        tools=get_tools(),
    )

    the_crew = Crew(
        agents=agents,
        tasks=[task],
        full_output=True,
        verbose=True,
    )

    result = the_crew.kickoff()["final_output"]

    get_chat_message_history().add_message(
        HumanMessage(content=[llm_input])
    )
    get_chat_message_history().add_message(AIMessage(content=[result]))

    return result


def assistant(
    llm_input, client, screenshot_path=None, dont_save_image=False, just_screenshot=False
):
    
    the_chat_history = ChatHistory()

    the_model = load_model_settings()

    print("LLM INPUT", llm_input)

    if llm_settings[the_model]["tools"]:
        llm_input = llm_input 


    human_first_message = {"type": "text", "text": f"{llm_input}"}
    the_chat_history.add_message("human", human_first_message)


    the_message = [
        human_first_message
    ]



    human_second_message = None

    if screenshot_path:
        base64_image = encode_image(screenshot_path)
        if llm_settings[the_model]["provider"] == "ollama":

            human_second_message = {
                "type": "image_url",
                "image_url": base64_image,
            }
            the_chat_history.add_message("human", human_second_message, auto_delete=10)


        else:
            human_second_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
            the_chat_history.add_message("human", human_second_message, auto_delete=10)


        print("LEN OF IMAGE", len(base64_image))


    if human_second_message:
        the_message.append(human_second_message)





    the_message = HumanMessage(content=the_message)


    

    llm_history = the_chat_history.get_chat()




    

    if (
        llm_settings[the_model]["provider"] == "openai"
        or llm_settings[the_model]["provider"] == "azureai"
        or llm_settings[the_model]["provider"] == "anthropic"
    ):
        if just_screenshot:
            msg = {"messages": llm_history + [the_message]}
            time.sleep(1)
        else:

            msg = get_agent_executor().invoke(
                {"messages": llm_history + [the_message]}, config=config
            )
      




    the_last_messages = msg["messages"]



    the_chat_history.add_message("assistant", the_last_messages[-1].content)






    return_value = the_last_messages[-1].content
    if isinstance(return_value, list):
        the_text = ""
        for each in return_value:
            the_text += str(each)
        return_value = the_text

    if return_value == "":
        return_value = "No response "
        return_value += str(random_charachter())


    if just_screenshot:
        return "OK"

    return return_value
