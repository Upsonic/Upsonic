import base64
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .chat_history import *
from .agent import *
from ..screen.shot import *

from ..utils.db import load_model_settings

config = {"configurable": {"thread_id": "abc123"}}

def assistant(llm_input, llm_history, client, screenshot_path=None):

    print("LLM INPUT", llm_input)


    the_message = [
                {"type": "text", "text": f"{llm_input}"},
                
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


    if the_model == "gpt-4o":
        msg = get_agent_executor().invoke({"messages":llm_history + [the_message]}, config=config)

    elif the_model == "llava" or the_model == "bakllava":

        get_agent_executor().invoke(
            {
                "input": the_message,
                "chat_history": llm_history,
            }
        )



    return msg["messages"]