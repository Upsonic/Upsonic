import base64
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .chat_history import *
from .agent import *
from ..screen.shot import *



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
    chat_message_history.add_message(the_message)



    msg = get_agent_executor().invoke({"messages":llm_history + [the_message]}, config=config)




    return msg["messages"]