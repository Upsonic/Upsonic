from langchain_core.messages import SystemMessage

try:
    from .chat_history import *
    from ..llm_settings import first_message
except ImportError:
    from agent.chat_history import *
    from llm_settings import first_message


def llm_history_oiginal():
    return [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": first_message(),
                }
            ]
        ),
    ]
