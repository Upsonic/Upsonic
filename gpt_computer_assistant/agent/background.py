from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .chat_history import *
from ..llm_settings import first_message


llm_history_oiginal = [
    SystemMessage(
        content=[
            {
                "type": "text",
                "text": first_message,
            }
        ]
    ),    
]
