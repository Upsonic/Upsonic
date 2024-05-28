from langchain_community.chat_message_histories import SQLChatMessageHistory
from .background import llm_history_oiginal

import os
from ..utils.db import get_history_db



def get_chat_message_history():
    print("HISTORY DB", get_history_db())
    return SQLChatMessageHistory(
    session_id="abc123", connection_string=f"sqlite:///{get_history_db()}")



if not os.path.exists(get_history_db()):
    get_chat_message_history().add_message(llm_history_oiginal[0])
    

def clear_chat_history():
    get_chat_message_history().clear()
    get_chat_message_history().add_message(llm_history_oiginal[0])
