from langchain_community.chat_message_histories import SQLChatMessageHistory
from .background import llm_history_oiginal

import os
from ..utils.db import history_db

chat_message_history = SQLChatMessageHistory(
    session_id="abc123", connection_string=f"sqlite:///test.db"
)


if not os.path.exists(history_db):
    chat_message_history.add_message(llm_history_oiginal[0])
    

def clear_chat_history():
    chat_message_history.clear()
    chat_message_history.add_message(llm_history_oiginal[0])
