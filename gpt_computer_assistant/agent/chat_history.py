from langchain_community.chat_message_histories import SQLChatMessageHistory
from .background import llm_history_oiginal
try:
    from ..utils.db import get_history_db
except ImportError:
    from utils.db import get_history_db


def get_chat_message_history():

    connection = SQLChatMessageHistory(
        session_id="abc123", connection_string=f"sqlite:///{get_history_db()}"
    )
    if len(connection.messages) == 0:
        connection.add_message(llm_history_oiginal[0])

    return connection


def clear_chat_history():
    get_chat_message_history().clear()
    get_chat_message_history().add_message(llm_history_oiginal[0])
