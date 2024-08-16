from langchain_community.chat_message_histories import SQLChatMessageHistory
from .background import llm_history_oiginal

try:
    from ..utils.db import get_history_db
    from ..utils.db import load_model_settings, agents
    from ..llm_settings import each_message_extension, llm_settings
except ImportError:
    from utils.db import get_history_db
    from utils.db import load_model_settings
    from llm_settings import llm_settings


def get_chat_message_history():
    connection = SQLChatMessageHistory(
        session_id="abc123", connection_string=f"sqlite:///{get_history_db()}"
    )
    if len(connection.messages) == 0:
        the_model = load_model_settings()
        if llm_settings[the_model]["tools"]:
            connection.add_message(llm_history_oiginal()[0])

    return connection


def clear_chat_history():
    get_chat_message_history().clear()

    the_model = load_model_settings()
    if llm_settings[the_model]["tools"]:
        get_chat_message_history().add_message(llm_history_oiginal()[0])
