
from kot import KOT

try:
    from .folder import currently_dir, artifacts_dir, media_dir
except:
    from folder import currently_dir, artifacts_dir, media_dir









from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


try:
    from .db import *

except:
    from db import *


import time



class Human:
    def __init__(self, content, the_time, auto_delete:int=None):

        self.that_was_empty = False

        if isinstance(content, dict):
            if "text" in content:
                if content["text"] == "":
                    self.that_was_empty = True
                    content["text"] = "No response"

        if isinstance(content, list):
            for i in range(len(content)):
                if "text" in content[i]:
                    if content[i]["text"] == "":
                        self.that_was_empty = True
                        content[i]["text"] = "No response"


        self.content = content
        self.timestamp = the_time
        self.auto_delete = auto_delete

    def __dict__(self):
        current_time = time.time()

        if self.auto_delete is not None:
            print(current_time, self.timestamp, self.auto_delete)
            if current_time - self.timestamp > self.auto_delete:
                return {"type": "human", "content": "This content deleted.", "timestamp": self.timestamp, "auto_delete": self.auto_delete, "that_was_empty": self.that_was_empty}

        return {"type": "human", "content": self.content, "timestamp": self.timestamp, "auto_delete": self.auto_delete, "that_was_empty": self.that_was_empty}

class Assistant:
    def __init__(self, content, the_time):

        self.that_was_empty = False

        if isinstance(content, dict):
            if "text" in content:
                if content["text"] == "":
                    self.that_was_empty = True
                    content["text"] = "No response"

        if isinstance(content, list):
            for i in range(len(content)):
                if "text" in content[i]:
                    if content[i]["text"] == "":
                        self.that_was_empty = True
                        content[i]["text"] = "No response"
        self.content = content
        self.timestamp = the_time

    def __dict__(self):
        return {"type": "assistant", "content": self.content, "timestamp": self.timestamp, "that_was_empty": self.that_was_empty}

class System:
    def __init__(self, content, the_time):

        self.that_was_empty = False

        if isinstance(content, dict):
            if "text" in content:
                if content["text"] == "":
                    self.that_was_empty = True
                    content["text"] = "No response"

        if isinstance(content, list):
            for i in range(len(content)):
                if "text" in content[i]:
                    if content[i]["text"] == "":
                        self.that_was_empty = True
                        content[i]["text"] = "No response"


        self.content = content
        self.timestamp = the_time

    def __dict__(self):
        return {"type": "system", "content": self.content, "timestamp": self.timestamp, "that_was_empty": self.that_was_empty}


class ChatHistory:

    def __init__(self):
        self.chat_id = get_profile()
        self.db = KOT(f"chat_history_{self.chat_id}", folder=artifacts_dir, enable_hashing=True)

        if self.db.get("chat") is None:
            self.db.set("chat", [])



    def add_message(self, message_type:str, content, auto_delete:int=None):

        the_time = time.time()


        if content == []:
            content = {"type":"text", "text": "No response"}


        if message_type == "human":
            message = Human(content, the_time, auto_delete)
        elif message_type == "assistant":
            print("ASSISTANT", content)
            message = Assistant(content, the_time)
        elif message_type == "system":
            print("SYSTEM", content)
            message = System(content, the_time)
        else:
            raise ValueError("Invalid message type")

       
        chat = self.db.get("chat")
        chat.append(message.__dict__())


        self.db.set("chat", chat)

    def get_chat(self):
        chat = self.db.get("chat")
        chat = sorted(chat, key=lambda x: x["timestamp"])

        # Transform dict to Message objects

        the_chat = []
        for message in chat:
            if message["type"] == "human":
                the_chat.append(Human(content=message["content"], the_time=message["timestamp"], auto_delete=message["auto_delete"]))
            elif message["type"] == "assistant":
                the_chat.append(Assistant(content=message["content"], the_time=message["timestamp"]))
            elif message["type"] == "system":
                the_chat.append(System(content=message["content"], the_time=message["timestamp"]))

        
        last_chat = []
        for message in the_chat:
            if message.that_was_empty:
                continue
            last_chat.append(message.__dict__())


        chat = last_chat

        langchain_messages = []

        for message in chat:

            if isinstance(message["content"], tuple):
                message["content"] = list(message["content"])
            if isinstance(message["content"], dict):
                message["content"] = [message["content"]]
            








            if message["type"] == "human":
                
                langchain_messages.append(HumanMessage(content=
                    message["content"]
                    ))
            elif message["type"] == "assistant":
                langchain_messages.append(AIMessage(content=
                    message["content"]
                    ))
            elif message["type"] == "system":
                langchain_messages.append(SystemMessage(content=
                    message["content"]
                    ))


        return langchain_messages
    

    def clear_chat(self):
        self.db.set("chat", [])

    

        
