import textwrap
import requests

import time
import sys
from upsonic import Tiger


the_upsonic_ = None


def the_upsonic():
    global the_upsonic_

    if not the_upsonic_:
        the_upsonic_ = Tiger()
    
    return the_upsonic_

class Remote_Client:
    def __init__(self, url):
        self.url = url

        if self.status != True:
            raise Exception("The server is not running")


    def send_request(self, path, data, dont_error=False):
        try:
            response = requests.post(self.url+path, json=data)
            if response.status_code != 200:
                try:
                    print(response.json())
                except:
                    print(response.text)
                
                raise Exception("Request failed", response.status_code, path)
            return response.json()
        except Exception as e:
            if dont_error:
                return {"response": str(e)}
            else:
                raise e


    @property
    def status(self):
        data = {}
        response = self.send_request("/status", data, dont_error=True)
        return response["response"]

    def input(self, text:str, screen:bool=False, talk:bool=False) -> str:
        data = {"text": text, "screen": str(screen).lower(), "talk": str(talk).lower()}
        response = self.send_request("/input", data)
        return response["response"]

    def just_screenshot(self) -> str:
        data = {}
        response = self.send_request("/screenshot", data)
        return response["response"]

    def talk(self, text:str) -> str:
        data = {"text": text}
        response = self.send_request("/tts", data)
        return response["response"]

    def profile(self, profile:str) -> str:
        data = {"profile": profile}
        response = self.send_request("/profile", data)
        return response["response"]

    def reset_memory(self) -> str:
        response = self.send_request("/reset_memory", {})
        return response["response"]

    def enable_predefined_agents(self) -> str:
        response = self.send_request("/activate_predefined_agents", {})
        return response["response"]

    def disable_predefined_agents(self) -> str:
        response = self.send_request("/deactivate_predefined_agents", {})
        return response["response"]

    def enable_online_tools(self) -> str:
        response = self.send_request("/activate_online_tools", {})
        return response["response"]

    def disable_online_tools(self) -> str:
        response = self.send_request("/deactivate_online_tools", {})
        return response["response"]


    def change_name(self, new_name:str) -> str:
        data = {"new_name": new_name}
        response = self.send_request("/change_name", data)
        return response["response"]
    
    def change_developer(self, new_developer:str) -> str:
        data = {"new_developer": new_developer}
        response = self.send_request("/change_developer", data)
        return response["response"]


    def install_library(self, library:str) -> str:
        data = {"library": library}
        response = self.send_request("/library_install", data)
        return response["response"]
    
    def uninstall_library(self, library:str) -> str:
        data = {"library": library}
        response = self.send_request("/library_uninstall", data)
        return response["response"]


    def custom_tool(self, func):
        the_code = textwrap.dedent(the_upsonic().extract_source(func))
        # Remove the first line

        if the_code.startswith("@remote.custom_tool"):
            the_code = the_code[the_code.find("\n")+1:]
        
        data = {"code": the_code}
        response = self.send_request("/custom_tool", data)
        return response["response"]


    def top_bar_animation(self, text):
        data = {"text": text}
        response = self.send_request("/top_bar_activate", data)


    def stop_top_bar_animation(self, text):
        data = {"text": text}
        response = self.send_request("/top_bar_deactivate", data)


    def boop(self):
        data = {}
        response = self.send_request("/boop_sound", data)


    def ask(self, question, wait_for_answer=None):
        data = {"question":question, "wait_for_answer":wait_for_answer}
        response = self.send_request("/ask_to_user", data)
        return response["response"]


    def set_text(self, text):
        data = {"text": text}
        response = self.send_request("/set_text", data)
        return response["response"]

    class OperationContext:
        def __init__(self, client, text):
            self.client = client
            self.text = text

        def __enter__(self):
            self.client.top_bar_animation(self.text)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.client.stop_top_bar_animation(self.text)

    def operation(self, text):
        return self.OperationContext(self, text)


    def set_background_color(self, color):
        data = {"color": color}
        response = self.send_request("/set_background_color", data)
        return response["response"]
    
    def set_opacity(self, opacity):
        data = {"opacity": opacity}
        response = self.send_request("/set_opacity", data)
        return response["response"]

    def set_border_radius(self, radius):
        data = {"radius": radius}
        response = self.send_request("/set_border_radius", data)
        return response["response"]


    def wait(self, second):
        time.sleep(second)




remote = Remote_Client("http://localhost:7541")
