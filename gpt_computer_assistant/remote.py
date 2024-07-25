import textwrap
import requests

import time

from upsonic import Tiger

the_upsonic = Tiger()

class Remote_Client:
    def __init__(self, url):
        self.url = url

    def send_request(self, path, data):
        response = requests.post(self.url+path, json=data)
        if response.status_code != 200:
            try:
                print(response.json())
            except:
                print(response.text)

            raise Exception("Request failed", response.status_code, path)
        return response.json()

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
        the_code = textwrap.dedent(the_upsonic.extract_source(func))
        # Remove the first line

        if the_code.startswith("@remote.custom_tool"):
            the_code = the_code[the_code.find("\n")+1:]
        
        data = {"code": the_code}
        response = self.send_request("/custom_tool", data)
        return response["response"]


    def wait(self, second):
        time.sleep(second)




remote = Remote_Client("http://localhost:7541")
