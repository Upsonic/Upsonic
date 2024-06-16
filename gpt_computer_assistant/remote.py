import requests



class Remote_Client:
    def __init__(self, url):
        self.url = url

    def send_request(self, path, data):
        response = requests.post(self.url+path, json=data)
        return response.json()

    def input(self, text:str, screen:bool=False) -> str:
        data = {"text": text, "screen": str(screen).lower()}
        response = self.send_request("/input", data)
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
    
    
        


remote = Remote_Client("http://localhost:7541")