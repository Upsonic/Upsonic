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
        


remote = Remote_Client("http://localhost:7541")