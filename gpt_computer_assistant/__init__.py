try:
    from .start import start

    from .agentic import Agent

    from .tooler import Tool
except:
    pass
__version__ = '0.23.16'  # fmt: skip


import os
import time
import subprocess
import requests



class instance:
    def __init__(self, url):
        self.url = url


    def request(self):
        pass





class interface:
    pass



class local_instance(instance):
    def __init__(self):
        super().__init__("http://localhost:7541")
        from .remote import Remote_Client

        self.client = Remote_Client(self.url)

    def request(self, the_request, the_response):

        return self.client.request(the_request, the_response)


    def start(self):
        command = "python -c 'from gpt_computer_assistant import start; start(True);'"
        self.process = subprocess.Popen(command, shell=True)


    def close(self):
        self.process.terminate()
        self.process.wait()

    def client_status(self):
        return self.client.status




class local(interface):

    @staticmethod
    def instance():
        the_instance = local_instance()
        the_instance.start()

        time.sleep(5)

        client_status = the_instance.client_status()

        if not client_status:
            raise Exception("Failed to start the local instance")
        
        return the_instance
        


class cloud_instance(instance):
    def __init__(self):
        super().__init__("https://free_cloud_1.gca.dev/")


    def request(self, the_request, the_response):
        response = requests.post(self.url+"request", data={"request": the_request, "response": the_response}, verify=False)
        json_response = response.json()
        return json_response["result"]

    def start(self):
        pass


    def close(self):
        pass

    def client_status(self):
        return True



class cloud(interface):

    @staticmethod
    def instance():
        return cloud_instance()