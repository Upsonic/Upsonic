try:
    from .start import start

    from .agentic import Agent

    from .tooler import Tool
except:
    pass
__version__ = '0.24.30'  # fmt: skip


import os
import time
import subprocess
import requests


from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)



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

    def request(self, the_request, the_response, screen=False):

        return self.client.request(the_request, the_response, screen)


    def start(self):
        command = "python -c 'from gpt_computer_assistant import start; start(True);'"
        self.process = subprocess.Popen(command, shell=True)


    def close(self):
        try:
            self.client.stop_server()
        except:
            pass

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


    def request(self, the_request, the_response, screen=False):
        screen = "false" if not screen else "true"

        response = requests.post(self.url+"request", data={"request": the_request, "response": the_response, "screen":screen, "instance":self.instance_id}, verify=False)
        try:
            json_response = response.json()
            return json_response["result"]
        except:
            return response.text
        


    def current_screenshot(self):
        response = requests.post(self.url+"screenshot_instance", data={"instance":self.instance_id}, verify=False)

        its_an_error = False

        try:
            the_json = response.json()
            if "result" in the_json:
                its_an_error = True
        except:
            pass


        if not its_an_error:
            with open('current_screenshot.png', 'wb') as file:
                file.write(response.content)
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            img = mpimg.imread('current_screenshot.png')
            plt.imshow(img)
            plt.axis('off')
            plt.show()





    def start(self):
        req = requests.get(self.url+"start_instance", verify=False)
        the_json = req.json()

        self.instance_id = the_json["result"]


    def close(self):
        req = requests.post(self.url+"stop_instance", data={"instance": self.instance_id}, verify=False)
        the_json = req.json()
        return the_json["result"]

    def client_status(self):
        return True



class cloud(interface):

    @staticmethod
    def instance():
        the_instance = cloud_instance()
        the_instance.start()

        return the_instance