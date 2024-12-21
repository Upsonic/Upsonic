try:
    from .start import start

    from .agentic import Agent

    from .tooler import Tool
except:
    pass
__version__ = '0.26.10'  # fmt: skip



from .classes import BaseClass, BaseVerifier, TypeVerifier, Task

import os
import time
import subprocess
import requests


from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)



class instance:
    def __init__(self, url, tasks=[]):
        self.url = url
        self.task = []
        for t in tasks:
            self.add_task(t)


    def request(self):
        pass

    def add_task(self, task):
        if isinstance(task, list):
            for t in task:
                self.task.append(t)
        else:
            self.task.append(task)

        for t in self.task:
            t.add_client(self)

    def run(self):
        for t in self.task:
            t.run()

        results = []
        for t in self.task:
            results.append(t.result)

        return results

    




class interface:
    pass



class local_instance(instance):
    def __init__(self, *args, **kwargs):
        super().__init__("http://localhost:7541", *args, **kwargs)
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
    def instance( *args, **kwargs):
        the_instance = local_instance( *args, **kwargs)
        the_instance.start()

        time.sleep(5)

        client_status = the_instance.client_status()

        if not client_status:
            raise Exception("Failed to start the local instance")
        
        return the_instance
        


class cloud_instance(instance):
    def __init__(self, *args, **kwargs):
        super().__init__("https://free_cloud_1.gca.dev/", *args, **kwargs)


    def request(self, the_request, the_response, screen=False):
        screen = "false" if not screen else "true"

        response = requests.post(self.url+"request", data={"request": the_request, "response": the_response, "screen":screen, "instance":self.instance_id}, verify=False)
        json_response = response.json()
        request_id = json_response["request_id"]
        try:
            while True:
                response = requests.post(self.url+"request_result", data={"request_id": request_id}, verify=False)
                the_json = response.json()
                if the_json["status"] == True:
                    return the_json["result"]
                time.sleep(1)
        except:
            return response.text
        

    def reset_memory(self):
        response = requests.post(self.url+"reset_memory", data={"instance":self.instance_id}, verify=False)
        the_json = response.json()
        return the_json["result"]

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
    def instance(*args, **kwargs):
        the_instance = cloud_instance( *args, **kwargs)
        the_instance.start()

        return the_instance
    






class docker_instance(instance):
    def __init__(self, url, *args, **kwargs):
        super().__init__(url, *args, **kwargs)
        from .remote import Remote_Client

        self.client = Remote_Client(self.url)

    def request(self, the_request, the_response, screen=False):

        return self.client.request(the_request, the_response, screen)


    def start(self):
        pass


    def close(self):
        pass



    def client_status(self):
        return self.client.status




class docker(interface):

    @staticmethod
    def instance(url, *args, **kwargs):
        the_instance = docker_instance(url, *args, **kwargs)
        the_instance.start()


        client_status = the_instance.client_status()

        if not client_status:
            raise Exception("Failed to start the docker instance")
        
        return the_instance
         