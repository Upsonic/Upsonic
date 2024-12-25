import traceback
import json
import re
import hashlib

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

from .remote import Remote_Client

# Create a global Console object for styled output
console = Console()


def extract_json(llm_output):

    # Use regex to extract the json then transform it to a python dict object ```json ````
    json_str = re.search(r'```json\n(.*?)```', llm_output, re.DOTALL).group(1)


    # transform the json string to a python dict object
    transformed_json = json.loads(json_str)



    return transformed_json



    
    

class BaseClass:
    def __init__(self, screen_task=False):
        self.screen_task = screen_task

    def add_client(self, client: Remote_Client):

        self.client = client

        if hasattr(self, "verifier"):

            self.verifier.add_client(client)

    def add_task(self, task):
        self.task = task




    def sha_hash(self, text):
        
        return hashlib.sha256(text.encode()).hexdigest()



class BaseVerifier(BaseClass):
    def __init__(self, try_count=5, exception_return=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.try_count = try_count
        self.exception_return = exception_return


class TypeVerifier(BaseVerifier):
    def __init__(self, type, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.type = type

    def verify(self, description, result):
        console.print(
            Panel(
                "[bold yellow]Verifying result with TypeVerifier...[/bold yellow]",
                title="Verifier",
                style=Style(color="bright_white", bgcolor="black", bold=True)
            )
        )

        console.print(f"[bold]Expected type:[/bold] [green]{self.type}[/green]\n")


        control_point = self.client.request(
            f"""
User Request:
{description}

AI Result:
{result}


Now critically analyze the result of the task you just completed. If the request is impossible to complete you can respond with "This task is impossible to complete" and "Reason:" to stop the task.

Getting current state:
- See the screen
- Read the history of conversation to understand the context of the task.


If the result fails respond onyl with “I am sorry” and "Reason:" to trigger a retry.
            """
            , "", screen=self.screen_task)


        if "I am sorry" in control_point:

            raise Exception(f"Not satisfied with the result {self.task.description}")


        
        prompt = """
        Hi, now your responsibility is returning the answer in the requested format.

        User only wants the result in the format of """f"""{self.type}"""+""".
        Dont use any other format or any other type of data.

        Format recipe:
        1. list
            Return the user want like this:
            ```json
            ["element1", "element2", "element3"]
            ```

        2. dict
            Return the user want like this:
            ```json
            {"key1": "value1", "key2": "value2", "key3": "value3"}
            ```
        
        3. list in list
            Return the user want like this:
            ```json
            [["element1", "element2", "element3"], ["element4", "element5", "element6"]]
            ```

        4. dict in list
            Return the user want like this:
            ```json
            [{"key1": "value1", "key2": "value2", "key3": "value3"}, {"key4": "value4", "key5": "value5", "key6": "value6"}]
            ```
        
        5. string
            Return the user want like this:
            ```json
            "This is a string"
            ```

        6. integer
            Return the user want like this:
            ```json
            123
            ```
        
        7. float
            Return the user want like this:
            ```json
            123.456
            ```
        
        8. bool
            Return the user want like this:
            ```json
            true
            ```

            

        End of the day return result in ```json ``` format.
        """

        self.client.change_profile(self.task.hash)
        result = self.client.request(prompt, "", screen=self.screen_task)



        console.print("[bold green]Verification successful![/bold green]\n")

        result = extract_json(result)
        return result



            


class Task(BaseClass):
    def __init__(self, description, verifier: BaseVerifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = description
        self.output = ""
        self.verifier = verifier
        if self.verifier:
            self.verifier.add_task(self)

        self.result = None

        self.hash = self.sha_hash(description)



    def run(self):

        console.print(
            Panel(
                f"[bold green]Starting Task[/bold green]\n[b]Task Description:[/b] {self.description}",
                title="Task",
                style=Style(color="bright_white", bgcolor="black", bold=True)
            )
        )

        

        # Verify the result
        if self.verifier:

            try_count = 0

            while try_count < self.verifier.try_count:
                try_count += 1

                console.print(
                    Panel(
                        f"[yellow]Attempt {try_count}[/yellow] for Task: {self.description}\n",
                        title="Retry",
                        style=Style(color="bright_white", bgcolor="black", bold=True)
                    )
                )

                if try_count > 1:
                    self.output = "User is not satisfied with the result. Please try again."
                self.client.change_profile(self.hash)
                result = self.client.request(self.description, self.output, screen=self.screen_task)
                ai_result = result
                try:
                    self.client.change_profile(self.hash+"VERIFY")
                    result = self.verifier.verify(self.description, result)
                    break
                except Exception as e:
                    traceback.print_exc()
                    console.print(
                        Panel(
                            f"[red]Verification failed[/red]\nReason: {e}\nAI Output: {ai_result}",
                            title="Verification Error",
                            style=Style(color="bright_white", bgcolor="black", bold=True)
                        )
                    )

                    result = self.verifier.exception_return
                    

            

        console.print(
            Panel(
                "[bold green]Task Completed[/bold green]\n[bold]Final result ready.[/bold]",
                title="Task Finished",
                style=Style(color="bright_white", bgcolor="black", bold=True)
            )
        )

        self.result = result
        return result
