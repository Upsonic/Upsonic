import time
import traceback
import json
import re
import hashlib

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

from .remote import Remote_Client
from .version import get_version


import sentry_sdk
sentry_sdk.init(
    dsn="https://eed76b3c8eb23bbe1c2f6a796a03f1a9@o4508336623583232.ingest.us.sentry.io/4508556319195136",   
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    release=f"gcs@{get_version()}",
    server_name="gca_framework",
)
from .utils.user_id import load_user_id
sentry_sdk.set_user({"id": load_user_id()})








# Create a global Console object for styled output
console = Console()


def extract_json(llm_output):

    # Use regex to extract the json then transform it to a python dict object ```json ````
    json_str = re.search(r'```json\n(.*?)```', llm_output, re.DOTALL).group(1)


    # transform the json string to a python dict object
    transformed_json = json.loads(json_str)



    return transformed_json



def current_date_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    
    

class BaseClass:
    def __init__(self, screen=False):
        self.screen_task = screen

    def add_client(self, client: Remote_Client):

        self.client = client

        if hasattr(self, "verifier"):
            if self.verifier:
                self.verifier.screen_task = self.screen_task
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
        self.feedback = None

    
    def verify(self, description, result):
        with sentry_sdk.start_transaction(op="task", name="Verify"):
            console.print(
                Panel(
                    "[bold yellow]Verifying result with TypeVerifier...[/bold yellow]",
                    title="Verifier",
                    style=Style(color="bright_white", bgcolor="black", bold=True)
                )
            )

            console.print(f"[bold]Expected type:[/bold] [green]{self.type}[/green]\n")

            the_ai_result_if_we_have = ""
            if not result.startswith("No response"):
                the_ai_result_if_we_have = f"AI Result:\n{result}\n"

            control_point_span = sentry_sdk.start_span(name="Control Point")
            control_point = self.client.request(
                f"""
    User Request:
    {description}

    {the_ai_result_if_we_have}


    Now critically analyze the result of the task you just completed.

    Getting current state:
    - See the screen (Optional, If the ai output is not enough)
    - Read the history of conversation to understand the context of the task.


    If the result not true, respond onyl with “I am sorry” and "Reason:" and "Feedback to resolve:" to trigger a retry.

    If the result is true, respond with "I am satisfied" to continue to the next task.

    Current Date Time: {current_date_time()}
                """
                , "", screen=self.screen_task)
            
            control_point_span.finish()

            if "I am sorry" in control_point:
                self.feedback = control_point

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

            extracting_output_span = sentry_sdk.start_span(name="Extracting Output")
            result = self.client.request(prompt, "", screen=self.screen_task)
            extracting_output_span.finish()



        

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
        with sentry_sdk.start_transaction(op="task", name="Run"):
            console.print(
                Panel(
                    f"[bold green]Starting Task[/bold green]\n[b]Task Description:[/b] {self.description}",
                    title="Task",
                    style=Style(color="bright_white", bgcolor="black", bold=True)
                )
            )

            
            requesting_ai_span = sentry_sdk.start_span(name="Requesting AI")

            requesting_ai_span.set_data("verifier", self.verifier)

            # Verify the result
            if self.verifier:

                try_count = 0

                while try_count < self.verifier.try_count:
                    try_count += 1

                    console.print(
                        Panel(
                            f"[yellow]Attempt {try_count}[/yellow]  ",
                            title="Retry",
                            style=Style(color="bright_white", bgcolor="black", bold=True)
                        )
                    )

                    if try_count > 1:
                        self.output = "User is not satisfied with the result. Please try again." if self.verifier.feedback is None else  self.verifier.feedback
                    self.client.change_profile(self.hash)
                    
                    result = self.client.request(self.description, self.output, screen=self.screen_task)
                    
                    time.sleep(1)
                    ai_result = result
                    try:
                        self.client.change_profile(self.hash+"VERIFY")
                        result = self.verifier.verify(self.description, result)
                        console.print("[bold green]Verification successful![/bold green]\n")
                        break
                    except Exception as e:
                        console.print(
                            Panel(
                                f"[red]Verification failed[/red]\nAI Output: {ai_result}\nFeedback: {self.verifier.feedback}",
                                title="Verification Error",
                                style=Style(color="bright_white", bgcolor="black", bold=True)
                            )
                        )

                        result = self.verifier.exception_return
                        

            else:
                result = self.client.request(self.description, self.output, screen=self.screen_task)


            requesting_ai_span.finish()



            console.print(
                Panel(
                    "[bold green]Task Completed[/bold green]\n[bold]Final result ready.[/bold]",
                    title="Task Finished",
                    style=Style(color="bright_white", bgcolor="black", bold=True)
                )
            )

            self.result = result
            return result
