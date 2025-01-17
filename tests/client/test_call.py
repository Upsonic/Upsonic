from upsonic import UpsonicClient
from pydantic import BaseModel
from upsonic import Task
from upsonic import StrResponse, IntResponse, FloatResponse, BoolResponse, ObjectResponse, StrInListResponse


server = UpsonicClient("http://localhost:7541")
server.set_default_llm_model("gpt-4o-azure")

class Human(ObjectResponse):
    name: str
    surname: str
    gender: str


class City(ObjectResponse):
    name: str
    country: str
    population: int


class Tool(ObjectResponse):
    raw_name: str


class ToolList(ObjectResponse):
    tools: list[Tool]

def test_gpt4o_call():
    task = Task(description="Hi, I am Onur Atakan ULUSOY and I am a male", response_format=Human)
    server.call(task)
    print(task.response)
    assert task.response.name.lower() == "onur atakan"
    assert task.response.surname.lower() == "ulusoy"
    assert task.response.gender.lower() == "male"


def test_gpt4o_call_without_return_type():
    task = Task(description="Say hi to me")
    server.call(task)
    potential_keywords = ["hi", "hello", "hey"]
    assert any(keyword in task.response.lower() for keyword in potential_keywords)


def test_call_with_response_format():
    task = Task(description="What is capital of Turkey?", response_format=City)
    server.call(task)
    print(task.response)
    assert task.response.name.lower() == "ankara"
    assert task.response.country.lower() == "turkey"
    assert task.response.population > 0


def test_call_with_tools():
    task = Task(description="What are your tools?", tools=["add_numbers"], response_format=StrInListResponse("tools_raw_names"))
    server.call(task)

    any_found_add_numbers = False
    for tool in task.response:
        if "add_numbers" in tool.lower():
            any_found_add_numbers = True
    assert any_found_add_numbers



