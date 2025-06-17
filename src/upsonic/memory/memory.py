from .memory_manager import save_memory, get_memory, reset_memory


from pydantic_core import to_jsonable_python


from pydantic_ai.messages import ModelMessagesTypeAdapter 

def save_agent_memory(agent, answer):
    history_step_1 = answer.all_messages()
    as_python_objects = to_jsonable_python(history_step_1)
    save_memory(agent.get_agent_id(), as_python_objects)


def get_agent_memory(agent):
    the_json = get_memory(agent.get_agent_id())
    history = ModelMessagesTypeAdapter.validate_python(the_json)

    return history
