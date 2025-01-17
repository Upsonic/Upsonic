

from pydantic import BaseModel
from pydantic_ai.result import ResultData

from typing import Any, Optional



from ...storage.configuration import Configuration



from ..level_utilized.memory import save_temporary_memory, get_temporary_memory

from ..level_utilized.utility import agent_creator

class AgentManager:
    def agent(
        self,
        agent_id: str,
        prompt: str,
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = "gpt-4o",
        system_prompt: Optional[Any] = None,
        retries: int = 1,
        memory: bool = False
    ) -> ResultData:

        
        roulette_agent = agent_creator(response_format, tools, context, llm_model, system_prompt)

        roulette_agent.retries = retries
        

        message_history = None
        if memory:
            message_history = get_temporary_memory(agent_id)
        

        
        message = [                   {
                        "type": "text",
                        "text": f"{prompt}"
                    }]





        if "claude-3-5-sonnet" in llm_model:
            print("Tools", tools)
            if "ComputerUse.*" in tools:
                try:
                    from ..level_utilized.cu import ComputerUse__screenshot
                    message.append(ComputerUse__screenshot())
                except Exception as e:
                    print("Error", e)

        



        result = roulette_agent.run_sync(message, message_history=message_history)

        usage = result.usage()

        if memory:
            save_temporary_memory(result.all_messages(), agent_id)

        return {"status_code": 200, "result": result.data, "usage": {"input_tokens": usage.request_tokens, "output_tokens": usage.response_tokens}}


Agent = AgentManager()