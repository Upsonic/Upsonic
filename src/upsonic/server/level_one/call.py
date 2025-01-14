

from pydantic import BaseModel
from pydantic_ai.result import ResultData

from typing import Any, Optional



from ...storage.configuration import Configuration




from ..level_utilized.utility import agent_creator

class CallManager:
    def gpt_4o(
        self,
        prompt: str,
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = "gpt-4o",
        system_prompt: Optional[Any] = None 
    ) -> ResultData:

        print("context", context)
        
        roulette_agent = agent_creator(response_format, tools, context, llm_model, system_prompt)
    

        result = roulette_agent.run_sync(prompt)

        return {"status_code": 200, "result": result.data}


Call = CallManager()
