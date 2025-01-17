

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

        
        roulette_agent = agent_creator(response_format, tools, context, llm_model, system_prompt)
        
        message = [                   {
                        "type": "text",
                        "text": f"{prompt}"
                    }]



        try:

            if "claude-3-5-sonnet" in llm_model:
                print("Tools", tools)
                if "ComputerUse.*" in tools:
                    try:
                        from ..level_utilized.cu import ComputerUse_screenshot_tool
                        result_of_screenshot = ComputerUse_screenshot_tool()
                        message.append(result_of_screenshot)
                    except Exception as e:
                        print("Error", e)

        
            print("Message", message)


            result = roulette_agent.run_sync(message)

            usage = result.usage()

            return {"status_code": 200, "result": result.data, "usage": {"input_tokens": usage.request_tokens, "output_tokens": usage.response_tokens}}
        except AttributeError:
            return roulette_agent

Call = CallManager()
