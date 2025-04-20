from pydantic import BaseModel
from typing import Any, Optional, List
from pydantic_ai.messages import ImageUrl

from ...storage.configuration import Configuration

from ..level_utilized.utility import (
    agent_creator, 
    prepare_message_history, 
    process_error_traceback,
    format_response,
    handle_compression_retry
)

import openai
import traceback


class CallManager:
    async def gpt_4o(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = "openai/gpt-4o",
        system_prompt: Optional[Any] = None 
    ):
        try:
            roulette_agent = agent_creator(response_format, tools, context, llm_model, system_prompt)
            if isinstance(roulette_agent, dict) and "status_code" in roulette_agent:
                return roulette_agent  # Return error from agent_creator

            message_history = prepare_message_history(prompt, images, llm_model, tools)

            try:
                print("I sent the request1")
                result = await roulette_agent.run(message_history)
                print("I got the response1")
                return format_response(result)
            except openai.BadRequestError as e:
                str_e = str(e)
                if "400" in str_e:
                    # Try to compress the message prompt
                    try:
                        result = await handle_compression_retry(
                            prompt, images, tools, llm_model, 
                            response_format, context, system_prompt
                        )
                        return format_response(result)
                    except Exception as e:
                        traceback.print_exc()
                        return process_error_traceback(e)
                else:
                    return process_error_traceback(e)
        except Exception as e:
            return process_error_traceback(e)

Call = CallManager()
