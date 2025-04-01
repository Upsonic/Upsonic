import traceback
import anthropic
import openai
from pydantic import BaseModel
import os
from pydantic_ai.messages import ImageUrl

from typing import Any, Optional, List

from ...storage.configuration import Configuration

from ..level_utilized.memory import save_temporary_memory, get_temporary_memory

from ..level_utilized.utility import (
    agent_creator, 
    prepare_message_history,
    process_error_traceback,
    format_response,
    handle_compression_retry
)

from ...client.tasks.tasks import Task
from ...client.tasks.task_response import ObjectResponse

from ..level_one.call import Call


class AgentManager:
    async def agent(
        self,
        agent_id: str,
        prompt: str,
        images: Optional[List[str]] = None,
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = "openai/gpt-4o",
        system_prompt: Optional[Any] = None,
        retries: int = 1,
        context_compress: bool = False,
        memory: bool = False
    ):
        try:
            roulette_agent = agent_creator(
                response_format=response_format, 
                tools=tools, 
                context=context, 
                llm_model=llm_model, 
                system_prompt=system_prompt,
                context_compress=context_compress
            )
            
            if isinstance(roulette_agent, dict) and "status_code" in roulette_agent:
                return roulette_agent  # Return error from agent_creator

            roulette_agent.retries = retries
            agent_memory = []
            if memory:
                agent_memory = get_temporary_memory(agent_id)

            message_history = prepare_message_history(prompt, images, llm_model, tools)
                
            feedback = ""
            satisfied = False
            total_request_tokens = 0
            total_response_tokens = 0
            total_retries = 0

            while not satisfied:
                if feedback:
                    current_message = prompt + "\n\n" + feedback
                    # Update the message history with the new message that includes feedback
                    if message_history and isinstance(message_history[0], str):
                        message_history[0] = current_message
                
                print("message:", message_history[0] if message_history and isinstance(message_history[0], str) else "")

                try:
                    print("I sent the request3")
                    print("message_history", message_history)
                    result = await roulette_agent.run(message_history, message_history=agent_memory)
                    print("I got the response3")
                except (openai.BadRequestError, anthropic.BadRequestError) as e:
                    str_e = str(e)
                    if "400" in str_e and context_compress:
                        try:
                            result = await handle_compression_retry(
                                prompt, images, tools, llm_model,
                                response_format, context, system_prompt, agent_memory
                            )
                        except Exception as e:
                            return process_error_traceback(e)
                    else:
                        return process_error_traceback(e)

                total_request_tokens += result.usage().request_tokens
                total_response_tokens += result.usage().response_tokens

                if retries == 1:
                    satisfied = True
                elif total_retries >= retries:
                    satisfied = True
                else:
                    total_retries += 1
                    print("Retrying", total_retries)

                    try:
                        class Satisfying(ObjectResponse):
                            satisfied: bool
                            feedback: str
                            
                        from ...client.level_two.agent import OtherTask
                        other_task = OtherTask(task=prompt, result=result.data)

                        satify_result = await Call.gpt_4o(
                            "Check if the result is satisfied", 
                            response_format=Satisfying, 
                            context=other_task, 
                            llm_model=llm_model
                        )
                        feedback = satify_result["result"].feedback

                        satisfied = satify_result["result"].satisfied
                    except Exception as e:
                        traceback.print_exc()
                        satisfied = True  # Break the loop on error

            if memory:
                save_temporary_memory(result.all_messages(), agent_id)

            return {
                "status_code": 200, 
                "result": result.data, 
                "usage": {
                    "input_tokens": total_request_tokens, 
                    "output_tokens": total_response_tokens
                }
            }

        except Exception as e:
            return process_error_traceback(e)


Agent = AgentManager()