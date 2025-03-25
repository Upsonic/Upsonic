import traceback
import anthropic
import openai
from pydantic import BaseModel
import os
from pydantic_ai.messages import ImageUrl

from typing import Any, Optional, List

from ...storage.configuration import Configuration

from ..level_utilized.memory import save_temporary_memory, get_temporary_memory

from ..level_utilized.utility import agent_creator, summarize_system_prompt, summarize_message_prompt

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

            roulette_agent.retries = retries
            
            if memory:
                message_history = get_temporary_memory(agent_id)
                if message_history == None:
                    message_history = []
            else:
                message_history = []
                
            message = prompt
            message_history.append(prompt)

            if images:
                for image in images:
                    message_history.append(ImageUrl(url=f"data:image/jpeg;base64,{image}"))

            if "claude-3-5-sonnet" in llm_model:
                print("Tools", tools)
                if "ComputerUse.*" in tools:
                    try:
                        from ..level_utilized.cu import ComputerUse_screenshot_tool
                        result_of_screenshot = ComputerUse_screenshot_tool()
                        message_history.append(ImageUrl(url=result_of_screenshot["image_url"]["url"]))
                    except Exception as e:
                        print("Error", e)

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
                    message = current_message
                
                print("message: ", message)

                try:
                    print("I sent the request3")
                    result = await roulette_agent.run(message_history)
                    print("I got the response3")
                except (openai.BadRequestError, anthropic.BadRequestError) as e:
                    str_e = str(e)
                    if "400" in str_e and context_compress:
                        try:
                            # These functions are not async, so don't await them
                            compressed_prompt = summarize_system_prompt(system_prompt, llm_model)
                            if compressed_prompt:
                                print("compressed_prompt", compressed_prompt)
                                
                            # Compress the message and update message_history
                            compressed_message = summarize_message_prompt(message, llm_model)
                            if compressed_message:
                                print("compressed_message", compressed_message)
                                message = compressed_message
                                
                                # Reset message history with compressed message
                                message_history = [compressed_message]
                                if images:
                                    for image in images:
                                        message_history.append(ImageUrl(url=f"data:image/jpeg;base64,{image}"))

                            roulette_agent = agent_creator(
                                response_format=response_format,
                                tools=tools,
                                context=context,
                                llm_model=llm_model,
                                system_prompt=compressed_prompt,
                                context_compress=False
                            )
                            print("I sent the request4")
                            result = await roulette_agent.run(message_history)
                            print("I got the response4")
                        except Exception as e:
                            tb = traceback.extract_tb(e.__traceback__)
                            file_path = tb[-1].filename
                            if "Upsonic/src/" in file_path:
                                file_path = file_path.split("Upsonic/src/")[1]
                            line_number = tb[-1].lineno
                            error_response = {"status_code": 403, "detail": f"Error processing Agent request in {file_path} at line {line_number}: {str(e)}"}
                            return error_response
                    else:
                        tb = traceback.extract_tb(e.__traceback__)
                        file_path = tb[-1].filename
                        if "Upsonic/src/" in file_path:
                            file_path = file_path.split("Upsonic/src/")[1]
                        line_number = tb[-1].lineno
                        error_response = {"status_code": 403, "detail": f"Error processing Agent request in {file_path} at line {line_number}: {str(e)}"}
                        return error_response

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
                        tb = traceback.extract_tb(e.__traceback__)
                        file_path = tb[-1].filename
                        if "Upsonic/src/" in file_path:
                            file_path = file_path.split("Upsonic/src/")[1]
                        line_number = tb[-1].lineno
                        traceback.print_exc()
                        satisfied = True  # Break the loop on error

            if memory:
                save_temporary_memory(result.all_messages(), agent_id)

            # Changed from direct dictionary return to consistent style with error returns
            success_response = {
                "status_code": 200, 
                "result": result.data, 
                "usage": {
                    "input_tokens": total_request_tokens, 
                    "output_tokens": total_response_tokens
                }
            }
            return success_response

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            file_path = tb[-1].filename
            if "Upsonic/src/" in file_path:
                file_path = file_path.split("Upsonic/src/")[1]
            line_number = tb[-1].lineno
            traceback.print_exc()
            error_response = {"status_code": 500, "detail": f"Error processing Agent request in {file_path} at line {line_number}: {str(e)}"}
            return error_response


Agent = AgentManager()