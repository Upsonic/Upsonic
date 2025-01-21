import traceback
import anthropic
import openai
from pydantic import BaseModel
from pydantic_ai.result import ResultData

from typing import Any, Optional

from ...storage.configuration import Configuration

from ..level_utilized.memory import save_temporary_memory, get_temporary_memory

from ..level_utilized.utility import agent_creator, summarize_system_prompt, summarize_message_prompt

from ...client.tasks.tasks import Task
from ...client.tasks.task_response import ObjectResponse

from ..level_one.call import Call

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
        context_compress: bool = False,
        memory: bool = False
    ) -> ResultData:

        
        roulette_agent = agent_creator(
            response_format=response_format, 
            tools=tools, 
            context=context, 
            llm_model=llm_model, 
            system_prompt=system_prompt,
            context_compress=context_compress
        )

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

        
        feedback = ""


        satisfied = False
        total_request_tokens = 0
        total_response_tokens = 0
        total_retries = 0
        while not satisfied:
            
            #adding_feedback
            message[0]["text"] = message[0]["text"] + "\n\n" + feedback


            print("message: ", message)

            try:
                result = roulette_agent.run_sync(message, message_history=message_history)
            except (openai.BadRequestError, anthropic.BadRequestError) as e:
                str_e = str(e)
                if "400" in str_e and context_compress:
                    # Try to compress both system prompt and message prompt
                    try:
                        compressed_prompt = summarize_system_prompt(system_prompt, llm_model)
                        if compressed_prompt:
                            print("compressed_prompt", compressed_prompt)
                        message[0]["text"] = summarize_message_prompt(message[0]["text"], llm_model)
                        if message[0]["text"]:
                            print("compressed_message", message[0]["text"])

                        roulette_agent = agent_creator(
                            response_format=response_format,
                            tools=tools,
                            context=context,
                            llm_model=llm_model,
                            system_prompt=compressed_prompt,
                            context_compress=False  # Prevent infinite recursion
                        )
                        # Also compress the message prompt
                        

                        print("agent side")
                        print("len of message", len(message))
                        print("len of message[0]", len(message[0]["text"]))
                        print("message", message)
                        try:
                            print("len of message_history", len(message_history))
                        except:
                            pass
                        
                        result = roulette_agent.run_sync(message, message_history=message_history)
                    except (openai.BadRequestError, anthropic.BadRequestError) as e:
                        traceback.print_exc()
                        return {"status_code": 403, "detail": "Error processing Agent request: " + str(e)}
                elif "400" in str_e:
                    return {"status_code": 402, "detail": "The model context window is small for this task. Please try to make the task shorter or enable context compression."}
                else:
                    return {"status_code": 403, "detail": "Error processing Agent request: " + str(e)}
                

                
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


                    satify_result = Call.gpt_4o("Check if the result is satisfied", response_format=Satisfying, context=other_task)
                    feedback = satify_result["result"].feedback
      

                    if satify_result["result"].satisfied:
                        satisfied = True
                    else:
                        satisfied = False

                    
                except Exception as e:
                    traceback.print_exc()

            
        usage = result.usage()

        if memory:
            save_temporary_memory(result.all_messages(), agent_id)

        return {"status_code": 200, "result": result.data, "usage": {"input_tokens": total_request_tokens, "output_tokens": total_response_tokens}}


Agent = AgentManager()