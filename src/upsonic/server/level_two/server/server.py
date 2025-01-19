from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import traceback
from ...api import app, timeout
from ..agent import Agent
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import base64


prefix = "/level_two"


class AgentRequest(BaseModel):
    agent_id: str
    prompt: str
    response_format: Optional[Any] = []
    tools: Optional[Any] = []
    context: Optional[Any] = None
    llm_model: Optional[Any] = "gpt-4o"
    system_prompt: Optional[Any] = None
    retries: Optional[Any] = 1
    context_compress: Optional[Any] = False
    memory: Optional[Any] = False

def run_sync_agent(agent_id, prompt, response_format, tools, context, llm_model, system_prompt, retries, context_compress, memory):
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return Agent.agent(
            agent_id=agent_id,
            prompt=prompt,
            response_format=response_format,
            tools=tools,
            context=context,
            llm_model=llm_model,
            system_prompt=system_prompt,
            retries=retries,
            context_compress=context_compress,
            memory=memory
        )
    finally:
        loop.close()


@app.post(f"{prefix}/agent")
@timeout(500.0)  # 5 minutes timeout for AI operations
async def call_agent(request: AgentRequest):
    """
    Endpoint to call GPT-4 with optional tools and MCP servers.

    Args:
        request: GPT4ORequest containing prompt and optional parameters

    Returns:
        The response from the AI model
    """


    try:
        # Handle pickled response format
        if request.response_format != "str":
            try:
                # Decode and unpickle the response format
                pickled_data = base64.b64decode(request.response_format)
                response_format = cloudpickle.loads(pickled_data)
            except Exception as e:
       
                # Fallback to basic type mapping if unpickling fails
                type_mapping = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                }
                response_format = type_mapping.get(request.response_format, str)
        else:
            response_format = str

        if request.context is not None:
            try:
                pickled_context = base64.b64decode(request.context)
                context = cloudpickle.loads(pickled_context)
            except Exception as e:
                context = None
        else:
            context = None



        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                run_sync_agent,
                request.agent_id,
                request.prompt,
                response_format,
                request.tools,
                context,
                request.llm_model,
                request.system_prompt,
                request.retries,
                request.context_compress,   
                request.memory
            )

        if request.response_format != "str" and result["status_code"] == 200:

            result["result"] = cloudpickle.dumps(result["result"])
            result["result"] = base64.b64encode(result["result"]).decode('utf-8')
        return {"result": result, "status_code": 200}
    except Exception as e:
        traceback.print_exc()


        return {"result": {"status_code": 500, "detail": f"Error processing Agent request: {str(e)}"}, "status_code": 500}
