from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import traceback

import pydantic_ai
from ...api import app, timeout, handle_server_errors
from ..agent import Agent
import asyncio
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import base64


prefix = "/level_two"


class AgentRequest(BaseModel):
    agent_id: str
    prompt: str
    images: Optional[List[str]] = None
    response_format: Optional[Any] = []
    tools: Optional[Any] = []
    context: Optional[Any] = None
    llm_model: Optional[Any] = "openai/gpt-4o"
    system_prompt: Optional[Any] = None
    context_compress: Optional[Any] = False
    memory: Optional[Any] = False


@app.post(f"{prefix}/agent")
@handle_server_errors
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

        result = await Agent.agent(
            agent_id=request.agent_id,
            prompt=request.prompt,
            images=request.images,
            response_format=response_format,
            tools=request.tools,
            context=context,
            llm_model=request.llm_model,
            system_prompt=request.system_prompt,
            context_compress=request.context_compress,
            memory=request.memory
        )

        if request.response_format != "str" and result["status_code"] == 200:
            result["result"] = cloudpickle.dumps(result["result"])
            result["result"] = base64.b64encode(result["result"]).decode('utf-8')
        return {"result": result, "status_code": 200}

    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        return {"result": {"status_code": 500, "detail": f"Change your response format to a simple format or improve your task description. Your response format is too hard for the model to understand (Dont use 'dict' like things in your response format, define everything explicitly). Try to make it more small parts.", "status_code": 500}}

    except Exception as e:
        traceback.print_exc()
        return {"result": {"status_code": 500, "detail": f"Error processing Agent request: {str(e)}"}, "status_code": 500}
