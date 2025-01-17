from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import traceback
from ...api import app, timeout
from ..call import Call
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import base64


prefix = "/level_one"


class GPT4ORequest(BaseModel):
    prompt: str
    response_format: Optional[Any] = []
    tools: Optional[Any] = []
    context: Optional[Any] = None
    llm_model: Optional[Any] = "gpt-4o"
    system_prompt: Optional[Any] = None


def run_sync_gpt4o(prompt, response_format, tools, context, llm_model, system_prompt):
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return Call.gpt_4o(
            prompt=prompt,
            response_format=response_format,
            tools=tools,
            context=context,
            llm_model=llm_model,
            system_prompt=system_prompt
        )
    finally:
        loop.close()


@app.post(f"{prefix}/gpt4o")
@timeout(300.0)  # 5 minutes timeout for AI operations
async def call_gpt4o(request: GPT4ORequest):
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
                traceback.print_exc()
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
                traceback.print_exc()
                context = None
        else:
            context = None



        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                run_sync_gpt4o,
                request.prompt,
                response_format,
                request.tools,
                context,
                request.llm_model,
                request.system_prompt
            )

        if request.response_format != "str" and result["status_code"] == 200:

            result["result"] = cloudpickle.dumps(result["result"])
            result["result"] = base64.b64encode(result["result"]).decode('utf-8')
        return {"result": result, "status_code": 200}
    except Exception as e:
        traceback.print_exc()


        return {"result": {"status_code": 500, "detail": f"Error processing Call request: {str(e)}"}, "status_code": 500}
