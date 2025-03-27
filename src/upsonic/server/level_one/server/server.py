from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import traceback
from ...api import app, timeout, handle_server_errors
from ..call import Call
import asyncio
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import base64
import os


prefix = "/level_one"


class GPT4ORequest(BaseModel):
    prompt: str
    images: Optional[List[str]] = None
    response_format: Optional[Any] = []
    tools: Optional[Any] = []
    context: Optional[Any] = None
    llm_model: Optional[Any] = "openai/gpt-4o"
    system_prompt: Optional[Any] = None


@app.post(f"{prefix}/gpt4o")
@handle_server_errors
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
                tb = traceback.extract_tb(e.__traceback__)
                file_path = tb[-1].filename
                if "Upsonic/src/" in file_path:
                    file_path = file_path.split("Upsonic/src/")[1]
                line_number = tb[-1].lineno
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
                tb = traceback.extract_tb(e.__traceback__)
                file_path = tb[-1].filename
                if "Upsonic/src/" in file_path:
                    file_path = file_path.split("Upsonic/src/")[1]
                line_number = tb[-1].lineno
                traceback.print_exc()
                context = None
        else:
            context = None

        result = await Call.gpt_4o(
            prompt=request.prompt,
            images=request.images,
            response_format=response_format,
            tools=request.tools,
            context=context,
            llm_model=request.llm_model,
            system_prompt=request.system_prompt
        )

        if request.response_format != "str" and result["status_code"] == 200:
            result["result"] = cloudpickle.dumps(result["result"])
            result["result"] = base64.b64encode(result["result"]).decode('utf-8')
        return {"result": result, "status_code": 200}
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        file_path = tb[-1].filename
        if "Upsonic/src/" in file_path:
            file_path = file_path.split("Upsonic/src/")[1]
        line_number = tb[-1].lineno
        traceback.print_exc()
        return {"result": {"status_code": 500, "detail": f"Error processing Call request in {file_path} at line {line_number}: {str(e)}"}, "status_code": 500}
