from fastapi import FastAPI, HTTPException
from functools import wraps
import asyncio
import httpx

from fastapi import FastAPI, HTTPException, Request, Response
import asyncio
from functools import wraps
from ..exception import TimeoutException
import inspect
from starlette.responses import JSONResponse
import threading
import time
import traceback
import logging
import os

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

# Remove the middleware and use exception handlers instead
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    tb = traceback.extract_tb(exc.__traceback__)
    file_path = tb[-1].filename
    if "Upsonic/src/" in file_path:
        file_path = file_path.split("Upsonic/src/")[1]
    line_number = tb[-1].lineno
    logging.error(f"Error in {file_path} at line {line_number}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error in {file_path} at line {line_number}: {str(exc)}"}
    )

def handle_server_errors(func):
    """
    Decorator to catch internal server errors, print the traceback,
    and return a standardized error response.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            file_path = tb[-1].filename
            if "Upsonic/src/" in file_path:
                file_path = file_path.split("Upsonic/src/")[1]
            line_number = tb[-1].lineno
            traceback.print_exc()
            return {"result": {"status_code": 500, "detail": f"Error processing Call request in {file_path} at line {line_number}: {str(e)}"}, "status_code": 500}

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            file_path = tb[-1].filename
            if "Upsonic/src/" in file_path:
                file_path = file_path.split("Upsonic/src/")[1]
            line_number = tb[-1].lineno
            traceback.print_exc()
            return {"result": {"status_code": 500, "detail": f"Error processing Call request in {file_path} at line {line_number}: {str(e)}"}, "status_code": 500}

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

@app.get("/status")
async def get_status():
    return {"status": "Server is running"}


def timeout(seconds: float):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Create a task for the function
                task = asyncio.create_task(func(*args, **kwargs))
                # Wait for the task to complete with timeout
                result = await asyncio.wait_for(task, timeout=seconds)
                return result
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail=f"Function timed out after {seconds} seconds"
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we'll use a thread-based approach
            result = []
            error = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)  # Wait for the specified timeout
            
            if thread.is_alive():
                raise HTTPException(
                    status_code=408,
                    detail=f"Function timed out after {seconds} seconds"
                )
            
            if error:
                raise error[0]
            
            return result[0]

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator