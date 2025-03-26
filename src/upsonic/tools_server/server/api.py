from fastapi import FastAPI, HTTPException, Request, Response
import asyncio
from functools import wraps
from ...exception import TimeoutException
import inspect
from starlette.responses import JSONResponse
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

# Remove the middleware and use exception handlers instead
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logging.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Import the cleanup function from server_utils instead of tools
from .server_utils import cleanup_all_servers

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up all server instances when the application shuts down.
    """
    await cleanup_all_servers()


async def timeout_handler(duration: float, coro):
    try:
        return await asyncio.wait_for(coro, timeout=duration)
    except asyncio.TimeoutError:
        raise TimeoutException(f"Operation timed out after {duration} seconds")

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

@app.get("/status")
async def get_status():
    return {"status": "Server is running"}
