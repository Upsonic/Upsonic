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
import signal
from concurrent.futures import ThreadPoolExecutor
import threading
import time


app = FastAPI()



@app.get("/status")
async def get_status():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8086/status")
        if response.status_code == 200:
            return {"status": "Server is running"}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to reach the server at localhost:8086"
            )

def timeout(seconds: float):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"Function timed out after {seconds} seconds")

            # Set the signal handler and a timeout
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))

            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except TimeoutException as e:
                raise HTTPException(
                    status_code=408,
                    detail=str(e)
                )
            finally:
                # Disable the alarm
                signal.alarm(0)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"Function timed out after {seconds} seconds")

            # Set the signal handler and a timeout
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))

            try:
                return func(*args, **kwargs)
            except TimeoutException as e:
                raise HTTPException(
                    status_code=408,
                    detail=str(e)
                )
            finally:
                # Disable the alarm
                signal.alarm(0)

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator