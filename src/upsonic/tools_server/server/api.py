from fastapi import FastAPI, HTTPException
import asyncio
from functools import wraps

app = FastAPI()


class TimeoutException(Exception):
    pass


async def timeout_handler(duration: float, coro):
    try:
        return await asyncio.wait_for(coro, timeout=duration)
    except asyncio.TimeoutError:
        raise TimeoutException(f"Operation timed out after {duration} seconds")


# Timeout decorator
def timeout(duration: float):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):

            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=duration)

                return result
            except asyncio.TimeoutError:

                raise HTTPException(
                    status_code=408,
                    detail=f"Operation timed out after {duration} seconds",
                )

        return wrapper

    return decorator


@app.get("/status")
async def get_status():
    return {"status": "Server is running"}
