from fastapi import FastAPI, HTTPException
from functools import wraps
import asyncio
import httpx

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

def timeout(duration: float):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=duration)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail=f"Operation timed out after {duration} seconds",
                )
            except Exception as e:
                raise e

        return wrapper

    return decorator
