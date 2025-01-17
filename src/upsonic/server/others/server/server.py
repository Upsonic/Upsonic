from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import traceback
from ...api import app, timeout

import asyncio
from concurrent.futures import ThreadPoolExecutor
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import base64
from ....storage.configuration import Configuration
import os
import tempfile
from markitdown import MarkItDown

from fastapi.responses import FileResponse
import uuid

md = MarkItDown()


prefix = "/others"



@app.get(f"{prefix}/take_screenshot")
async def take_screenshot():
    """
    Takes a screenshot using pyautogui and returns it to the client.

    Returns:
        The screenshot image file
    """
    import pyautogui
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), "upsonic_screenshots")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate a unique filename
        filename = f"screenshot_{uuid.uuid4()}.png"
        file_path = os.path.join(temp_dir, filename)

        # Take the screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)

        # Return the file and clean up after sending
        return FileResponse(
            file_path,
            media_type="image/png",
            filename=filename,
            background=asyncio.create_task(cleanup_screenshot(file_path))
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_screenshot(file_path: str):
    """
    Cleanup function to remove the screenshot file after it's been sent.
    """
    try:
        await asyncio.sleep(1)  # Wait a bit to ensure the file has been sent
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up screenshot: {e}")
