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



prefix = "/markdown"



@app.post(f"{prefix}/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file and convert it to markdown.

    Args:
        file: The file to convert to markdown

    Returns:
        The markdown content
    """
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), "upsonic_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Convert to markdown
        from markitdown import MarkItDown

        md = MarkItDown()
        markdown_content = md.convert(file_path).text_content

        # Add filename as heading
        markdown_with_filename = f"# {file.filename}\n\n{markdown_content}"

        # Clean up
        os.remove(file_path)

        return {"markdown": markdown_with_filename}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

