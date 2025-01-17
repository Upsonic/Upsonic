import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel
import os
import tempfile


class Markdown:
    def markdown(self, file_path: str) -> str:
        """
        Upload a file and convert it to markdown.

        Args:
            file_path: Path to the file to convert or a URL to download and convert

        Returns:
            The markdown content
        """
        if file_path.startswith("http"):
            # Download file
            response = httpx.get(file_path)
            response.raise_for_status()

            # Save to temporary .html file
            fd, tmp_path = tempfile.mkstemp(suffix=".html")
            os.write(fd, response.content)
            os.close(fd)

            file_path = tmp_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the file and prepare for upload
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = self.send_request("/markdown/upload", {}, files=files)
            
        if file_path.startswith("/tmp"):
            os.remove(file_path)  # Delete temporary HTML file

        return response.get("markdown")
