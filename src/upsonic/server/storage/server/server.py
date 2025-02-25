from fastapi import HTTPException
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


prefix = "/storage"



class ConfigGetRequest(BaseModel):
    key: str

class ConfigSetRequest(BaseModel):
    key: str
    value: str

class BulkConfigSetRequest(BaseModel):
    configs: Dict[str, str]


@app.post(f"{prefix}/config/get")
async def get_config(request: ConfigGetRequest):
    """
    Endpoint to get a configuration value by key using POST.

    Args:
        key: The configuration key

    Returns:
        The configuration value or a default message if not found
    """
    value = Configuration.get(request.key)
    return {"key": request.key, "value": value}


@app.post(f"{prefix}/config/set")
async def set_config(request: ConfigSetRequest):
    """
    Endpoint to set a configuration value.

    Args:
        key: The configuration key
        value: The configuration value

    Returns:
        A success message
    """
    Configuration.set(request.key, request.value)
    return {"message": "Configuration updated successfully"}

@app.post(f"{prefix}/config/bulk_set")
async def bulk_set_config(request: BulkConfigSetRequest):
    """
    Endpoint to set multiple configuration values at once.

    Args:
        configs: Dictionary of configuration key-value pairs

    Returns:
        A success message
    """
    for key, value in request.configs.items():
        Configuration.set(key, value)
    return {"message": "Bulk configuration updated successfully"}

