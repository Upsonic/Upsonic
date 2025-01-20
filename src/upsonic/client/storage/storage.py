import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel



class Storage:



    def get_config(self, key: str) -> Any:
        """
        Get a configuration value by key from the server.

        Args:
            key: The configuration key

        Returns:
            The configuration value
        """
        from ..trace import sentry_sdk
        with sentry_sdk.start_transaction(op="task", name="Storage.get_config") as transaction:
            with sentry_sdk.start_span(op="send_request"):
                data = {"key": key}
                response = self.send_request("/storage/config/get", data=data)
            return response.get("value")

    def set_config(self, key: str, value: str) -> str:
        """
        Set a configuration value on the server.

        Args:
            key: The configuration key
            value: The configuration value

        Returns:
            A success message
        """
        from ..trace import sentry_sdk
        with sentry_sdk.start_transaction(op="task", name="Storage.set_config") as transaction:
            with sentry_sdk.start_span(op="send_request"):
                data = {"key": key, "value": value}
                response = self.send_request("/storage/config/set", data=data)
            return response.get("message")
