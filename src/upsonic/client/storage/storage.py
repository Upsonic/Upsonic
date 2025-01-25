import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
import os
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel, Field


from dotenv import load_dotenv
load_dotenv()


class ClientConfig(BaseModel):
    DEFAULT_LLM_MODEL: str = Field(default="openai/gpt-4o")
    
    OPENAI_API_KEY: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    ANTHROPIC_API_KEY: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    
    AZURE_OPENAI_ENDPOINT: str | None = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT"))
    AZURE_OPENAI_API_VERSION: str | None = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION"))
    AZURE_OPENAI_API_KEY: str | None = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY"))
    
    AWS_ACCESS_KEY_ID: str | None = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    AWS_SECRET_ACCESS_KEY: str | None = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    AWS_REGION: str | None = Field(default_factory=lambda: os.getenv("AWS_REGION"))

    DEEPSEEK_API_KEY: str | None = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))



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


    def set_default_llm_model(self, llm_model: str):
        self.default_llm_model = llm_model


    def config(self, config: ClientConfig):
        print(config)
        # Parse it and use set_config to set it 
        # Exclude the default_llm_model and None values
        for key, value in config.model_dump().items():
            if key != "default_llm_model" and value is not None:
                self.set_config(key, value)
        
        self.default_llm_model = config.DEFAULT_LLM_MODEL
