import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
import os
import asyncio
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel, Field


from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))


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

    GOOGLE_GLA_API_KEY: str | None = Field(default_factory=lambda: os.getenv("GOOGLE_GLA_API_KEY"))
    
    OPENROUTER_API_KEY: str | None = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))



class Storage:



    def get_config(self, key: str) -> Any:
        """
        Get a configuration value by key from the server.

        Args:
            key: The configuration key

        Returns:
            The configuration value
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    from ..base import run_coroutine_in_new_thread
                    return run_coroutine_in_new_thread(self.get_config_async(key))
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.get_config_async(key))
        except Exception as e:
            raise e

    async def get_config_async(self, key: str) -> Any:
        """
        Get a configuration value by key from the server asynchronously.

        Args:
            key: The configuration key

        Returns:
            The configuration value
        """
        from ..trace import sentry_sdk
        with sentry_sdk.start_transaction(op="task", name="Storage.get_config_async") as transaction:
            with sentry_sdk.start_span(op="send_request_async"):
                data = {"key": key}
                response = await self.send_request_async("/storage/config/get", data=data)
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
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    from ..base import run_coroutine_in_new_thread
                    return run_coroutine_in_new_thread(self.set_config_async(key, value))
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.set_config_async(key, value))
        except Exception as e:
            raise e

    async def set_config_async(self, key: str, value: str) -> str:
        """
        Set a configuration value on the server asynchronously.

        Args:
            key: The configuration key
            value: The configuration value

        Returns:
            A success message
        """
        from ..trace import sentry_sdk
        with sentry_sdk.start_transaction(op="task", name="Storage.set_config_async") as transaction:
            with sentry_sdk.start_span(op="send_request_async"):
                data = {"key": key, "value": value}
                response = await self.send_request_async("/storage/config/set", data=data)
            return response.get("message")

    def bulk_set_config(self, configs: Dict[str, str]) -> str:
        """
        Set multiple configuration values on the server at once.

        Args:
            configs: Dictionary of configuration key-value pairs

        Returns:
            A success message
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    from ..base import run_coroutine_in_new_thread
                    return run_coroutine_in_new_thread(self.bulk_set_config_async(configs))
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.bulk_set_config_async(configs))
        except Exception as e:
            raise e

    async def bulk_set_config_async(self, configs: Dict[str, str]) -> str:
        """
        Set multiple configuration values on the server at once asynchronously.

        Args:
            configs: Dictionary of configuration key-value pairs

        Returns:
            A success message
        """
        data = {"configs": configs}
        response = await self.send_request_async("/storage/config/bulk_set", data=data)
        return response.get("message")

    def set_default_llm_model(self, llm_model: str):
        self.default_llm_model = llm_model

    def config(self, config: ClientConfig):
        """
        Configure the client.
        
        Args:
            config: ClientConfig object with configuration values
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    from ..base import run_coroutine_in_new_thread
                    return run_coroutine_in_new_thread(self.config_async(config))
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.config_async(config))
        except Exception as e:
            raise e

    async def config_async(self, config: ClientConfig):
        """
        Configure the client asynchronously.
        
        Args:
            config: ClientConfig object with configuration values
        """
        # Create a dictionary of non-None values excluding default_llm_model
        config_dict = {
            key: str(value) for key, value in config.model_dump().items() 
            if key != "DEFAULT_LLM_MODEL" and value is not None
        }
        
        # Bulk set the configurations if there are any
        if config_dict:
            await self.bulk_set_config_async(config_dict)
        
        self.default_llm_model = config.DEFAULT_LLM_MODEL
