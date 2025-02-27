from dotenv import load_dotenv
load_dotenv()

from ....storage.configuration import Configuration

import asyncio
import atexit



class BrowserManager:
    _instance = None
    _browser = None
    _loop = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            # Register the cleanup function
            atexit.register(cls._cleanup)
        return cls._instance

    @classmethod
    async def initialize(cls):
        instance = cls.get_instance()
        if instance._browser is None:
            from browser_use import Browser
            browser = Browser()
            instance._browser = browser
            instance._loop = asyncio.get_event_loop()
            return browser
        return instance._browser

    @classmethod
    async def get_context(cls):
        """Get a new browser context for isolation"""
        instance = cls.get_instance()
        if instance._browser:
            context = await instance._browser.new_context()
            return context
        return None

    @classmethod
    def _cleanup(cls):
        """Cleanup function that will be called when the Python process exits"""
        instance = cls.get_instance()
        if instance._browser and instance._loop:
            # Create a new event loop if the main one is closed
            try:
                loop = instance._loop if instance._loop.is_running() else asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(instance._browser.close())
            except:
                pass  # Suppress any errors during shutdown

    @classmethod
    async def close(cls):
        """Manual close method - only use if you explicitly need to close the browser"""
        instance = cls.get_instance()
        if instance._browser:
            await instance._browser.close()
            instance._browser = None

    @classmethod
    def get_browser(cls):
        instance = cls.get_instance()
        return instance._browser


class LLMManager:
    _instance = None
    _llm_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_model(cls, model):
        instance = cls.get_instance()
        instance._llm_model = model
        print("SETTING THE LLM MODEL TO:", model)

    @classmethod
    def get_model(cls):
        instance = cls.get_instance()
        print("GETTING THE LLM MODEL:", instance._llm_model)
        return instance._llm_model


def get_llm():
    llm_model = LLMManager.get_model()
    print("THE LLM MODEL IS", llm_model)
    
    if not llm_model:
        raise ValueError("LLM model not set before calling get_llm()")
    
    # Map our model names to standard model names
    openai_model_mapping = {
        "openai/gpt-4o": "gpt-4o",
        "gpt-4o": "gpt-4o",
        "openai/o3-mini": "o3-mini",
        "openai/gpt-4o-mini": "gpt-4o",
        "azure/gpt-4o": "gpt-4o",
        "azure/gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o-azure": "gpt-4o"
    }

    claude_model_mapping = {
        "claude/claude-3-5-sonnet": "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
        "bedrock/claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-sonnet-aws": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    }

    deepseek_model_mapping = {
        "deepseek/deepseek-chat": "deepseek-chat"
    }
    
    # Handle Azure OpenAI
    if llm_model in ["azure/gpt-4o", "gpt-4o-azure", "azure/gpt-4o-mini"]:
        azure_endpoint = Configuration.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = Configuration.get("AZURE_OPENAI_API_KEY")
        azure_api_version = Configuration.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        

            
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            model="gpt-4o",
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key
        )
    
    # Handle regular OpenAI
    elif llm_model in openai_model_mapping:
        openai_api_key = Configuration.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in configuration")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name=openai_model_mapping[llm_model],
            openai_api_key=openai_api_key,
        )

    # Handle Claude (Anthropic)
    elif llm_model in claude_model_mapping:
        if llm_model in ["bedrock/claude-3-5-sonnet", "claude-3-5-sonnet-aws"]:
            # AWS Bedrock configuration
            aws_access_key_id = Configuration.get("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = Configuration.get("AWS_SECRET_ACCESS_KEY")
            aws_region = Configuration.get("AWS_REGION")
            
            if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
                raise ValueError("AWS credentials not found in configuration")
            from langchain_community.chat_models import BedrockChat
            llm = BedrockChat(
                model_id=claude_model_mapping[llm_model],
                credentials_profile_name=None,
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Regular Anthropic configuration
            anthropic_api_key = Configuration.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("Anthropic API key not found in configuration")
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=claude_model_mapping[llm_model],
                anthropic_api_key=anthropic_api_key,
                temperature=0.0,
                timeout=100
            )

    # Handle DeepSeek models
    elif llm_model in deepseek_model_mapping:
        deepseek_api_key = Configuration.get("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DeepSeek API key not found in configuration")
            
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name=deepseek_model_mapping[llm_model],
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
    
    else:
        raise ValueError(f"Unsupported model for browser use: {llm_model}")
    
    return llm


async def BrowserUse__browser_agent(task: str, expected_output: str):
    """An AI agent that can browse the web, extract information, and perform actions."""
    from browser_use import Agent
    
    # Get or create the browser instance
    browser = await BrowserManager.initialize()
    
    # Create a new context for this agent run
    context = await BrowserManager.get_context()
    
    # Create the agent with the browser context
    agent = Agent(
        task=task+"\n\nExpected Output: "+expected_output,
        llm=get_llm(),
        browser=browser,
        browser_context=context,  # Use persistent context
        generate_gif=False
    )
    
    try:
        result = await agent.run()
        return result.final_result()
    finally:
        # Clean up the context after the agent is done
        if context:
            await context.close()


# List of all browser use tools
BrowserUse_tools = [
    BrowserUse__browser_agent
]
