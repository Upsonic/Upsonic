from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from upsonic.model_base import Model
    from upsonic.usage import RequestUsage, RunUsage



MODEL_CONTEXT_WINDOWS: Dict[str, Optional[int]] = {
    # GPT-5 family
    'gpt-5.2': 400000,
    'gpt-5.1': 400000,
    'gpt-5': 400000,
    'gpt-5-mini': 400000,
    'gpt-5-nano': 400000,
    'gpt-5.2-chat-latest': 128000,
    'gpt-5.1-chat-latest': 128000,
    'gpt-5-chat-latest': 128000,
    'gpt-5.1-codex-max': 400000,
    'gpt-5.1-codex': 400000,
    'gpt-5.1-codex-mini': 400000,
    'gpt-5-codex': 400000,
    'gpt-5.2-pro': 400000,
    'gpt-5-pro': 400000,
    'gpt-5-search-api': 128000,
    
    # GPT-4.1 family
    'gpt-4.1': 1047576,
    'gpt-4.1-mini': 1047576,
    'gpt-4.1-nano': 1047576,
    
    # GPT-4o family
    'gpt-4o': 128000,
    'gpt-4o-2024-05-13': 128000,
    'gpt-4o-mini': 128000,
    'gpt-4o-search-preview': 128000,
    'gpt-4o-mini-search-preview': 128000,
    
    # GPT Realtime models
    'gpt-realtime': 32000,
    'gpt-realtime-mini': 32000,
    'gpt-4o-realtime-preview': 32000,
    'gpt-4o-mini-realtime-preview': 16000,
    
    # GPT Audio models
    'gpt-audio': 128000,
    'gpt-audio-mini': 128000,
    'gpt-4o-audio-preview': 128000,
    'gpt-4o-mini-audio-preview': 128000,
    
    # O-series reasoning models
    'o1': 200000,
    'o1-pro': 200000,
    'o1-mini': 128000,
    'o3': 200000,
    'o3-pro': 200000,
    'o3-mini': 200000,
    'o3-deep-research': 200000,
    'o4-mini': 200000,
    'o4-mini-deep-research': 200000,
    
    # Codex and special models
    'codex-mini-latest': 200000,
    'computer-use-preview': 8192,
    
    # Image models (no text context window)
    'gpt-image-1.5': None,
    'chatgpt-image-latest': None,
    'gpt-image-1': None,
    'gpt-image-1-mini': None,
    
    'claude-3-5-sonnet-20241022': 200000,
    'claude-3-5-sonnet-latest': 200000,
    'claude-3-5-sonnet-20240620': 200000,
    'claude-3-5-haiku-20241022': 200000,
    'claude-3-5-haiku-latest': 200000,
    'claude-3-7-sonnet-20250219': 200000,
    'claude-3-7-sonnet-latest': 200000,
    'claude-3-opus-20240229': 200000,
    'claude-3-opus-latest': 200000,
    'claude-3-haiku-20240307': 200000,
    'claude-4-opus-20250514': 200000,
    'claude-4-sonnet-20250514': 200000,
    'claude-opus-4-0': 200000,
    'claude-opus-4-1-20250805': 200000,
    'claude-opus-4-20250514': 200000,
    'claude-sonnet-4-0': 200000,
    'claude-sonnet-4-20250514': 200000,
    
    'gemini-3-pro-preview': 1048576,
    'gemini-3-pro-image-preview': 65536,
    'gemini-3-flash-preview': 1048576,
    'gemini-2.5-flash': 1048576,
    'gemini-2.5-flash-preview-09-2025': 1048576,
    'gemini-2.5-flash-image': 65536,
    'gemini-2.5-flash-lite': 1048576,
    'gemini-2.5-flash-lite-preview-09-2025': 1048576,
    'gemini-2.5-pro': 1048576,
    'gemini-2.0-flash': 1048576,
    'gemini-2.0-flash-lite': 1048576,
    'gemini-1.5-pro': 1048576,
    'gemini-1.5-flash': 1048576,
    'gemini-1.0-pro': 32000,
    
    'llama-3-8b': 8000,
    'llama-3-70b': 8000,
    'llama-3.1-8b': 128000,
    'llama-3.1-8b-instant': 128000,
    'llama-3.1-70b': 128000,
    'llama-3.1-405b': 128000,
    'llama-3.2-1b': 128000,
    'llama-3.2-3b': 128000,
    'llama-3.2-11b-vision': 128000,
    'llama-3.2-90b-vision': 128000,
    'llama-3.3-70b': 128000,
    'llama-3.3-70b-versatile': 128000,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'llama-4-maverick': 128000,
    'llama-4-maverick-17b-128e-instruct': 128000,
    'llama-4-scout': 10000000,
    'llama-4-scout-17b-16e-instruct': 10000000,
    'meta-llama/Llama-3.3-70B-Instruct': 128000,
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct': 128000,
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': 10000000,
    
    'grok-3': 131000,
    'grok-3-mini': None,
    'grok-3-fast': 131000,
    'grok-3-mini-fast': None,
    'grok-4': 256000,
    'grok-4-0709': 256000,
    'grok-4-fast-reasoning': None,
    'grok-4.1': 256000,
    'grok-4-1-fast-reasoning': 2000000,
    'grok-4-1-fast-non-reasoning': 2000000,
    'grok-code-fast-1': 2000000,
    
    'qwen-max': 32000,
    'qwen-plus': 1000000,
    'qwen-turbo': 1000000,
    'qwen-flash': 1000000,
    'qwen-long': 10000000,
    'qwen3-0.6b': 32000,
    'qwen3-1.7b': 32000,
    'qwen3-4b': 32000,
    'qwen3-8b': 128000,
    'qwen3-14b': 128000,
    'qwen3-32b': 128000,
    'qwen-3-32b': 128000,
    'qwen3-235b-a22b': 128000,
    'qwen-3-235b-a22b-instruct-2507': 128000,
    'qwen3-235b-a22b-thinking': 128000,
    'qwen-3-235b-a22b-thinking-2507': 128000,
    'qwen3-coder-plus': 1000000,
    'qwen-3-coder-480b': 1000000,
    'qwen-vl-plus': None,
    'qwen-vl-max': None,
    'qwen-audio-turbo': None,
    'Qwen/QwQ-32B': 128000,
    'Qwen/Qwen2.5-72B-Instruct': 128000,
    'Qwen/Qwen3-235B-A22B': 128000,
    'Qwen/Qwen3-32B': 128000,
    
    'deepseek-v2-base': 128000,
    'deepseek-v2-chat': 128000,
    'deepseek-v3-base': 128000,
    'deepseek-v3': 64000,
    'deepseek-chat': 64000,
    'deepseek-v3.1': 128000,
    'deepseek-v3.2-exp': 128000,
    'deepseek-r1': 128000,
    'deepseek-reasoner': 64000,
    'deepseek-r1-zero': 128000,
    'deepseek-r1-distill-qwen-1.5b': 128000,
    'deepseek-r1-distill-qwen-7b': 128000,
    'deepseek-r1-distill-llama-8b': 128000,
    'deepseek-r1-distill-qwen-14b': 128000,
    'deepseek-r1-distill-qwen-32b': 128000,
    'deepseek-r1-distill-llama-70b': 128000,
    'deepseek-coder-1.3b': 16000,
    'deepseek-coder-6.7b': 16000,
    'deepseek-coder-33b': 16000,
    'deepseek-coder-v2-lite-base': 128000,
    'deepseek-coder-v2-base': 128000,
    'deepseek-coder-v2-instruct': 128000,
    'deepseek-math-v2': 128000,
    'deepseek-prover-v2-7b': 128000,
    'deepseek-prover-v2-671b': 128000,
    'deepseek-vl-7b-chat': None,
    'deepseek-ocr-3b': None,
    'janus-pro-7b': None,
    'deepseek-ai/DeepSeek-R1': 128000,
    
    'mistral-7b-v0.1': 8000,
    'open-mistral-7b': 8000,
    'mistral-7b-v0.2': 32000,
    'mistral-7b-v0.3': 32000,
    'mistral-small': 32000,
    'mistral-small-latest': 128000,
    'mistral-small-2409': 128000,
    'mistral-small-3-1': 128000,
    'mistral-small-2506': 128000,
    'mistral-medium-3': 128000,
    'mistral-medium-3-1': 128000,
    'mistral-large': 32000,
    'mistral-large-latest': 128000,
    'mistral-large-2407': 128000,
    'mistral-large-2411': 131000,
    'mistral-large-3': 256000,
    'mistral-nemo': 128000,
    'mixtral-8x7b': 32000,
    'open-mixtral-8x7b': 32000,
    'mixtral-8x7b-32768': 32768,
    'mixtral-8x22b': 64000,
    'open-mixtral-8x22b': 64000,
    'mixtral-8x22b-v0.3': 64000,
    'ministral-3b-2410': 128000,
    'ministral-8b-2410': 128000,
    'ministral-14b': 128000,
    'codestral-2405': 256000,
    'codestral-latest': 256000,
    'codestral-2501': 256000,
    'codestral-2508': 256000,
    'codestral-mamba': None,
    'devstral-small-2507': None,
    'devstral-medium-2507': None,
    'mathstral-7b-v0.1': 32000,
    'pixtral-12b-2409': 128000,
    'pixtral-large-2411': 131000,
    'magistral-small-2507': None,
    'magistral-medium-2507': None,
    'voxtral-small-2507': 32000,
    'voxtral-mini-2507': None,
    'command': 4096,
    'command-light': 4096,
    'command-r': 128000,
    'command-r-plus': 128000,
    'gemma2-9b-it': 8192,
    'gpt-oss-120b': None,
    'llama3.1-8b': 128000,

    'test': None,
}

DEFAULT_CONTEXT_WINDOW: Optional[int] = None

PROVIDER_PREFIXES: list[str] = [
    'anthropic:',
    'google-gla:',
    'google-vertex:',
    'groq:',
    'mistral:',
    'cohere:',
    'deepseek:',
    'grok:',
    'moonshotai:',
    'cerebras:',
    'huggingface:',
    'heroku:',
    'bedrock:',
    'openai:',
]


def get_model_name(model: Union["Model", str]) -> str:
    """
    Extract the model name from a Model instance or string.
    
    Args:
        model: Model instance or model name string (e.g., "openai/gpt-4o-mini")
    
    Returns:
        The extracted model name without provider prefix.
    """
    if isinstance(model, str):
        # Handle provider/model format like "openai/gpt-4o-mini"
        if '/' in model:
            return model.split('/', 1)[1]
        return model
    elif hasattr(model, 'model_name'):
        model_name = model.model_name
        # Handle case where model_name might be a coroutine (in tests)
        if hasattr(model_name, '__await__'):
            return "test-model"
        return model_name
    else:
        return str(model)


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name by stripping provider prefixes.
    
    Args:
        model_name: The model name to normalize.
    
    Returns:
        The normalized model name without provider prefixes.
    """
    # Handle case where model_name might be a coroutine (in tests)
    if hasattr(model_name, '__await__'):
        model_name = "test-model"
    
    # Ensure model_name is a string
    model_name = str(model_name)
    
    # Strip provider prefixes
    for prefix in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]
            break
    
    return model_name


def get_model_context_window(model: Union["Model", str]) -> Optional[int]:
    """
    Get the context window size for a model.
    
    Args:
        model: Model instance or model name string.
    
    Returns:
        The context window size in tokens, or None if unknown.
    """
    model_name = get_model_name(model)
    normalized_name = normalize_model_name(model_name)
    
    return MODEL_CONTEXT_WINDOWS.get(normalized_name, DEFAULT_CONTEXT_WINDOW)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: Union["Model", str],
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """
    Calculate the cost in dollars based on token usage and model.
    
    Uses the genai_prices library for accurate pricing calculations.
    
    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        model: Model instance or model name string.
        cache_write_tokens: Number of cache write tokens (if applicable).
        cache_read_tokens: Number of cache read tokens (if applicable).
        reasoning_tokens: Number of reasoning tokens (for o1/o3 models).
    
    Returns:
        The calculated cost as a float (in dollars).
    
    Raises:
        ImportError: If genai_prices library is not installed.
    """
    if input_tokens is None or output_tokens is None:
        return 0.0
    
    try:
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        cache_write_tokens = max(0, int(cache_write_tokens or 0))
        cache_read_tokens = max(0, int(cache_read_tokens or 0))
        reasoning_tokens = max(0, int(reasoning_tokens or 0))
    except (ValueError, TypeError):
        return 0.0
    
    from genai_prices import calc_price
    from upsonic.usage import RequestUsage

    usage = RequestUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )

    model_name = get_model_name(model)
    price_calc = calc_price(usage, model_name)
    cost = float(price_calc.total_price)

    if reasoning_tokens > 0:
        reasoning_usage = RequestUsage(
            input_tokens=0,
            output_tokens=reasoning_tokens,
            cache_write_tokens=0,
            cache_read_tokens=0,
        )
        reasoning_price = calc_price(reasoning_usage, model_name)
        cost += float(reasoning_price.total_price)

    return cost


def calculate_cost_from_usage(
    usage: Union[Dict[str, int], "RequestUsage", "RunUsage"],
    model: Union["Model", str]
) -> float:
    """
    Calculate cost from a usage object (RequestUsage, RunUsage, or dict).
    
    Args:
        usage: Usage object or dictionary with token counts.
        model: Model instance or model name string.
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    if isinstance(usage, dict):
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cache_write_tokens = usage.get('cache_write_tokens', 0)
        cache_read_tokens = usage.get('cache_read_tokens', 0)
        reasoning_tokens = usage.get('reasoning_tokens', 0)
    else:
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        cache_write_tokens = getattr(usage, 'cache_write_tokens', 0)
        cache_read_tokens = getattr(usage, 'cache_read_tokens', 0)
        reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
    
    return calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def calculate_cost_from_run_output(
    run_output: Any,
    model: Union["Model", str]
) -> float:
    """
    Calculate cost from an AgentRunOutput object.
    
    If the run_output has a usage attribute with cost already set, return it.
    Otherwise, calculate from the messages in the run output.
    
    Args:
        run_output: AgentRunOutput object.
        model: Model instance or model name string.
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    # First check if usage already has cost
    if hasattr(run_output, 'usage') and run_output.usage:
        if hasattr(run_output.usage, 'cost') and run_output.usage.cost is not None:
            return run_output.usage.cost
        # Calculate from usage tokens
        return calculate_cost_from_usage(run_output.usage, model)
    
    # Fall back to calculating from messages
    total_input_tokens = 0
    total_output_tokens = 0
    
    if hasattr(run_output, 'all_messages'):
        messages = run_output.all_messages()
        for message in messages:
            if hasattr(message, 'usage') and message.usage:
                if hasattr(message, 'kind') and message.kind == 'response':
                    usage = message.usage
                    total_input_tokens += getattr(usage, 'input_tokens', 0)
                    total_output_tokens += getattr(usage, 'output_tokens', 0)
    
    return calculate_cost(total_input_tokens, total_output_tokens, model)


def calculate_cost_from_agent(agent: Any) -> float:
    """
    Calculate cost from an Agent's current run or session.
    
    Args:
        agent: Agent instance.
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    # Try to get session usage first
    if hasattr(agent, 'get_session_usage'):
        session_usage = agent.get_session_usage()
        if session_usage and hasattr(session_usage, 'cost') and session_usage.cost is not None:
            return session_usage.cost
    
    # Fall back to run output
    if hasattr(agent, 'get_run_output'):
        run_output = agent.get_run_output()
        if run_output:
            model = getattr(agent, 'model', 'gpt-4o-mini')
            return calculate_cost_from_run_output(run_output, model)
    
    return 0.0


def format_cost(cost: float, approximate: bool = True) -> str:
    """
    Format a cost value as a string.
    
    Args:
        cost: The cost value in dollars.
        approximate: Whether to prefix with "~" for approximate values.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123" or "$0.0123").
    """
    prefix = "~" if approximate else ""
    
    if cost < 0.0001:
        return f"{prefix}${cost:.6f}"
    elif cost < 0.01:
        return f"{prefix}${cost:.5f}"
    else:
        return f"{prefix}${cost:.4f}"


def get_estimated_cost(
    input_tokens: int,
    output_tokens: int,
    model: Union["Model", str]
) -> str:
    """
    Calculate and format estimated cost as a string.
    
    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        model: Model instance or model name string.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost(input_tokens, output_tokens, model)
    return format_cost(cost, approximate=True)


def get_estimated_cost_from_usage(
    usage: Union[Dict[str, int], Any],
    model: Union["Model", str]
) -> str:
    """
    Calculate and format estimated cost from usage data as a string.
    
    Args:
        usage: Usage object or dictionary with token counts.
        model: Model instance or model name string.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost_from_usage(usage, model)
    return format_cost(cost, approximate=True)


def get_estimated_cost_from_run_output(
    run_output: Any,
    model: Union["Model", str]
) -> str:
    """
    Calculate and format estimated cost from AgentRunOutput as a string.
    
    Args:
        run_output: AgentRunOutput object.
        model: Model instance or model name string.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost_from_run_output(run_output, model)
    return format_cost(cost, approximate=True)


def get_estimated_cost_from_agent(agent: Any) -> str:
    """
    Calculate and format estimated cost from Agent as a string.
    
    Args:
        agent: Agent instance.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost_from_agent(agent)
    return format_cost(cost, approximate=True)
