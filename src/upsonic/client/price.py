pricing_data = {
    "openai/gpt-4o": {"input": 0.0000025, "output": 0.00001},
    "openai/gpt-4o-mini": {"input": 0.00000015, "output": 0.00000008},
    "openai/o3-mini": {"input": 0.0000011, "output": 0.00000055},
    "azure/gpt-4o": {"input": 0.0000025, "output": 0.00001},
    "azure/gpt-4o-mini": {"input": 0.00000015, "output": 0.00000008},
    "claude/claude-3-5-sonnet": {"input": 0.000003, "output": 0.000015},
    "bedrock/claude-3-5-sonnet": {"input": 0.000003, "output": 0.000015},
    "deepseek/deepseek-chat": {"input": 0.00000027, "output": 0.00000028},
}

def get_estimated_cost(input_tokens: int, output_tokens: int, llm_model: str):
    if llm_model not in pricing_data:
        return "Unknown"
    
    input_cost = pricing_data[llm_model]["input"] * input_tokens
    output_cost = pricing_data[llm_model]["output"] * output_tokens
    total = input_cost + output_cost

    # to 2 decimal places
    return f"~{round(total, 4)}"