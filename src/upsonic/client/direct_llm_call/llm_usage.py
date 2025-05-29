def llm_usage(model_response):
    return {"input_tokens": model_response.usage().request_tokens, "output_tokens": model_response.usage().response_tokens}