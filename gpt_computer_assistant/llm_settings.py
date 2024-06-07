llm_settings = {
    "gpt-4o": {"vision":True, "transcription":True, "provider":"openai"},
    "gpt-3.5-turbo": {"vision":False, "transcription":True, "provider":"openai"},
    "llava": {"vision":True, "transcription":False, "provider":"ollama"},
    "bakllava": {"vision":True, "transcription":False, "provider":"ollama"},
    "mixtral-8x7b-groq": {"vision":False, "transcription":False, "provider":"groq"},
}

llm_show_name = {
    "gpt-4o (OpenAI)": "gpt-4o",
    "gpt-3.5-turbo (OpenAI)": "gpt-3.5-turbo",
    "Llava (Ollama)": "llava",
    "BakLLaVA (Ollama)": "bakllava",
    "Mixtral 8x7b (Groq)": "mixtral-8x7b-groq",
}