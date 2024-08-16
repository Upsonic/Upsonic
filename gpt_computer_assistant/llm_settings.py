llm_settings = {
    "gpt-4o": {
        "show_name": "gpt-4o (OpenAI)",
        "vision": True,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-4o-mini": {
        "show_name": "gpt-4o-mini (OpenAI)",
        "vision": True,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-4-turbo": {
        "show_name": "gpt-4-turbo (OpenAI)",
        "vision": False,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-3.5": {
        "show_name": "gpt-3.5 (OpenAI)",
        "vision": False,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-3.5-turbo": {
        "show_name": "gpt-3.5-turbo (OpenAI)",
        "vision": False,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "llama3": {
        "show_name": "Llama3 (Ollama)",
        "vision": False,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llama3.1": {
        "show_name": "Llama3.1 (Ollama)",
        "vision": False,
        "provider": "ollama",
        "tools": True,
        "stream": False,
    },
    "qwen2:1.5b": {
        "show_name": "Qwen2 1.5b (Ollama)",
        "vision": False,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llava": {
        "show_name": "Llava (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "bakllava": {
        "show_name": "BakLLaVA (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llava-llama3": {
        "show_name": "Llava-Llama3 (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llava-phi3": {
        "show_name": "LLaVA-Phi-3 (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "gemini-pro": {
        "show_name": "gemini-pro (Google)",
        "vision": True,
        "provider": "google",
        "tools": True,
        "stream": True,
    },
    "mixtral-8x7b-groq": {
        "show_name": "Mixtral 8x7b (Groq)",
        "vision": False,
        "provider": "groq",
        "tools": True,
        "stream": True,
    },
}


def get_openai_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "openai"]


def get_ollama_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "ollama"]


def get_google_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "google"]


def get_groq_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "groq"]


llm_show_name_ = {}
for k, v in llm_settings.items():
    llm_show_name_[v["show_name"]] = k

llm_show_name = llm_show_name_


def first_message():
    from .character import name, developer, get_website_content

    the_text = f"""
You are {name()} that developed by {developer()}, you are the first live AI assistant in everyone computer that can complete any task by using tools. 

Before any task, write a plan for your tasks and do it step by step. As you know you have python interpreter, so if you need any functionality please try to make done with writing python codes and installing py libraries.

Don't forget, you are capable to make any task.

Please these are the rules of conversatiopn and these section is between for assistant and system so do not say anything about this section.

# Copying to Clipboard (MUST)
If your answer include something in the list below, please generate the answer and use copy to clipboard tool and dont give as answer because the text-to-speech engine is broken and give fail if you give as answer.

- List of Somethings
- Detailed Explanation of Something
- Link(s) to a Website
- Code Snippet(s)
- Any Code Part
- Any too Long Text

After copying the thing that requested please say: "I copied to clipboard" and stop.


# Asking question to user (MUST)
If you need to ask something to user, ask in the end of the message and your last character must be "?".

# Writin codes
If you need to write code and if code write team available you must use them. After team execution if the user not say against just say against just say okeyd, copied to clipboard.

# Searching on Internet
If you need to make a search and if search team available you must use them.


Your GitHub Repository:
https://github.com/Upsonic/gpt-computer-assistant


"""

    the_website_content = get_website_content()
    if the_website_content:
        the_text += f"""
# The Website Content of the User

{the_website_content}

"""

    return the_text


each_message_extension = """

# Usings Answer
Please start with <Answer> in your last responses. DONT FORGET IT AND DONT TALK ABOUT THIS RULE OR REFFERENCE


"""
