import json
from pathlib import Path


def init_command() -> int:
    """
    Initialize a new Upsonic agent project.
    
    Prompts the user for an agent name and creates:
    - src/io.py: Input/output schema definitions
    - src/main.py: Main agent file
    - upsonic_configs.json: Configuration file with agent name
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Lazy import printer functions
        from upsonic.cli.printer import (
            prompt_agent_name,
            print_error,
            print_file_created,
            confirm_overwrite,
            print_cancelled,
            print_init_success,
        )
        
        # Prompt for agent name
        agent_name = prompt_agent_name()
        
        if not agent_name:
            print_error("Agent name cannot be empty.")
            return 1
        
        # Get current directory
        current_dir = Path.cwd()
        
        # Create src directory
        src_dir = current_dir / "src"
        
        # Check if src directory already exists and has files
        io_py_path = src_dir / "io.py"
        main_py_path = src_dir / "main.py"
        config_json_path = current_dir / "upsonic_configs.json"
        
        if io_py_path.exists():
            if not confirm_overwrite(io_py_path):
                print_cancelled()
                return 1
        
        if main_py_path.exists():
            if not confirm_overwrite(main_py_path):
                print_cancelled()
                return 1
        
        if config_json_path.exists():
            if not confirm_overwrite(config_json_path):
                print_cancelled()
                return 1
        
        # Create src directory if it doesn't exist
        src_dir.mkdir(exist_ok=True)
        
        # Create io.py
        io_py_content = """inputs = [
    {
        "name": "question",
        "description": "The mail to analyze",
        "type": "string",
        "default": None,
        "required": True
    }
]

outputs = [
    {
        "name": "answer",
        "description": "The action list to the person",
        "type": "string"
    }
]
"""
        
        io_py_path.write_text(io_py_content, encoding="utf-8")
        print_file_created(io_py_path)
        
        # Create main.py (renamed from agent.py)
        main_py_content = """from upsonic import Task, Agent


async def main(inputs):
    question = inputs.get("question")
    answering_task = Task(f"Answer the user question {question}")
    agent = Agent()
    result = await agent.print_do_async(answering_task)
    return {
        "answer": result
    }
"""
        
        main_py_path.write_text(main_py_content, encoding="utf-8")
        print_file_created(main_py_path)
        
        # Create upsonic_configs.json
        config_data = {
            "envinroment_variables": {
                "UPSONIC_WORKERS_AMOUNT": {
                    "type": "number",
                    "description": "The number of workers for the Upsonic API",
                    "default": 1
                },
                "API_WORKERS": {
                    "type": "number",
                    "description": "The number of workers for the Upsonic API",
                    "default": 1
                },
                "RUNNER_CONCURRENCY": {
                    "type": "number",
                    "description": "The number of runners for the Upsonic API",
                    "default": 1
                }
            },
            "machine_spec": {
                "cpu": 2,
                "memory": 4096,
                "storage": 1024
            },
            "agent_name": agent_name,
            "description": "An Upsonic AI agent that processes user inputs and generates intelligent responses",
            "icon": "book",
            "language": "book",
            "streamlit": False,
            "proxy_agent": False,
            "dependencies": {
                "api": [
                    "fastapi>=0.115.12",
                    "uvicorn>=0.34.2",
                    "aiofiles>=24.1.0",
                    "celery>=5.5.2",
                    "sqlalchemy>=2.0.40",
                    "psycopg2-binary>=2.9.9",
                    "upsonic",
                    "pytz>=2025.2",
                    "psutil>=5.9.8",
                    "fire>=0.7.0",
                    "ruamel.yaml>=0.18.5",
                    "redis>=5.0.0"
                ],
                "streamlit": [
                    "streamlit==1.40.0",
                    "pandas",
                    "numpy==2.2.6"
                ],
                "development": [
                    "watchdog",
                    "python-dotenv",
                    "ipdb",
                    "pytest",
                    "streamlit-autorefresh"
                ]
            },
            "input_schema": {
                "inputs": {
                    "question": {
                        "type": "string",
                        "description": "The question of the User",
                        "required": True,
                        "default": None
                    }
                }
            },
            "output_schema": {
                "answer": {
                    "type": "string",
                    "description": "Answer of the agent"
                }
            }
        }
        
        config_json_path.write_text(
            json.dumps(config_data, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )
        print_file_created(config_json_path)
        
        # Print success message with created files
        print_init_success(agent_name, [str(io_py_path), str(main_py_path), str(config_json_path)])
        return 0
        
    except KeyboardInterrupt:
        from upsonic.cli.printer import print_cancelled
        print_cancelled()
        return 1
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        return 1

