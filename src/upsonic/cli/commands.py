import importlib.util
import json
import subprocess
import sys
import zipfile
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy import cache for heavy dependencies
_FASTAPI_IMPORTS = None

# Cache for config files to avoid repeated I/O
_CONFIG_CACHE = {}


def _get_fastapi_imports():
    """Lazy load FastAPI dependencies only when needed."""
    global _FASTAPI_IMPORTS
    if _FASTAPI_IMPORTS is None:
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            import uvicorn
            
            _FASTAPI_IMPORTS = {
                'FastAPI': FastAPI,
                'Request': Request,
                'JSONResponse': JSONResponse,
                'uvicorn': uvicorn,
            }
        except ImportError:
            return None
    return _FASTAPI_IMPORTS


def _load_config(config_path: Path, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load and parse config file with caching.
    
    Args:
        config_path: Path to upsonic_configs.json
        use_cache: Whether to use cached config (default: True)
    
    Returns:
        Parsed config dictionary or None if error
    """
    cache_key = str(config_path.absolute())
    
    if use_cache and cache_key in _CONFIG_CACHE:
        # Check if file has been modified
        try:
            current_mtime = config_path.stat().st_mtime
            cached_mtime, cached_data = _CONFIG_CACHE[cache_key]
            if current_mtime == cached_mtime:
                return cached_data
        except Exception:
            pass
    
    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # Cache it
        if use_cache:
            try:
                mtime = config_path.stat().st_mtime
                _CONFIG_CACHE[cache_key] = (mtime, config_data)
            except Exception:
                pass
        
        return config_data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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


def add_command(library: str, section: str) -> int:
    """
    Add a dependency to upsonic_configs.json.
    
    Args:
        library: Library name with version (e.g., "x_library==0.52.0")
        section: Section name in dependencies (e.g., "api", "streamlit", "development")
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Lazy import printer functions
        from upsonic.cli.printer import (
            print_config_not_found,
            print_error,
            print_invalid_section,
            print_dependency_added,
        )
        
        # Get current directory
        current_dir = Path.cwd()
        config_json_path = current_dir / "upsonic_configs.json"
        
        # Check if config file exists
        if not config_json_path.exists():
            print_config_not_found()
            return 1
        
        # Read the config file (don't use cache since we're modifying)
        config_data = _load_config(config_json_path, use_cache=False)
        if config_data is None:
            print_error("Invalid JSON in upsonic_configs.json")
            return 1
        
        # Validate dependencies section exists
        if "dependencies" not in config_data:
            print_error("'dependencies' section not found in upsonic_configs.json")
            return 1
        
        dependencies = config_data["dependencies"]
        
        # Validate section exists
        if section not in dependencies:
            available_sections = list(dependencies.keys())
            print_invalid_section(section, available_sections)
            return 1
        
        # Check if dependency already exists
        if library in dependencies[section]:
            print_error(f"Dependency '{library}' already exists in dependencies.{section}")
            return 1
        
        # Add the dependency
        dependencies[section].append(library)
        
        # Write back to file
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        
        # Print success message
        print_dependency_added(library, section)
        return 0
        
    except KeyboardInterrupt:
        from upsonic.cli.printer import print_cancelled
        print_cancelled()
        return 1
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        return 1




def _map_inputs_props(inputs_schema: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Map input schema to JSON and multipart properties for OpenAPI schema.
    
    Args:
        inputs_schema: List of input schema items
        
    Returns:
        Tuple of (json_props, multipart_props, required_fields)
    """
    json_props = {}
    multipart_props = {}
    required = []
    
    for item in inputs_schema:
        name = item["name"]
        itype = item.get("type", "string")
        default = item.get("default")
        
        if itype in ("files",):
            json_props[name] = {"type": "array", "items": {"type": "string"}}
            multipart_props[name] = {"type": "array", "items": {"type": "string", "format": "binary"}}
        elif itype in ("file", "binary", "string($binary)"):
            json_props[name] = {"type": "string"}
            multipart_props[name] = {"type": "string", "format": "binary"}
        elif itype == "number":
            json_props[name] = {"type": "number"}
            multipart_props[name] = {"type": "number"}
        elif itype == "integer":
            json_props[name] = {"type": "integer"}
            multipart_props[name] = {"type": "integer"}
        elif itype in ("boolean", "bool"):
            json_props[name] = {"type": "boolean"}
            multipart_props[name] = {"type": "boolean"}
        elif itype in ("list", "array"):
            json_props[name] = {"type": "array", "items": {"type": "string"}}
            multipart_props[name] = {"type": "array", "items": {"type": "string"}}
        elif itype in ("json",):
            json_props[name] = {"type": "object"}
            multipart_props[name] = {"type": "object"}
        else:
            json_props[name] = {"type": "string"}
            multipart_props[name] = {"type": "string"}
        
        # Add default values if present
        if default is not None:
            try:
                json_props[name]["default"] = default
            except Exception:
                pass
        
        if item.get("required", False):
            required.append(name)
    
    return json_props, multipart_props, required


def _map_output_props(output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map output schema to OpenAPI properties.
    
    Args:
        output_schema: Output schema dictionary
        
    Returns:
        Dictionary of OpenAPI properties
    """
    props = {}
    for k, v in (output_schema or {}).items():
        t = v.get("type", "string")
        if t == "number":
            props[k] = {"type": "number", "description": v.get("description", "")}
        elif t == "integer":
            props[k] = {"type": "integer", "description": v.get("description", "")}
        elif t in ("boolean", "bool"):
            props[k] = {"type": "boolean", "description": v.get("description", "")}
        elif t in ("list", "array"):
            props[k] = {"type": "array", "items": {"type": "string"}, "description": v.get("description", "")}
        elif t in ("json", "object"):
            props[k] = {"type": "object", "description": v.get("description", "")}
        else:
            props[k] = {"type": "string", "description": v.get("description", "")}
    return props


def _modify_openapi_schema(
    schema: Dict[str, Any],
    inputs_schema: List[Dict[str, Any]],
    output_schema_dict: Dict[str, Any],
    path: str = "/call",
) -> Dict[str, Any]:
    """
    Modify OpenAPI schema to include multipart/form-data and application/json.
    
    Args:
        schema: The OpenAPI schema dictionary
        inputs_schema: List of input schema items
        output_schema_dict: Output schema dictionary
        path: The endpoint path to modify
        
    Returns:
        Modified schema dictionary
    """
    paths = schema.get("paths", {})
    if path not in paths:
        return schema
    
    post_op = paths[path].get("post", {})
    
    # Map props
    json_props, multipart_props, required_fields = _map_inputs_props(inputs_schema)
    
    # Build RequestModel for JSON
    request_model = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": json_props
            }
        }
    }
    if required_fields:
        request_model["properties"]["data"]["required"] = required_fields
    
    # Add components/schemas
    components = schema.setdefault("components", {})
    comps_schemas = components.setdefault("schemas", {})
    comps_schemas["RequestModel"] = request_model
    comps_schemas["JobStatus"] = {
        "type": "object",
        "properties": _map_output_props(output_schema_dict)
    }
    
    # Build content with multipart/form-data FIRST
    content = {}
    multipart_schema = {
        "type": "object",
        "properties": multipart_props,
    }
    if required_fields:
        multipart_schema["required"] = required_fields
    content["multipart/form-data"] = {"schema": multipart_schema}
    content["application/json"] = {"schema": {"$ref": "#/components/schemas/RequestModel"}}
    
    # Set responses
    responses = post_op.setdefault("responses", {})
    responses["200"] = {
        "description": "Successful Response",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/JobStatus"}
            }
        }
    }
    
    # Replace requestBody
    post_op["requestBody"] = {
        "required": True,
        "content": content
    }
    
    # Ensure tags
    if not post_op.get("tags"):
        post_op["tags"] = ["jobs"]
    
    return schema


def _install_dependencies(dependencies: list[str], quiet: bool = False) -> bool:
    """
    Install dependencies using uv or pip.
    
    Args:
        dependencies: List of dependency strings (e.g., ["fastapi>=0.115.12", "uvicorn>=0.34.2"])
        quiet: If True, suppress output messages
    
    Returns:
        True if successful, False otherwise.
    """
    if not dependencies:
        return True
    
    try:
        # Lazy import printer functions only when needed
        if not quiet:
            from upsonic.cli.printer import print_info, print_success, print_error
            print_info(f"Installing {len(dependencies)} dependencies...")
        
        # Try uv first (preferred for this project)
        try:
            result = subprocess.run(
                ["uv", "add"] + dependencies,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                if not quiet:
                    from upsonic.cli.printer import print_success
                    print_success("Dependencies installed successfully")
                return True
            # If uv fails, fall back to pip
        except FileNotFoundError:
            pass
        
        # Fall back to pip
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + dependencies,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                if not quiet:
                    from upsonic.cli.printer import print_success
                    print_success("Dependencies installed successfully")
                return True
            else:
                if not quiet:
                    from upsonic.cli.printer import print_error
                    print_error(f"Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            if not quiet:
                from upsonic.cli.printer import print_error
                print_error(f"Error installing dependencies: {str(e)}")
            return False
            
    except Exception as e:
        if not quiet:
            from upsonic.cli.printer import print_error
            print_error(f"Error installing dependencies: {str(e)}")
        return False


def install_command(section: Optional[str] = None) -> int:
    """
    Install dependencies from upsonic_configs.json.
    
    Args:
        section: Specific section to install ("api", "streamlit", "development", "all", or None for "api")
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Lazy import printer functions
        from upsonic.cli.printer import (
            print_config_not_found,
            print_error,
            print_invalid_section,
            print_info,
        )
        
        # Get current directory
        current_dir = Path.cwd()
        config_json_path = current_dir / "upsonic_configs.json"
        
        # Check if config file exists
        if not config_json_path.exists():
            print_config_not_found()
            return 1
        
        # Read config (use cache since we're only reading)
        config_data = _load_config(config_json_path)
        if config_data is None:
            print_error("Invalid JSON in upsonic_configs.json")
            return 1
        
        # Get dependencies
        all_dependencies = config_data.get("dependencies", {})
        if not all_dependencies:
            print_error("No dependencies section found in upsonic_configs.json")
            return 1
        
        # Determine which sections to install
        if section is None or section == "api":
            sections_to_install = ["api"]
        elif section == "all":
            sections_to_install = list(all_dependencies.keys())
        else:
            sections_to_install = [section]
        
        # Validate sections
        for sec in sections_to_install:
            if sec not in all_dependencies:
                available_sections = list(all_dependencies.keys())
                print_invalid_section(sec, available_sections)
                return 1
        
        # Collect all dependencies to install
        dependencies_to_install = []
        for sec in sections_to_install:
            dependencies_to_install.extend(all_dependencies[sec])
        
        if not dependencies_to_install:
            print_info("No dependencies to install")
            return 0
        
        # Install dependencies
        if _install_dependencies(dependencies_to_install):
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        from upsonic.cli.printer import print_cancelled
        print_cancelled()
        return 1
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        return 1


def run_command(host: str = "0.0.0.0", port: int = 8000) -> int:
    """
    Run the agent as a FastAPI server.

    Dynamically builds OpenAPI for both multipart/form-data and application/json
    from upsonic_configs.json input_schema/output_schema so /docs shows editable form fields.
    """
    try:
        # Lazy import printer functions
        from upsonic.cli.printer import (
            print_config_not_found,
            print_error,
            print_success,
            print_info,
        )
        
        # Get current directory
        current_dir = Path.cwd()
        config_json_path = current_dir / "upsonic_configs.json"

        # Check if config file exists
        if not config_json_path.exists():
            print_config_not_found()
            return 1

        # Read config (use cache for faster startup)
        config_data = _load_config(config_json_path)
        if config_data is None:
            print_error("Invalid JSON in upsonic_configs.json")
            return 1

        # Get FastAPI imports (lazy loaded)
        fastapi_imports = _get_fastapi_imports()
        if fastapi_imports is None:
            print_error("FastAPI dependencies not found. Please run: upsonic install")
            return 1
        
        # Extract imports from cache
        FastAPI = fastapi_imports['FastAPI']
        JSONResponse = fastapi_imports['JSONResponse']
        uvicorn = fastapi_imports['uvicorn']
        request_fastapi = fastapi_imports['Request']


        # Agent metadata
        agent_name = config_data.get("agent_name", "Upsonic Agent")
        description = config_data.get("description", "An Upsonic AI agent")

        # Load main.py from src directory
        agent_py_file = config_data.get("agent_py", "src/main.py")
        agent_py_path = current_dir / agent_py_file
        if not agent_py_path.exists():
            print_error(f"Agent file not found: {agent_py_path}")
            return 1

        spec = importlib.util.spec_from_file_location("main", agent_py_path)
        if spec is None or spec.loader is None:
            print_error(f"Failed to load agent module from {agent_py_path}")
            return 1

        agent_module = importlib.util.module_from_spec(spec)
        sys.modules["main"] = agent_module
        spec.loader.exec_module(agent_module)

        if not hasattr(agent_module, "main"):
            print_error(f"main function not found in {agent_py_file}")
            return 1
        main_func = agent_module.main

        # Build inputs_schema list from config
        input_schema_dict = config_data.get("input_schema", {}).get("inputs", {}) or {}
        inputs_schema = []
        for field_name, field_config in input_schema_dict.items():
            inputs_schema.append({
                "name": field_name,
                "type": field_config.get("type", "string"),
                "required": bool(field_config.get("required", False)),
                "default": field_config.get("default"),
                "description": field_config.get("description", "") or ""
            })

        # Build output schema
        output_schema_dict = config_data.get("output_schema", {}) or {}

        # Create app
        app = FastAPI(title=f"{agent_name} - Upsonic", description=description, version="0.1.0")

        # Import necessary types
        # Create unified endpoint that handles BOTH multipart/form-data AND application/json
        @app.post("/call", summary="Call Main", operation_id="call_main_call_post", tags=["jobs"])
        async def call_endpoint_unified(request: request_fastapi):
            """
            Unified endpoint - accepts BOTH:
            - multipart/form-data (for forms and files)
            - application/json (for JSON APIs)
            """
            try:
                content_type = request.headers.get("content-type", "").lower()
                
                if "application/json" in content_type:
                    # Handle JSON request
                    inputs = await request.json()
                elif "multipart/form-data" in content_type:
                    # Handle multipart/form-data request
                    form_data = await request.form()
                    inputs = {}
                    
                    for key, value in form_data.items():
                        if value is None:
                            continue
                        # Check if it's a file upload
                        if hasattr(value, 'read'):
                            # It's an UploadFile
                            try:
                                inputs[key] = await value.read()
                            except Exception:
                                inputs[key] = None
                        else:
                            # Regular form field
                            inputs[key] = value
                else:
                    # Default to form data for other content types
                    form_data = await request.form()
                    inputs = {k: v for k, v in form_data.items() if v is not None}
                
                # Call main function
                result = await main_func(inputs)
                return JSONResponse(content=result)
                
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "type": type(e).__name__}
                )

        # Override openapi() to return modified schema
        original_openapi = app.openapi
        
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
            
            # Generate and modify schema once
            schema = original_openapi()
            schema = _modify_openapi_schema(schema, inputs_schema, output_schema_dict, "/call")
            app.openapi_schema = schema
            
            return app.openapi_schema
        
        app.openapi = custom_openapi

        # Startup messages
        print_success(f"Starting {agent_name} server...")
        display_host = "localhost" if host == "0.0.0.0" else host
        print_info(f"Server will be available at http://{display_host}:{port}")
        print_info(f"API documentation: http://{display_host}:{port}/docs")
        print_info("Press CTRL+C to stop the server")
        print()

        # Run uvicorn - it handles SIGINT/SIGTERM properly
        # Use log_level="info" to reduce noise, but keep important messages
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except KeyboardInterrupt:
            # This should be caught by uvicorn, but just in case
            from upsonic.cli.printer import print_info
            print()
            print_info("Server stopped by user")
        
        return 0

    except KeyboardInterrupt:
        from upsonic.cli.printer import print_info
        print()
        print_info("Server stopped by user")
        return 0
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def zip_command(output_file: Optional[str] = None) -> int:
    """
    Create a zip file containing the context of the current directory.
    
    This command captures all files in the current directory.
    The resulting zip file can be extracted to restore the complete context.
    
    Args:
        output_file: Name of the output zip file (default: "upsonic_context_<timestamp>.zip")
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Lazy import printer functions
        from upsonic.cli.printer import (
            print_error,
            print_success,
            print_info,
        )
        
        # Get current directory
        current_dir = Path.cwd()
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"upsonic_context_{timestamp}.zip"
        
        # Ensure .zip extension
        if not output_file.endswith('.zip'):
            output_file += '.zip'
        
        output_path = current_dir / output_file
        
        print_info(f"Creating context zip file: {output_file}")
        print_info(f"Scanning directory: {current_dir}")
        
        # Collect files to include
        files_to_zip = []
        total_size = 0
        
        for item in current_dir.rglob('*'):
            if item.is_file() and item != output_path:
                try:
                    size = item.stat().st_size
                    files_to_zip.append(item)
                    total_size += size
                except (PermissionError, OSError) as e:
                    print_info(f"Skipping {item.name}: {str(e)}")
        
        if not files_to_zip:
            print_error("No files found to include in the zip")
            return 1
        
        print_info(f"Found {len(files_to_zip)} files ({total_size / 1024 / 1024:.2f} MB)")
        
        # Create the zip file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files
            for file_path in files_to_zip:
                relative_path = file_path.relative_to(current_dir)
                try:
                    zipf.write(file_path, arcname=relative_path)
                except Exception as e:
                    print_info(f"Warning: Could not add {relative_path}: {str(e)}")
        
        # Print success
        final_size = output_path.stat().st_size
        print_success(f"Context zip created successfully: {output_file}")
        print_info(f"Archive size: {final_size / 1024 / 1024:.2f} MB")
        print_info(f"Files included: {len(files_to_zip)}")
        print_info(f"Extract this file to restore the complete context")
        
        return 0
        
    except KeyboardInterrupt:
        from upsonic.cli.printer import print_cancelled
        print_cancelled()
        return 1
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
