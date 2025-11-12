import contextlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from upsonic.cli.printer import (
    prompt_agent_name,
    print_error,
    print_file_created,
    confirm_overwrite,
    print_cancelled,
    print_init_success,
    print_dependency_added,
    print_config_not_found,
    print_invalid_section,
    print_info,
    print_success,
)


def init_command() -> int:
    """
    Initialize a new Upsonic agent project.
    
    Prompts the user for an agent name and creates:
    - agent.py: Template agent file
    - upsonic_config.json: Configuration file with agent name
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Prompt for agent name
        agent_name = prompt_agent_name()
        
        if not agent_name:
            print_error("Agent name cannot be empty.")
            return 1
        
        # Get current directory
        current_dir = Path.cwd()
        
        # Check if files already exist
        agent_py_path = current_dir / "agent.py"
        config_json_path = current_dir / "upsonic_config.json"
        
        if agent_py_path.exists():
            if not confirm_overwrite(agent_py_path):
                print_cancelled()
                return 1
        
        if config_json_path.exists():
            if not confirm_overwrite(config_json_path):
                print_cancelled()
                return 1
        
        # Create agent.py
        agent_py_content = """from upsonic import Task, Agent


async def main(inputs):
    question = inputs.get("question")
    answering_task = Task(f"Answer the user question {question}")
    agent = Agent()
    result = await agent.print_do_async(answering_task)
    return {
        "answer": result
    }
"""
        
        agent_py_path.write_text(agent_py_content)
        print_file_created(agent_py_path)
        
        # Create upsonic_config.json
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
                    "ruamel.yaml>=0.18.5"
                ],
                "streamlit": [
                    "streamlit==1.32.2",
                    "pandas==2.2.1",
                    "numpy==1.26.4"
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
            },
            "agent_py": "agent.py",
            "streamlit_app_py": "streamlit_app.py"
        }
        
        config_json_path.write_text(json.dumps(config_data, indent=4))
        print_file_created(config_json_path)
        
        # Print success message with created files
        print_init_success(agent_name, [str(agent_py_path), str(config_json_path)])
        return 0
        
    except KeyboardInterrupt:
        print_cancelled()
        return 1
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        return 1


def add_command(library: str, section: str) -> int:
    """
    Add a dependency to upsonic_config.json.
    
    Args:
        library: Library name with version (e.g., "x_library==0.52.0")
        section: Section name in dependencies (e.g., "api", "streamlit", "development")
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Get current directory
        current_dir = Path.cwd()
        config_json_path = current_dir / "upsonic_config.json"
        
        # Check if config file exists
        if not config_json_path.exists():
            print_config_not_found()
            return 1
        
        # Read the config file
        try:
            with open(config_json_path, "r") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in upsonic_config.json: {str(e)}")
            return 1
        
        # Validate dependencies section exists
        if "dependencies" not in config_data:
            print_error("'dependencies' section not found in upsonic_config.json")
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
        with open(config_json_path, "w") as f:
            json.dump(config_data, f, indent=4)
        
        # Print success message
        print_dependency_added(library, section)
        return 0
        
    except KeyboardInterrupt:
        print_cancelled()
        return 1
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        return 1


def _build_endpoint_parameters(
    inputs_schema: List[Dict[str, Any]],
    UploadFile: Any,
    File: Any,
    Form: Any,
    Annotated: Any,
) -> List[inspect.Parameter]:
    """
    Build function parameters with proper annotations for FastAPI endpoint.
    
    Args:
        inputs_schema: List of input schema items
        UploadFile: FastAPI UploadFile type
        File: FastAPI File dependency
        Form: FastAPI Form dependency
        Annotated: typing.Annotated type
        
    Returns:
        List of inspect.Parameter objects
    """
    from inspect import Parameter
    
    parameters = []
    for item in inputs_schema:
        name = item["name"]
        itype = item.get("type", "string")
        required = item.get("required", False)
        default = item.get("default")
        desc = item.get("description", "")
        
        # Determine annotation and default
        if itype in ("file", "binary", "string($binary)"):
            annotation = Annotated[UploadFile, File(description=desc)]
            param_default = Parameter.empty if required else None
        elif itype == "files":
            annotation = Annotated[List[UploadFile], File(description=desc)]
            param_default = Parameter.empty if required else None
        else:
            # Map basic types to Python types
            if itype == "number":
                py_type = float
            elif itype == "integer":
                py_type = int
            elif itype in ("boolean", "bool"):
                py_type = bool
            else:
                py_type = str
            
            annotation = Annotated[py_type, Form(description=desc)]
            if required:
                param_default = Parameter.empty
            elif default is not None:
                param_default = default
            else:
                param_default = None
        
        param = Parameter(
            name=name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=param_default,
            annotation=annotation,
        )
        parameters.append(param)
    
    return parameters


def _process_form_inputs(**kwargs: Any) -> Dict[str, Any]:
    """
    Process form inputs from FastAPI endpoint parameters.
    
    Handles UploadFile objects, lists of UploadFiles, and regular values.
    
    Args:
        **kwargs: Form parameters from FastAPI
        
    Returns:
        Dictionary of processed inputs
    """
    from fastapi.datastructures import UploadFile
    
    inputs = {}
    for key, value in kwargs.items():
        if value is None:
            continue
            
        if isinstance(value, UploadFile):
            # Single file - read content
            try:
                # Note: This will be awaited in the async function
                inputs[key] = value
            except Exception:
                inputs[key] = None
        elif isinstance(value, list) and value and all(
            isinstance(x, UploadFile) for x in value
        ):
            # List of files
            inputs[key] = value
        else:
            # Regular value
            inputs[key] = value
    
    return inputs


async def _read_upload_files(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read UploadFile objects asynchronously.
    
    Args:
        inputs: Dictionary that may contain UploadFile objects
        
    Returns:
        Dictionary with UploadFile objects replaced with their content
    """
    from fastapi.datastructures import UploadFile
    
    processed = {}
    for key, value in inputs.items():
        if isinstance(value, UploadFile):
            try:
                processed[key] = await value.read()
            except Exception:
                processed[key] = None
        elif isinstance(value, list) and value and all(
            isinstance(x, UploadFile) for x in value
        ):
            file_contents = []
            for upload_file in value:
                try:
                    file_contents.append(await upload_file.read())
                except Exception:
                    file_contents.append(None)
            processed[key] = file_contents
        else:
            processed[key] = value
    
    return processed


def _create_endpoint_function(
    parameters: List[inspect.Parameter],
    main_func: Callable,
    JSONResponse: Any,
) -> Callable:
    """
    Create the FastAPI endpoint function dynamically.
    
    Args:
        parameters: List of function parameters
        main_func: The agent's main function to call
        JSONResponse: FastAPI JSONResponse class
        
    Returns:
        The endpoint function
    """
    from types import FunctionType
    
    # Create function signature
    sig = inspect.Signature(parameters)
    
    # Create the function code
    async def endpoint_function(*args, **kwargs):
        # Process form inputs
        inputs = _process_form_inputs(**kwargs)
        
        # Read upload files asynchronously
        inputs = await _read_upload_files(inputs)
        
        # Call main function
        try:
            result = await main_func(inputs)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "type": type(e).__name__}
            )
    
    # Set the signature
    endpoint_function.__signature__ = sig
    
    # Set annotations from parameters
    annotations = {param.name: param.annotation for param in parameters}
    endpoint_function.__annotations__ = annotations
    
    return endpoint_function


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


def _install_dependencies(dependencies: list[str]) -> bool:
    """
    Install dependencies using uv or pip.
    
    Args:
        dependencies: List of dependency strings (e.g., ["fastapi>=0.115.12", "uvicorn>=0.34.2"])
    
    Returns:
        True if successful, False otherwise.
    """
    if not dependencies:
        return True
    
    try:
        print_info(f"Installing {len(dependencies)} dependencies...")
        
        # Try uv first (preferred for this project)
        try:
            result = subprocess.run(
                ["uv", "pip", "install"] + dependencies,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
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
                print_success("Dependencies installed successfully")
                return True
            else:
                print_error(f"Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            print_error(f"Error installing dependencies: {str(e)}")
            return False
            
    except Exception as e:
        print_error(f"Error installing dependencies: {str(e)}")
        return False


def run_command(host: str = "0.0.0.0", port: int = 8000) -> int:
    """
    Run the agent as a FastAPI server.

    Dynamically builds OpenAPI for both multipart/form-data and application/json
    from upsonic_config.json input_schema/output_schema so /docs shows editable form fields.
    """
    try:
        # Get current directory
        current_dir = Path.cwd()
        config_json_path = current_dir / "upsonic_config.json"

        # Check if config file exists
        if not config_json_path.exists():
            print_config_not_found()
            return 1

        # Read config
        try:
            with open(config_json_path, "r") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in upsonic_config.json: {str(e)}")
            return 1

        # Install dependencies
        dependencies = config_data.get("dependencies", {}).get("api", [])
        if dependencies:
            if not _install_dependencies(dependencies):
                print_error("Failed to install required dependencies")
                return 1


        # Import FastAPI pieces
        try:
            from fastapi import FastAPI, File, Form, Request
            from fastapi.responses import JSONResponse
            from fastapi.datastructures import UploadFile
            import uvicorn
        except ImportError:
            print_error("FastAPI dependencies not found. Please install with: pip install upsonic[run]")
            return 1

        # Agent metadata
        agent_name = config_data.get("agent_name", "Upsonic Agent")
        description = config_data.get("description", "An Upsonic AI agent")

        # Load agent.py
        agent_py_file = config_data.get("agent_py", "agent.py")
        agent_py_path = current_dir / agent_py_file
        if not agent_py_path.exists():
            print_error(f"Agent file not found: {agent_py_path}")
            return 1

        spec = importlib.util.spec_from_file_location("agent", agent_py_path)
        if spec is None or spec.loader is None:
            print_error(f"Failed to load agent module from {agent_py_path}")
            return 1

        agent_module = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = agent_module
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

        # Store a flag to track if schema is modified (shared between lifespan and custom_openapi)
        _schema_modified_flag = {"modified": False}
        
        # Create lifespan first (will be used when creating app)
        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI):
            # Modify schema immediately on startup
            try:
                print_info("=== LIFESPAN: Modifying OpenAPI schema on startup ===")
                # Force schema generation (this will call custom_openapi())
                schema = app.openapi()
                
                # Modify schema using helper function
                schema = _modify_openapi_schema(schema, inputs_schema, output_schema_dict, "/call")
                
                # Update the schema and mark as modified
                app.openapi_schema = schema
                _schema_modified_flag["modified"] = True
                
                # Verify
                paths = schema.get("paths", {})
                if "/call" in paths:
                    post_op = paths["/call"].get("post", {})
                    verify_rb = post_op.get("requestBody", {})
                    verify_content = verify_rb.get("content", {})
                    print_info(f"=== LIFESPAN: Content types set: {list(verify_content.keys())} ===")
            except Exception as e:
                print_error(f"=== LIFESPAN: Error updating OpenAPI schema: {e} ===")
                import traceback
                traceback.print_exc()
            
            yield
        
        # Create app with lifespan
        app = FastAPI(title=f"{agent_name} - Upsonic", description=description, version="0.1.0", lifespan=lifespan)

        # Build endpoint parameters using helper function
        from typing import Annotated
        parameters = _build_endpoint_parameters(
            inputs_schema, UploadFile, File, Form, Annotated
        )
        
        # Create endpoint function using helper function
        call_endpoint_form = _create_endpoint_function(parameters, main_func, JSONResponse)
        
        # Debug: Check function signature
        sig = inspect.signature(call_endpoint_form)
        print_info(f"Function signature: {sig}")
        print_info(f"Function annotations: {call_endpoint_form.__annotations__}")
        for param_name, param in sig.parameters.items():
            print_info(f"  Param {param_name}: annotation={param.annotation}, default={param.default}")

        # Register endpoint
        app.post("/call", summary="Call Main", operation_id="call_main_call_post", tags=["jobs"])(
            call_endpoint_form
        )

        # Override openapi() to ALWAYS return our modified schema
        original_openapi = app.openapi
        
        def custom_openapi():
            # If already modified, ALWAYS return the cached version - NEVER regenerate!
            if _schema_modified_flag["modified"] and app.openapi_schema:
                print_info("Returning cached modified schema (already modified)")
                # Double-check it has multipart
                try:
                    paths = app.openapi_schema.get("paths", {})
                    if "/call" in paths:
                        post_op = paths["/call"].get("post", {})
                        rb = post_op.get("requestBody", {})
                        content = rb.get("content", {})
                        if "multipart/form-data" in content:
                            print_info(f"Cached schema verified - has multipart/form-data. Content types: {list(content.keys())}")
                            return app.openapi_schema
                        else:
                            print_error("Cached schema missing multipart/form-data! Regenerating...")
                except Exception as e:
                    print_error(f"Error checking cached schema: {e}")
                    # Fall through to regenerate
            
            # Generate base schema (only if not already modified)
            print_info("Generating base OpenAPI schema in custom_openapi()...")
            schema = original_openapi()
            
            print_info(f"Base schema generated. Paths: {list(schema.get('paths', {}).keys())}")
            
            try:
                # Modify schema using helper function
                schema = _modify_openapi_schema(schema, inputs_schema, output_schema_dict, "/call")
                
                # Store the modified schema and mark as modified
                app.openapi_schema = schema
                _schema_modified_flag["modified"] = True
                print_info("Schema modification complete and cached")
            except Exception as e:
                print_error(f"Error in custom_openapi: {e}")
                import traceback
                traceback.print_exc()
            
            return app.openapi_schema
        
        app.openapi = custom_openapi

        # Startup messages
        print_success(f"Starting {agent_name} server...")
        print_info(f"Server will be available at http://{host}:{port}")
        print_info(f"API documentation: http://{host}:{port}/docs")
        print_info("Press CTRL+C to stop the server")
        print()

        # Run uvicorn - it handles SIGINT/SIGTERM properly
        # Use log_level="info" to reduce noise, but keep important messages
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except KeyboardInterrupt:
            # This should be caught by uvicorn, but just in case
            print()
            print_info("Server stopped by user")
        
        return 0

    except KeyboardInterrupt:
        print()
        print_info("Server stopped by user")
        return 0
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


