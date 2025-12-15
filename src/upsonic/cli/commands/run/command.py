import importlib.util
import sys
from pathlib import Path

from upsonic.cli.commands.shared.config import load_config
from upsonic.cli.commands.shared.fastapi_imports import get_fastapi_imports
from upsonic.cli.commands.shared.openapi import modify_openapi_schema


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
        config_data = load_config(config_json_path)
        if config_data is None:
            print_error("Invalid JSON in upsonic_configs.json")
            return 1

        # Get FastAPI imports (lazy loaded)
        fastapi_imports = get_fastapi_imports()
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

        # Load agent file from src directory
        # Priority: entrypoints.api_file
        entrypoints = config_data.get("entrypoints", {})
        agent_py_file = entrypoints.get("api_file")
        
        if not agent_py_file:
            print_error("entrypoints.api_file not found in upsonic_configs.json")
            return 1
            
        agent_py_path = current_dir / agent_py_file
        if not agent_py_path.exists():
            print_error(f"Agent file not found: {agent_py_path}")
            return 1


        agent_dir = agent_py_path.parent.absolute()
        project_root = current_dir.absolute()
        
        # Add agent directory to sys.path (handles same-dir and subdirectory imports)
        agent_dir_str = str(agent_dir)
        if agent_dir_str not in sys.path:
            sys.path.insert(0, agent_dir_str)
        
        # Add project root to sys.path (handles project-level imports)
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        # Determine package structure for relative imports
        # Calculate relative path from project root to agent file
        try:
            relative_path = agent_py_path.relative_to(project_root)
            # Remove .py extension and convert to package path
            package_parts = relative_path.parts[:-1]  # Exclude filename
            module_package = '.'.join(package_parts) if package_parts else None
        except ValueError:
            # If agent file is outside project root, use None
            module_package = None

        # Create spec with "main" as the module name (loader requires this to match)
        spec = importlib.util.spec_from_file_location("main", agent_py_path)
        if spec is None or spec.loader is None:
            print_error(f"Failed to load agent module from {agent_py_path}")
            return 1

        agent_module = importlib.util.module_from_spec(spec)
        
        # Set __package__ for relative imports to work correctly
        # This is the key for relative imports (from .module import X)
        if module_package:
            agent_module.__package__ = module_package
        else:
            # If no package structure, set to empty string to allow relative imports
            agent_module.__package__ = ""
        
        # Keep __name__ as "main" to match the spec (loader requirement)
        # The __package__ attribute is sufficient for relative imports to work
        agent_module.__name__ = "main"
        
        # Register the module
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
            schema = modify_openapi_schema(schema, inputs_schema, output_schema_dict, "/call")
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

