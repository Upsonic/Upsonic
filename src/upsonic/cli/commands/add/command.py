import json
from pathlib import Path

from upsonic.cli.commands.shared.config import load_config


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
        config_data = load_config(config_json_path, use_cache=False)
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

