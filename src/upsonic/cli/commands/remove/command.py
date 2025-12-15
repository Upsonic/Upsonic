import json
import re
from pathlib import Path

from upsonic.cli.commands.shared.config import load_config


def get_package_name(dependency_str: str) -> str:
    """Extract package name from dependency string."""
    # Split by common version specifiers and extras
    # Matches: ==, >=, <=, >, <, ~=, !=, [, ;
    delimiters = ["==", ">=", "<=", ">", "<", "~=", "!=", "[", ";"]
    pattern = "|".join(map(re.escape, delimiters))
    return re.split(pattern, dependency_str)[0].strip().lower()


def remove_command(library: str, section: str) -> int:
    """
    Remove a dependency from upsonic_configs.json.
    
    Args:
        library: Library name (e.g., "fastapi", "upsonic[storage]")
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
            print_dependency_removed,
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
        
        section_deps = dependencies[section]
        target_lib_lower = library.lower()
        target_pkg_name = get_package_name(library)
        
        # Find the dependency to remove
        # Priority 1: Exact string match (case-insensitive)
        # Priority 2: Package name match
        
        to_remove = None
        
        # Check for exact match first
        for dep in section_deps:
            if dep.lower() == target_lib_lower:
                to_remove = dep
                break
        
        # If not found, check for package name match
        if to_remove is None:
            for dep in section_deps:
                if get_package_name(dep) == target_pkg_name:
                    to_remove = dep
                    break
        
        if to_remove is None:
            print_error(f"Dependency '{library}' not found in dependencies.{section}")
            return 1
        
        # Remove the dependency
        dependencies[section].remove(to_remove)
        
        # Write back to file
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        
        # Print success message
        print_dependency_removed(to_remove, section)
        return 0
        
    except KeyboardInterrupt:
        from upsonic.cli.printer import print_cancelled
        print_cancelled()
        return 1
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        return 1
