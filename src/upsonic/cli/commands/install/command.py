from pathlib import Path
from typing import Optional

from upsonic.cli.commands.shared.config import load_config
from upsonic.cli.commands.shared.dependencies import install_dependencies


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
        config_data = load_config(config_json_path)
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
        if install_dependencies(dependencies_to_install):
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

