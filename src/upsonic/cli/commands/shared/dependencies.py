import subprocess
import sys
from typing import List


def install_dependencies(dependencies: list[str], quiet: bool = False) -> bool:
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

