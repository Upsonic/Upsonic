import subprocess
import sys
from typing import List


def _ensure_pip_available() -> bool:
    """
    Ensure pip is available in the current Python environment.
    Uses ensurepip to install pip if it's not available.
    
    Returns:
        True if pip is available, False otherwise.
    """
    try:
        # Check if pip is available
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return True
        
        # Try to install pip using ensurepip
        result = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            # Verify pip is now available
            verify_result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            return verify_result.returncode == 0
        
        return False
    except Exception:
        return False


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
        
        # Fall back to pip - ensure pip is available first
        if not _ensure_pip_available():
            if not quiet:
                from upsonic.cli.printer import print_error
                print_error("Failed to ensure pip is available. Please install pip manually.")
            return False
        
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
                    error_msg = result.stderr if result.stderr else result.stdout
                    print_error(f"Failed to install dependencies: {error_msg}")
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

