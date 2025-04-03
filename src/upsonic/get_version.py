import toml
import os
from importlib.metadata import version, PackageNotFoundError

def get_library_version() -> str:
    """
    Get the version of the upsonic library.
    
    Returns:
        The version of the upsonic library as a string.
    """
    # First try: Using importlib.metadata (most reliable for installed packages)
    try:
        return version("upsonic")
    except:
        pass
    
    # Second try: Using pyproject.toml
    try:
        # Try relative to the current file
        pyproject_path = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
        with open(pyproject_path, 'r') as file:
            pyproject_data = toml.load(file)
            return pyproject_data['project']['version']
    except (FileNotFoundError, KeyError):
        # Third try: Look for pyproject.toml in current working directory
        try:
            with open('pyproject.toml', 'r') as file:
                pyproject_data = toml.load(file)
                return pyproject_data['project']['version']
        except (FileNotFoundError, KeyError):
            return "Version information not available."
