import toml
import os

def get_library_version() -> str:
    """
    Get the version of the upsonic library from pyproject.toml.

    Returns:
        The version of the upsonic library as a string.
    """
    try:
        # Assuming the pyproject.toml is located at the root of the project
        pyproject_path = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
        with open(pyproject_path, 'r') as file:
            pyproject_data = toml.load(file)
            return pyproject_data['project']['version']
    except (FileNotFoundError, KeyError):
        return "Version information not available."
