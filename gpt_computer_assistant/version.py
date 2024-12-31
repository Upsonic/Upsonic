# Get the current dir then open __init__.py then extract the version from it

# __version__ = '0.27.5'  

import os


current_dir = os.path.dirname(os.path.abspath(__file__))

def get_version():
    with open(os.path.join(current_dir, "__init__.py")) as f:
        for line in f:
            if "__version__" in line:
                the_str = line.split("=")[1].strip().strip("'")
                the_str = the_str.replace("'  # fmt: skip", "")
                return the_str
    return None


