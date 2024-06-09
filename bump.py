"""Module for managing the version updates of a python package."""

import os
import sys
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_version():
    """
    Gets and extracts the version number from the '__init__.py' file of
    a Python package.

    Returns:
        str or None: The version number of the package if found, otherwise None.
    """
    with open('gpt_computer_assistant/__init__.py', 'r') as file:
        for line in file:
            match = re.search(r"__version__ = '(.*)'", line)
            if match:
                return match.group(1)


def increment_version(part, version):
    """
    Simple function that increments the version number based on the given part
    i.e., ('major', 'minor', or 'patch').

    Notes:
        Splits the version string into major, minor, and patch components, then
        increments the specified part by one

    Args:
        part (str): The part of the version number to increment
            ('major', 'minor', or 'patch').
        version (str): The current version number in 'major.minor.patch' format.

    Returns:
        str: String containing new changes made to the version.
    """
    major, minor, patch = map(int, version.split('.'))
    if part == 'major':
        major += 1
        minor = 0
        patch = 0
    elif part == 'minor':
        minor += 1
        patch = 0
    elif part == 'patch':
        patch += 1
    return f'{major}.{minor}.{patch}'


def write_version(version):
    """
    Updates the `__version__` variable in the `__init__.py` file of the
    `gpt_computer_assistant` package.

    Args:
        version (str): The new version number to replace the existing one.
    """
    with open('gpt_computer_assistant/__init__.py', 'r+') as file:
        content = file.read()
        content = re.sub(r"__version__ = '.*'", f"__version__ = '{version}'", content)
        file.seek(0)
        file.write(content)


def update_version(version):
    """
    Updates the version number found in a list of files.

    Args:
        version (str): The new version number to replace the existing one.
    """
    files = ['setup.py']
    for file in files:
        with open(file, 'r+') as f:
            content = f.read()
            content = re.sub(r'    version=".*"', f'    version="{version}"', content)
            f.seek(0)
            f.write(content)


def create_tag(version):
    """
    Uses the `os.system()` to create a `Git tag` for a specified version.

    Args:
        version (str): The version number for the git tag.
    """
    os.system(f"git tag v{version}")


def create_commit(version):
    """
    Uses `os.system()` to add and commit the changed version number
    to the Git repository.

    Args:
        version (str): Version number included in the commit message.
    """
    os.system("git add .")
    os.system(f"git commit -m 'Changed version number with v{version}'")


def push():
    """Pushes changes and tags to the repository."""
    os.system("git push")
    os.system("git push --tag")


def main():
    """The main function for managing version updates."""
    valid_parts = ['major', 'minor', 'patch']
    if len(sys.argv) != 2 or sys.argv[1] not in valid_parts:
        logger.error(f"Usage: python version.py <{'|'.join(valid_parts)}>")
        sys.exit(1)
        
    part = sys.argv[1]
    version = read_version()
    new_version = increment_version(part, version)
    write_version(new_version)
    update_version(new_version)
    create_commit(new_version)
    create_tag(new_version)
    push()


if __name__ == '__main__':
    main()
