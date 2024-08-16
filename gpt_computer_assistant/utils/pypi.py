import subprocess


def install_library(library):
    try:
        result = subprocess.run(
            ["pip", "install", library],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def uninstall_library(library):
    try:
        result = subprocess.run(
            ["pip", "uninstall", "-y", library],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
