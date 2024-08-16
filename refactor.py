import os


def install_refactor_tool():
    os.system("pip install ruff==0.6.0")


def refactor():
    os.system("ruff check --fix")
    os.system("ruff format")


def create_commit():
    os.system("git add .")
    os.system("git commit -m 'refactor: Scheduled refactoring'")


def push():
    os.system("git push")


if __name__ == "__main__":
    install_refactor_tool()
    refactor()
    create_commit()
    push()
