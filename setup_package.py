from cx_Freeze import setup, Executable

import sys
sys.setrecursionlimit(10000)

setup(
    name="GPT_Computer_Assistant",
    version="0.1",
    description="",
    executables=[Executable("run.py")])
