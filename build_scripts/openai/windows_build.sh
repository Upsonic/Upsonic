#!/bin/sh
# Requireed installations
python -m pip install ".[base]"
python -m pip install '.[agentic]'
python -m pip install pyinstaller==6.9.0


# Pyinstaller
pyinstaller --hidden-import="PyQt5" --recursive-copy-metadata gpt_computer_assistant run.py --windowed --add-data="gpt_computer_assistant/utils/media/*":"gpt_computer_assistant/utils/media" --icon="gpt_computer_assistant/utils/media/icon.ico" --name="GPT Computer Assistant" --onefile

