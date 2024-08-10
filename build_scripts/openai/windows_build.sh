#!/bin/sh
# Requireed installations


python -m pip install pyinstaller==6.9.0


# Pyinstaller
pip3 install .



pip3 install crewai==0.30.11

pip3 install langgraph==0.0.51
pip3 install pyqt5==5.15.10
pip3 install scipy==1.13.1
pip3 install pygame==2.5.2
pip3 install soundcard==0.4.3
pip3 install openai==1.30.3
pip3 install langchain-google-genai==1.0.4
pip3 install python-dotenv==1.0.0
pip3 install upsonic==0.28.4
pip3 install pyautogui==0.9.54
pip3 install sounddevice==0.4.6
pip3 install soundfile==0.12.1
pip3 install pydub==0.25.1
pip3 install pyscreeze==0.1.30
pip3 install pyperclip==1.8.2
pip3 install pydantic==2.7.2
pip3 install pillow==10.3.0
pip3 install langchainhub==0.1.18
pip3 install langchain-experimental==0.0.58
pip3 install opentelemetry-sdk==1.24.0
pip3 install opentelemetry-exporter-otlp==1.24.0
pip3 install langchain-groq==0.1.5
pip3 install langchain-openai==0.1.6
pip3 install open-interpreter==0.2.6
pip3 install langchain==0.1.20
pip3 install langchain-community==0.0.38
pip3 install langchain-core==0.1.52

# custom tools
pip3 install pyperclip==1.8.2
pip3 install google==3.0.0
pip3 install duckduckgo-search==5.3.0
pip3 install beautifulsoup4==4.12.3

pip3 install pytesseract==0.3.10
pip3 install pywifi-controls==0.7

pip3 install pynput==1.7.7



pip3 uninstall numpy -y
pip3 install numpy

pyinstaller --recursive-copy-metadata gpt_computer_assistant run.py --onefile --add-data="gpt_computer_assistant/utils/media/*":"gpt_computer_assistant/utils/media" --icon="gpt_computer_assistant/utils/media/icon.ico" --name="GPT_Computer_Assistant"
