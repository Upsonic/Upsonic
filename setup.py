#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup
import platform


# Read the requirements from the requirements.txt file
with open("requirements.txt") as fp:
    install_requires = fp.read().splitlines()

if platform.system() in ["Windows"]:
    install_requires.append("AppOpener==1.7")

elif platform.system() == "Darwin":  # Darwin is the system name for macOS
    install_requires.append(
        "MacAppOpener==0.0.5"
    )  # Replace with actual macOS specific package


setup(
    name="gpt_computer_assistant",
    version="0.22.3",
    description="""GPT""",
    long_description="".join(open("README.md", encoding="utf-8").readlines()),
    long_description_content_type="text/markdown",
    url="https://github.com/onuratakan/gpt-computer-assistant",
    author="Onur Atakan ULUSOY",
    author_email="atadogan06@gmail.com",
    license="MIT",
    packages=[
        "gpt_computer_assistant",
        "gpt_computer_assistant.agent",
        "gpt_computer_assistant.gui",
        "gpt_computer_assistant.screen",
        "gpt_computer_assistant.utils",
        "gpt_computer_assistant.audio",
        "gpt_computer_assistant.audio.tts_providers",
        "gpt_computer_assistant.audio.stt_providers",
    ],
    include_package_data=True,
    install_requires=[],
    entry_points={
        "console_scripts": ["computerassistant=gpt_computer_assistant.start:start"],
    },
    python_requires=">= 3.9",
    zip_safe=False,
    extras_require={
        "base": install_requires,
        "default": install_requires,
        "agentic": ["crewai==0.30.11"],
        "wakeword": ["pvporcupine", "pyaudio"],
        "api": [
            "flask==3.0.3",
        ],
        "local_tts": [
            "tensorflow==2.17.0",
            "datasets[audio]==2.20.0",
            "sentencepiece==0.2.0",
            "torch==2.4.0",
            "transformers==4.43.3",
        ],
        "local_stt": [
            "openai-whisper==20231117",
        ],
    },
)
