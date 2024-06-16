#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup

# Read the requirements from the requirements.txt file
with open("requirements.txt") as fp:
    install_requires = fp.read().splitlines()

setup(
    name="gpt_computer_assistant",
    version="0.17.0",
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
        "api": ["flask==3.0.3",],
    },
)
