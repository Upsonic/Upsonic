#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()
setup(
    name="gpt_computer_assistant",
    version="0.0.0",
    description="""GPT""",
    long_description="".join(open("README.md", encoding="utf-8").readlines()),
    long_description_content_type="text/markdown",
    url="https://github.com/onuratakan/gpt-computer-assistant",
    author="Onur Atakan ULUSOY",
    author_email="atadogan06@gmail.com",
    license="MIT",
    packages=["gpt_assistant", "gpt_assistant.agent", "gpt_assistant.gui", "gpt_assistant.screen", "gpt_assistant.utils", "gpt_assistant.audio"],
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["computerassistant=gpt_assistant.gpt_assistant:start"],
    },      
    python_requires=">= 3",
    zip_safe=False,
)


