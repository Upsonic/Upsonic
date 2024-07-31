#!/bin/sh
# Requireed installations
pip install ".[base]"
pip install '.[agentic]'
brew install create-dmg
pip install pyinstaller==6.9.0


# Pyinstaller
pyinstaller --recursive-copy-metadata gpt_computer_assistant run.py --windowed --add-data="gpt_computer_assistant/utils/media/*":"gpt_computer_assistant/utils/media" --icon="gpt_computer_assistant/utils/media/icon.icns" --name="GPT Computer Assistant"
# Create a DMG
# Create a folder (named dmg) to prepare our DMG in (if it doesn't already exist).
mkdir -p dist/dmg
# Empty the dmg folder.
rm -r dist/dmg/*
# Copy the app bundle to the dmg folder.
cp -r "dist/GPT Computer Assistant.app" dist/dmg
# If the DMG already exists, delete it.
test -f "dist/GPT Computer Assistant.dmg" && rm "dist/GPT Computer Assistant.dmg"
create-dmg \
  --volname "GPT Computer Assistant" \
  --volicon "gpt_computer_assistant/utils/media/icon.icns" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --icon "GPT Computer Assistant.app" 175 120 \
  --hide-extension "GPT Computer Assistant.app" \
  --app-drop-link 425 120 \
  "dist/GPT Computer Assistant.dmg" \
  "dist/dmg/"