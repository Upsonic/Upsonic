# Read the contents of setup.py
with open("setup.py", "r") as file:
    setup_content = file.read()

# Replace the project name
setup_content = setup_content.replace(
    """name="gpt_computer_assistant",""", """name="gcadev","""
)

# Write the modified content to gca_setup.py
with open("gca_setup.py", "w") as file:
    file.write(setup_content)
