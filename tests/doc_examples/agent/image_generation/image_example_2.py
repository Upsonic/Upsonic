from upsonic import Agent, Task
from upsonic.tools.builtin_tools import ImageGenerationTool
from upsonic.utils.image import save_image_to_folder, open_image_file

# Setup
agent = Agent("openai-responses/gpt-4o")

# Generate image
task = Task(
    description="Generate an image of a futuristic city at night.",
    tools=[ImageGenerationTool()]
)
result = agent.do(task)

# Save and open (folder created automatically)
if isinstance(result, bytes):
    saved_path = save_image_to_folder(
        image_data=result,
        folder_path="generated_images",
        filename="city.png",
        is_base64=False
    )
    open_image_file(saved_path)