from upsonic import Agent, Task
from upsonic.tools.builtin_tools import ImageGenerationTool
from upsonic.utils.image import save_image_to_folder

# Create agent with image generation capability
agent = Agent(
    model="openai-responses/gpt-4o",
)

# Create task with ImageGenerationTool
task = Task(
    description="Generate a beautiful landscape image of a sunset over mountains.",
    tools=[ImageGenerationTool()]
)

# Execute task - result will be image bytes
result = agent.do(task)

# Save the generated image (folder created automatically)
saved_path = save_image_to_folder(
    image_data=result,
    folder_path="my_images",
    filename="sunset.png",
    is_base64=False  # Agent returns bytes directly
)

print(f"Image saved to: {saved_path}")