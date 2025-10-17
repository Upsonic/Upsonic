import re
from typing import Optional, Any

from upsonic.models import infer_model, Model


class Canvas:
    def __init__(self, canvas_name: str, model: Optional[Model | str] = None):
        """Initializes the Canvas."""
        self.canvas_name = canvas_name
        
        if model is None:
            self.model = infer_model("openai/gpt-4o-mini")
        else:
            self.model = infer_model(model) if isinstance(model, str) else model

        self._clean_canvas()
    
    def _clean_canvas(self):
        """Clean the canvas text"""
        canvas_text = self.get_current_state_of_canvas()
        canvas_text = re.sub(r'```.*?```', '', canvas_text, flags=re.DOTALL)
        self._save_canvas(canvas_text)

    def _save_canvas(self, canvas_text: str):
        """Save the canvas text to a file."""
        normalized_name = re.sub(r'[^\w\s-]', '', self.canvas_name).strip().replace(' ', '_')
        filename = f"{normalized_name}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(canvas_text)

    def _load_canvas(self) -> str:
        """Load the canvas text from a file."""
        normalized_name = re.sub(r'[^\w\s-]', '', self.canvas_name).strip().replace(' ', '_')
        filename = f"{normalized_name}.txt"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def get_current_state_of_canvas(self) -> str:
        """Get the current state of the text canvas"""
        result = self._load_canvas()
        return "Empty Canvas" if result == "" else result

    async def change_in_canvas(self, new_text_of_part: str, part_definition: str) -> str:
        """Change the text of a part of the canvas"""
        from upsonic import Task
        from upsonic.agent.agent import Agent
        
        agent = Agent(model=self.model, name="Canvas Editor")
        
        current_canvas = self.get_current_state_of_canvas()
        
        if current_canvas == "Empty Canvas" or current_canvas == "":
            self._save_canvas(new_text_of_part)
            return new_text_of_part

        prompt = (
            f"I have a text document with the following content:\n\n{current_canvas}\n\n"
            f"If there is a line or section that contains '{part_definition}', replace it with exactly:\n"
            f"{new_text_of_part}\n\n"
            f"If the document does NOT contain a section with '{part_definition}', append the following as a new section at the end of the document:\n"
            f"{new_text_of_part}\n\n"
            f"Return only the complete updated text document without any explanations, code blocks, or additional formatting."
        )
        
        task = Task(description=prompt)
        await agent.do_async(task)
        result = task.response
        
        self._save_canvas(result)
        return result


    def functions(self):
        return [self.get_current_state_of_canvas, self.change_in_canvas]