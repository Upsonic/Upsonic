from .tasks.tasks import Task
from .direct_llm_call.direct_llm_cal import Direct

from typing import Optional

class Language:
    def __init__(self, language: str, task: Task, llm_model: str):
        self.language = language
        self.task = task
        self.llm_model = llm_model

    async def transform(self):
        language_transformation_task = Task(
            f"User task is completed but we want to change the language of the task to {self.language}. Just return the translated result of task. Dont say or put anything to your return. Make one to one translation.",
            context=[self.task],
            response_format=self.task.response_format,
        )

        direct = Direct(self.llm_model)
        await direct.do_async(language_transformation_task)
        return language_transformation_task.response
