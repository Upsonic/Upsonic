from dataclasses import Field
import uuid
from pydantic import BaseModel


from typing import Any, List, Dict, Optional, Type, Union



class KnowledgeBaseMarkdown(BaseModel):
    knowledges: Dict[str, str]


class KnowledgeBase(BaseModel):
    sources: List[str] = []

    def add_file(self, file_path: str):
        self.sources.append(file_path)

    def remove_file(self, file_path: str):
        self.sources.remove(file_path)


    def markdown(self, client):
        knowledge_base = KnowledgeBaseMarkdown(knowledges={})
        the_list_of_files = self.sources
        

        for each in the_list_of_files:
            markdown_content = client.markdown(each)

            knowledge_base.knowledges[each] = markdown_content



        return knowledge_base