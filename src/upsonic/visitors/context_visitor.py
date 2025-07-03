from abc import ABC, abstractmethod

class ContextVisitor(ABC):
    @abstractmethod
    def visit_task(self, task): pass

    @abstractmethod
    def visit_direct(self, agent): pass

    @abstractmethod
    def visit_defaultprompt(self, prompt): pass

    @abstractmethod
    def visit_knowledgebase(self, kb): pass