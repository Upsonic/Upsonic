import warnings
import importlib
from typing import Any

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

__version__ = "0.1.0"

_lazy_imports = {}

def _lazy_import(module_name: str, class_name: str = None):
    """Lazy import function to defer heavy imports until actually needed."""
    def _import():
        if module_name not in _lazy_imports:
            _lazy_imports[module_name] = importlib.import_module(module_name)
        
        if class_name:
            return getattr(_lazy_imports[module_name], class_name)
        return _lazy_imports[module_name]
    
    return _import

def _get_Task():
    return _lazy_import("upsonic.tasks.tasks", "Task")()

def _get_KnowledgeBase():
    return _lazy_import("upsonic.knowledge_base.knowledge_base", "KnowledgeBase")()

def _get_Agent():
    return _lazy_import("upsonic.agent.agent", "Agent")()

def _get_AgentRunResult():
    return _lazy_import("upsonic.agent.run_result", "AgentRunResult")()

def _get_OutputDataT():
    return _lazy_import("upsonic.agent.run_result", "OutputDataT")()

def _get_Graph():
    return _lazy_import("upsonic.graph.graph", "Graph")()

def _get_DecisionFunc():
    return _lazy_import("upsonic.graph.graph", "DecisionFunc")()

def _get_DecisionLLM():
    return _lazy_import("upsonic.graph.graph", "DecisionLLM")()

def _get_TaskNode():
    return _lazy_import("upsonic.graph.graph", "TaskNode")()

def _get_TaskChain():
    return _lazy_import("upsonic.graph.graph", "TaskChain")()

def _get_State():
    return _lazy_import("upsonic.graph.graph", "State")()

def _get_Canvas():
    return _lazy_import("upsonic.canvas.canvas", "Canvas")()

def _get_Team():
    return _lazy_import("upsonic.team.team", "Team")()

def _get_tool():
    return _lazy_import("upsonic.tools", "tool")()

def _get_safety_engine_components():
    """Lazy import of safety engine components to avoid circular imports."""
    try:
        from upsonic.safety_engine import (
            Policy, RuleBase, ActionBase, PolicyInput, RuleOutput, PolicyOutput,
            RuleInput, ActionResult, DisallowedOperation, AdultContentBlockPolicy,
            AnonymizePhoneNumbersPolicy, CryptoBlockPolicy, CryptoRaiseExceptionPolicy,
            SensitiveSocialBlockPolicy, SensitiveSocialRaiseExceptionPolicy,
            AdultContentBlockPolicy_LLM, AdultContentBlockPolicy_LLM_Finder,
            AdultContentRaiseExceptionPolicy, AdultContentRaiseExceptionPolicy_LLM,
            SensitiveSocialBlockPolicy_LLM, SensitiveSocialBlockPolicy_LLM_Finder,
            SensitiveSocialRaiseExceptionPolicy_LLM, AnonymizePhoneNumbersPolicy_LLM_Finder
        )
        
        return {
            "Policy": Policy,
            "RuleBase": RuleBase,
            "ActionBase": ActionBase,
            "PolicyInput": PolicyInput,
            "RuleOutput": RuleOutput,
            "PolicyOutput": PolicyOutput,
            "RuleInput": RuleInput,
            "ActionResult": ActionResult,
            "DisallowedOperation": DisallowedOperation,
            "AdultContentBlockPolicy": AdultContentBlockPolicy,
            "AnonymizePhoneNumbersPolicy": AnonymizePhoneNumbersPolicy,
            "CryptoBlockPolicy": CryptoBlockPolicy,
            "CryptoRaiseExceptionPolicy": CryptoRaiseExceptionPolicy,
            "SensitiveSocialBlockPolicy": SensitiveSocialBlockPolicy,
            "SensitiveSocialRaiseExceptionPolicy": SensitiveSocialRaiseExceptionPolicy,
            "AdultContentBlockPolicy_LLM": AdultContentBlockPolicy_LLM,
            "AdultContentBlockPolicy_LLM_Finder": AdultContentBlockPolicy_LLM_Finder,
            "AdultContentRaiseExceptionPolicy": AdultContentRaiseExceptionPolicy,
            "AdultContentRaiseExceptionPolicy_LLM": AdultContentRaiseExceptionPolicy_LLM,
            "SensitiveSocialBlockPolicy_LLM": SensitiveSocialBlockPolicy_LLM,
            "SensitiveSocialBlockPolicy_LLM_Finder": SensitiveSocialBlockPolicy_LLM_Finder,
            "SensitiveSocialRaiseExceptionPolicy_LLM": SensitiveSocialRaiseExceptionPolicy_LLM,
            "AnonymizePhoneNumbersPolicy_LLM_Finder": AnonymizePhoneNumbersPolicy_LLM_Finder,
        }
    except ImportError:
        return {}

def _get_exception_classes():
    """Lazy import of exception classes."""
    from upsonic.utils.package.exception import (
        UupsonicError, 
        AgentExecutionError, 
        ModelConnectionError, 
        TaskProcessingError, 
        ConfigurationError, 
        RetryExhaustedError,
        NoAPIKeyException
    )
    
    return {
        'UupsonicError': UupsonicError,
        'AgentExecutionError': AgentExecutionError,
        'ModelConnectionError': ModelConnectionError,
        'TaskProcessingError': TaskProcessingError,
        'ConfigurationError': ConfigurationError,
        'RetryExhaustedError': RetryExhaustedError,
        'NoAPIKeyException': NoAPIKeyException,
    }

def hello() -> str:
    return "Hello from upsonic!"

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    
    if name == "Task":
        return _get_Task()
    elif name == "KnowledgeBase":
        return _get_KnowledgeBase()
    elif name == "Agent":
        return _get_Agent()
    elif name == "AgentRunResult":
        return _get_AgentRunResult()
    elif name == "OutputDataT":
        return _get_OutputDataT()
    elif name == "Graph":
        return _get_Graph()
    elif name == "DecisionFunc":
        return _get_DecisionFunc()
    elif name == "DecisionLLM":
        return _get_DecisionLLM()
    elif name == "TaskNode":
        return _get_TaskNode()
    elif name == "TaskChain":
        return _get_TaskChain()
    elif name == "State":
        return _get_State()
    elif name == "Canvas":
        return _get_Canvas()
    elif name == "Team":
        return _get_Team()
    elif name == "tool":
        return _get_tool()
    
    safety_components = _get_safety_engine_components()
    if name in safety_components:
        return safety_components[name]
    
    exception_classes = _get_exception_classes()
    if name in exception_classes:
        return exception_classes[name]
    
    if name == "MultiAgent":
        return _get_Team()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "hello", 
    "Task", 
    "KnowledgeBase", 
    "Agent",
    "AgentRunResult",
    "OutputDataT",
    "Graph",
    "DecisionFunc",
    "DecisionLLM",
    "TaskNode",
    "TaskChain",
    "State",
    "Canvas",
    "MultiAgent",
    "Team",
    "UupsonicError",
    "AgentExecutionError", 
    "ModelConnectionError", 
    "TaskProcessingError", 
    "ConfigurationError", 
    "RetryExhaustedError",
    "NoAPIKeyException",
    "Policy",
    "RuleBase",
    "ActionBase",
    "PolicyInput", 
    "RuleOutput",
    "PolicyOutput",
    "RuleInput",
    "ActionResult",
    "DisallowedOperation",
    "AdultContentBlockPolicy",
    "AnonymizePhoneNumbersPolicy",
    "CryptoBlockPolicy",
    "CryptoRaiseExceptionPolicy",
    "SensitiveSocialBlockPolicy",
    "SensitiveSocialRaiseExceptionPolicy",
    "AdultContentBlockPolicy_LLM",
    "AdultContentBlockPolicy_LLM_Finder",
    "AdultContentRaiseExceptionPolicy",
    "AdultContentRaiseExceptionPolicy_LLM",
    "SensitiveSocialBlockPolicy_LLM",
    "SensitiveSocialBlockPolicy_LLM_Finder",
    "SensitiveSocialRaiseExceptionPolicy_LLM",
    "AnonymizePhoneNumbersPolicy_LLM_Finder",
    "tool",
]