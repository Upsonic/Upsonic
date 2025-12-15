from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runnable import Runnable
    from .sequence import RunnableSequence
    from .prompt import ChatPromptTemplate
    from .passthrough import RunnablePassthrough
    from .parallel import RunnableParallel
    from .lambda_runnable import RunnableLambda
    from .branch import RunnableBranch
    from .decorator import chain
    from .output_parser import BaseOutputParser, StrOutputParser, PydanticOutputParser

def _get_uel_classes():
    """Lazy import of UEL classes."""
    from .runnable import Runnable
    from .sequence import RunnableSequence
    from .prompt import ChatPromptTemplate
    from .passthrough import RunnablePassthrough
    from .parallel import RunnableParallel
    from .lambda_runnable import RunnableLambda
    from .branch import RunnableBranch
    from .decorator import chain
    from .output_parser import BaseOutputParser, StrOutputParser, PydanticOutputParser
    
    return {
        'Runnable': Runnable,
        'RunnableSequence': RunnableSequence,
        'ChatPromptTemplate': ChatPromptTemplate,
        'RunnablePassthrough': RunnablePassthrough,
        'RunnableParallel': RunnableParallel,
        'RunnableLambda': RunnableLambda,
        'RunnableBranch': RunnableBranch,
        'chain': chain,
        'BaseOutputParser': BaseOutputParser,
        'StrOutputParser': StrOutputParser,
        'PydanticOutputParser': PydanticOutputParser,
    }

def _get_itemgetter():
    """Lazy import of itemgetter function."""
    import operator
    from .lambda_runnable import RunnableLambda
    
    _original_itemgetter = operator.itemgetter
    
    def itemgetter(*items):
        """Create an itemgetter that supports the pipe operator.
        
        This is a drop-in replacement for operator.itemgetter that works with UEL chains.
        It returns a RunnableLambda, so it can be used directly in chains with the pipe operator.
        
        Example:
            ```python
            from upsonic.uel import itemgetter
            
            chain = itemgetter("key") | (lambda x: f"Value: {x}")
            result = chain.invoke({"key": "test"})  # Returns "Value: test"
            ```
        """
        getter = _original_itemgetter(*items)
        return RunnableLambda(getter)
    
    return itemgetter

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # UEL classes
    uel_classes = _get_uel_classes()
    if name in uel_classes:
        return uel_classes[name]
    
    # Itemgetter function
    if name == "itemgetter":
        return _get_itemgetter()
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    'Runnable',
    'RunnableSequence', 
    'RunnableParallel',
    'RunnableLambda',
    'RunnableBranch',
    'ChatPromptTemplate',
    'RunnablePassthrough',
    'chain',
    'BaseOutputParser',
    'StrOutputParser',
    'PydanticOutputParser',
    'itemgetter',
]
