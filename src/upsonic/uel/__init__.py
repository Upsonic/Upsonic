from upsonic.uel.runnable import Runnable
from upsonic.uel.sequence import RunnableSequence
from upsonic.uel.prompt import ChatPromptTemplate
from upsonic.uel.passthrough import RunnablePassthrough
from upsonic.uel.parallel import RunnableParallel
from upsonic.uel.lambda_runnable import RunnableLambda
from upsonic.uel.branch import RunnableBranch
from upsonic.uel.decorator import chain
from upsonic.uel.output_parser import BaseOutputParser, StrOutputParser, PydanticOutputParser

# Provide itemgetter that works with pipe operator
import operator
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
