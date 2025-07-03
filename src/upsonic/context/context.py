from ..visitors.context_string_builder import ContextStringBuilder
from .default_prompt import default_prompt


def context_proceess(context):
    if context is None:
        context = []

    context.append(default_prompt())

    builder = ContextStringBuilder()

    for item in context:
        item.accept(builder)

    return builder.build()

    



