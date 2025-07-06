from upsonic.context.schemas import ContextSections
from .default_prompt import default_prompt
from .context_protocol import ContextItem


def context_proceess(context: list[ContextItem] | None):
    if context is None:
        context = []

    context.append(default_prompt())

    sections: dict[ContextSections, list[str]] = {
        section: [] for section in ContextSections
    }

    for item in context:
        section_name = item.get_context_section()
        content = item.to_context_string()

        if section_name in sections:
            sections[section_name].append(content)

    result = "<Context>"
    result += f"<Agents>{''.join(sections[ContextSections.AGENTS])}</Agents>"
    result += f"<Tasks>{''.join(sections[ContextSections.TASKS])}</Tasks>"
    result += f"<Default Prompt>{''.join(sections[ContextSections.DEFAULT_PROMPT])}</Default Prompt>"
    result += f"<Knowledge Base>{''.join(sections[ContextSections.KNOWLEDGE_BASE])}</Knowledge Base>"
    result += "</Context>"

    return result
