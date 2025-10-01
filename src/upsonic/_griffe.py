from __future__ import annotations as _annotations

import logging
import re
from collections.abc import Callable
from contextlib import contextmanager
from inspect import Signature
from typing import TYPE_CHECKING, Any, Literal, cast

from griffe import Docstring, DocstringSectionKind, Object as GriffeObject

if TYPE_CHECKING:
    from upsonic.tools import DocstringFormat

DocstringStyle = Literal['google', 'numpy', 'sphinx']


def doc_descriptions(
    func: Callable[..., Any],
    sig: Signature,
    *,
    docstring_format: DocstringFormat,
) -> tuple[str | None, dict[str, str]]:
    """Extract function description and parameter descriptions from docstring.

    Returns:
        Tuple of main description and parameter descriptions dict.
    """
    doc = func.__doc__
    if doc is None:
        return None, {}

    parent = cast(GriffeObject, sig)

    docstring_style = _infer_docstring_style(doc) if docstring_format == 'auto' else docstring_format
    docstring = Docstring(
        doc,
        lineno=1,
        parser=docstring_style,
        parent=parent,
        parser_options={'returns_named_value': False, 'returns_multiple_items': False},
    )
    with _disable_griffe_logging():
        sections = docstring.parse()

    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ''
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    if return_ := next((p for p in sections if p.kind == DocstringSectionKind.returns), None):
        return_statement = return_.value[0]
        return_desc = return_statement.description
        return_type = return_statement.annotation
        type_tag = f'<type>{return_type}</type>\n' if return_type else ''
        return_xml = f'<returns>\n{type_tag}<description>{return_desc}</description>\n</returns>'

        if main_desc:
            main_desc = f'<summary>{main_desc}</summary>\n{return_xml}'
        else:
            main_desc = return_xml

    return main_desc, params


def _infer_docstring_style(doc: str) -> DocstringStyle:
    """Infer docstring style from content."""
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (
            re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE) for replacement in replacements
        )
        if any(matches):
            return style
    return 'google'


_docstring_style_patterns: list[tuple[str, list[str], DocstringStyle]] = [
    (
        r'\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n',
        [
            'param',
            'parameter',
            'arg',
            'argument',
            'key',
            'keyword',
            'type',
            'var',
            'ivar',
            'cvar',
            'vartype',
            'returns',
            'return',
            'rtype',
            'raises',
            'raise',
            'except',
            'exception',
        ],
        'sphinx',
    ),
    (
        r'\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+',
        [
            'args',
            'arguments',
            'params',
            'parameters',
            'keyword args',
            'keyword arguments',
            'other args',
            'other arguments',
            'other params',
            'other parameters',
            'raises',
            'exceptions',
            'returns',
            'yields',
            'receives',
            'examples',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
            'warns',
            'warnings',
        ],
        'google',
    ),
    (
        r'\n[ \t]*{0}\n[ \t]*---+\n',
        [
            'deprecated',
            'parameters',
            'other parameters',
            'returns',
            'yields',
            'receives',
            'raises',
            'warns',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
        ],
        'numpy',
    ),
]


@contextmanager
def _disable_griffe_logging():
    old_level = logging.root.getEffectiveLevel()
    logging.root.setLevel(logging.ERROR)
    yield
    logging.root.setLevel(old_level)