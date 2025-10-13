from __future__ import annotations as _annotations

import asyncio
import functools
import inspect
import re
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from functools import partial
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeGuard, TypeVar, get_args, get_origin, overload

from anyio.to_thread import run_sync
from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import (
    ParamSpec,
    TypeIs,
    is_typeddict,
)
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from upsonic.utils.package.exception import UserError


if TYPE_CHECKING:
    from upsonic.messages import messages as _messages
    from upsonic.tools import ObjectJsonSchema

_P = ParamSpec('_P')
_R = TypeVar('_R')


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    wrapped_func = partial(func, *args, **kwargs)
    return await run_sync(wrapped_func)


def is_model_like(type_: Any) -> bool:
    """Check if type is a pydantic model, dataclass or typedict.

    These generate JSON Schema with {"type": "object"} and are usable as function parameters.
    """
    return (
        isinstance(type_, type)
        and not isinstance(type_, GenericAlias)
        and (
            issubclass(type_, BaseModel)
            or is_dataclass(type_)  # pyright: ignore[reportUnknownArgumentType]
            or is_typeddict(type_)  # pyright: ignore[reportUnknownArgumentType]
            or getattr(type_, '__is_model_like__', False)  # pyright: ignore[reportUnknownArgumentType]
        )
    )


def check_object_json_schema(schema: JsonSchemaValue) -> 'ObjectJsonSchema':

    if schema.get('type') == 'object':
        return schema
    elif schema.get('$ref') is not None:
        maybe_result = schema.get('$defs', {}).get(schema['$ref'][8:])  # This removes the initial "#/$defs/".

        if "'$ref': '#/$defs/" in str(maybe_result):
            return schema
        return maybe_result
    else:
        raise UserError('Schema must be an object')


T = TypeVar('T')


@dataclass
class Some(Generic[T]):
    """Rust's Option::Some equivalent."""

    value: T


Option: TypeAlias = Some[T] | None
"""Rust's Option equivalent: Option[Thing] = Some[Thing] | None."""


class Unset:
    """Singleton representing unset value."""

    pass


UNSET = Unset()


def is_set(t_or_unset: T | Unset) -> TypeGuard[T]:
    return t_or_unset is not UNSET


@asynccontextmanager
async def group_by_temporal(
    aiterable: AsyncIterable[T], soft_max_interval: float | None
) -> AsyncIterator[AsyncIterable[list[T]]]:
    """Group async iterable items into lists based on time interval.

    Debounces the iterator. Returns context manager for cancellation on error.

    Args:
        aiterable: Async iterable to group.
        soft_max_interval: Max interval for grouping. If None, no grouping performed.

    Returns:
        Context manager yielding async iterable of item lists.
    """
    if soft_max_interval is None:

        async def async_iter_groups_noop() -> AsyncIterator[list[T]]:
            async for item in aiterable:
                yield [item]

        yield async_iter_groups_noop()
        return

    # we might wait for the next item more than once, so we store the task to await next time
    task: asyncio.Task[T] | None = None

    async def async_iter_groups() -> AsyncIterator[list[T]]:
        nonlocal task

        assert soft_max_interval is not None and soft_max_interval >= 0, 'soft_max_interval must be a positive number'
        buffer: list[T] = []
        group_start_time = time.monotonic()

        aiterator = aiterable.__aiter__()
        while True:
            if group_start_time is None:
                wait_time = soft_max_interval
            else:
                wait_time = soft_max_interval - (time.monotonic() - group_start_time)

            if task is None:
                task = asyncio.create_task(aiterator.__anext__())

            done, _ = await asyncio.wait((task,), timeout=wait_time)

            if done:
                try:
                    item = done.pop().result()
                except StopAsyncIteration:
                    if buffer:
                        yield buffer
                    task = None
                    break
                else:
                    buffer.append(item)
                    task = None
                    if group_start_time is None:
                        group_start_time = time.monotonic()
            elif buffer:
                yield buffer
                buffer = []
                group_start_time = None

    try:
        yield async_iter_groups()
    finally:
        if task:
            task.cancel('Cancelling due to error in iterator')
            with suppress(asyncio.CancelledError):
                await task


def sync_anext(iterator: Iterator[T]) -> T:
    """Get next item from sync iterator, raising StopAsyncIteration if exhausted.

    Useful for iterating over sync iterator in async context.
    """
    try:
        return next(iterator)
    except StopIteration as e:
        raise StopAsyncIteration() from e


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def guard_tool_call_id(
    t: _messages.ToolCallPart
    | _messages.ToolReturnPart
    | _messages.RetryPromptPart
    | _messages.BuiltinToolCallPart
    | _messages.BuiltinToolReturnPart,
) -> str:
    """Return tool call id or generate new one if None."""
    return t.tool_call_id or generate_tool_call_id()


def generate_tool_call_id() -> str:
    """Generate unique tool call id."""
    return f'upsonic_{uuid.uuid4().hex}'


class PeekableAsyncStream(Generic[T]):
    """Wraps async iterable allowing peek at next item without consuming.

    Buffers one item at a time. Single-pass stream.
    """

    def __init__(self, source: AsyncIterable[T]):
        self._source = source
        self._source_iter: AsyncIterator[T] | None = None
        self._buffer: T | Unset = UNSET
        self._exhausted = False

    async def peek(self) -> T | Unset:
        """Return next item without consuming it.

        Returns UNSET if stream exhausted.
        """
        if self._exhausted:
            return UNSET

        if not isinstance(self._buffer, Unset):
            return self._buffer

        if self._source_iter is None:
            self._source_iter = self._source.__aiter__()

        try:
            self._buffer = await self._source_iter.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            return UNSET

        return self._buffer

    async def is_exhausted(self) -> bool:
        """Return True if stream exhausted, False otherwise."""
        return isinstance(await self.peek(), Unset)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        """Yield buffered item or fetch next from source.

        Raises StopAsyncIteration if stream exhausted.
        """
        if self._exhausted:
            raise StopAsyncIteration

        if not isinstance(self._buffer, Unset):
            item = self._buffer
            self._buffer = UNSET
            return item

        if self._source_iter is None:
            self._source_iter = self._source.__aiter__()

        try:
            return await self._source_iter.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            raise



def dataclasses_no_defaults_repr(self: Any) -> str:
    """Exclude fields with values equal to field default."""
    kv_pairs = (
        f'{f.name}={getattr(self, f.name)!r}' for f in fields(self) if f.repr and getattr(self, f.name) != f.default
    )
    return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'


_datetime_ta = TypeAdapter(datetime)


def number_to_datetime(x: int | float) -> datetime:
    return _datetime_ta.validate_python(x)


AwaitableCallable = Callable[..., Awaitable[T]]


@overload
def is_async_callable(obj: AwaitableCallable[T]) -> TypeIs[AwaitableCallable[T]]: ...


@overload
def is_async_callable(obj: Any) -> TypeIs[AwaitableCallable[Any]]: ...


def is_async_callable(obj: Any) -> Any:
    """Check if callable is async."""
    while isinstance(obj, functools.partial):
        obj = obj.func

    return inspect.iscoroutinefunction(obj) or (callable(obj) and inspect.iscoroutinefunction(obj.__call__))  # type: ignore


def _update_mapped_json_schema_refs(s: dict[str, Any], name_mapping: dict[str, str]) -> None:
    """Update $refs in schema to use new names from name_mapping."""
    if '$ref' in s:
        ref = s['$ref']
        if ref.startswith('#/$defs/'):
            original_name = ref[8:]
            new_name = name_mapping.get(original_name, original_name)
            s['$ref'] = f'#/$defs/{new_name}'

    if 'properties' in s:
        props: dict[str, dict[str, Any]] = s['properties']
        for prop in props.values():
            _update_mapped_json_schema_refs(prop, name_mapping)

    if 'items' in s and isinstance(s['items'], dict):
        items: dict[str, Any] = s['items']
        _update_mapped_json_schema_refs(items, name_mapping)
    if 'prefixItems' in s:
        prefix_items: list[dict[str, Any]] = s['prefixItems']
        for item in prefix_items:
            _update_mapped_json_schema_refs(item, name_mapping)

    for union_type in ['anyOf', 'oneOf']:
        if union_type in s:
            union_items: list[dict[str, Any]] = s[union_type]
            for item in union_items:
                _update_mapped_json_schema_refs(item, name_mapping)


def merge_json_schema_defs(schemas: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Merge $defs from different JSON schemas into single deduplicated $defs.

    Handles name collisions and rewrites $refs to point to new $defs.

    Returns:
        Tuple of rewritten schemas and new $defs dictionary.
    """
    all_defs: dict[str, dict[str, Any]] = {}
    rewritten_schemas: list[dict[str, Any]] = []

    for schema in schemas:
        if '$defs' not in schema:
            rewritten_schemas.append(schema)
            continue

        schema = schema.copy()
        defs = schema.pop('$defs', None)
        schema_name_mapping: dict[str, str] = {}

        for name, def_schema in defs.items():
            if name not in all_defs:
                all_defs[name] = def_schema
                schema_name_mapping[name] = name
            elif def_schema != all_defs[name]:
                new_name = name
                if title := schema.get('title'):
                    new_name = f'{title}_{name}'

                i = 1
                original_new_name = new_name
                new_name = f'{new_name}_{i}'
                while new_name in all_defs:
                    i += 1
                    new_name = f'{original_new_name}_{i}'

                all_defs[new_name] = def_schema
                schema_name_mapping[name] = new_name

        _update_mapped_json_schema_refs(schema, schema_name_mapping)
        rewritten_schemas.append(schema)

    return rewritten_schemas, all_defs


def validate_empty_kwargs(_kwargs: dict[str, Any]) -> None:
    """Validate no unknown kwargs remain after processing.

    Args:
        _kwargs: Dictionary of remaining kwargs after processing.

    Raises:
        UserError: If unknown kwargs remain.
    """
    if _kwargs:
        unknown_kwargs = ', '.join(f'`{k}`' for k in _kwargs.keys())
        raise UserError(f'Unknown keyword arguments: {unknown_kwargs}')


def strip_markdown_fences(text: str) -> str:
    if text.startswith('{'):
        return text

    regex = r'```(?:\w+)?\n(\{.*\})\n```'
    match = re.search(regex, text, re.DOTALL)
    if match:
        return match.group(1)

    return text


def _unwrap_annotated(tp: Any) -> Any:
    origin = get_origin(tp)
    while typing_objects.is_annotated(origin):
        tp = tp.__origin__
        origin = get_origin(tp)
    return tp


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `tp` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    tp = _unwrap_annotated(tp)
    origin = get_origin(tp)
    if is_union_origin(origin):
        return tuple(_unwrap_annotated(arg) for arg in get_args(tp))
    else:
        return ()