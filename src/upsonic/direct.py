from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional, Union

from upsonic.models.settings import ModelSettings
from upsonic.tasks.tasks import Task
from upsonic.profiles import ModelProfileSpec
from upsonic.providers import Provider


class Direct:
    """Simplified, high-speed interface for LLM interactions.

    Direct executes a task as a *bare* LLM call — no memory, knowledge base,
    tools, reflection, reliability, or policies. It is a thin facade over a
    minimally-configured ``Agent`` running the reduced ``"direct"`` pipeline
    profile, and delegates execution, output extraction, usage/cost/timing
    metrics, and printing entirely to that pipeline.

    Example:
        ```python
        from upsonic import Direct, Task
        from pydantic import BaseModel

        my_direct = Direct(model="openai/gpt-4o")

        class MyResponse(BaseModel):
            tax_number: str

        my_task = Task(
            "Read the paper and return me the tax number",
            context=["my.pdf", "my.png"],
            response_format=MyResponse
        )

        result = my_direct.do(my_task)
        print(result)
        ```
    """

    def __init__(
        self,
        model: Union[str, Any, None] = None,
        *,
        settings: Optional[ModelSettings] = None,
        profile: Optional[ModelProfileSpec] = None,
        provider: Optional[Union[str, Provider]] = None,
        print: Optional[bool] = None
    ):
        """Initialize the Direct instance.

        Args:
            model: Model name (e.g., "openai/gpt-4o"), Model instance, or None.
            settings: Optional model settings.
            profile: Optional model profile.
            provider: Optional provider name or Provider instance.
            print: Print behaviour, forwarded to the internal Agent (identical
                semantics to ``Agent``'s ``print``): if None, ``do()`` does not
                print unless ``UPSONIC_AGENT_PRINT=true``, while ``print_do()``
                does; if set, it overrides the method default. ``UPSONIC_AGENT_PRINT``
                overrides everything.
        """
        self._model: Any = None
        self._settings = settings
        self._profile = profile
        self._provider = provider
        # Forwarded to the internal Agent, which owns all print resolution
        # (constructor ``print`` / ``UPSONIC_AGENT_PRINT`` / method default).
        self._print = print
        # Internal Agent that runs the reduced pipeline. Built lazily and cached;
        # the per-instance config is fixed (with_* returns new instances), so
        # reuse across do() calls is safe.
        self._internal_agent: Optional[Any] = None

        if model is not None:
            self._set_model(model)

    def _set_model(self, model: Union[str, Any]) -> None:
        """Set the model for this Direct instance."""
        if isinstance(model, str):
            from upsonic.models import infer_model
            self._model = infer_model(model)
        elif hasattr(model, 'request'):  # Check if it's a Model-like object
            self._model = model
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
        # A new model invalidates any cached internal agent.
        self._internal_agent = None

    def with_model(self, model: Union[str, Any]) -> "Direct":
        """Create a new Direct instance with the specified model.

        Args:
            model: Model name or Model instance

        Returns:
            New Direct instance with the specified model
        """
        new_direct = Direct(
            settings=self._settings,
            profile=self._profile,
            provider=self._provider,
            print=self._print
        )
        new_direct._set_model(model)
        return new_direct

    def with_settings(self, settings: ModelSettings) -> "Direct":
        """Create a new Direct instance with the specified settings.

        Args:
            settings: Model settings

        Returns:
            New Direct instance with the specified settings
        """
        new_direct = Direct(
            model=self._model,
            settings=settings,
            profile=self._profile,
            provider=self._provider,
            print=self._print
        )
        return new_direct

    def with_profile(self, profile: ModelProfileSpec) -> "Direct":
        """Create a new Direct instance with the specified profile.

        Args:
            profile: Model profile

        Returns:
            New Direct instance with the specified profile
        """
        new_direct = Direct(
            model=self._model,
            settings=self._settings,
            profile=profile,
            provider=self._provider,
            print=self._print
        )
        return new_direct

    def with_provider(self, provider: Union[str, Provider]) -> "Direct":
        """Create a new Direct instance with the specified provider.

        Args:
            provider: Provider name or Provider instance

        Returns:
            New Direct instance with the specified provider
        """
        new_direct = Direct(
            model=self._model,
            settings=self._settings,
            profile=self._profile,
            provider=provider,
            print=self._print
        )
        return new_direct

    def _resolve_model(self) -> Any:
        """Resolve and cache the concrete model, applying settings and profile.

        Creates the default model when none was supplied and applies any
        configured settings/profile onto the model object.

        Returns:
            The resolved model instance, also stored on ``self._model``.
        """
        if self._model is None:
            from upsonic.models import infer_model
            self._model = infer_model("openai/gpt-4o")

        if self._settings is not None:
            self._model._settings = self._settings

        if self._profile is not None:
            self._model._profile = self._profile

        return self._model

    def _build_internal_agent(self) -> Any:
        """Build and cache the internal Agent that runs the reduced pipeline.

        The Agent is configured for a bare LLM call: no memory, knowledge base,
        tools, reflection, reliability, or policies. The ``print`` setting is
        forwarded so the Agent owns all print resolution.

        Returns:
            The cached internal Agent, configured with the ``"direct"`` profile.
        """
        if self._internal_agent is None:
            from upsonic.agent.agent import Agent

            model = self._resolve_model()
            agent = Agent(
                model=model,
                memory=None,
                tools=[],
                reflection=False,
                print=self._print,
            )
            agent._pipeline_profile = "direct"
            self._internal_agent = agent
        return self._internal_agent

    def do(self, task: Task) -> Any:
        """Execute a task synchronously (does not print by default).

        Print behaviour is identical to ``Agent.do``: nothing is printed unless
        the constructor ``print=True`` or ``UPSONIC_AGENT_PRINT=true``.

        Args:
            task: Task object containing description, context, and response format.

        Returns:
            The model's response (extracted output).
        """
        return self._build_internal_agent().do(task)

    async def do_async(
        self,
        task: Task,
        state: Optional[Any] = None,
        *,
        graph_execution_id: Optional[str] = None
    ) -> Any:
        """Execute a task asynchronously by delegating to the internal Agent.

        The Agent pipeline owns execution, output extraction, metrics, and
        printing. Printing follows ``Agent.do_async`` semantics (does not print
        by default). ``TaskOutputSource`` items in ``task.context`` are resolved
        inside the pipeline from ``state``.

        Args:
            task: Task object containing description, context, and response format.
            state: Optional Graph execution state (resolves ``TaskOutputSource``).
            graph_execution_id: Optional graph execution ID (Graph compatibility).

        Returns:
            The model's response (extracted output).
        """
        return await self._build_internal_agent().do_async(
            task,
            state=state,
            graph_execution_id=graph_execution_id,
            _print_method_default=False,
        )

    def print_do(self, task: Task) -> Any:
        """Execute a task synchronously and print the result.

        Print behaviour is identical to ``Agent.print_do``: the result is printed
        unless ``print=False`` or ``UPSONIC_AGENT_PRINT=false`` overrides it.

        Args:
            task: Task object containing description, context, and response format.

        Returns:
            The model's response (extracted output).
        """
        return self._build_internal_agent().print_do(task)

    async def print_do_async(
        self,
        task: Task,
        state: Optional[Any] = None,
        *,
        graph_execution_id: Optional[str] = None
    ) -> Any:
        """Execute a task asynchronously and print the result.

        Print behaviour is identical to ``Agent.print_do_async``: the result is
        printed unless overridden by ``print=False`` / ``UPSONIC_AGENT_PRINT``.

        Args:
            task: Task object containing description, context, and response format.
            state: Optional Graph execution state (resolves ``TaskOutputSource``).
            graph_execution_id: Optional graph execution ID (Graph compatibility).

        Returns:
            The model's response (extracted output).
        """
        return await self._build_internal_agent().do_async(
            task,
            state=state,
            graph_execution_id=graph_execution_id,
            _print_method_default=True,
        )

    def stream(
        self,
        task: Task,
        *,
        events: bool = False,
        state: Optional[Any] = None,
    ) -> Iterator[Union[str, Any]]:
        """Stream a task synchronously, yielding output as it arrives.

        Delegates to ``Agent.stream`` on the internal Agent, which runs the
        reduced ``"direct"`` streaming pipeline — a bare streaming LLM call with
        no memory, knowledge base, tools, reflection, reliability, or policies.
        For async streaming use ``astream``.

        Args:
            task: Task object containing description, context, and response format.
            events: If ``True`` yield ``AgentStreamEvent`` objects; if ``False``
                (default) yield text chunks (``str``).
            state: Optional Graph execution state (resolves ``TaskOutputSource``).

        Returns:
            An iterator over text chunks, or stream events when ``events=True``.
        """
        return self._build_internal_agent().stream(task, events=events, state=state)

    def astream(
        self,
        task: Task,
        *,
        events: bool = False,
        state: Optional[Any] = None,
    ) -> AsyncIterator[Union[str, Any]]:
        """Stream a task asynchronously, yielding output as it arrives.

        Delegates to ``Agent.astream`` on the internal Agent, which runs the
        reduced ``"direct"`` streaming pipeline — a bare streaming LLM call with
        no memory, knowledge base, tools, reflection, reliability, or policies.

        Print behaviour follows ``Agent`` streaming: the start/metrics panels are
        shown only when the constructor ``print=True`` or
        ``UPSONIC_AGENT_PRINT=true``.

        Args:
            task: Task object containing description, context, and response format.
            events: If ``True`` yield ``AgentStreamEvent`` objects; if ``False``
                (default) yield text chunks (``str``).
            state: Optional Graph execution state (resolves ``TaskOutputSource``).

        Returns:
            An async iterator over text chunks, or stream events when
            ``events=True``.
        """
        return self._build_internal_agent().astream(task, events=events, state=state)

    @property
    def model(self) -> Optional[Any]:
        """Get the current model."""
        return self._model

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model's name, or ``None`` if no model is set yet."""
        return getattr(self._model, "model_name", None)

    @property
    def settings(self) -> Optional[ModelSettings]:
        """Get the current settings."""
        return self._settings

    @property
    def profile(self) -> Optional[ModelProfileSpec]:
        """Get the current profile."""
        return self._profile

    @property
    def provider(self) -> Optional[Union[str, Provider]]:
        """Get the current provider."""
        return self._provider

    @property
    def usage(self) -> Any:
        """Aggregated token / cost / timing usage for this Direct instance.

        Delegates to the internal Agent's :pyattr:`Agent.usage` — the single
        source of truth — so it is shape-compatible with ``Agent.usage``
        (input_tokens, output_tokens, requests, cost, duration, ...) and
        accumulates across every ``do()`` / ``do_async()`` call.

        Returns:
            An ``AggregatedUsage`` view of this instance's usage.
        """
        return self._build_internal_agent().usage
