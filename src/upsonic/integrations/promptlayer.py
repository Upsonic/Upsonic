"""PromptLayer integration for Upsonic prompt management and observability.

Provides prompt registry, versioning, request tracking, scoring, and metadata
through PromptLayer's REST API.  When passed to ``Agent(promptlayer=...)``,
every agent execution is automatically logged to PromptLayer with full model
parameters, token counts, and cost.

Usage::

    from upsonic import Agent
    from upsonic.integrations.promptlayer import PromptLayer

    pl = PromptLayer(api_key="pl_...")

    agent = Agent(
        "openai/gpt-4o",
        system_prompt=pl.get_prompt("my-agent-v2"),
        promptlayer=pl,
    )
    result = agent.do("What is the capital of Japan?")

    pl.shutdown()
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from upsonic.utils.logging_config import get_logger

if TYPE_CHECKING:
    import httpx as _httpx
    import threading as _threading

logger = get_logger(__name__)


class PromptLayer:
    """PromptLayer integration for prompt management and observability.

    Connects to PromptLayer's REST API for prompt registry (versioned prompt
    templates), request tracking, scoring, and metadata tagging.

    Args:
        api_key: PromptLayer API key (``pl_...``).
            Falls back to ``PROMPTLAYER_API_KEY`` env var.
        base_url: PromptLayer API base URL.
            Falls back to ``PROMPTLAYER_BASE_URL`` env var,
            then defaults to ``https://api.promptlayer.com``.

    Raises:
        ValueError: If api_key cannot be resolved.
    """

    _DEFAULT_BASE_URL: str = "https://api.promptlayer.com"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key: str = api_key or os.getenv("PROMPTLAYER_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "PromptLayer api_key is required. "
                "Pass it as an argument or set PROMPTLAYER_API_KEY env var."
            )
        resolved_url: str = base_url or os.getenv(
            "PROMPTLAYER_BASE_URL", self._DEFAULT_BASE_URL
        )
        self._base_url: str = resolved_url.rstrip("/")
        self._client: Optional[_httpx.Client] = None
        self._async_client: Optional[_httpx.AsyncClient] = None
        self._last_prompt_name: Optional[str] = None
        self._last_prompt_id: Optional[int] = None
        self._last_prompt_version: Optional[int] = None
        self._created_workflows: Dict[str, int] = {}  # name -> workflow_id

        from threading import Lock, Thread  # noqa: F811
        self._pending_threads: list[Thread] = []
        self._threads_lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stringify_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        import json as _json

        out: Dict[str, str] = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, (dict, list, tuple)):
                out[str(k)] = _json.dumps(v, default=str)
            else:
                out[str(k)] = str(v)
        return out

    @staticmethod
    def _parse_provider_model(name: str) -> Tuple[str, str]:
        """Extract ``(provider, model)`` from ``provider/model`` format.

        Handles ``"openai/gpt-4o"``, ``"accuracy_eval:anthropic/claude-sonnet-4-6"``,
        and plain names like ``"reliability_eval"``.
        """
        cleaned: str = name
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[1].strip()
        if "/" in cleaned:
            parts: List[str] = cleaned.split("/", 1)
            return parts[0], parts[1]
        return "custom", name

    @staticmethod
    def _epoch_to_iso(epoch: float) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_client(self) -> "_httpx.Client":
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
        return self._client

    def _get_async_client(self) -> "_httpx.AsyncClient":
        if self._async_client is None:
            import httpx
            self._async_client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        return self._async_client

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().get(path, params=params, headers=headers)
        if response.status_code >= 400:
            logger.warning("GET %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _aget(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().get(path, params=params, headers=headers)
        if response.status_code >= 400:
            logger.warning("GET %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    def _post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().post(path, json=body, headers=headers)
        if response.status_code >= 400:
            logger.warning("POST %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _apost(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().post(path, json=body, headers=headers)
        if response.status_code >= 400:
            logger.warning("POST %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    def _patch(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().patch(path, json=body, headers=headers)
        if response.status_code >= 400:
            logger.warning("PATCH %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _apatch(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().patch(path, json=body, headers=headers)
        if response.status_code >= 400:
            logger.warning("PATCH %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().delete(path, params=params, headers=headers)
        if response.status_code >= 400:
            logger.warning("DELETE %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _adelete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().delete(path, params=params, headers=headers)
        if response.status_code >= 400:
            logger.warning("DELETE %s returned %s: %s", path, response.status_code, response.text)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    def _log_post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().post("/log-request", json=body, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _alog_post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().post("/log-request", json=body, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    # ------------------------------------------------------------------
    # Prompt registry
    # ------------------------------------------------------------------

    def get_prompt(
        self,
        prompt_name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Fetch a prompt template from PromptLayer by name.

        Args:
            prompt_name: Name of the prompt in PromptLayer registry.
            version: Specific version number. ``None`` for latest.
            label: Label to fetch (e.g. ``"production"``, ``"staging"``).
            variables: Template variables to fill in the prompt template.
            return_metadata: If ``True``, returns ``(prompt_text, metadata)`` tuple.

        Returns:
            The rendered prompt string, or ``(prompt_string, metadata)``
            when *return_metadata* is ``True``.
        """
        body: Dict[str, Any] = {}
        if version is not None:
            body["version"] = version
        if label is not None:
            body["label"] = label
        if variables is not None:
            body["input_variables"] = variables

        result: Dict[str, Any] = self._post(
            f"/prompt-templates/{prompt_name}", body or None
        )
        prompt_text: str = self._extract_prompt_text(result)

        self._last_prompt_name = prompt_name
        self._last_prompt_id = result.get("id")
        self._last_prompt_version = result.get("version")

        if return_metadata:
            metadata: Dict[str, Any] = {
                "id": result.get("id"),
                "version": result.get("version"),
                "label": result.get("label"),
            }
            return prompt_text, metadata
        return prompt_text

    async def aget_prompt(
        self,
        prompt_name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Async variant of :meth:`get_prompt`."""
        body: Dict[str, Any] = {}
        if version is not None:
            body["version"] = version
        if label is not None:
            body["label"] = label
        if variables is not None:
            body["input_variables"] = variables

        result: Dict[str, Any] = await self._apost(
            f"/prompt-templates/{prompt_name}", body or None
        )
        prompt_text: str = self._extract_prompt_text(result)

        self._last_prompt_name = prompt_name
        self._last_prompt_id = result.get("id")
        self._last_prompt_version = result.get("version")

        if return_metadata:
            metadata: Dict[str, Any] = {
                "id": result.get("id"),
                "version": result.get("version"),
                "label": result.get("label"),
            }
            return prompt_text, metadata
        return prompt_text

    @staticmethod
    def _extract_prompt_text(result: Dict[str, Any]) -> str:
        prompt_template: Dict[str, Any] = result.get("prompt_template", {})

        messages: Optional[List[Dict[str, Any]]] = prompt_template.get("messages")
        if messages and isinstance(messages, list):
            parts: List[str] = []
            for msg in messages:
                content: Any = msg.get("content", "")
                if isinstance(content, list):
                    text_parts: List[str] = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)
                parts.append(str(content))
            return "\n\n".join(parts)

        content_list: Optional[List[Dict[str, Any]]] = prompt_template.get("content")
        if content_list and isinstance(content_list, list):
            text_parts: List[str] = [
                p.get("text", "")
                for p in content_list
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            if text_parts:
                return "\n".join(text_parts)

        template: Optional[str] = prompt_template.get("template")
        if template:
            return str(template)

        return str(prompt_template)

    # ------------------------------------------------------------------
    # Unified log / alog
    # ------------------------------------------------------------------

    def _build_log_body(
        self,
        *,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        start_time: Optional[float],
        end_time: Optional[float],
        input_tokens: int,
        output_tokens: int,
        price: float,
        parameters: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        score: Optional[int],
        status: str,
        function_name: Optional[str],
        prompt_name: Optional[str],
        prompt_id: Optional[int],
        prompt_version: Optional[int],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now: float = time.time()

        # ── Input messages ────────────────────────────────────────────
        input_messages: List[Dict[str, Any]] = []
        if system_prompt:
            input_messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        input_messages.append(
            {"role": "user", "content": [{"type": "text", "text": input_text}]}
        )

        input_body: Dict[str, Any] = {"type": "chat", "messages": input_messages}
        if tools:
            input_body["tools"] = tools

        # ── Output messages ───────────────────────────────────────────
        output_messages: List[Dict[str, Any]] = []

        # If there were tool calls, build: assistant (with tool_calls) → tool results → final assistant
        if tool_calls and tool_results:
            # Assistant message requesting tool calls (no final text yet)
            tc_assistant: Dict[str, Any] = {
                "role": "assistant",
                "content": [],
                "tool_calls": tool_calls,
            }
            output_messages.append(tc_assistant)

            # Tool result messages
            for tr in tool_results:
                output_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_call_id", ""),
                    "name": tr.get("name", ""),
                    "content": [{"type": "text", "text": tr.get("content", "")}],
                })

            # Final assistant message with the output text
            output_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": output_text}],
            })
        elif tool_calls:
            # Tool calls but no results recorded
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": [{"type": "text", "text": output_text}],
                "tool_calls": tool_calls,
            }
            output_messages.append(assistant_message)
        else:
            # Simple response, no tools
            output_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": output_text}],
            })

        body: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "input": input_body,
            "output": {"type": "chat", "messages": output_messages},
            "request_start_time": self._epoch_to_iso(start_time or now),
            "request_end_time": self._epoch_to_iso(end_time or now),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "price": price,
            "status": status,
        }

        if function_name is not None:
            body["function_name"] = function_name
        if parameters:
            body["parameters"] = parameters
        if tags:
            body["tags"] = tags
        if metadata:
            body["metadata"] = self._stringify_metadata(metadata)
        if score is not None:
            body["score"] = max(0, min(100, int(round(score))))
        if prompt_name is not None:
            body["prompt_name"] = prompt_name
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if prompt_version is not None:
            body["prompt_version_number"] = prompt_version

        return body

    def log(
        self,
        *,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[int] = None,
        status: str = "SUCCESS",
        function_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_id: Optional[int] = None,
        prompt_version: Optional[int] = None,
        scores: Optional[Dict[str, int]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Log a request to PromptLayer via ``/log-request``.

        This is the single entry point for all PromptLayer logging -- agent
        runs, accuracy evals, reliability evals, performance evals.  Callers
        construct the appropriate arguments for their use case.

        Args:
            provider: LLM provider (e.g. ``"openai"``, ``"anthropic"``).
            model: Model name (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
            input_text: The input prompt or query.
            output_text: The model's output or response.
            start_time: Request start time (epoch seconds).
            end_time: Request end time (epoch seconds).
            input_tokens: Number of input/prompt tokens used.
            output_tokens: Number of output/completion tokens used.
            price: Cost of the request in USD.
            parameters: Model parameters (temperature, max_tokens, etc.).
            tags: Tags for organizing requests.
            metadata: Metadata dictionary (values are stringified).
            score: Primary score (0--100, clamped).
            status: Request status (``SUCCESS``, ``WARNING``, ``ERROR``).
            function_name: Function name for dashboard display.
            prompt_name: PromptLayer prompt template name.
            prompt_id: PromptLayer prompt template ID.
            prompt_version: Prompt template version number.
            scores: Named scores dict (``{name: value}``) to attach via
                ``/rest/track-score`` after the initial log.
            system_prompt: System prompt text (added as a system message).
            tools: Tool definitions available to the model.
            tool_calls: Tool calls made by the model during the run.
            tool_results: Tool execution results (``tool_call_id``, ``name``, ``content``).

        Returns:
            The PromptLayer ``request_id``.
        """
        try:
            body: Dict[str, Any] = self._build_log_body(
                provider=provider,
                model=model,
                input_text=input_text,
                output_text=output_text,
                start_time=start_time,
                end_time=end_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price=price,
                parameters=parameters,
                tags=tags,
                metadata=metadata,
                score=score,
                status=status,
                function_name=function_name,
                prompt_name=prompt_name,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                system_prompt=system_prompt,
                tools=tools,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            result: Dict[str, Any] = self._log_post(body)
            request_id: int = result.get("id", 0)

            if scores and request_id:
                for score_name, score_value in scores.items():
                    self.score(request_id, score_value, name=score_name)

            return request_id
        except Exception as e:
            logger.warning("Error in log: %s", e)
            return 0

    async def alog(
        self,
        *,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[int] = None,
        status: str = "SUCCESS",
        function_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_id: Optional[int] = None,
        prompt_version: Optional[int] = None,
        scores: Optional[Dict[str, int]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Async variant of :meth:`log`."""
        try:
            body: Dict[str, Any] = self._build_log_body(
                provider=provider,
                model=model,
                input_text=input_text,
                output_text=output_text,
                start_time=start_time,
                end_time=end_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price=price,
                parameters=parameters,
                tags=tags,
                metadata=metadata,
                score=score,
                status=status,
                function_name=function_name,
                prompt_name=prompt_name,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                system_prompt=system_prompt,
                tools=tools,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            result: Dict[str, Any] = await self._alog_post(body)
            request_id: int = result.get("id", 0)

            if scores and request_id:
                for score_name, score_value in scores.items():
                    await self.ascore(request_id, score_value, name=score_name)

            return request_id
        except Exception as e:
            logger.warning("Error in alog: %s", e)
            return 0

    # ------------------------------------------------------------------
    # Post-hoc score / metadata
    # ------------------------------------------------------------------

    def score(
        self,
        request_id: int,
        score: Union[int, float],
        *,
        name: str = "quality",
    ) -> bool:
        """Score a previously logged request.

        PromptLayer requires integer scores in the range 0--100.
        Float values are rounded automatically and clamped to [0, 100].

        Args:
            request_id: The PromptLayer request ID to score.
            score: Numerical score value (rounded and clamped to 0--100).
            name: Name of the score metric (e.g. ``"accuracy"``, ``"quality"``).

        Returns:
            ``True`` if scoring was successful.
        """
        try:
            clamped: int = max(0, min(100, int(round(score))))
            body: Dict[str, Any] = {
                "request_id": request_id,
                "score": clamped,
                "score_name": name,
            }
            result: Dict[str, Any] = self._post("/rest/track-score", body)
            return bool(result.get("success", False))
        except Exception as e:
            logger.warning("Error in score: %s", e)
            return False

    async def ascore(
        self,
        request_id: int,
        score: Union[int, float],
        *,
        name: str = "quality",
    ) -> bool:
        """Async variant of :meth:`score`."""
        try:
            clamped: int = max(0, min(100, int(round(score))))
            body: Dict[str, Any] = {
                "request_id": request_id,
                "score": clamped,
                "score_name": name,
            }
            result: Dict[str, Any] = await self._apost("/rest/track-score", body)
            return bool(result.get("success", False))
        except Exception as e:
            logger.warning("Error in ascore: %s", e)
            return False

    def add_metadata(
        self,
        request_id: int,
        metadata: Dict[str, Any],
    ) -> bool:
        """Add metadata to a previously logged request.

        Args:
            request_id: The PromptLayer request ID.
            metadata: Metadata dictionary to attach (values are stringified).

        Returns:
            ``True`` if metadata was added successfully.
        """
        try:
            body: Dict[str, Any] = {
                "request_id": request_id,
                "metadata": self._stringify_metadata(metadata),
            }
            result: Dict[str, Any] = self._post("/rest/track-metadata", body)
            return bool(result.get("success", False))
        except Exception as e:
            logger.warning("Error in add_metadata: %s", e)
            return False

    async def aadd_metadata(
        self,
        request_id: int,
        metadata: Dict[str, Any],
    ) -> bool:
        """Async variant of :meth:`add_metadata`."""
        try:
            body: Dict[str, Any] = {
                "request_id": request_id,
                "metadata": self._stringify_metadata(metadata),
            }
            result: Dict[str, Any] = await self._apost("/rest/track-metadata", body)
            return bool(result.get("success", False))
        except Exception as e:
            logger.warning("Error in aadd_metadata: %s", e)
            return False



    def list_workflows(
        self,
        *,
        page: int = 1,
        per_page: int = 30,
    ) -> Dict[str, Any]:
        """List all workflows (agents) in the workspace.

        Args:
            page: Pagination page number (min 1).
            per_page: Items per page (1--100, default 30).

        Returns:
            Paginated response with ``items``, ``page``, ``per_page``,
            ``total``, ``pages``, ``has_next``, ``has_prev``, ``next_num``,
            ``prev_num``.
        """
        try:
            params: Dict[str, Any] = {"page": page, "per_page": per_page}
            return self._get("/workflows", params=params)
        except Exception as e:
            logger.warning("Error in list_workflows: %s", e)
            return {}

    async def alist_workflows(
        self,
        *,
        page: int = 1,
        per_page: int = 30,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`list_workflows`."""
        try:
            params: Dict[str, Any] = {"page": page, "per_page": per_page}
            return await self._aget("/workflows", params=params)
        except Exception as e:
            logger.warning("Error in alist_workflows: %s", e)
            return {}

    def create_workflow(
        self,
        *,
        nodes: List[Dict[str, Any]],
        name: Optional[str] = None,
        workflow_id: Optional[int] = None,
        workflow_name: Optional[str] = None,
        folder_id: Optional[int] = None,
        commit_message: Optional[str] = None,
        required_input_variables: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        release_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new workflow (agent) or a new version of an existing one.

        Args:
            nodes: List of workflow node objects. Each node requires ``name``,
                ``node_type``, ``configuration``, and ``is_output_node``.
                At least one node must have ``is_output_node=True``.
            name: Name for a new workflow (1--255 chars).
                Cannot be used with *workflow_id* or *workflow_name*.
            workflow_id: ID of an existing workflow to create a new version of.
            workflow_name: Name of an existing workflow to create a new version of.
            folder_id: Folder placement ID.
            commit_message: Description of the version change.
            required_input_variables: Variable name-to-type mappings.
            edges: Conditional node connections. Each edge requires
                ``source_node_name``, ``target_node_name``, ``is_and``,
                and ``conditionals``.
            release_labels: Version labels (e.g. ``["production"]``).

        Returns:
            Response with ``success``, ``workflow_id``, ``workflow_name``,
            ``workflow_version_id``, ``version_number``, ``base_version``,
            ``release_labels``, ``nodes``, ``required_input_variables``.
        """
        try:
            body: Dict[str, Any] = {"nodes": nodes}
            if name is not None:
                body["name"] = name
            if workflow_id is not None:
                body["workflow_id"] = workflow_id
            if workflow_name is not None:
                body["workflow_name"] = workflow_name
            if folder_id is not None:
                body["folder_id"] = folder_id
            if commit_message is not None:
                body["commit_message"] = commit_message
            if required_input_variables is not None:
                body["required_input_variables"] = required_input_variables
            if edges is not None:
                body["edges"] = edges
            if release_labels is not None:
                body["release_labels"] = release_labels
            return self._post("/rest/workflows", body)
        except Exception as e:
            logger.warning("Error in create_workflow: %s", e)
            return {}

    async def acreate_workflow(
        self,
        *,
        nodes: List[Dict[str, Any]],
        name: Optional[str] = None,
        workflow_id: Optional[int] = None,
        workflow_name: Optional[str] = None,
        folder_id: Optional[int] = None,
        commit_message: Optional[str] = None,
        required_input_variables: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        release_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`create_workflow`."""
        try:
            body: Dict[str, Any] = {"nodes": nodes}
            if name is not None:
                body["name"] = name
            if workflow_id is not None:
                body["workflow_id"] = workflow_id
            if workflow_name is not None:
                body["workflow_name"] = workflow_name
            if folder_id is not None:
                body["folder_id"] = folder_id
            if commit_message is not None:
                body["commit_message"] = commit_message
            if required_input_variables is not None:
                body["required_input_variables"] = required_input_variables
            if edges is not None:
                body["edges"] = edges
            if release_labels is not None:
                body["release_labels"] = release_labels
            return await self._apost("/rest/workflows", body)
        except Exception as e:
            logger.warning("Error in acreate_workflow: %s", e)
            return {}

    def patch_workflow(
        self,
        workflow_id_or_name: Union[int, str],
        *,
        base_version: Optional[int] = None,
        commit_message: Optional[str] = None,
        nodes: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        required_input_variables: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        release_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Patch an existing workflow (agent) to create a new version.

        Args:
            workflow_id_or_name: Workflow identifier (integer ID or string name).
            base_version: Version number to base changes on (defaults to latest).
            commit_message: Description of changes.
            nodes: Node updates keyed by name. Set a node to ``None`` to remove it.
                Provide new node dicts (with ``node_type``, ``configuration``) to add.
            required_input_variables: Replaces input variables entirely if provided.
            edges: Replaces edges entirely if provided.
            release_labels: Labels for the new version.

        Returns:
            Response with ``success``, ``workflow_id``, ``workflow_name``,
            ``workflow_version_id``, ``version_number``, ``base_version``,
            ``release_labels``, ``nodes``, ``required_input_variables``.
        """
        try:
            body: Dict[str, Any] = {}
            if base_version is not None:
                body["base_version"] = base_version
            if commit_message is not None:
                body["commit_message"] = commit_message
            if nodes is not None:
                body["nodes"] = nodes
            if required_input_variables is not None:
                body["required_input_variables"] = required_input_variables
            if edges is not None:
                body["edges"] = edges
            if release_labels is not None:
                body["release_labels"] = release_labels
            return self._patch(f"/rest/workflows/{workflow_id_or_name}", body)
        except Exception as e:
            logger.warning("Error in patch_workflow: %s", e)
            return {}

    async def apatch_workflow(
        self,
        workflow_id_or_name: Union[int, str],
        *,
        base_version: Optional[int] = None,
        commit_message: Optional[str] = None,
        nodes: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        required_input_variables: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        release_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`patch_workflow`."""
        try:
            body: Dict[str, Any] = {}
            if base_version is not None:
                body["base_version"] = base_version
            if commit_message is not None:
                body["commit_message"] = commit_message
            if nodes is not None:
                body["nodes"] = nodes
            if required_input_variables is not None:
                body["required_input_variables"] = required_input_variables
            if edges is not None:
                body["edges"] = edges
            if release_labels is not None:
                body["release_labels"] = release_labels
            return await self._apatch(f"/rest/workflows/{workflow_id_or_name}", body)
        except Exception as e:
            logger.warning("Error in apatch_workflow: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def create_dataset_group(
        self,
        name: str,
        *,
        workspace_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new dataset group.

        An initial draft dataset (version_number=-1) is created automatically.

        Args:
            name: Unique dataset group name within the workspace.
            workspace_id: Optional workspace ID (defaults to API key's workspace).

        Returns:
            ``{"success": bool, "dataset_group": {...}, "dataset": {...}}``
        """
        body: Dict[str, Any] = {"name": name}
        if workspace_id is not None:
            body["workspace_id"] = workspace_id
        return self._post("/api/public/v2/dataset-groups", body)

    async def acreate_dataset_group(
        self,
        name: str,
        *,
        workspace_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`create_dataset_group`."""
        body: Dict[str, Any] = {"name": name}
        if workspace_id is not None:
            body["workspace_id"] = workspace_id
        return await self._apost("/api/public/v2/dataset-groups", body)

    def list_datasets(
        self,
        *,
        dataset_group_id: Optional[int] = None,
        name: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """List datasets.

        Args:
            dataset_group_id: Filter by dataset group.
            name: Case-insensitive partial match on group name.
            status: ``"active"``, ``"deleted"``, or ``"all"`` (default ``"active"``).
            page: Page number.
            per_page: Results per page (max 100).

        Returns:
            ``{"datasets": [...], "page": int, "per_page": int, "total": int, "pages": int}``
        """
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if dataset_group_id is not None:
            params["dataset_group_id"] = dataset_group_id
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        return self._get("/api/public/v2/datasets", params)

    async def alist_datasets(
        self,
        *,
        dataset_group_id: Optional[int] = None,
        name: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`list_datasets`."""
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if dataset_group_id is not None:
            params["dataset_group_id"] = dataset_group_id
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        return await self._aget("/api/public/v2/datasets", params)

    def create_dataset_version_from_file(
        self,
        dataset_group_id: int,
        file_name: str,
        file_content_base64: str,
    ) -> Dict[str, Any]:
        """Create a dataset version by uploading a CSV or JSON file.

        Args:
            dataset_group_id: The dataset group to add a version to.
            file_name: File name ending with ``.csv`` or ``.json``.
            file_content_base64: Base64-encoded file content (max 100MB).

        Returns:
            ``{"success": bool, "dataset_id": int}``
        """
        body: Dict[str, Any] = {
            "dataset_group_id": dataset_group_id,
            "file_name": file_name,
            "file_content_base64": file_content_base64,
        }
        return self._post("/api/public/v2/dataset-versions/from-file", body)

    async def acreate_dataset_version_from_file(
        self,
        dataset_group_id: int,
        file_name: str,
        file_content_base64: str,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`create_dataset_version_from_file`."""
        body: Dict[str, Any] = {
            "dataset_group_id": dataset_group_id,
            "file_name": file_name,
            "file_content_base64": file_content_base64,
        }
        return await self._apost("/api/public/v2/dataset-versions/from-file", body)

    def create_dataset_version_from_filter(
        self,
        dataset_group_id: int,
        *,
        variables_to_parse: Optional[List[str]] = None,
        prompt_id: Optional[int] = None,
        prompt_version_id: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a dataset version from filter parameters (historical requests).

        Args:
            dataset_group_id: The dataset group to add a version to.
            variables_to_parse: Variables to extract from request logs.
            prompt_id: Filter by prompt template ID.
            prompt_version_id: Filter by prompt version ID.
            start_time: ISO datetime start filter.
            end_time: ISO datetime end filter.
            tags: Filter by tags.
            metadata: Filter by metadata key-value pairs.
            scores: Filter by score ranges.

        Returns:
            ``{"success": bool, "dataset_id": int, "dataset_group_id": int, "version_number": int}``
        """
        body: Dict[str, Any] = {"dataset_group_id": dataset_group_id}
        if variables_to_parse is not None:
            body["variables_to_parse"] = variables_to_parse
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if prompt_version_id is not None:
            body["prompt_version_id"] = prompt_version_id
        if start_time is not None:
            body["start_time"] = start_time
        if end_time is not None:
            body["end_time"] = end_time
        if tags is not None:
            body["tags"] = tags
        if metadata is not None:
            body["metadata"] = metadata
        if scores is not None:
            body["scores"] = scores
        return self._post("/api/public/v2/dataset-versions/from-filter-params", body)

    async def acreate_dataset_version_from_filter(
        self,
        dataset_group_id: int,
        *,
        variables_to_parse: Optional[List[str]] = None,
        prompt_id: Optional[int] = None,
        prompt_version_id: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`create_dataset_version_from_filter`."""
        body: Dict[str, Any] = {"dataset_group_id": dataset_group_id}
        if variables_to_parse is not None:
            body["variables_to_parse"] = variables_to_parse
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if prompt_version_id is not None:
            body["prompt_version_id"] = prompt_version_id
        if start_time is not None:
            body["start_time"] = start_time
        if end_time is not None:
            body["end_time"] = end_time
        if tags is not None:
            body["tags"] = tags
        if metadata is not None:
            body["metadata"] = metadata
        if scores is not None:
            body["scores"] = scores
        return await self._apost("/api/public/v2/dataset-versions/from-filter-params", body)

    # ------------------------------------------------------------------
    # Reports / Evaluations
    # ------------------------------------------------------------------

    def create_report(
        self,
        dataset_group_id: int,
        *,
        name: Optional[str] = None,
        folder_id: Optional[int] = None,
        dataset_version_number: Optional[int] = None,
        columns: Optional[List[Dict[str, Any]]] = None,
        score_configuration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new evaluation report (pipeline).

        Args:
            dataset_group_id: ID of the dataset group to evaluate.
            name: Optional report name (1-255 chars).
            folder_id: Optional folder ID.
            dataset_version_number: Optional dataset version.
            columns: Optional list of column definitions.
            score_configuration: Optional custom scoring config.

        Returns:
            ``{"success": bool, "report_id": int, "report_columns": [...]}``
        """
        body: Dict[str, Any] = {"dataset_group_id": dataset_group_id}
        if name is not None:
            body["name"] = name
        if folder_id is not None:
            body["folder_id"] = folder_id
        if dataset_version_number is not None:
            body["dataset_version_number"] = dataset_version_number
        if columns is not None:
            body["columns"] = columns
        if score_configuration is not None:
            body["score_configuration"] = score_configuration
        return self._post("/reports", body)

    async def acreate_report(
        self,
        dataset_group_id: int,
        *,
        name: Optional[str] = None,
        folder_id: Optional[int] = None,
        dataset_version_number: Optional[int] = None,
        columns: Optional[List[Dict[str, Any]]] = None,
        score_configuration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`create_report`."""
        body: Dict[str, Any] = {"dataset_group_id": dataset_group_id}
        if name is not None:
            body["name"] = name
        if folder_id is not None:
            body["folder_id"] = folder_id
        if dataset_version_number is not None:
            body["dataset_version_number"] = dataset_version_number
        if columns is not None:
            body["columns"] = columns
        if score_configuration is not None:
            body["score_configuration"] = score_configuration
        return await self._apost("/reports", body)

    def get_report(self, report_id: int) -> Dict[str, Any]:
        """Get a report by ID.

        Returns:
            Report object with ``report``, ``status``, ``stats`` fields.
        """
        return self._get(f"/reports/{report_id}")

    async def aget_report(self, report_id: int) -> Dict[str, Any]:
        """Async variant of :meth:`get_report`."""
        return await self._aget(f"/reports/{report_id}")

    def get_report_score(self, report_id: int) -> Dict[str, Any]:
        """Get the score for a report.

        Returns:
            ``{"success": bool, "score": {"overall_score": ..., "score_type": ..., ...}}``
        """
        return self._get(f"/reports/{report_id}/score")

    async def aget_report_score(self, report_id: int) -> Dict[str, Any]:
        """Async variant of :meth:`get_report_score`."""
        return await self._aget(f"/reports/{report_id}/score")

    def add_report_column(
        self,
        report_id: int,
        column_type: str,
        name: str,
        configuration: Dict[str, Any],
        *,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add a column to a report.

        Args:
            report_id: The report ID.
            column_type: Column type (e.g. ``"CODE_EXECUTION"``, ``"LLM_ASSERTION"``).
            name: Column name (unique per report, 1-255 chars).
            configuration: Column configuration object.
            position: Optional column position.

        Returns:
            ``{"success": true, "report_column": {...}}``
        """
        body: Dict[str, Any] = {
            "report_id": report_id,
            "column_type": column_type,
            "name": name,
            "configuration": configuration,
        }
        if position is not None:
            body["position"] = position
        return self._post("/report-columns", body)

    async def aadd_report_column(
        self,
        report_id: int,
        column_type: str,
        name: str,
        configuration: Dict[str, Any],
        *,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`add_report_column`."""
        body: Dict[str, Any] = {
            "report_id": report_id,
            "column_type": column_type,
            "name": name,
            "configuration": configuration,
        }
        if position is not None:
            body["position"] = position
        return await self._apost("/report-columns", body)

    def update_report_score_card(
        self,
        report_id: int,
        column_names: List[str],
        *,
        code: Optional[str] = None,
        code_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure the score card for a report.

        Args:
            report_id: The report ID.
            column_names: List of column names to include in scoring.
            code: Optional custom scoring code (Python or JavaScript).
            code_language: ``"PYTHON"`` or ``"JAVASCRIPT"`` (default ``"PYTHON"``).

        Returns:
            Report object with updated ``score_configuration``.
        """
        body: Dict[str, Any] = {"column_names": column_names}
        if code is not None:
            body["code"] = code
        if code_language is not None:
            body["code_language"] = code_language
        return self._post(f"/reports/{report_id}/score-card", body)

    async def aupdate_report_score_card(
        self,
        report_id: int,
        column_names: List[str],
        *,
        code: Optional[str] = None,
        code_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`update_report_score_card`."""
        body: Dict[str, Any] = {"column_names": column_names}
        if code is not None:
            body["code"] = code
        if code_language is not None:
            body["code_language"] = code_language
        return await self._apost(f"/reports/{report_id}/score-card", body)

    def run_report(
        self,
        report_id: int,
        name: str,
        *,
        dataset_id: Optional[int] = None,
        refresh_dataset: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run a report (execute the evaluation pipeline).

        Args:
            report_id: The report/blueprint ID to run.
            name: Name for the resulting report (1-255 chars).
            dataset_id: Optional dataset ID override.
            refresh_dataset: Whether to refresh a dynamic dataset.

        Returns:
            ``{"success": bool, "report_id": int}``
        """
        body: Dict[str, Any] = {"name": name}
        if dataset_id is not None:
            body["dataset_id"] = dataset_id
        if refresh_dataset is not None:
            body["refresh_dataset"] = refresh_dataset
        return self._post(f"/reports/{report_id}/run", body)

    async def arun_report(
        self,
        report_id: int,
        name: str,
        *,
        dataset_id: Optional[int] = None,
        refresh_dataset: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`run_report`."""
        body: Dict[str, Any] = {"name": name}
        if dataset_id is not None:
            body["dataset_id"] = dataset_id
        if refresh_dataset is not None:
            body["refresh_dataset"] = refresh_dataset
        return await self._apost(f"/reports/{report_id}/run", body)

    def delete_report_by_name(self, report_name: str) -> Dict[str, Any]:
        """Archive all reports with the given name.

        Args:
            report_name: Name of the report(s) to archive.

        Returns:
            ``{"success": bool, "message": str}``
        """
        return self._delete(f"/reports/name/{report_name}")

    async def adelete_report_by_name(self, report_name: str) -> Dict[str, Any]:
        """Async variant of :meth:`delete_report_by_name`."""
        return await self._adelete(f"/reports/name/{report_name}")

    def list_evaluations(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """List evaluations in the workspace.

        Args:
            name: Optional name filter (case-insensitive partial match).
            status: ``"active"``, ``"deleted"``, or ``"all"`` (default ``"active"``).
            page: Page number (default 1).
            per_page: Results per page (default 10, max 100).

        Returns:
            ``{"evaluations": [...], "page": int, "per_page": int, "total": int, "pages": int}``
        """
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        return self._get("/api/public/v2/evaluations", params)

    async def alist_evaluations(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`list_evaluations`."""
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        return await self._aget("/api/public/v2/evaluations", params)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _register_thread(self, thread: "_threading.Thread") -> None:
        with self._threads_lock:
            self._pending_threads = [t for t in self._pending_threads if t.is_alive()]
            self._pending_threads.append(thread)

    def _drain_threads(self, timeout: float = 10.0) -> None:
        with self._threads_lock:
            threads = list(self._pending_threads)
            self._pending_threads.clear()
        for t in threads:
            t.join(timeout=timeout)

    def shutdown(self) -> None:
        """Wait for pending background logs, then close HTTP clients.

        Safe to call multiple times.  The async client is created inside
        background threads (each with their own event loop via
        ``asyncio.run``), so by the time ``_drain_threads`` returns the
        underlying transport is already finished.  We drop the reference
        instead of attempting ``aclose`` on a dead loop.
        """
        self._drain_threads()
        if self._client is not None:
            self._client.close()
            self._client = None
        self._async_client = None

    async def ashutdown(self) -> None:
        """Async variant of :meth:`shutdown`."""
        self._drain_threads()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"PromptLayer(base_url={self._base_url!r})"
