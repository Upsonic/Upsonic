"""
Apify Actor Toolkit for Upsonic Framework.

This module provides Apify platform integration, allowing you to run any Apify Actor
as a tool. Actors are dynamically registered based on their input schemas fetched
from the Apify API.

Required Environment Variables:
-----------------------------
- APIFY_API_TOKEN: Apify API token from https://console.apify.com

Example Usage:
    ```python
    from upsonic import Agent, Task
    from upsonic.tools.custom_tools.apify import ApifyTools

    agent = Agent(
        "openai/gpt-4o",
        tools=[
            ApifyTools(
                actors=["apify/rag-web-browser"],
                apify_api_token="your_apify_api_key",
                actor_defaults={
                    "apify/rag-web-browser": {
                        "maxResults": 3,
                        "outputFormats": ["markdown"],
                    }
                },
            )
        ],
    )
    task = Task("What info can you find on https://example.com?")
    agent.print_do(task)
    ```
"""

import inspect
import json
import types
from os import getenv
from typing import Any, Dict, List, Optional, Union

from upsonic.tools.base import ToolKit
from upsonic.tools.config import ToolConfig
from upsonic.utils.integrations.apify import (
    MAX_DESCRIPTION_LEN,
    actor_id_to_tool_name,
    create_apify_client,
    get_actor_latest_build,
    props_to_json_schema,
    prune_actor_input_schema,
)
from upsonic.utils.printing import error_log

try:
    from apify_client import ApifyClient
    _APIFY_AVAILABLE = True
except ImportError:
    ApifyClient = None
    _APIFY_AVAILABLE = False


_SCHEMA_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _make_actor_function(
    actor_id: str,
    client,
    properties: Dict[str, Any],
    required: List[str],
    fixed_defaults: Optional[Dict[str, Any]] = None,
):
    """Create a tool function for an Apify Actor with a proper typed signature.

    Builds a real Python function whose ``inspect.signature`` contains one
    ``Parameter`` per schema property (with correct type annotations and
    defaults).  This lets ``function_schema`` generate the right JSON schema
    so the LLM knows which arguments to pass.

    Parameters that are pre-set via ``fixed_defaults`` are excluded from the
    signature so the LLM never sees them.  At call time the fixed values are
    merged underneath whatever the LLM provides (LLM args win).
    """
    fixed_defaults = fixed_defaults or {}

    # -- 1.  Build inspect.Parameter list --------------------------------
    params: List[inspect.Parameter] = []
    for name, meta in properties.items():
        # Skip params that are pre-set via actor_defaults
        if name in fixed_defaults:
            continue
        annotation = _SCHEMA_TYPE_MAP.get(meta.get("type", ""), Any)
        if name in required:
            default = inspect.Parameter.empty
        else:
            default = meta.get("default", meta.get("prefill", None))
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )

    sig = inspect.Signature(params, return_annotation=str)

    # -- 2.  The actual implementation -----------------------------------
    def actor_function(self, **kwargs: Any) -> str:
        """Run an Apify Actor."""
        try:
            # Merge fixed defaults underneath LLM-provided args
            run_input = {**fixed_defaults, **kwargs}

            details = client.actor(actor_id=actor_id).call(run_input=run_input)
            if details is None:
                raise ValueError(
                    f"Actor: {actor_id} was not started properly and "
                    "details about the run were not returned"
                )

            run_id = details.get("id")
            if run_id is None:
                raise ValueError(f"Run ID not found in the run details for Actor: {actor_id}")

            run = client.run(run_id=run_id)
            results = run.dataset().list_items(clean=True).items

            return json.dumps(results)
        except Exception as e:
            error_log(f"Error running Apify Actor {actor_id}: {e}")
            return json.dumps([{"error": f"Error running Apify Actor {actor_id}: {e}"}])

    # -- 3.  Patch the signature so schema generation works --------------
    actor_function.__signature__ = sig
    annotations: Dict[str, type] = {"return": str}
    for p in params:
        if p.annotation is not inspect.Parameter.empty:
            annotations[p.name] = p.annotation
    actor_function.__annotations__ = annotations

    return actor_function


class ApifyTools(ToolKit):
    """Apify Actor toolkit. Dynamically registers Apify Actors as tools."""

    def __init__(
        self,
        actors: Optional[Union[str, List[str]]] = None,
        apify_api_token: Optional[str] = None,
        actor_defaults: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ApifyTools toolkit.

        Args:
            actors: Single Actor ID or list of Actor IDs to register as tools.
            apify_api_token: Apify API token. Falls back to APIFY_API_TOKEN env var.
            actor_defaults: Per-actor default input values.  Keys are actor IDs,
                values are dicts of parameter name → value.  These parameters
                are always sent to the actor and hidden from the LLM schema.
                Example::

                    actor_defaults={
                        "apify/rag-web-browser": {
                            "maxResults": 3,
                            "outputFormats": ["markdown"],
                        }
                    }
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)

        if not _APIFY_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="apify-client",
                install_command="pip install apify-client",
                feature_name="Apify tools",
            )

        self.apify_api_token: str = apify_api_token or getenv("APIFY_API_TOKEN", "")
        if not self.apify_api_token:
            raise ValueError(
                "Apify API token is required. Set APIFY_API_TOKEN environment "
                "variable or pass apify_api_token parameter."
            )

        self.client = create_apify_client(self.apify_api_token)
        self._actor_defaults: Dict[str, Dict[str, Any]] = actor_defaults or {}

        if actors:
            actor_list = [actors] if isinstance(actors, str) else actors
            for actor_id in actor_list:
                self._register_actor(actor_id)

    def _register_actor(self, actor_id: str) -> None:
        """Register an Apify Actor as a tool method on this toolkit.

        Args:
            actor_id: ID of the Apify Actor (e.g. 'apify/web-scraper').
        """
        try:
            build = get_actor_latest_build(self.client, actor_id)
            tool_name = actor_id_to_tool_name(actor_id)

            actor_description = build.get("actorDefinition", {}).get("description", "")
            if len(actor_description) > MAX_DESCRIPTION_LEN:
                actor_description = actor_description[:MAX_DESCRIPTION_LEN] + "...(TRUNCATED)"

            actor_input = build.get("actorDefinition", {}).get("input")
            if not actor_input:
                raise ValueError(f"Input schema not found in the Actor build for Actor: {actor_id}")

            properties, required = prune_actor_input_schema(actor_input)

            # Resolve fixed defaults for this actor
            fixed_defaults = self._actor_defaults.get(actor_id, {})

            # Remove fixed-default params from required list
            visible_required = [r for r in required if r not in fixed_defaults]

            # Build visible properties (exclude pre-set params)
            visible_properties = {
                k: v for k, v in properties.items() if k not in fixed_defaults
            }

            # Build docstring only for visible (LLM-facing) parameters
            docstring = f"{actor_description}\n\nArgs:\n"
            for param_name, param_info in visible_properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "No description available")
                required_marker = "(required)" if param_name in visible_required else "(optional)"
                docstring += f"    {param_name} ({param_type}): {required_marker} {param_desc}\n"
            docstring += "\nReturns:\n    str: JSON string containing the Actor's output dataset\n"

            # Create the function with fixed defaults baked in
            func = _make_actor_function(
                actor_id, self.client, properties, required, fixed_defaults
            )
            func.__name__ = tool_name
            func.__qualname__ = f"ApifyTools.{tool_name}"
            func.__doc__ = docstring

            # Mark as tool (same attributes the @tool decorator sets)
            func._upsonic_tool_config = ToolConfig()
            func._upsonic_is_tool = True
            # Provide a rich JSON schema override using only visible properties
            func._json_schema_override = props_to_json_schema(
                visible_properties, visible_required
            )

            # Bind as a method on this instance so inspect.ismethod picks it up
            bound_method = types.MethodType(func, self)
            setattr(self, tool_name, bound_method)

        except Exception as e:
            error_log(f"Failed to register Apify Actor '{actor_id}': {e}")
