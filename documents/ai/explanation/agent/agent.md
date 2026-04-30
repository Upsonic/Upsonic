---
name: agent-execution-pipeline
description: Use when working on the core agent orchestration in `src/upsonic/agent/`, including the 24-step direct pipeline, the 22-step streaming pipeline, context managers, policy enforcement, OTel instrumentation, or the AutonomousAgent and DeepAgent subclasses. Use when a user asks to add or modify a pipeline Step, debug HITL resumption, wire up tool execution and tool policies, change system-prompt assembly, adjust context-window overflow handling, plug in reliability/reflection layers, or extend the prebuilt autonomous/deep agents. Trigger when the user mentions Agent, Clanker, Direct, BaseAgent, AutonomousAgent, DeepAgent, AgentRunOutput, AgentRunInput, PipelineManager, Step, StepResult, StepStatus, ModelExecutionStep, StreamModelExecutionStep, CallManager, MemoryManager, SystemPromptManager, ContextManager, ContextManagementMiddleware, LLMManager, TaskManager, ReliabilityManager, PolicyManager, ToolPolicyManager, PolicyScope, AgentOTelManager, RunRequirement, ConfirmationPause, UserInputPause, ExternalExecutionPause, _handle_model_response, _run_in_bg_loop, do/do_async, stream/astream, continue_run, write_todos, FilesystemToolKit, AutonomousFilesystemToolKit, AutonomousShellToolKit, BackendProtocol, StateBackend, MemoryBackend, CompositeBackend, SubagentToolKit, PlanningToolKit, HITL resume, durable execution, or Langfuse Baggage propagation in the agent module.
---

# `src/upsonic/agent/` — Deep Dive

## 1. What this folder is

`src/upsonic/agent/` is the **execution heart of the Upsonic framework**. Every time a user calls `agent.do(task)`, `agent.stream(task)`, or `agent.continue_run(...)`, control flows through the classes defined here. The folder owns:

- The user-facing **`Agent`** class (and its alias `Clanker`) — the main object users instantiate.
- A **24-step direct pipeline** and a **22-step streaming pipeline** that turn a `Task` into a model response, with cache lookups, safety policies, memory injection, system-prompt assembly, model execution, tool execution, reflection, reliability layers, and finalization stitched together as discrete `Step` objects.
- **Context managers** (`SystemPromptManager`, `ContextManager`, `MemoryManager`, `LLMManager`, `CallManager`, `TaskManager`, `ReliabilityManager`, `ContextManagementMiddleware`) — small async classes that handle one cross-cutting concern each and are wired into the pipeline.
- **Safety/policy plumbing** — `PolicyManager` and `ToolPolicyManager` enforce input/output and tool-level rules with feedback loops, blocking, and reversible anonymization.
- **OpenTelemetry instrumentation** — `AgentOTelManager` centralizes span creation for `agent.run`, `pipeline.execute`, `pipeline.step.*`, and `tool.execute` with Langfuse-aware Baggage propagation.
- Three **specialized agent variants** that subclass `Agent`:
  - `AutonomousAgent` — adds default storage/memory + filesystem + shell toolkits sandboxed to a workspace.
  - `DeepAgent` — adds task decomposition (`write_todos`), a virtual filesystem with pluggable backends, and subagent delegation.
- A re-export layer (`events.py`) that surfaces every pipeline/run event class from `upsonic.run.events` for backward compatibility.

In other words, everything else in Upsonic (models, tools, memory, knowledge bases, safety) plugs into the orchestration code that lives here.

---

## 2. Top-level layout

```
src/upsonic/agent/
├── __init__.py                      # Lazy facade — exposes Agent, Clanker, BaseAgent,
│                                    # DeepAgent, AutonomousAgent and 30+ event classes
├── base.py                          # BaseAgent abstract class (anti-circular-import shim)
├── agent.py                         # Agent (5,489 lines) — main orchestrator
├── events.py                        # Backward-compat re-export of run event classes
├── otel_manager.py                  # AgentOTelManager — OTel span/attribute helpers
├── policy_manager.py                # PolicyManager + PolicyResult + PolicyScope
├── tool_policy_manager.py           # ToolPolicyManager + ToolPolicyResult
├── context_managers/                # 8 lifecycle managers (memory, prompt, LLM, etc.)
├── pipeline/                        # Step base + 27 concrete steps + PipelineManager
├── autonomous_agent/                # AutonomousAgent + filesystem & shell toolkits
└── deepagent/                       # DeepAgent + planning, filesystem, subagent toolkits
```

`__init__.py` deliberately uses **module-level `__getattr__`** to lazy-load every public class. Importing `from upsonic.agent import Agent` doesn't drag in the pipeline, deepagent, or autonomous_agent code paths until you actually touch them — important because the agent module is imported on every cold start and the heavy submodules transitively pull in OTel, MCP, vector stores, etc.

---

## 3. Top-level files

### 3.1 `base.py` — `BaseAgent`

A 7-line abstract base class with no methods. Its only purpose is to break circular imports between `Graph` and the concrete `Agent`/`DeepAgent`/`AutonomousAgent` implementations. Anything that wants to be used as a node in `upsonic.graph.Graph` simply inherits from `BaseAgent`.

### 3.2 `agent.py` — `Agent` (and `Clanker = Agent` alias)

The single biggest file in the repo. Every "give me an LLM-driven agent" entry point in Upsonic ultimately winds through this class. Highlights:

#### 3.2.1 Persistent background event loop

```python
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
def _get_bg_loop() -> asyncio.AbstractEventLoop: ...
def _run_in_bg_loop(coro): ...
```

`asyncio.run()` opens and closes a loop per call, which kills cached `httpx.AsyncClient` connections inside the OpenAI/Anthropic SDKs and produces `RuntimeError: Event loop is closed` on the second sync call. The agent module spins up **one daemon thread** running an event loop for the lifetime of the process and submits coroutines to it via `asyncio.run_coroutine_threadsafe`. If you're already inside that loop, `nest_asyncio` is applied to avoid deadlock.

#### 3.2.2 The `Agent` class

`Agent.__init__` takes ~70 keyword arguments and wires together every capability:

| Concern | Constructor params |
| --- | --- |
| Model | `model`, `settings`, `profile`, `reasoning_effort`, `reasoning_summary`, `thinking_enabled`, `thinking_budget`, `thinking_include_thoughts`, `reasoning_format` |
| Identity / persona | `name`, `role`, `goal`, `instructions`, `education`, `work_experience`, `company_*`, `metadata`, `culture` |
| Memory / storage | `memory`, `db`, `session_id`, `user_id`, `feed_tool_call_results` |
| Context window mgmt | `context_management`, `context_management_keep_recent`, `context_management_model` |
| Tools | `tools`, `skills`, `tool_call_limit`, `enable_thinking_tool`, `enable_reasoning_tool`, `show_tool_calls` |
| Reliability | `reliability_layer`, `reflection`, `reflection_config`, `retry`, `mode` |
| Safety | `user_policy`, `agent_policy`, `tool_policy_pre`, `tool_policy_post`, `*_feedback`, `*_feedback_loop`, scoped `user_policy_apply_to_*` flags |
| Observability | `instrument` (OTel), `promptlayer`, `debug`, `debug_level`, `print` |
| Workspace | `workspace` — pre-loads `AGENTS.md` and runs a one-shot greeting on first task |
| Misc | `canvas`, `model_selection_criteria`, `use_llm_for_selection` |

Reasoning settings are mapped to model-specific keys by `_get_model_specific_reasoning_settings()` — e.g. `reasoning_effort` becomes `openai_reasoning_effort` for OpenAI, `thinking_enabled` becomes `anthropic_thinking={"type": "enabled"}` for Anthropic, `google_thinking_config` for Gemini, and `groq_reasoning_format` for Groq.

#### 3.2.3 Public execution surface

| Method | Behavior |
| --- | --- |
| `do(task)` / `do_async(task)` | Run a single task or list of tasks through the **direct pipeline**. Returns task content (default) or full `AgentRunOutput` when `return_output=True`. Supports `timeout` (raises `ExecutionTimeoutError`) and `partial_on_timeout` (silently switches to streaming so partial text can be returned). |
| `print_do` / `print_do_async` | Same, but defaults to printing the run summary. |
| `stream(task)` / `astream(task)` | Run through the **streaming pipeline**, yielding either text chunks or `AgentEvent` objects. Sync `stream()` runs the async generator on the persistent bg loop and bridges it via a `queue.Queue`. |
| `continue_run` / `continue_run_async` | Resume a paused/cancelled/error run. Loads `RunData` from storage if needed, injects HITL results (external tools, user confirmation, user input), and re-enters the pipeline at the failed step. Supports a `hitl_handler` callback that the loop calls per `RunRequirement`. |
| `add_tools` / `remove_tools` / `get_tool_defs` | Dynamic tool management (validates with `tool_policy_pre` on add). |
| `recommend_model_for_task` / `_async` | Calls `upsonic.models.model_selector` to suggest a better model for a task (rule-based or LLM-based). |
| `cancel_run(run_id)` | Cancels a specific run via the global cancel registry. |
| `as_mcp(name)` | Returns a `fastmcp.FastMCP` server with a single `do(task: str)` tool, letting any agent be exposed as an MCP server. |
| `instrument_all(...)` (classmethod) | Sets a global `_TracingProvider` so every subsequent `Agent` instance auto-instruments. |
| `cost` (property) | Aggregated tokens + USD cost across every task this agent has executed (mirrors `Task.get_total_cost`). |
| `get_session_usage` / `aget_session_usage` | Session-level `TaskUsage` from the configured `Memory`. |
| `execute_workspace_greeting` / `_async` | When `workspace=` is set, runs a hidden greeting task on first call. |

#### 3.2.4 Pipeline construction

`_create_direct_pipeline_steps()` returns the canonical 24-step direct pipeline:

```
0  InitializationStep         12 MessageAssemblyStep
1  StorageConnectionStep      13 CallManagerSetupStep
2  CacheCheckStep             14 ModelExecutionStep        ← HITL resume target
3  LLMManagerStep             15 ResponseProcessingStep
4  ModelSelectionStep         16 ReflectionStep
5  ToolSetupStep              17 TaskManagementStep
6  MemoryPrepareStep          18 ReliabilityStep
7  SystemPromptBuildStep      19 AgentPolicyStep
8  ContextBuildStep           20 CacheStorageStep
9  ChatHistoryStep            21 FinalizationStep
10 UserPolicyStep             22 MemorySaveStep
11 UserInputBuildStep         23 CallManagementStep        ← prints metrics last
```

`_create_streaming_pipeline_steps()` returns the 22-step streaming variant: it swaps `ModelExecutionStep` + `ResponseProcessingStep` for `StreamModelExecutionStep` (which has `supports_streaming=True` and yields events live), drops `TaskManagementStep`/`MemorySaveStep` in favor of `StreamMemoryMessageTrackingStep`, and ends with `StreamFinalizationStep` + `CallManagementStep`.

#### 3.2.5 Internal orchestration helpers

- `_handle_model_response(response, messages)` — recursive function that drives the tool-call loop. It (a) handles culture repeat injection, (b) detects `finish_reason == 'length'` truncation and re-prompts with smaller output, (c) executes regular tool calls, (d) re-applies `ContextManagementMiddleware` before each follow-up `model.request()`, and (e) recurses until no tool calls remain. It catches `ExternalExecutionPause`/`ConfirmationPause`/`UserInputPause` for HITL.
- `_execute_tool_calls(tool_calls)` — splits calls into sequential vs parallel buckets (based on `ToolDefinition.sequential`), runs sequential ones in order, and `asyncio.gather`s the parallel ones. Each call is wrapped in an OTel `tool.execute` span, validated against `tool_policy_post`, and on failure produces a `ToolReturnPart` with the error so the model can recover.
- `_apply_user_policy(task, context, system_prompt_manager)` — collects every text source the user is allowed to scope to (description, KB context, system prompt, chat history) and feeds them into `PolicyManager.execute_policies_async()` with source tracking. Replaces/anonymizes are written back into the right places, originals are stashed on `task._policy_originals` for de-anonymization in `FinalizationStep`. Generates a `[PRIVACY MODE ACTIVE: ...]` notice when anonymization actually changes content.
- `_apply_agent_policy(task, context)` — the output-side counterpart; supports the feedback-loop retry mechanism by returning a `feedback_message` to the caller (handled in `AgentPolicyStep`).
- `_execute_with_guardrail(task, memory_handler, state)` — runs the model with a `task.guardrail` callable in a retry loop (`task.guardrail_retries + 1` attempts), feeding back the failure reason as a correction prompt.
- `_inject_hitl_results(output, requirements)` — for resumption: walks resolved `RunRequirement`s and turns them into `ToolReturnPart`s that get appended to the chat history before the pipeline resumes at the model-execution step.
- `_log_to_promptlayer_unified(...)` + `_create_promptlayer_workflow(...)` — fire-and-forget logging with a node-graph workflow representation (input → system_prompt → step nodes → tool branches → output) so PromptLayer can render the agent run as a DAG.

#### 3.2.6 Notable design choices

- **AgentRunOutput is the single source of truth.** Every step reads/writes the same `AgentRunOutput` instance. Pipeline state, chat history, step results, requirements, and usage all live there — that is what gets serialized for HITL/durable execution.
- **Steps don't share globals.** Cross-step state goes through the `PipelineManager._managers` registry (`memory_manager`, `call_manager`, etc.) via `set_manager`/`get_manager`.
- **Sync wrappers always go through `_run_in_bg_loop`.** Never `asyncio.run`.
- **Backward compatibility**: `Clanker = Agent` is the only documented alias; `events.py` re-exports every event class so external users of the old import paths keep working.

### 3.3 `events.py`

A 100-line shim that re-exports ~40 event classes (`AgentEvent`, `PipelineStartEvent`, `PipelineEndEvent`, `StepStartEvent`/`StepEndEvent`, `CacheCheckEvent`/`CacheHitEvent`/`CacheMissEvent`, `PolicyCheckEvent`, `PolicyFeedbackEvent`, `ModelSelectedEvent`, `ToolsConfiguredEvent`, `MessagesBuiltEvent`, `ModelRequestStartEvent`/`ModelResponseEvent`, `ToolCallEvent`/`ToolResultEvent`, `ExternalToolPauseEvent`, `ReflectionEvent`, `MemoryUpdateEvent`/`CultureUpdateEvent`, `ReliabilityEvent`, `CacheStoredEvent`, `ExecutionCompleteEvent`, `RunStartedEvent`/`RunCompletedEvent`/`RunPausedEvent`/`RunCancelledEvent`, `TextDeltaEvent`/`TextCompleteEvent`/`ThinkingDeltaEvent`/`ToolCallDeltaEvent`/`FinalOutputEvent`) from their canonical home in `upsonic.run.events.events`. Existing import paths like `from upsonic.agent.events import AgentEvent` continue to work.

### 3.4 `otel_manager.py` — `AgentOTelManager`

Central place for OpenTelemetry instrumentation of the agent. The class is intentionally degraded-on-disable: when `settings is None` every method returns `nullcontext(None)`, so the agent code never needs `if otel_enabled:` branches.

Key span types:

| Span | When | Kind |
| --- | --- | --- |
| `agent.run` | Wraps a whole `do_async`/`astream` call (root span) | `SERVER` |
| `pipeline.execute` | Wraps the sequential step loop in `PipelineManager` | `INTERNAL` |
| `pipeline.step.<name>` | One per pipeline step | `INTERNAL` |
| `tool.execute` | One per tool invocation | `INTERNAL` |

The agent root span attaches **OTel Baggage** (`langfuse.trace.name`, `langfuse.user.id`/`session.id`, plus generic `user.id`/`session.id`) via a custom `_BaggageSpanContext`, which is detected by the `BaggageSpanProcessor` and propagated to every child span — required for Langfuse trace-level grouping.

`finalize_agent_run(span, output, ...)` extracts every relevant metric from `AgentRunOutput` (token usage with `gen_ai.*` semantic conventions, tool call count, cost, model name/provider, status, pipeline step timing/statuses, tool definitions, executed tool calls with input/output) and writes it to the root span before flushing. `extract_trace_id(span)` returns the hex 32-char trace ID so the agent can stash it on `AgentRunOutput.trace_id` for cross-system correlation.

### 3.5 `policy_manager.py` — `PolicyManager` + `PolicyResult` + `PolicyScope`

Drives the **safety engine** for input (`user_policy`) and output (`agent_policy`).

`PolicyScope(description, context, system_prompt, chat_history, tool_outputs)` is a dataclass of per-source booleans. `resolve_policy_scope(policy, task, agent)` resolves them with **Policy > Task > Agent** priority — so an individual `Policy` can opt out of scanning chat history even if the agent globally enables it.

`PolicyResult` is the aggregated outcome:
- `action_taken` ∈ `{"ALLOW", "BLOCK", "REPLACE", "ANONYMIZE", "DISALLOWED_EXCEPTION"}`
- `was_blocked`, `disallowed_exception`
- `transformation_map`: `{idx: {"original": "...", "anonymous": "...", "pii_type": "..."}}` for **reversible anonymization** (de-anonymization happens in `FinalizationStep`).
- `feedback_message`, `requires_retry`, `original_content`, `violated_policy_name`, `violation_reason` for the **feedback loop** (`should_retry_with_feedback()`).
- `output_texts`, `source_keys`: post-transformation text per source.

`PolicyManager.execute_policies_async(policy_input, check_type, source_keys, task, agent)`:
1. Iterates policies sequentially.
2. In **scoped mode** (when `task` and `agent` are provided), filters input texts by each policy's scope.
3. Stops on first `BLOCK` (the most restrictive). `REPLACE`/`ANONYMIZE` are accumulated — transformation maps are merged with offset re-keying so multiple policies can chain.
4. Optionally generates LLM-driven feedback via `UpsonicLLMProvider.generate_policy_feedback_async()` when `enable_feedback=True` and retries remain (`feedback_loop_count`).

`drain_accumulated_usage()` collects `RunUsage` from every policy's `base_llm`/`text_finder_llm` plus the feedback LLM, so the parent agent can fold sub-agent token cost into `AgentRunOutput.usage`.

### 3.6 `tool_policy_manager.py` — `ToolPolicyManager` + `ToolPolicyResult`

Parallel structure to `PolicyManager`, specialized for tools.

- `execute_tool_validation_async(tool_info, ...)` — runs at **tool registration time** (`tool_policy_pre`) before a tool can ever be exposed to the model. If a policy returns `BLOCK` or raises `DisallowedOperation`, the tool is removed from the agent's `ToolManager`.
- `execute_tool_call_validation_async(tool_call_info, ...)` — runs **right before** each actual tool invocation (`tool_policy_post`). Catches malicious arguments at runtime.
- Both pass the tool name, description, JSON schema, args, and call ID into the policy's `extra_data` so rules can inspect them.

Result objects: `is_safe`, `was_blocked`, `threat_details: {policy_name → {content_type, details, confidence}}`.

---

## 4. `context_managers/` — Lifecycle managers

Eight thin async classes, each owning one responsibility. The pattern is identical: every class exposes `aprepare()`, `afinalize()`, sync wrappers, and a legacy `manage_*()` async context manager kept for backward compatibility. The pipeline calls `aprepare()` at the start and `afinalize()` at the end of the relevant phase.

| File | Class | Responsibility |
| --- | --- | --- |
| `__init__.py` | — | Lazy facade for the eight managers |
| `call_manager.py` | `CallManager` | Tracks call start/end times, prints the **Tool Calls** + **LLM Result** + **Task Metrics** panels, records `price_id` summary. The single printing entry-point for completed runs. |
| `task_manager.py` | `TaskManager` | Stores the model response and calls `task.task_response(response)` in `afinalize`. Task start/end is now handled by `InitializationStep`/`FinalizationStep` — this class is mostly a no-op kept for API parity. |
| `reliability_manager.py` | `ReliabilityManager` | Lazy-imports `ReliabilityProcessor` and runs the configured reliability layer over the task. |
| `llm_manager.py` | `LLMManager` | Selects the model. Reads `LLM_MODEL_KEY` env var, supports a Celery-driven `bypass_llm_model` override, calls `infer_model()`, then writes `model_name`/`provider`/`profile` onto `AgentRunOutput`. |
| `memory_manager.py` | `MemoryManager` | Calls `memory.prepare_inputs_for_task(agent_metadata=...)` to load message history, summary context, user-profile system-prompt injection, and metadata injection. Saves the session via `memory.save_session_async()` in `afinalize`. Builds an `<AgentMetadata>...</AgentMetadata>` block from agent metadata. |
| `system_prompt_manager.py` | `SystemPromptManager` | Builds the **final system prompt**. Order: (optional reflective `Operation Deliberate Thought`/`Operation Blueprint` mission briefings if thinking/reasoning tools are on) → today's date → memory user-profile injection → workspace `AGENTS.md` → user `system_prompt` → role/goal/instructions/education/work_experience/company_* → cultural knowledge (always last so it has final authority) → `<YourCharacter>` (other agents passed as context) → skills section → `<ToolInstructions>` collected from agent and task `ToolManager`s. Decides whether to re-include the system prompt on follow-up turns based on whether `<UserProfile>`/`<CulturalKnowledge>` content is dynamic. |
| `context_manager.py` | `ContextManager` | Builds the **task-specific context block**. Iterates `task.context` items: other `Task`s become `<Tasks>`, `KnowledgeBase`s trigger `setup_async()` + optional `query_async()` and become `<rag source=...>` blocks with metadata-rich chunk formatting, raw strings become `<Additional Context>`, `TaskOutputSource` items pull from `Graph.State` and become `<PreviousTaskNodeOutput>`. Also exposes `get_knowledge_base_health_status()` and `get_context_summary()` for diagnostics. |
| `context_management_middleware.py` | `ContextManagementMiddleware` | Applies the context-window-overflow strategy. (See §4.1.) |

### 4.1 `ContextManagementMiddleware` — the overflow strategy

When `context_management=True` on the agent, this middleware runs before every `model.request()` and every follow-up tool-call request. Its `apply(messages)` method is a 3-stage funnel:

1. **Estimate tokens.** Sum every `ModelResponse.usage.input_tokens + output_tokens` across the message list (since the list can span multiple runs). Fall back to `len(text) // 4` when no usage data exists. Compare against `model_context_window × safety_margin_ratio` (default 0.90).
2. **Step 1 — Tool pruning** (only if any `ToolCallPart`/`ToolReturnPart`/`RetryPromptPart` is present). Identifies "tool rounds" as `(ModelResponse with ToolCallPart, ModelRequest with matching ToolReturnPart)` pairs and drops all but the `keep_recent_count` most recent ones. Messages with no remaining parts are dropped entirely.
3. **Step 2 — LLM summarization.** Splits messages into conversation pairs, keeps the last `keep_recent_count` verbatim, serializes the rest into a structured prompt and asks the LLM (the `context_compression_model` if configured, else the agent's primary model) to return a JSON `ConversationSummary` matching a strict Pydantic schema (`SummarizedRequest`/`SummarizedResponse` with strict part-kind preservation rules — the LLM is told not to reorder, drop, or merge parts; tool-call IDs must be preserved exactly). Reconstructs `ModelRequest`/`ModelResponse` objects from the summary.
4. **Step 3 — Context full.** If the result still exceeds the limit, signals `context_full=True`. The agent injects a fixed `[SYSTEM] The conversation context window has been exceeded...` `ModelResponse` and stops the pipeline.

Summarization usage is captured in `_last_summarization_usage` so the parent agent can drain it via `_propagate_context_management_usage()` and account for the cost.

---

## 5. `pipeline/` — Step machinery

### 5.1 Files

```
pipeline/
├── __init__.py    # Lazy facade for Step, StepResult, StepStatus, PipelineManager,
│                  # and 27 concrete step classes
├── step.py        # 388 lines — Step ABC, StepResult, StepStatus enum,
│                  # error-injection registry for tests
├── steps.py       # 3,925 lines — every concrete step implementation
└── manager.py     # 927 lines — PipelineManager (sequential + streaming execution)
```

### 5.2 `step.py` — the abstract base

`StepStatus` is a `str` Enum: `RUNNING`, `COMPLETED`, `PAUSED`, `CANCELLED`, `ERROR`, `SKIPPED`. `StepStatus.to_run_status()` maps it onto `RunStatus` from `upsonic.run.base` so step status flows up to the run status without ambiguity.

`StepResult` (Pydantic): `name`, `step_number`, `status`, `message`, `execution_time` — serializable for storage / HITL resumption.

`Step` is an ABC. Subclasses must implement:
- `name` (property) — stable identifier, used by `_get_step_index_by_name` for HITL resumption.
- `description` (property) — human-readable, surfaced to OTel and logs.
- `supports_streaming` — `True` only for `StreamModelExecutionStep`.
- `execute(context, task, agent, model, step_number, pipeline_manager)` — the real work, returning a `StepResult`.

The default `run()` wrapper calls `check_error_injection(self.name)` before the step body — `inject_error_into_step("model_execution", trigger_count=1)` lets tests force a step to fail N times to exercise durable execution. After the step returns, `_finalize_step_result()` appends the `StepResult` to `context.step_results` and increments `context.execution_stats`.

Streaming steps yield events instead of returning. The default `execute_stream()` snapshots `len(context.events)` before calling `execute()` and yields anything new — so non-streaming steps can still emit events into a streaming pipeline by appending to `context.events`. Dedicated streaming steps override `execute_stream()` and emit deltas live (e.g. `TextDeltaEvent`, `TextCompleteEvent`).

### 5.3 `manager.py` — `PipelineManager`

Owns step execution. Two main entry points:

- `execute(context, start_step_index=0)` — non-streaming. Wraps the loop in (a) an OTel `pipeline.execute` span, (b) a Sentry transaction tagged with step counts, debug, streaming, task type/description. Each step gets its own OTel span and Sentry span. After every step, OTel attributes are written via `set_step_result(...)`. On exception the manager dispatches:
  - `ConfirmationPause` → `_handle_confirmation_pause` (creates `RunRequirement(requires_confirmation=True)` per call, marks `paused`, saves session, returns normally).
  - `UserInputPause` → `_handle_user_input_pause` (similar with `user_input_schema`).
  - `ExternalExecutionPause` → `_handle_external_tool_pause` (similar with `external_execution_required=True`).
  - `RunCancelledException` → `_ahandle_cancellation` (marks cancelled, saves session, returns normally).
  - Any other exception → `_ahandle_durable_execution_error` (saves checkpoint, re-raises). In all cases the failed step lives in `context.step_results`, so `output.get_problematic_step()` finds the resume point.
- `execute_stream(context, start_step_index=0)` — async generator. Emits `RunStartedEvent`, `PipelineStartEvent`, then per-step `StepStartEvent`/`StepEndEvent`, plus all events steps push to `context.events` (or yield directly when `step.supports_streaming`). Always yields `PipelineEndEvent` and `RunCompletedEvent` (or `RunCancelledEvent`) in `finally`.

`_managers: Dict[str, Any]` is the cross-step registry. `MemoryPrepareStep` registers the `MemoryManager`, `CallManagerSetupStep` registers the `CallManager`, etc. Steps that need a manager call `pipeline_manager.get_manager('memory_manager')`.

`_save_session(output)` is the centralized `agent.memory.save_session_async(output)` call — used for normal completion AND for every problematic state (paused/cancelled/error) so durable execution + HITL just read the same `RunData` back out of storage.

### 5.4 `steps.py` — the 27 concrete steps

The order of steps in the **direct pipeline** matches the numbered list in §3.2.4. Each step is implemented as:

```python
class XxxStep(Step):
    @property
    def name(self): return "xxx"
    @property
    def description(self): return "..."
    async def execute(self, context, task, agent, model, step_number, pipeline_manager=None):
        try:
            raise_if_cancelled(agent.run_id)
            ...business logic...
            step_result = StepResult(name=..., status=COMPLETED, ...)
            return step_result
        except RunCancelledException: ...status=CANCELLED... raise
        except (ExternalExecutionPause, ConfirmationPause, UserInputPause): ...status=PAUSED... raise
        except Exception: ...status=ERROR... raise
        finally:
            if step_result: self._finalize_step_result(step_result, context)
```

Step-by-step responsibilities:

| # | Step (direct) | What it does |
| --- | --- | --- |
| 0 | `InitializationStep` | `task.task_start(agent)`, attach `task._usage` to context, set `current_task`, emit `AgentInitializedEvent`. |
| 1 | `StorageConnectionStep` | Connects the storage backend (DB / Redis / etc) so memory has a live connection. |
| 2 | `CacheCheckStep` | If `task.enable_cache`, runs `task.get_cached_response(input, model)` (exact or vector-search). On hit, sets `task._cached_result=True`, fills `context.output`, ends task, emits `CacheHitEvent`. Drains the cache LLM's usage into context usage. |
| 3 | `LLMManagerStep` | Constructs and prepares `LLMManager` (handles env-var / Celery model overrides). |
| 4 | `ModelSelectionStep` | If `model_selection_criteria` is set, calls `select_model_async(...)` and may switch the model. Emits `ModelSelectedEvent`. |
| 5 | `ToolSetupStep` | `agent._setup_task_tools(task)` — registers task-level tools, applies `tool_policy_pre` to both agent and task tool managers. Emits `ToolsConfiguredEvent`. |
| 6 | `MemoryPrepareStep` | Creates and `aprepare()`s the `MemoryManager`. Registers it in the pipeline registry. Emits `MemoryPreparedEvent` / `ChatHistoryLoadedEvent`. |
| 7 | `SystemPromptBuildStep` | Builds the system prompt via `SystemPromptManager`. Emits `SystemPromptBuiltEvent`. |
| 8 | `ContextBuildStep` | Builds the task context (KB queries, memory injection, prior tasks). Emits `ContextBuiltEvent`. |
| 9 | `ChatHistoryStep` | Loads chat history from memory into `context.chat_history` and calls `output.start_new_run()` so the run-boundary is recorded. **Critical for HITL resume** — `_get_step_index_by_name("chat_history")` is used to decide whether to call `start_new_run` again on resumption. |
| 10 | `UserPolicyStep` | Calls `agent._apply_user_policy(...)`. Stops the pipeline (returns `should_continue=False`) on BLOCK; transforms inputs on REPLACE/ANONYMIZE. |
| 11 | `UserInputBuildStep` | Builds the final user-input message (description + context + attachments) onto `AgentRunInput`. Emits `UserInputBuiltEvent`. |
| 12 | `MessageAssemblyStep` | Assembles the `ModelRequest` (`SystemPromptPart` + `UserPromptPart`) and runs `ContextManagementMiddleware.apply()`. If middleware reports `context_full`, sets `context._context_window_full=True` and short-circuits subsequent steps. |
| 13 | `CallManagerSetupStep` | Constructs and `aprepare()`s the `CallManager`, registers it in the pipeline registry. |
| 14 | `ModelExecutionStep` | The big one. Runs `model.request(...)` (or `agent._execute_with_guardrail(...)` if guardrail set), then `agent._handle_model_response(response, chat_history)` — which recursively executes tool calls. Adds model execution time to context. Catches HITL pauses → `StepStatus.PAUSED` → re-raised so `PipelineManager` can convert them to `RunRequirement`s. **HITL resume target** — paused runs resume here. |
| 15 | `ResponseProcessingStep` | `agent._extract_output(task, response)` decides text vs structured-output vs image. Saves `ThinkingPart`s and binary image content separately. |
| 16 | `ReflectionStep` | If `reflection=True`, runs `agent.reflection_processor` over the response. Emits `ReflectionEvent`. |
| 17 | `TaskManagementStep` | Calls `TaskManager.afinalize()` which propagates the model response onto the task. |
| 18 | `ReliabilityStep` | Runs the configured `reliability_layer` (verifier/editor agents). Emits `ReliabilityEvent`. |
| 19 | `AgentPolicyStep` | Calls `agent._apply_agent_policy(task, context)`. If feedback is enabled and a violation triggers a feedback message, this step retries the model up to `agent_policy_feedback_loop` times by re-entering the model-execution path with the feedback prompt prepended. Emits `PolicyCheckEvent`/`PolicyFeedbackEvent`. |
| 20 | `CacheStorageStep` | If `task.enable_cache` and not a cache hit, persists `(input, output)` to the task's cache backend. Emits `CacheStoredEvent`. |
| 21 | `FinalizationStep` | Last-mile cleanup: de-anonymizes output, response, chat history, and tool args using `task._anonymization_map`; restores originals stashed in `task._policy_originals`; closes any task-level MCP handlers; emits `ExecutionCompleteEvent`. |
| 22 | `MemorySaveStep` | Calls `MemoryManager.afinalize()` → `memory.save_session_async()`. Emits `MemoryUpdateEvent`. |
| 23 | `CallManagementStep` | **Last step.** Calls `CallManager.alog_completion(context)` which prints the tool-call panel + LLM result + task metrics, records `price_id_summary`, and `task.task_end()`. |

Streaming-specific steps:

- `StreamModelExecutionStep` — `supports_streaming=True`. Uses `model.request_stream(...)` and yields `ModelRequestStartEvent`, then `TextDeltaEvent` for every partial token, `TextCompleteEvent` at the end, `FinalOutputEvent` carrying the full text. If the response was cached it yields the cached content character-by-character so the UX is identical.
- `StreamMemoryMessageTrackingStep` — combines `TaskManagementStep` + `MemorySaveStep` for the streaming pipeline; saves the session and ends the task.
- `StreamFinalizationStep` — streaming variant of `FinalizationStep` (de-anonymization + ExecutionComplete event).

---

## 6. `autonomous_agent/` — Production-ready coding/devops agent

### 6.1 Files

```
autonomous_agent/
├── __init__.py
├── autonomous_agent.py      # 573 lines — AutonomousAgent class
├── filesystem_toolkit.py    # 942 lines — AutonomousFilesystemToolKit
└── shell_toolkit.py         # 332 lines — AutonomousShellToolKit
```

### 6.2 `AutonomousAgent`

Subclass of `Agent` that bakes in everything you need to build a coding/devops agent without manual wiring:

- **Default storage/memory.** If `db is None and memory is None and storage is None`, it instantiates `InMemoryStorage()` and wraps it in a `Memory(...)` with `full_session_memory=True` (chat history on by default), `summary_memory=False`, `user_analysis_memory=False`. All memory feature flags from `Memory.__init__` are exposed as kwargs.
- **Workspace sandboxing.** `workspace` is resolved to an absolute `Path`, created if it doesn't exist, and **every** filesystem and shell tool is restricted to it. Defaults to `Path.cwd()` if not provided.
- **Default toolkits.** When `enable_filesystem=True` (default) and `enable_shell=True` (default), instantiates `AutonomousFilesystemToolKit(workspace=...)` and `AutonomousShellToolKit(workspace=..., default_timeout=120, max_output_length=10000, blocked_commands=...)` and prepends them to the user's `tools=`.
- **Dynamic system prompt.** `_build_autonomous_system_prompt()` builds an `<AutonomousAgent>...</AutonomousAgent>` wrapped prompt with sections for Capabilities (per enabled toolkit), Guidelines (when to use each), Best Practices (numbered, e.g. "Read before edit", "Use precise edits", "Stay in workspace"), and Security Restrictions. If the user passes a custom `system_prompt`, that gets wrapped instead.
- **Heartbeat support.** `heartbeat`, `heartbeat_period` (minutes), `heartbeat_message` — `aexecute_heartbeat()` runs the heartbeat message as a hidden `do_async` call (with `print=False`) and returns the agent's text response, suitable for UI integrations that want a periodic "what's the agent doing?" probe.
- **`print` defaults to `True`.** Unlike base `Agent`, `do()` prints task metrics by default — pass `print=False` to silence.
- **`reset_filesystem_tracking()`.** Clears the `_read_files` set used by `edit_file` for read-before-edit safety.

### 6.3 `AutonomousFilesystemToolKit`

11 `@tool` methods, all sandboxed via `_validate_path(path)` which resolves to an absolute path and ensures `resolved.relative_to(self.workspace)` succeeds (raises on path traversal):

| Tool | Purpose |
| --- | --- |
| `read_file(file_path, offset=None, limit=None)` | Read with pagination. Tracks the file in `self._read_files` so `edit_file` can enforce read-before-edit. |
| `write_file(path, content)` | Create / overwrite. Auto-creates parent directories. |
| `edit_file(path, old_string, new_string, replace_all=False)` | Precise string replacement. Errors if the file wasn't read first or if `old_string` isn't unique (forces context-rich anchors). |
| `list_files(path=".", pattern=None, recursive=False)` | Directory listing with glob filtering. |
| `search_files(name_pattern, path=".")` | Find by filename. |
| `grep_files(pattern, path=".", glob=None, regex=False)` | Find by content. |
| `move_file`, `copy_file`, `delete_file` | Self-explanatory. |
| `file_info(path)` | Size, permissions, timestamps. |
| `create_directory(path)` | Recursive `mkdir`. |

### 6.4 `AutonomousShellToolKit`

3 `@tool` methods, also sandboxed:

| Tool | Purpose |
| --- | --- |
| `run_command(command, timeout=None, env=None, shell=True)` | Subprocess execution with combined stdout/stderr capture, configurable timeout, output truncation at `max_output_length`. |
| `run_python(code)` | One-shot Python execution. |
| `check_command_exists(command)` | `shutil.which`-style check. |

`_validate_command(command)` enforces a configurable `blocked_commands` blacklist (defaults: `rm -rf /`, `rm -rf /*`, `:(){:|:&};:`, `mkfs`, `dd if=/dev/zero`) and an optional `allowed_commands` whitelist. Commands always run with `cwd=self.workspace`.

---

## 7. `deepagent/` — Planning + virtual filesystem + subagents

### 7.1 Files

```
deepagent/
├── __init__.py
├── deepagent.py              # 322 lines — DeepAgent class
├── constants.py              # 438 lines — system prompts + tool descriptions
├── backends/
│   ├── __init__.py
│   ├── protocol.py           # BackendProtocol (async read/write/delete/exists/list_dir/glob)
│   ├── state_backend.py      # In-memory dict storage
│   ├── memory_backend.py     # Persistent storage via Upsonic Storage (collection: "deepagent_filesystem")
│   └── composite_backend.py  # Path-prefix-based routing
└── tools/
    ├── __init__.py
    ├── filesystem_toolkit.py # FilesystemToolKit — 6 tools using BackendProtocol
    ├── planning_toolkit.py   # PlanningToolKit + Todo + TodoList (write_todos)
    └── subagent_toolkit.py   # SubagentToolKit (task tool)
```

### 7.2 `DeepAgent`

Subclass of `Agent` aimed at long-horizon, complex tasks. It composes three orthogonal capabilities, all toggleable:

- **Planning** (`enable_planning=True`, default). Adds `PlanningToolKit` providing a single `write_todos` tool. The tool itself is a "cognitive forcing function" — its computation is minimal; the value comes from making the LLM commit to a structured plan. Todos are validated with Pydantic (min 2, unique IDs, statuses ∈ `{pending, in_progress, completed, cancelled}`). The active todo list is stored on the current task and surfaced via `agent.get_current_plan()`.
- **Virtual filesystem** (`enable_filesystem=True`, default). Adds `FilesystemToolKit` with 6 tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) that all delegate to a `BackendProtocol`. Default backend is `StateBackend` (per-instance dict, no I/O); switch to `MemoryBackend(storage=...)` for persistence; combine with `CompositeBackend(default=..., routes={"/memories/": MemoryBackend(...), "/tmp/": StateBackend()})` for hybrid layouts. All paths must be absolute (`/...`); `..` and null bytes are blocked at the backend level. Read-before-edit is enforced via `_read_files`.
- **Subagent delegation** (`enable_subagents=True` by default — auto-enabled if you pass `subagents=[...]`). Adds `SubagentToolKit` with a single `task(task_description, subagent_type)` tool that looks up an `Agent` in `self.subagents` by `name` and runs `subagent.do_async(...)` on it. Each subagent has `memory=None` (complete isolation). If no subagents are passed, `DeepAgent` auto-creates a `general-purpose` subagent that inherits the parent's tools (minus `SubagentToolKit` to prevent recursion). The task tool's docstring is dynamically rebuilt every time `add_subagent()` is called so the model sees the up-to-date roster.

`DeepAgent._build_deep_system_prompt(user_prompt)` concatenates: `BASE_AGENT_PROMPT` ("DO NOT STOP UNTIL YOU HAVE COMPLETED THE TASK!") → user's prompt → `FILESYSTEM_SYSTEM_PROMPT` → `WRITE_TODOS_SYSTEM_PROMPT` → `TASK_SYSTEM_PROMPT`. Default `tool_call_limit=20` (vs base Agent's 100) since subagents amplify each call into many.

Public API: `add_subagent(agent)`, `get_subagent_names()`, `get_current_plan()`, `get_filesystem_stats()`.

### 7.3 Backends

`BackendProtocol` is a `@runtime_checkable Protocol` with five async methods: `read`, `write`, `delete`, `exists`, `list_dir`, `glob`. Path conventions are uniform across all backends: absolute paths only, `/` separators, no `..`, case-sensitive.

| Backend | Storage | Persistence | Use case |
| --- | --- | --- | --- |
| `StateBackend` | `Dict[path → content]` (None for directory markers) | Per-process | Scratch space inside one run |
| `MemoryBackend` | Upsonic `Storage` collection `deepagent_filesystem`, value type `FilesystemEntry` (Pydantic with `path`, `content`, `is_directory`, `created_at`, `modified_at`, `size`) | Survives restarts; can be shared across agents | Long-term memory, important artifacts |
| `CompositeBackend` | First-match-wins prefix routing over a list of `(prefix, backend)` plus a default | Per-route | Hybrid setups (e.g. `/memories/` persistent, `/tmp/` ephemeral) |

### 7.4 Constants (`constants.py`)

438 lines of carefully-tuned prompt strings: full tool docstrings (`READ_FILE_TOOL_DESCRIPTION`, `EDIT_FILE_TOOL_DESCRIPTION` with read-before-edit enforcement, `WRITE_FILE_TOOL_DESCRIPTION`, `LIST_FILES_TOOL_DESCRIPTION`, `GLOB_TOOL_DESCRIPTION`, `GREP_TOOL_DESCRIPTION`, `WRITE_TODOS_TOOL_DESCRIPTION`, `TASK_TOOL_DESCRIPTION` with an `{available_agents}` placeholder filled in dynamically) plus the system-prompt blocks (`FILESYSTEM_SYSTEM_PROMPT`, `WRITE_TODOS_SYSTEM_PROMPT`, `TASK_SYSTEM_PROMPT`, `BASE_AGENT_PROMPT`, `DEFAULT_SUBAGENT_PROMPT`, `DEFAULT_GENERAL_PURPOSE_DESCRIPTION`).

---

## 8. End-to-end flow of a single `agent.do(task)` call

1. **Sync entry.** `agent.do(task)` resolves a list-vs-single, converts strings to `Task`, then submits `agent.do_async(...)` to the persistent background loop via `_run_in_bg_loop`.
2. **Setup.** `do_async` validates the task isn't already completed/problematic, generates a `run_id`, registers it with the global cancel registry, builds an `AgentRunInput` (splits attachments by mime-type into images vs documents), constructs the `AgentRunOutput` factory, and applies any per-call model override.
3. **OTel + PromptLayer.** Wraps the rest in `agent._otel.agent_run_span(...)` (with Langfuse Baggage attached) and records the start time for PromptLayer.
4. **Pipeline.** Constructs the 24-step `PipelineManager` via `_create_direct_pipeline_steps()` and calls `pipeline.execute(context, start_step_index)`. (If `partial_on_timeout=True`, switches to streaming + `asyncio.wait_for` so partial text can be salvaged on timeout.)
5. **Step loop.** `PipelineManager.execute` walks each step inside its own OTel/Sentry span. Steps mutate `AgentRunOutput` directly: chat history, system prompt, requirements, step results, usage, output, events.
6. **Tool round-trip.** `ModelExecutionStep` calls `model.request(...)`, then `_handle_model_response()` recursively executes tool calls (parallel where possible), reapplies `ContextManagementMiddleware`, and re-calls `model.request(...)` with the tool returns until no tool calls remain. HITL pauses raise `ExternalExecutionPause`/`ConfirmationPause`/`UserInputPause`, propagating up to the manager, which converts them into `RunRequirement`s and saves a checkpoint.
7. **Post-processing.** `ResponseProcessingStep` extracts the final output (text, structured Pydantic, image bytes). `ReflectionStep`/`ReliabilityStep` may rewrite it. `AgentPolicyStep` may ask for one or more retries with feedback.
8. **Finalization.** `CacheStorageStep` writes the result to the cache backend if enabled. `FinalizationStep` de-anonymizes output and chat history, restores any `_policy_originals`, and closes task-level MCP handlers. `MemorySaveStep` calls `memory.save_session_async`. `CallManagementStep` prints panels and records `price_id_summary`.
9. **Wrap-up.** `_otel.finalize_agent_run(...)` writes every metric to the root span and flushes. `_log_to_promptlayer_background(...)` fires off a thread to log the run + create/patch a workflow node graph in PromptLayer. Run ID is cleaned up unless the task is paused (HITL keeps it alive).
10. **Return.** Either `AgentRunOutput` (if `return_output=True`) or `context.output`.

---

## 9. Class hierarchy

```
BaseAgent (abstract, base.py)
└── Agent (agent.py)                           ← Clanker = Agent
    ├── AutonomousAgent (autonomous_agent.py)
    │       owns: AutonomousFilesystemToolKit, AutonomousShellToolKit
    └── DeepAgent (deepagent/deepagent.py)
            owns: FilesystemToolKit, PlanningToolKit, SubagentToolKit,
                  BackendProtocol (StateBackend | MemoryBackend | CompositeBackend),
                  List[Agent] subagents (each is itself an Agent)

Step (ABC, pipeline/step.py)
└── 27 concrete Step subclasses (pipeline/steps.py) used by:
    └── PipelineManager (pipeline/manager.py)

PolicyManager + ToolPolicyManager (policy_manager.py, tool_policy_manager.py)
SystemPromptManager / ContextManager / MemoryManager / LLMManager / CallManager /
TaskManager / ReliabilityManager / ContextManagementMiddleware
        (context_managers/*)

AgentOTelManager (otel_manager.py) — composed into every Agent instance
```

---

## 10. Why this structure exists

- **Pipeline composition** isolates each concern in a 100–200 line step. New features get added as a step, not as new branches inside an ever-growing `do()` method. Steps can be skipped (cache hit, policy block, context full) by returning `COMPLETED` early — the rest of the pipeline keeps running.
- **`AgentRunOutput` as the ledger.** Because every step writes to one object, persisting to storage gives you durable execution, HITL resumption, and cross-process cancel for free — there's no hidden state in local variables to lose.
- **Manager registry** keeps step interfaces small. Without it, every step would need 8+ kwargs to receive memory, prompt, LLM, call, task, reliability state.
- **Lazy loading** at every `__init__.py` keeps cold-start cost minimal for users who only need the base `Agent`.
- **OpenTelemetry as a no-op when off.** Every span helper returns `nullcontext(None)` when instrumentation is disabled, so the pipeline never branches on `if otel_enabled`.
- **Specialization via subclass + tool injection.** `AutonomousAgent` and `DeepAgent` add behavior almost entirely by injecting toolkits and a custom system prompt — they don't override pipeline machinery, which keeps them tracking the base agent automatically as it evolves.

