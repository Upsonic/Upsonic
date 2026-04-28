# Adding a New Prebuilt Autonomous Agent

This guide is the canonical reference for shipping a new prebuilt autonomous
agent inside Upsonic. A *prebuilt* is a ready-to-run agent that bundles its
own system prompt, first-message template, and skills as files under
`src/upsonic/prebuilt/<your_agent>/template/`, and exposes a small Python
class that wires those files to
[`PrebuiltAutonomousAgentBase`](../../src/upsonic/prebuilt/prebuilt_agent_base.py).

If you only want to *use* an existing prebuilt, see the README — this
document is for contributors adding new ones.

---

## 1. Mental model

```
┌──────────────────────────────────────────────────────────────────────┐
│ Runtime layer:  PrebuiltAutonomousAgentBase  (clones template into   │
│                                                a per-run workspace,  │
│                                                reads system_prompt   │
│                                                + first_message,      │
│                                                renders, runs)        │
├──────────────────────────────────────────────────────────────────────┤
│ Agent class:    YourAgent(PrebuiltAutonomousAgentBase)               │
│                 ├── AGENT_REPO   = "https://github.com/Upsonic/..."  │
│                 ├── AGENT_FOLDER = "src/upsonic/prebuilt/your/template"│
│                 └── new_<X>(...) — high-level, template-aware API    │
├──────────────────────────────────────────────────────────────────────┤
│ Template:       src/upsonic/prebuilt/<your_agent>/template/          │
│                 ├── system_prompt.md        (required)               │
│                 ├── first_message.md        (required, with {})      │
│                 └── skills/<skill_name>/SKILL.md  (one or more)      │
└──────────────────────────────────────────────────────────────────────┘
```

The base class clones the `template/` folder fresh from GitHub on every
`run()`. That means **the template you ship must live in the public Upsonic
repository on `master`** — local-only edits won't be picked up by the agent
at run time. (You can override `agent_repo` to point at a fork while
developing.)

---

## 2. Canonical file layout

Every prebuilt agent has the exact same shape:

```
src/upsonic/prebuilt/
├── __init__.py                          # re-exports your agent class
├── prebuilt_agent_base.py               # do NOT edit; shared base class
└── <your_agent>/
    ├── __init__.py                      # lazy imports for your sub-package
    ├── agent.py                         # YourAgent + any helper classes
    └── template/
        ├── system_prompt.md
        ├── first_message.md
        └── skills/
            ├── <skill_a>/SKILL.md
            ├── <skill_b>/SKILL.md
            └── ...
```

Use `applied_scientist/` as a working reference whenever this doc is
ambiguous — it is the original prebuilt and is kept canonical.

---

## 3. Step-by-step: shipping a new prebuilt

### 3.1 Pick the folder name

Use a snake-case noun describing the role of the agent — e.g.
`applied_scientist`, `code_reviewer`, `release_manager`. The folder name
becomes the user-facing import path
(`from upsonic.prebuilt import YourAgent`) so prefer something short and
unambiguous.

### 3.2 Author the template files

Create `src/upsonic/prebuilt/<your_agent>/template/` and put the prompts
inside.

**`system_prompt.md`** — full system prompt for the agent. This is rendered
verbatim, then wrapped by the autonomous-agent harness with workspace,
filesystem, and shell instructions. Keep it self-contained; it is what the
model sees on every turn.

**`first_message.md`** — the very first user-style message sent to the
agent. Use Python `str.format` placeholders (`{name}`, `{research_source}`,
…) for any value the caller supplies at run time. The base class extracts
the placeholder set and raises `ValueError` if the caller forgets one, so
you do not need to hand-validate kwargs.

```markdown
# first_message.md
New experiment.

**Experiment name:** {research_name}
**Research source:** {research_source}
**Current notebook:** {current_notebook}
```

**`skills/<skill_name>/SKILL.md`** — Anthropic-style skill files. Each is a
self-contained, scoped capability the agent loads on demand. The base class
copies the entire `skills/` tree into the workspace untouched, so anything
the system prompt references by relative path will be found.

### 3.3 Subclass the base in `agent.py`

```python
# src/upsonic/prebuilt/<your_agent>/agent.py
from __future__ import annotations
from typing import Any, Optional, Union, TYPE_CHECKING

from upsonic.prebuilt.prebuilt_agent_base import PrebuiltAutonomousAgentBase

if TYPE_CHECKING:
    from upsonic.models import Model


class YourAgent(PrebuiltAutonomousAgentBase):
    """Short docstring describing the agent and its high-level API."""

    AGENT_REPO: str = "https://github.com/Upsonic/Upsonic"
    AGENT_FOLDER: str = "src/upsonic/prebuilt/<your_agent>/template"

    def __init__(
        self,
        *,
        model: Union[str, "Model"] = "openai/gpt-4o",
        workspace: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # The base class accepts agent_repo/agent_folder; pin them here so
        # users only have to think about model + workspace.
        kwargs.pop("agent_repo", None)
        kwargs.pop("agent_folder", None)
        super().__init__(
            model=model,
            workspace=workspace,
            agent_repo=self.AGENT_REPO,
            agent_folder=self.AGENT_FOLDER,
            **kwargs,
        )
```

That alone is enough — users can already call
`agent.run(workspace="./ws", **template_params)`.

### 3.4 Add a high-level, template-aware API (optional)

The user should not have to remember the placeholder names in
`first_message.md`. Wrap them in a typed factory method, optionally
returning a small object that defers the actual run:

```python
def new_task(self, name: str, *, source: str, target: str) -> "YourTask":
    return YourTask(
        agent=self,
        template_params={"name": name, "source": source, "target": target},
    )
```

Use the
[`Experiment`](../../src/upsonic/prebuilt/applied_scientist/agent.py) object
in `applied_scientist` as a reference if your agent needs:

- Foreground vs. background runs (`run()` vs. `run_in_background()`).
- Live progress polling that's friendly to Jupyter cells.
- An on-disk record / registry of past runs.

The reason `applied_scientist` exposes a class instead of a thin function
is precisely so that callers in a notebook can keep a handle to a running
experiment, observe its progress bar, and inspect the result later — copy
that pattern only when you need it.

### 3.5 Wire up `__init__.py` files

**`src/upsonic/prebuilt/<your_agent>/__init__.py`** — lazy re-export of the
classes in `agent.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import YourAgent  # plus any helper classes


def _get_classes() -> dict[str, Any]:
    from .agent import YourAgent
    return {"YourAgent": YourAgent}


def __getattr__(name: str) -> Any:
    classes = _get_classes()
    if name in classes:
        return classes[name]
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(classes.keys())}"
    )


__all__ = ["YourAgent"]
```

**`src/upsonic/prebuilt/__init__.py`** — add your agent to the package-level
`_get_classes()` and `__all__` so `from upsonic.prebuilt import YourAgent`
works:

```python
def _get_classes() -> dict[str, Any]:
    from .prebuilt_agent_base import PrebuiltAutonomousAgentBase
    from .applied_scientist.agent import AppliedScientist, Experiment, ExperimentResult
    from .your_agent.agent import YourAgent          # ← new
    return {
        "PrebuiltAutonomousAgentBase": PrebuiltAutonomousAgentBase,
        "AppliedScientist": AppliedScientist,
        "Experiment": Experiment,
        "ExperimentResult": ExperimentResult,
        "YourAgent": YourAgent,                       # ← new
    }


__all__ = [
    "PrebuiltAutonomousAgentBase",
    "AppliedScientist",
    "Experiment",
    "ExperimentResult",
    "YourAgent",                                      # ← new
]
```

Lazy imports keep `import upsonic` cheap; do not move any of these to
top-of-file unconditional imports.

---

## 4. Testing your prebuilt

A minimal smoke test:

```python
from upsonic.prebuilt import YourAgent

agent = YourAgent(model="openai/gpt-4o", workspace="./ws")

# Verify constants and inheritance
assert agent.AGENT_FOLDER == "src/upsonic/prebuilt/your_agent/template"
assert agent.AGENT_REPO.endswith("/Upsonic")

# Run end-to-end (cheap model recommended for CI)
agent.run(source="...", target="...")
```

For local development against an unmerged template, override the repo:

```python
agent = YourAgent(
    model="openai/gpt-4o",
    workspace="./ws",
    agent_repo="https://github.com/<your_fork>/Upsonic",
)
```

The base class always clones fresh — there is no caching layer to bust.

---

## 5. Conventions to follow

- **Pin `AGENT_REPO` to `https://github.com/Upsonic/Upsonic`.** Users
  installing from PyPI must be able to reach the template; private repos
  break that contract.
- **Pin `AGENT_FOLDER` to a path under `src/upsonic/prebuilt/<your_agent>/template`.**
  This keeps every prebuilt's source code and template colocated.
- **Keep `agent.py` self-contained.** A reader landing on
  `src/upsonic/prebuilt/your_agent/agent.py` should see the agent class,
  any companion result/record classes, and nothing else. Do not edit
  `prebuilt_agent_base.py` to add agent-specific behaviour — extend or
  override in your subclass instead.
- **Pass placeholder kwargs through `**template_params`.** The base class
  validates them against the placeholder set extracted from
  `first_message.md`, so a typo in either the template or the call site
  surfaces as a clear `ValueError` listing what's missing.
- **Templates are runtime data, not Python.** No imports, no f-strings;
  treat them as the model's eyes and only use `str.format` placeholders.

---

## 6. Reviewer checklist

Before opening a PR adding `<your_agent>`:

- [ ] `src/upsonic/prebuilt/<your_agent>/__init__.py` lazy-imports from `agent.py`.
- [ ] `src/upsonic/prebuilt/__init__.py` re-exports the agent class and
      lists it in `__all__`.
- [ ] `agent.py` subclasses `PrebuiltAutonomousAgentBase`, sets
      `AGENT_REPO` and `AGENT_FOLDER`, and pops `agent_repo` /
      `agent_folder` from `kwargs` before calling `super().__init__`.
- [ ] `template/system_prompt.md` and `template/first_message.md` exist.
- [ ] Every placeholder in `first_message.md` is documented in the
      docstring of the high-level method that supplies it.
- [ ] Your agent appears in the README and any concept-level docs that
      enumerate the available prebuilts.
- [ ] A smoke test calls `from upsonic.prebuilt import YourAgent` and
      instantiates it (no API calls required).

That's it — once the PR lands on `master`, the next user who runs
`from upsonic.prebuilt import YourAgent` will pull the freshly merged
template at run time.
