---
name: agent-skills
description: Use when working on Upsonic's progressive skill discovery system in `src/upsonic/skills/` — building, loading, validating, caching, or executing SKILL.md packages with scripts/references/assets. Use when a user asks to add a new skill, load skills from a folder, GitHub repo, or URL archive, register skill tools on an Agent or Task, debug dependency cycles, version constraints, path-traversal safety, script execution, or system-prompt skill summaries. Trigger when the user mentions Skill, Skills, SKILL.md, LocalSkills, InlineSkills, BuiltinSkills, GitHubSkills, URLSkills, RemoteSkillLoader, SkillLoader, SkillMetrics, SkillCache, SkillVersion, VersionConstraint, get_skill_instructions, get_skill_reference, get_skill_script, get_skill_asset, allowed-tools, frontmatter, progressive disclosure, builtins (code-review, data-analysis, summarization), or skill dependency resolution.
---

# `src/upsonic/skills/` — Agent Skills (Progressive Tool / Instruction Discovery)

## 1. What this folder is

The `skills` package gives an Upsonic `Agent` (or a single `Task`) **structured
domain expertise** that it can discover and load on demand. A *skill* is a
self-contained package made of:

| Part            | Purpose                                                                 |
| --------------- | ----------------------------------------------------------------------- |
| `SKILL.md`      | Required. YAML frontmatter (`name`, `description`, …) plus instructions |
| `scripts/`      | Optional executable scripts (Python, bash, etc.) the agent can run      |
| `references/`   | Optional documentation files (style guides, cheatsheets, OWASP lists)   |
| `assets/`       | Optional supporting files (templates, fonts, icons, examples)           |

The design follows a **progressive discovery** workflow:

1. The agent receives only the *summaries* of available skills via the system
   prompt (skill name + one-line description + which scripts/references/assets
   exist).
2. When the agent decides a skill is relevant to the current task, it calls a
   tool function (`get_skill_instructions`) to load the full SKILL.md body.
3. As the body suggests, the agent can then call `get_skill_reference`,
   `get_skill_script` (read or execute) or `get_skill_asset` to pull only the
   pieces it actually needs.

This keeps the system prompt small while making a much larger library of
expertise available *just-in-time*. The exact rules the agent follows are
written into the prompt by `Skills.get_system_prompt_section`:

```text
1. Browse  — review the skill summaries below
2. Load    — call get_skill_instructions(skill_name) when a task matches
3. Reference — get_skill_reference for documentation
4. Scripts  — get_skill_script(execute=True/False) for code
```

The package also handles loading from many sources (local FS, inline objects,
built-in library, GitHub repos, generic URL archives), validation against the
Agent Skills spec, dependency resolution, optional embedding-based auto-select,
in-memory TTL caching, per-skill metrics, safety-engine integration, and
versioning with semver-style constraints.

---

## 2. Folder layout

```
src/upsonic/skills/
├── __init__.py              # Public API re-exports
├── skill.py                 # Skill dataclass (pure data container)
├── skills.py                # Skills container — loads, dispatches tools
├── validator.py             # Spec compliance for SKILL.md frontmatter
├── metrics.py               # SkillMetrics dataclass (per-skill counters)
├── version.py               # SkillVersion + VersionConstraint (semver)
├── cache.py                 # SkillCache (in-memory TTL cache)
├── utils.py                 # Path-safety, shebang parsing, run_script
├── dependency.py            # Dep cycle detection + topological sort
├── loader/
│   ├── __init__.py
│   ├── base.py              # SkillLoader ABC
│   ├── local.py             # LocalSkills (filesystem)
│   ├── inline.py            # InlineSkills (programmatic)
│   ├── builtin.py           # BuiltinSkills (ships with Upsonic)
│   ├── remote_base.py       # RemoteSkillLoader (cached download base)
│   ├── github.py            # GitHubSkills (repo tarball)
│   └── url.py               # URLSkills (.tar.gz / .zip)
└── builtins/                # Skills that ship with Upsonic
    ├── __init__.py
    ├── code-review/
    │   ├── SKILL.md
    │   └── references/
    │       ├── owasp-top-10.md
    │       └── severity-guide.md
    ├── data-analysis/
    │   ├── SKILL.md
    │   ├── scripts/profile_data.py
    │   └── references/statistical-tests-guide.md
    └── summarization/
        ├── SKILL.md
        └── references/summary-templates.md
```

---

## 3. Top-level files

### 3.1 `skill.py` — `Skill` data class

A pure data container. Loading and validation are handled elsewhere.

```python
@dataclass
class Skill:
    name: str
    description: str
    instructions: str           # Body of SKILL.md (everything after frontmatter)
    source_path: str            # Filesystem path to the skill folder
    scripts: List[str]          # Filenames inside scripts/
    references: List[str]       # Filenames inside references/
    assets: List[str]           # Filenames inside assets/
    metadata: Optional[Dict]
    license: Optional[str]
    compatibility: Optional[str]
    allowed_tools: Optional[List[str]]
    version: Optional[str]
    dependencies: List[str]
```

Helpers:

| Method        | Purpose                                              |
| ------------- | ---------------------------------------------------- |
| `to_dict()`   | Serialize for storage / cross-process transfer       |
| `from_dict()` | Reconstruct from a dict                              |
| `__repr__`    | Compact, e.g. `Skill(name='code-review', scripts=0…)` |

### 3.2 `skills.py` — `Skills` container

`Skills` is the orchestrator that the rest of Upsonic interacts with. It:

1. **Loads** skills by walking `loaders` (each implementing `SkillLoader`).
   Later loaders override earlier ones on name collision.
2. **Resolves dependencies**: warns or raises (`strict_deps=True`) on missing
   deps and dependency cycles.
3. **Provides accessors**: `get_skill`, `get_all_skills`, `get_skill_names`.
4. **Generates the system-prompt snippet** that lets the LLM browse skills.
5. **Generates the four tool callables** the LLM uses to actually load
   instructions / references / scripts / assets.
6. **Tracks metrics** per skill (`load_count`, `reference_access_count`, …).
7. **Optionally** caches results, applies safety policies, and selects the
   most relevant skills via embeddings.

Constructor signature:

```python
Skills(
    loaders: List[SkillLoader],
    strict_deps: bool = False,
    cache_ttl: Optional[int] = None,        # seconds; enables SkillCache
    on_load: Optional[Callable] = None,
    on_script_execute: Optional[Callable] = None,
    on_reference_access: Optional[Callable] = None,
    auto_select: bool = False,              # filter prompt to relevant skills
    max_skills: int = 5,                    # cap when auto_select is on
    embedding_provider: Optional[Any] = None,
    policy: Optional[Any] = None,           # safety_engine policy or list
)
```

#### `get_system_prompt_section(task_description=None)`

Renders an XML-tagged block listing the available skills along with the names
of their scripts/references/assets so the LLM knows what is callable. With
`auto_select=True` and `embedding_provider` supplied, the snippet is filtered
to the top-`max_skills` most semantically similar skills for that task.

Sample output:

```xml
<skills_system>

## What are Skills?
Skills are packages of domain expertise …

## IMPORTANT: How to Use Skills
1. get_skill_instructions(skill_name) - Load the full instructions
2. get_skill_reference(skill_name, reference_path) - Access documentation
3. get_skill_script(skill_name, script_path, execute=False) - Read or run scripts
4. get_skill_asset(skill_name, asset_path) - Read asset files

## Available Skills
<skill>
  <name>code-review</name>
  <description>Perform structured code reviews …</description>
  <scripts>none</scripts>
  <references>owasp-top-10.md, severity-guide.md</references>
</skill>
<skill>
  <name>data-analysis</name>
  <description>Analyze, explore, clean, and visualize datasets …</description>
  <scripts>profile_data.py</scripts>
  <references>statistical-tests-guide.md</references>
</skill>
…
</skills_system>
```

The result is cached (when `cache_ttl` is set) under
`f"system_prompt:{task_description or ''}"`.

#### `get_tools(prefix="")` — the four tool callables

```python
tools = skills.get_tools()        # plain Python callables
# returns four functions, in order:
# get_skill_instructions, get_skill_reference, get_skill_script, get_skill_asset
```

`prefix` rewrites `__name__` / `__qualname__` so that task-level tools don't
collide with agent-level ones. Upsonic uses `prefix="task_"` for
task-scoped skills (see Section 7).

| Tool                       | Required args               | Optional args                          | Returns (JSON string)                                       |
| -------------------------- | --------------------------- | -------------------------------------- | ----------------------------------------------------------- |
| `get_skill_instructions`   | `skill_name`                | —                                      | `{skill_name, description, instructions, available_scripts, available_references, available_assets, dependencies, version, recommended_tools?}` |
| `get_skill_reference`      | `skill_name, reference_path`| —                                      | `{skill_name, reference_path, content}`                     |
| `get_skill_script`         | `skill_name, script_path`   | `execute=False, args=None, timeout=30` | If `execute=False`: `{skill_name, script_path, content}`. If `execute=True`: `{stdout, stderr, returncode}` |
| `get_skill_asset`          | `skill_name, asset_path`    | —                                      | `{skill_name, asset_path, content}`                         |

Failure modes always return JSON (never raise) so the LLM can recover:

```json
{"error": "Skill 'foo' not found", "available_skills": "code-review, data-analysis, summarization"}
```

Path traversal attempts (`../../etc/passwd`) are rejected by `is_safe_path`
in `utils.py`. Script execution times out after `timeout` seconds and is
captured by `subprocess.run`.

#### `merge(*instances)` — class method

Snapshots multiple `Skills` instances into a new one whose only loader is an
`InlineSkills` containing the union of skills. Later instances win on name
collision. Used by the system-prompt manager when both the agent and the
task carry their own `Skills`.

```python
@classmethod
def merge(cls, *instances: "Skills") -> "Skills":
    combined: Dict[str, Skill] = {}
    for inst in instances:
        combined.update(inst._skills)
    return cls(loaders=[InlineSkills(list(combined.values()))])
```

#### Other helpers on `Skills`

| Method                    | Notes                                                        |
| ------------------------- | ------------------------------------------------------------ |
| `copy()`                  | Shallow copy that shares loaders/skills but **fresh** metrics — used by `Team._propagate_skills` so each agent measures itself |
| `reload()`                | Clears skill dict + cache, re-runs `_load_skills`            |
| `get_metrics()`           | `Dict[str, SkillMetrics]`                                    |
| `get_active_skill_tools()`| Union of `allowed_tools` from skills the agent has actually loaded — useful for tool-binding policies |
| `__len__`, `__contains__` | Convenience                                                  |

### 3.3 `validator.py` — spec enforcement

Implements the public `validate_skill_directory(path)` and
`validate_metadata(meta, skill_dir)` helpers. Returns a list of
human-readable errors (empty = valid).

Rules enforced:

| Field            | Rule                                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------------- |
| `name`           | Non-empty, ≤ 64 chars, lowercase, alphanumeric + hyphens, no leading/trailing/consecutive hyphens, must equal directory name |
| `description`    | Non-empty, ≤ 1024 chars, **must not contain `<` or `>`** (prevents prompt injection)                  |
| `compatibility`  | String, ≤ 500 chars                                                                                   |
| `license`        | String                                                                                                |
| `allowed-tools`  | List of strings                                                                                       |
| `dependencies`   | List of strings                                                                                       |
| `metadata`       | Dict                                                                                                  |
| Frontmatter keys | Only `{name, description, version, license, allowed-tools, metadata, compatibility, dependencies}`    |

If `pyyaml` isn't available it falls back to a tiny `_simple_yaml_parse` so
the validator works on minimal installs.

### 3.4 `metrics.py` — `SkillMetrics`

```python
@dataclass
class SkillMetrics:
    load_count: int = 0
    reference_access_count: int = 0
    script_execution_count: int = 0
    total_chars_loaded: int = 0
    last_used_timestamp: Optional[float] = None
```

Mutated by the `_get_skill_*` methods on `Skills` to give callers visibility
into which skills are actually paying for their place in the prompt.

### 3.5 `version.py` — `SkillVersion` + `VersionConstraint`

Semver-style versions and compound constraints (`>=1.0.0,<2.0.0`):

```python
SkillVersion.parse("1.2.3")            # -> SkillVersion(1, 2, 3)
VersionConstraint(">=1.0,<2").satisfies(SkillVersion(1, 4, 0))  # True
```

Operators supported in each comma-separated segment: `>=`, `<=`, `>`, `<`,
`==`, `!=`. Used by `LocalSkills(version_constraint="…")` to filter the
skills it returns.

### 3.6 `cache.py` — `SkillCache`

A trivial in-memory TTL cache keyed by string. Created by `Skills`
when `cache_ttl` is supplied. Caches:

- `system_prompt:<task_description>` — output of
  `get_system_prompt_section`
- `instructions:<skill_name>` — output of `_get_skill_instructions`

`invalidate()` clears everything (used by `Skills.reload()`).

### 3.7 `utils.py` — paths, shebangs, script execution

| Helper                         | Job                                                                |
| ------------------------------ | ------------------------------------------------------------------ |
| `is_safe_path(base, requested)`| Resolves `requested` and confirms it stays inside `base`            |
| `parse_shebang(script_path)`   | Returns `python3`, `bash`, `node`, … from `#!/usr/bin/env -S node` |
| `get_interpreter_command(name)`| Maps `python*` to `sys.executable` so the venv is preserved        |
| `run_script(...)`              | `subprocess.run` with timeout; on Windows builds the command from the parsed shebang because Windows can't run shebangs natively |
| `read_file_safe(path)`         | UTF-8 read with explicit error surface                             |

`ScriptResult` is a small dataclass `(stdout, stderr, returncode)` returned
by `run_script`.

### 3.8 `dependency.py` — graph algorithms

```python
get_missing_dependencies(skills) -> Dict[name, [missing_dep, …]]
detect_cycles(skills)            -> List[List[name]]   # 3-color DFS
resolve_load_order(skills)       -> List[name]         # Kahn's topological sort
```

`Skills._load_skills` calls `get_missing_dependencies` and `detect_cycles`
after loading. Either is fatal under `strict_deps=True`, otherwise logged as
a warning. `resolve_load_order` is exposed for callers that want to walk
skills in dependency order.

---

## 4. Subfolders walked through

### 4.1 `loader/` — pluggable sources

#### `loader/base.py` — `SkillLoader`

```python
class SkillLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> List[Skill]: ...
```

That's the entire contract. Anything implementing it can plug into
`Skills(loaders=[…])`.

#### `loader/local.py` — `LocalSkills`

The workhorse loader. Accepts a path that is either:

- a **single skill folder** (`SKILL.md` directly inside it), or
- a **parent folder** of skill folders.

It does the real parsing of `SKILL.md`:

1. Optional validation via `validate_skill_directory`.
2. Splits frontmatter (between two `---` lines) from instructions body.
3. Parses YAML with `pyyaml` (falls back to `_parse_simple_frontmatter`).
4. Reads the `version` field from either the top level or `metadata.version`.
5. Lists `scripts/`, `references/`, and `assets/` files (sorted, hidden
   files skipped).
6. Optionally filters by `version_constraint` using `VersionConstraint`.

#### `loader/inline.py` — `InlineSkills`

Wraps an in-memory list of `Skill` objects. Used:

- by user code to register skills programmatically without a filesystem;
- by `Skills.merge` as the snapshot loader.

`validate=True` runs name/description/dependency rules through
`validate_metadata`.

#### `loader/builtin.py` — `BuiltinSkills`

Resolves the path to `upsonic.skills.builtins/` (works in both editable
installs and wheel installs via `importlib.resources.files(...)`), then
delegates to `LocalSkills`. Optional `skills=[…]` filters which built-ins
to expose. `available_skills()` lists all built-in folder names that
contain a `SKILL.md`.

#### `loader/remote_base.py` — `RemoteSkillLoader`

Abstract base for any loader that downloads. Implements:

- a per-source cache directory at
  `~/.upsonic/skills_cache/<loader_name>/<sha256-of-source-key>/`,
- TTL-based freshness via a `.cache_meta.json` file,
- `force_refresh` to bypass the cache,
- delegation to `LocalSkills` after extraction.

Subclasses implement only `_download(target_dir)` and `_source_key()`.

#### `loader/github.py` — `GitHubSkills`

Downloads `https://api.github.com/repos/{owner}/{name}/tarball/{branch}`,
streams it through a `_MAX_DOWNLOAD_SIZE = 100 MB` guard, then extracts only
files under `path/` (default `skills/`). Honors `GITHUB_TOKEN` /
`GH_TOKEN` env vars. Symlinks and `..` traversal entries are skipped.
Optional `skills=[…]` filter restricts to specific skill folders inside
the tarball.

#### `loader/url.py` — `URLSkills`

Same shape, but for arbitrary `.tar.gz`/`.tgz` or `.zip` archives at any
URL. Auto-detects archive type from extension or from the zip magic
number `PK\x03\x04`. Same path-traversal/symlink protections.

### 4.2 `builtins/` — skills that ship with Upsonic

Three pre-built, pre-validated skills:

| Skill           | What it teaches the agent                                                | Scripts            | References                                  |
| --------------- | ------------------------------------------------------------------------ | ------------------ | ------------------------------------------- |
| `code-review`   | Multi-dimensional code review (correctness, security, performance, …)    | —                  | `severity-guide.md`, `owasp-top-10.md`      |
| `data-analysis` | Cleaning, exploring, statistical analysis, A/B tests, communication      | `profile_data.py`  | `statistical-tests-guide.md`                |
| `summarization` | Executive / technical / research / meeting / changelog summarization     | —                  | `summary-templates.md`                      |

Each `SKILL.md` has YAML frontmatter (`name`, `description`,
`metadata.version`, `metadata.author`, `metadata.tags`) followed by
detailed instructions written in the second person ("Read the entire code
before making any comments"). These bodies are what the agent receives
when it calls `get_skill_instructions`.

Example: `data-analysis/SKILL.md` references both its script and its
documentation, telling the LLM exactly which tool call to use:

```text
- Execute `profile_data.py` with a data file path to get a quick profile …
- Load `statistical-tests-guide.md` when choosing statistical tests …
```

The matching script (`builtins/data-analysis/scripts/profile_data.py`) is
a real, runnable Python program with `argparse` CLI — when the agent calls
`get_skill_script("data-analysis", "profile_data.py", execute=True,
args=["data.csv", "--output", "json"])`, `Skills` shells out to it via
`utils.run_script`.

---

## 5. Cross-file relationships

```
                       ┌─────────────────────┐
                       │     Skills          │
 (user code)──loaders──┤  (skills.py)        │
                       │                     │
                       │  • _load_skills() ──┼──► LocalSkills/Inline/Builtin/
                       │  • get_tools() ─────┼──►  GitHubSkills/URLSkills (loader/*.py)
                       │  • get_system_prompt│       │
                       │      _section()     │       └─► validator.validate_skill_directory
                       │  • merge()          │
                       │  • copy()           │
                       └────────┬────────────┘
                                │ uses
                                ▼
            ┌──────────────────────────────────────┐
            │ Skill (skill.py)                     │
            │ SkillMetrics (metrics.py)            │
            │ SkillCache (cache.py)                │
            │ SkillVersion / VersionConstraint      │
            │   (version.py)                       │
            │ utils.is_safe_path / run_script /    │
            │   parse_shebang                      │
            │ dependency.detect_cycles /           │
            │   get_missing_dependencies           │
            └──────────────────────────────────────┘
```

Key edges:

- `Skills.__init__` walks `self.loaders` (each `SkillLoader.load`) → builds
  `self._skills: Dict[str, Skill]`.
- After loading, `Skills` consults `dependency.py` to flag cycles/missing.
- `Skills._get_skill_script` calls `utils.run_script` — the only place the
  package shells out.
- `LocalSkills` (and therefore `BuiltinSkills` / `RemoteSkillLoader`) runs
  `validator.validate_skill_directory` when `validate=True`.
- `LocalSkills` honors `version_constraint` via `version.VersionConstraint`.
- `Skills.merge` builds an `InlineSkills` from a snapshot — the only way to
  express a frozen union of multiple `Skills` instances.
- `Skills` raises `SkillValidationError`, `SkillParseError` and surfaces
  `SkillDownloadError` from remote loaders. All three live in
  `upsonic/utils/package/exception.py` and are re-exported by
  `upsonic.skills`.

---

## 6. Public API

Everything below comes from `from upsonic.skills import …`:

```python
__all__ = [
    "Skill",
    "Skills",
    "SkillLoader",
    "LocalSkills",
    "InlineSkills",
    "BuiltinSkills",
    "RemoteSkillLoader",
    "GitHubSkills",
    "URLSkills",
    "SkillMetrics",
    "SkillError",
    "SkillParseError",
    "SkillValidationError",
]
```

| Name                  | Kind     | Use                                                               |
| --------------------- | -------- | ----------------------------------------------------------------- |
| `Skill`               | class    | Build a skill in code (`InlineSkills([Skill(...)])`)              |
| `Skills`              | class    | Container passed to `Agent(skills=...)` / `Task(skills=...)`      |
| `SkillLoader`         | ABC      | Subclass to add a custom source                                   |
| `LocalSkills`         | class    | Filesystem (single folder or parent folder)                       |
| `InlineSkills`        | class    | Programmatic / cross-process snapshot                             |
| `BuiltinSkills`       | class    | Load skills shipped with Upsonic                                  |
| `RemoteSkillLoader`   | ABC      | Cached-download base for new remote loaders                       |
| `GitHubSkills`        | class    | Download a skills folder from a GitHub repo                       |
| `URLSkills`           | class    | Download a `.tar.gz`/`.zip` archive of skills                     |
| `SkillMetrics`        | class    | Read counters from `Skills.get_metrics()`                         |
| `SkillError`          | exception| Generic skill-related error                                       |
| `SkillParseError`     | exception| `SKILL.md` malformed                                              |
| `SkillValidationError`| exception| Frontmatter / dependency / cycle problems                         |

Internal-but-callable extras (not in `__all__`):

- `upsonic.skills.validator.validate_skill_directory(path)`
- `upsonic.skills.dependency.{get_missing_dependencies, detect_cycles, resolve_load_order}`
- `upsonic.skills.version.{SkillVersion, VersionConstraint}`
- `upsonic.skills.cache.SkillCache`
- `upsonic.skills.utils.{is_safe_path, run_script, parse_shebang, …}`

---

## 7. Integration with the rest of Upsonic

### 7.1 `Agent.skills`

In `src/upsonic/agent/agent.py`:

```python
class Agent:
    def __init__(self, ..., skills: Optional["Skills"] = None, ...):
        ...
        self.skills = skills

        # Register skill tools if skills are provided
        if self.skills is not None:
            self.tools.extend(self.skills.get_tools())
```

So at construction time the four skill tools (`get_skill_instructions` …
`get_skill_asset`) get appended to the agent's tool list, exactly like any
other tool. `Agent.skill_metrics()` exposes `Skills.get_metrics()`.

### 7.2 `Task.skills`

In `src/upsonic/tasks/tasks.py`, `Task.__init__` accepts an optional
`skills`. At run time, `Agent` registers task-level skill tools with a
prefix:

```python
# agent.py, around line 2101
if hasattr(task, 'skills') and task.skills is not None:
    tools_to_register.extend(task.skills.get_tools(prefix="task_"))
```

This lets the same agent expose **two** independent skill libraries (its
own and the task's) without name collisions — the task-level versions are
named `task_get_skill_instructions`, `task_get_skill_reference`, etc.
Tasks also serialize their skills via cloudpickle in `Task._pickle` /
`_unpickle` so they survive cross-process scheduling.

### 7.3 System prompt construction

`src/upsonic/agent/context_managers/system_prompt_manager.py` decides
which skill summary section to inject:

```python
agent_skills = getattr(self.agent, 'skills', None)
task_skills  = getattr(self.task,  'skills', None) if self.task else None

if agent_skills is not None and task_skills is not None:
    from upsonic.skills import Skills
    merged = Skills.merge(agent_skills, task_skills)   # task overrides agent
    skills_section = merged.get_system_prompt_section()
elif agent_skills is not None:
    skills_section = agent_skills.get_system_prompt_section()
else:
    skills_section = task_skills.get_system_prompt_section()

if skills_section:
    prompt_parts.append(skills_section)
```

That is how the `<skills_system>` block lands in every system prompt.

### 7.4 `Team` propagation

`src/upsonic/team/team.py` walks the team graph in `_propagate_skills`:

- Each `Agent` entity gets its **own** `Skills.copy()` (so per-agent
  metrics are independent).
- If an entity already has skills, the team merges via `Skills.merge` (the
  entity's skills win on conflict).
- `entity.add_tools(skills.get_tools())` re-registers the four tools on
  every receiving agent.
- Sub-`Team`s recurse.

### 7.5 Autonomous agent

`src/upsonic/agent/autonomous_agent/autonomous_agent.py` passes its
`skills` argument straight through to the underlying `Agent` constructor
(line 330: `skills=skills`), so prebuilt autonomous agents inherit the
same progressive-discovery flow.

---

## 8. End-to-end flow

### 8.1 Setting up an agent with skills

```python
from upsonic import Agent, Task
from upsonic.skills import (
    Skills, LocalSkills, BuiltinSkills, GitHubSkills, InlineSkills, Skill
)

skills = Skills(
    loaders=[
        BuiltinSkills(),                                  # ships with Upsonic
        LocalSkills("/etc/team-skills"),                  # shared FS
        LocalSkills("./project-skills",                   # project-specific
                    version_constraint=">=1.0.0,<2.0.0"),
        GitHubSkills(repo="acme/skills-library",
                     branch="main", path="skills/"),
        InlineSkills([                                    # programmatic
            Skill(name="hello",
                  description="Say hi politely.",
                  instructions="Always greet the user by name.",
                  source_path=""),
        ]),
    ],
    cache_ttl=600,            # cache instructions for 10 min
    strict_deps=False,
    auto_select=False,        # set True + embedding_provider for filtering
)

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    skills=skills,
)
```

### 8.2 What the agent sees in its system prompt

The container injects (via `get_system_prompt_section`) a
`<skills_system>` block listing every loaded skill's summary, available
scripts, references and assets. The block also contains the rules for
*how* to use the skill tools and a 4-step "progressive discovery
workflow".

### 8.3 What happens when the LLM picks a skill

1. **Browse** — model reads the summary block.
2. **Load** — model emits a tool call:

   ```json
   {"tool": "get_skill_instructions", "args": {"skill_name": "data-analysis"}}
   ```

   `Skills._get_skill_instructions`:
   - Checks `SkillCache` (key `instructions:data-analysis`).
   - Looks up the `Skill` object.
   - Adds the name to `self._active_skills`.
   - Runs the safety policies (`_validate_content`) against the
     instructions body. If blocked, returns
     `{"error": "Content blocked by policy: …"}` instead.
   - Records `metrics.record_load(chars=len(result))`.
   - Fires `on_load(name, description)` callback if configured.
   - Caches and returns the JSON string.

3. **Reference** — model decides it needs the stats decision matrix:

   ```json
   {"tool": "get_skill_reference",
    "args": {"skill_name": "data-analysis",
             "reference_path": "statistical-tests-guide.md"}}
   ```

   Verified via `is_safe_path` against `<source_path>/references/`,
   read with `read_file_safe`, policy-checked, metrics updated, returned
   as JSON.

4. **Script** — model calls the bundled profiler:

   ```json
   {"tool": "get_skill_script",
    "args": {"skill_name": "data-analysis",
             "script_path": "profile_data.py",
             "execute": true,
             "args": ["data.csv", "--output", "json"],
             "timeout": 30}}
   ```

   `_get_skill_script`:
   - Validates `script_path` is in `skill.scripts`.
   - Resolves under `<source_path>/scripts/` via `is_safe_path`.
   - Calls `utils.run_script` — which parses the shebang
     (`#!/usr/bin/env python3`), maps `python3` to `sys.executable` so the
     current venv is used, and runs `subprocess.run(..., timeout=30,
     cwd=skill.source_path)`.
   - Returns `{stdout, stderr, returncode}` as JSON.
   - Bumps `metrics.script_execution_count`.

5. **Asset** — for assets the model calls
   `get_skill_asset(skill_name, asset_path)` and gets back the file
   contents (templates, fonts, sample CSVs, …).

### 8.4 Cleanup, observability, error paths

- `agent.skill_metrics()` returns a `Dict[str, dict]` of counters per
  skill, useful for telemetry and skill curation ("which skills earn
  their place in the prompt").
- `Skills.get_active_skill_tools()` returns the union of `allowed_tools`
  declared by skills the model actually loaded, letting the rest of
  Upsonic gate other tools on skill activation.
- All errors (skill-not-found, invalid path, traversal attempt, timeout,
  policy block, missing interpreter) are returned as JSON strings rather
  than raised — the LLM keeps the conversation flowing and can recover.
- `Skills.reload()` invalidates the cache and re-runs the loaders, useful
  when skills on disk are edited at runtime.
- Remote loaders cache on disk under
  `~/.upsonic/skills_cache/<loader>/<hash>/`, gated by `cache_ttl` on the
  loader (independent from `Skills(cache_ttl=…)` which is the in-memory
  prompt cache).

The net effect: an Upsonic agent ends up with a *library* of structured
expertise it can dip into selectively, paying the prompt-token cost only
for the skills it actually decides to use, while the framework keeps
loading, validation, sandboxing, caching, and observability out of the
agent author's way.
