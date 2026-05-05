---
name: upsonic-cli
description: Use when working on the `upsonic` command-line interface, its dispatcher, command modules, or shell scaffolding for Upsonic agent projects. Use when a user asks to add a CLI subcommand, fix `upsonic init`/`add`/`remove`/`install`/`run`/`zip`, debug FastAPI/uvicorn boot in `run_command`, adjust the OpenAPI schema customizer, tweak Rich-based help/printer output, or modify `upsonic_configs.json` handling. Trigger when the user mentions upsonic CLI, `upsonic init`, `upsonic run`, `upsonic install`, `upsonic add`, `upsonic remove`, `upsonic zip`, `_COMMAND_HANDLERS`, `init_command`, `run_command`, `install_command`, `add_command`, `remove_command`, `zip_command`, `load_config`, `install_dependencies`, `get_fastapi_imports`, `modify_openapi_schema`, `InterfaceManager`, `upsonic_configs.json`, console_script entry point, lazy imports, Rich printer, `cli/main.py`, `cli/printer.py`, or `commands/shared/`.
---

# `src/upsonic/cli/` — The Upsonic Command-Line Interface

This document is a deep technical reference to the `upsonic` CLI module. It is intended for contributors who need to understand, extend, or debug the behaviour of the command line surface that ships with the Upsonic framework.

## 1. What this folder is — its role in Upsonic

`src/upsonic/cli/` packages the implementation of the `upsonic` shell command that gets installed by Upsonic's `pyproject.toml`:

```toml
[project.scripts]
upsonic = "upsonic.cli.main:main"
```

After `uv sync` (or `pip install`) the user has an `upsonic` binary on their PATH. Invoking it eventually reaches `upsonic.cli.main:main`, which is the dispatcher implemented in this folder.

The CLI exists to give users a project-bootstrap and runtime workflow that mirrors the conventions of frameworks like `npm`, `cargo`, or `poetry`, but oriented around an *Upsonic agent project*. Concretely, it lets a user:

| Step | Command | What it accomplishes |
|------|---------|----------------------|
| 1 | `upsonic init` | Scaffolds an agent project (`main.py` + `upsonic_configs.json`). |
| 2 | `upsonic add <lib> <section>` | Records a new dependency in `upsonic_configs.json`. |
| 3 | `upsonic remove <lib> <section>` | Removes a dependency. |
| 4 | `upsonic install [section]` | Materializes the recorded dependencies via `uv` or `pip`. |
| 5 | `upsonic run [--host --port]` | Boots the agent as a FastAPI service (or `InterfaceManager` deployment). |
| 6 | `upsonic zip [filename]` | Zips up the project for sharing/backup. |

Two architectural ideas dominate the folder:

1. **Lazy imports everywhere.** The module is engineered so that simply running `upsonic --help` or `upsonic` without arguments does *not* import FastAPI, uvicorn, or even Rich until they are needed. This matters because the rest of Upsonic transitively imports a lot — the CLI's snappy startup is intentional.
2. **A flat, declarative dispatch table.** `main.py` does no `argparse` work. It uses a single dict (`_COMMAND_HANDLERS`) for O(1) command lookup and forwards `sys.argv` slices to small handler closures.

Everything else in the folder (printers, command modules, shared helpers) is downstream of those two ideas.

## 2. Folder layout

```text
src/upsonic/cli/
├── __init__.py                    # re-exports `main`
├── main.py                        # entry point + command dispatcher
├── printer.py                     # Rich-based pretty printing (lazy)
└── commands/
    ├── __init__.py                # facade re-exporting all *_command()
    ├── add/
    │   ├── __init__.py
    │   └── command.py             # add_command(library, section)
    ├── init/
    │   ├── __init__.py
    │   └── command.py             # init_command()
    ├── install/
    │   ├── __init__.py
    │   └── command.py             # install_command(section=None)
    ├── remove/
    │   ├── __init__.py
    │   └── command.py             # remove_command(library, section)
    ├── run/
    │   ├── __init__.py
    │   └── command.py             # run_command(host, port)
    ├── zip/
    │   ├── __init__.py
    │   └── command.py             # zip_command(output_file=None)
    └── shared/
        ├── __init__.py            # (empty marker)
        ├── config.py              # cached load_config() helper
        ├── dependencies.py        # uv→pip install bridge
        ├── fastapi_imports.py     # lazy FastAPI/uvicorn loader
        └── openapi.py             # OpenAPI schema customizer
```

Each command is its own subpackage (`commands/<name>/command.py`). The `__init__.py` files inside the command folders are intentionally trivial — the canonical import path is `upsonic.cli.commands.<name>.command:<name>_command`. The umbrella `commands/__init__.py` re-exports the public functions for backward compatibility with any code that previously imported `from upsonic.cli.commands import init_command`.

## 3. Top-level files — file-by-file walkthrough

### 3.1 `src/upsonic/cli/__init__.py`

A thin facade that hoists the entry point so `from upsonic.cli import main` works. Nothing else lives here.

```python
from upsonic.cli.main import main
__all__ = ["main"]
```

### 3.2 `src/upsonic/cli/main.py`

The brains of the CLI. Three concepts coexist here:

1. A static dispatch table — `_COMMAND_HANDLERS` — that maps argv\[0] to a handler.
2. Per-command `_handle_*` functions that perform argument parsing and *only then* import the real implementation (lazy imports).
3. The public `main(args)` entry point.

#### `_COMMAND_HANDLERS`

```python
_COMMAND_HANDLERS = {
    'init':    lambda args: _handle_init(args),
    'add':     lambda args: _handle_add(args),
    'remove':  lambda args: _handle_remove(args),
    'install': lambda args: _handle_install(args),
    'run':     lambda args: _handle_run(args),
    'zip':     lambda args: _handle_zip(args),
}
```

Lookup is via `_COMMAND_HANDLERS.get(command)` so unknown commands fall through to the unknown-command printer. There is no global `argparse.ArgumentParser` — each handler does the minimal parsing for its own flags.

#### Per-command handlers

| Function | Min args | Behaviour |
|----------|----------|-----------|
| `_handle_init(args)` | 1 | If `--help/-h` is the second arg, prints `print_help_init()` and returns 0. Otherwise calls `init_command()`. |
| `_handle_add(args)` | 3 | Validates `args[1]` (library) and `args[2]` (section); errors with usage hint otherwise. Forwards to `add_command(library, section)`. |
| `_handle_remove(args)` | 3 | Same shape as `_handle_add` but routes to `remove_command`. |
| `_handle_install(args)` | 1–2 | Optional `args[1]` is the section; calls `install_command(section)`. |
| `_handle_run(args)` | 1+ | Custom while-loop parser that walks args looking for `--host <h>` and `--port <p>`. Defaults `host="0.0.0.0"` and `port=8000`. Validates that `--port` is an int; on failure prints an error and returns 1. |
| `_handle_zip(args)` | 1–2 | Optional `args[1]` becomes `output_file`; calls `zip_command(output_file)`. |

Every handler honours `--help` / `-h` *before* importing the underlying command so help is essentially free in startup time.

#### `main(args: Optional[list[str]] = None) -> int`

```python
def main(args=None):
    if args is None:
        args = sys.argv[1:]
    if not args:
        from upsonic.cli.printer import print_usage
        print_usage(); return 0
    if args[0] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_general
        print_help_general(); return 0
    command = args[0]
    handler = _COMMAND_HANDLERS.get(command)
    if handler:
        return handler(args)
    from upsonic.cli.printer import print_unknown_command
    print_unknown_command(command); return 1
```

Important details:

- `main()` is callable from Python (tests pass `args=[...]`) and from the shell (default uses `sys.argv[1:]`).
- It returns an int exit code, and `pyproject.toml`'s console script picks that up.
- All printer imports are deferred to keep the no-args / `--help` path featherweight.

### 3.3 `src/upsonic/cli/printer.py`

A self-contained presentation layer built on the `rich` library. The module opts to delay importing Rich until the first call by stashing imports in a module-level `_RICH_IMPORTS` dict.

#### `_get_rich_imports()`

Memoizes a dict of references — `Console`, `Panel`, `Prompt`, `Confirm`, `escape`, `Table`, `Text`, `Style`, `box`, plus a forced-terminal `Console` instance under the `console` key. After the first call the dict is reused on every subsequent call. This is the single hottest path in the printer module.

#### `_escape_rich_markup(text)`

Wraps `rich.markup.escape` so user-supplied strings (paths, library names) cannot be misinterpreted as Rich tags.

#### Banner & usage

| Function | Output |
|----------|--------|
| `print_banner()` | Hand-crafted ANSI escape codes (no Rich) draw the `UPSONIC CLI` ASCII banner in bold green/blue. Used by both `print_usage` and `print_help_general`. |
| `print_usage()` | Banner + Rich `Table` listing the six commands. Triggered when the user runs `upsonic` with no args. |
| `print_help_general()` | Same content but with a `padding=(1,2)` panel and trailing usage hints. Triggered by `upsonic --help`. |
| `print_unknown_command(command)` | Red panel plus available-commands hint. |

#### Per-command help

Each command has a dedicated formatter:

| Function | Triggered by |
|----------|--------------|
| `print_help_init` | `upsonic init --help` |
| `print_help_add` | `upsonic add --help` |
| `print_help_remove` | `upsonic remove --help` |
| `print_help_install` | `upsonic install --help` |
| `print_help_run` | `upsonic run --help` |
| `print_help_zip` | `upsonic zip --help` |

These are pure presentation — they only read the static help strings baked into the source.

#### User-flow helpers

| Function | Purpose |
|----------|---------|
| `prompt_agent_name()` | Wraps `rich.Prompt.ask` to collect the agent name. Used only by `init_command()`. |
| `confirm_overwrite(file_path)` | Yellow warning panel + `rich.Confirm.ask` (defaulting to `False`). |
| `print_cancelled()` | Yellow panel saying the user cancelled. Used by every command's `KeyboardInterrupt` branch. |

#### Status / outcome helpers

| Function | Visual style |
|----------|--------------|
| `print_error(msg)` | Red bordered panel titled `❌ Error`. |
| `print_success(msg)` | Green panel titled `✅ Success`. |
| `print_info(msg)` | Single-line `ℹ` cyan info marker. |
| `print_file_created(path)` | Green check + cyan path. |
| `print_init_success(name, files)` | Big finishing banner with a `Table` listing all created files. |
| `print_dependency_added(lib, section)` | Green panel for `add_command`. |
| `print_dependency_removed(lib, section)` | Green panel for `remove_command`. |
| `print_config_not_found()` | Red panel pointing to `upsonic init`. |
| `print_invalid_section(section, available)` | Red panel listing valid sections. |

The printer module owns *all* user-facing strings; commands never `print()` directly except for the FastAPI server banner in `run_command`.

## 4. Subfolders — same treatment

### 4.1 `commands/__init__.py`

Acts as a public facade re-exporting all command functions:

```python
from upsonic.cli.commands.init.command    import init_command
from upsonic.cli.commands.add.command     import add_command
from upsonic.cli.commands.remove.command  import remove_command
from upsonic.cli.commands.install.command import install_command
from upsonic.cli.commands.run.command     import run_command
from upsonic.cli.commands.zip.command     import zip_command
```

This means every `_handle_*` in `main.py` can simply do `from upsonic.cli.commands import init_command` — but because of how `from upsonic.cli.commands import X` works, that statement still has to import all six. The current handlers in `main.py` therefore import `command` directly via the namespace (`from upsonic.cli.commands import init_command`), which is good enough because each command's `command.py` still defers its own heavy imports (FastAPI, rich, etc.).

### 4.2 `commands/init/command.py`

```python
def init_command() -> int
```

Responsibilities:

1. Lazy-import the printer helpers (`prompt_agent_name`, `print_error`, `print_file_created`, `confirm_overwrite`, `print_cancelled`, `print_init_success`).
2. `agent_name = prompt_agent_name()`. If empty, error and return 1.
3. Resolve `current_dir = Path.cwd()`.
4. Compute target paths: `main.py` and `upsonic_configs.json`.
5. For each, if it exists, ask `confirm_overwrite()`; on a `False` return print cancelled and exit 1.
6. Write a hard-coded `main.py` template:

   ```python
   from upsonic import Task, Agent

   async def main(inputs):
       user_query = inputs.get("user_query")
       answering_task = Task(f"Answer the user question {user_query}")
       agent = Agent()
       result = await agent.print_do_async(answering_task)
       return {"bot_response": result}
   ```
7. Write a richly-shaped `upsonic_configs.json` containing:
   - `envinroment_variables` (sic — note the typo is intentional in the file) with sample knobs (`UPSONIC_WORKERS_AMOUNT`, `API_WORKERS`, `RUNNER_CONCURRENCY`, `NEW_FEATURE_FLAG`).
   - `machine_spec` — `cpu`, `memory`, `storage`.
   - `agent_name`, `description`, `icon`, `language`.
   - `streamlit: false`, `proxy_agent: false`.
   - `dependencies` with three sections: `api`, `streamlit`, `development`.
   - `entrypoints.api_file` = `main.py`, `entrypoints.streamlit_file` = `streamlit_app.py`.
   - `input_schema.inputs.user_query` (string, required).
   - `output_schema.bot_response` (string).
8. Print `print_init_success(agent_name, [main_py_path, config_json_path])`.

Errors are caught in two layers: `KeyboardInterrupt` calls `print_cancelled` and returns 1; any other exception is funnelled through `print_error(f"An error occurred: {e}")` and returns 1.

### 4.3 `commands/add/command.py`

`add_command(library: str, section: str) -> int` mutates `upsonic_configs.json`:

1. If `upsonic_configs.json` does not exist → `print_config_not_found()` and return 1.
2. Load with `load_config(path, use_cache=False)` — caching is suppressed because we are about to write.
3. If the JSON failed to parse, error and return 1.
4. Validate `dependencies` exists.
5. Validate `section` is a key of `dependencies`; otherwise call `print_invalid_section`.
6. If the exact `library` string is already in `dependencies[section]`, treat it as a duplicate and error (this is a *string* match — `requests==1` and `requests==2` are considered different).
7. Append and serialize back with `json.dump(..., indent=4, ensure_ascii=False)`.
8. `print_dependency_added(library, section)`.

### 4.4 `commands/remove/command.py`

`remove_command(library, section)` is more sophisticated than `add_command` because PEP 440 specs allow a thousand syntactic variants (`requests`, `requests==2.31.0`, `requests[security]`, etc.).

#### `get_package_name(dependency_str: str) -> str`

Canonicalises a dependency to its bare package name (lower-cased) by splitting on the first occurrence of any of `==`, `>=`, `<=`, `>`, `<`, `~=`, `!=`, `[`, `;`. Examples:

| Input | Output |
|-------|--------|
| `"requests==2.31.0"` | `"requests"` |
| `"upsonic[storage]"` | `"upsonic"` |
| `"FastAPI>=0.115.12"` | `"fastapi"` |

#### Removal algorithm

1. Verify the config file exists / parses (same prelude as `add_command`).
2. Two-pass match:
   - **Pass 1 — exact match.** Iterate `section_deps`; lower-case compare against `target_lib_lower`.
   - **Pass 2 — package-name match.** Iterate again, comparing the *normalised* package name from `get_package_name(dep)` to `get_package_name(library)`.
3. The first match wins; if neither pass finds anything → `print_error(f"Dependency '{library}' not found in dependencies.{section}")`.
4. Mutate the list in place and write the file back.
5. `print_dependency_removed(to_remove, section)` — note this prints the *actual* string that lived in the config, not the user's input, so the user sees what was really removed.

This means `upsonic remove requests api` will remove `requests==2.31.0` if that's what's in the config.

### 4.5 `commands/install/command.py`

`install_command(section: Optional[str] = None) -> int` is essentially the marriage of `load_config` and `install_dependencies`.

Decision logic for which sections to install:

| `section` argument | Sections installed |
|--------------------|--------------------|
| `None` or `"api"` | `["api"]` |
| `"all"` | `list(all_dependencies.keys())` (every section) |
| anything else | `[section]` |

After resolving sections it validates each name is present in the loaded `dependencies` dict, and if not, calls `print_invalid_section`. It then flattens all dependencies via `extend()` and forwards the unified list to `install_dependencies(...)`. Empty list → `print_info("No dependencies to install"); return 0`.

This command is the only one that uses the read-cache (`use_cache=True`), because it never writes back.

### 4.6 `commands/run/command.py`

The most elaborate command. Two start-up modes:

#### Mode A — Interface mode

1. Statically scan the entrypoint source for the literal string `"InterfaceManager"` via `_is_interface_mode(source_path)`. This is intentionally a *grep*-like check — it does not import anything and has no side effects, so we don't accidentally launch the user's app just to detect mode.
2. If matched, call the user's `main` with empty inputs (`main({})`). Async functions are run with `asyncio.run`. The expected return is an `InterfaceManager` instance from `upsonic.interfaces.manager`.
3. If a valid manager is returned, list the registered interfaces (`iface.get_name()` for each `iface in manager.interfaces`) and call `interface_manager.serve(host=host, port=port)`. `KeyboardInterrupt` ends gracefully.

`_resolve_interface_manager(main_func)` is the helper that performs the call and isinstance-check; it returns `None` on any failure, which simply falls through to Mode B.

#### Mode B — FastAPI mode

If interface detection fails, `run_command` builds a FastAPI app on the fly:

1. Load FastAPI dependencies via the lazy `get_fastapi_imports()`. If unavailable, instruct the user to run `upsonic install` and return 1.
2. Read `input_schema.inputs` and `output_schema` from the config; flatten the input dict into a list of `{"name", "type", "required", "default", "description"}` entries.
3. Construct `FastAPI(title=f"{agent_name} - Upsonic", description=description, version="0.1.0")`.
4. Register a single endpoint:

   ```python
   @app.post("/call", summary="Call Main",
             operation_id="call_main_call_post", tags=["jobs"])
   async def call_endpoint_unified(request: Request): ...
   ```

   It branches on `Content-Type`:
   - `application/json` → `await request.json()`.
   - `multipart/form-data` → `await request.form()`; `UploadFile`-like values are read with `await value.read()`.
   - anything else → fallback to `request.form()` filtered to non-`None` values.

   It then dispatches the user's `main` (sync or async) with the parsed `inputs` dict and returns a `JSONResponse` with the result. On any exception it returns a 500 with `{"error": str(e), "type": type(e).__name__}`.
5. Wrap `app.openapi` with a custom builder that calls `modify_openapi_schema(...)` from `commands/shared/openapi.py` so the schema understands both content types.
6. Print server URLs (`http://<host>:<port>`, `http://<host>:<port>/docs`) and run `uvicorn.run(app, host=host, port=port, log_level="info")`. `KeyboardInterrupt` is caught at multiple levels for clean shutdown messages.

#### Module loading mechanics

`run_command` is also responsible for *importing the user's project*. It does the following before anything else:

| Step | Code |
|------|------|
| Inject the agent's directory into `sys.path` | `sys.path.insert(0, str(agent_dir))` |
| Inject the project root into `sys.path` | `sys.path.insert(0, str(project_root))` |
| Compute the would-be Python package path | `module_package = ".".join(relative_path.parts[:-1])` |
| Build a `ModuleSpec` from the file location | `importlib.util.spec_from_file_location("main", agent_py_path)` |
| Set `agent_module.__package__` and `__name__ = "main"` | so relative imports inside the user's `main.py` work |
| Register in `sys.modules["main"]` and execute | `spec.loader.exec_module(agent_module)` |

This is critical — it is what allows users to put their agent next to a sibling package and still have intra-project relative imports resolve.

#### Key helpers within `run/command.py`

| Helper | Purpose |
|--------|---------|
| `_is_interface_mode(source_path)` | Pure source string check for `"InterfaceManager"`. No execution. |
| `_resolve_interface_manager(main_func)` | Calls `main({})` (sync or async) and checks `isinstance(result, InterfaceManager)`. Returns `None` on failure. |

### 4.7 `commands/zip/command.py`

`zip_command(output_file: Optional[str] = None)`:

1. Default file name → `f"upsonic_context_{YYYYMMDD_HHMMSS}.zip"`.
2. If user-supplied name lacks `.zip` suffix, append it.
3. `Path.cwd().rglob('*')` walks every entry; skips directories and the in-progress output zip itself.
4. Tracks `total_size` for reporting. Per-file `PermissionError`/`OSError` causes a `print_info` warning but does not abort.
5. If no files were collected → `print_error("No files found...")` and return 1.
6. Open `zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED)`; write each file with its `relative_to(current_dir)` arcname so the archive preserves the relative tree. Failures per-file are warned, not fatal.
7. Print final stats — archive path, archive size in MB, file count.

`KeyboardInterrupt` → cancel; other exceptions print error + traceback for debuggability.

### 4.8 `commands/shared/`

A grab-bag of helpers shared between `add`, `remove`, `install`, and `run`.

#### 4.8.1 `shared/__init__.py`

Empty marker file — the directory is a package, but nothing is re-exported.

#### 4.8.2 `shared/config.py`

```python
def load_config(config_path: Path, use_cache: bool = True) -> Optional[Dict[str, Any]]
```

A small, mtime-aware in-process cache for `upsonic_configs.json`. `_CONFIG_CACHE` keys are absolute path strings and values are `(mtime, parsed_dict)` tuples.

Behaviour:

| `use_cache` | File mtime matches cache? | Result |
|-------------|---------------------------|--------|
| `True` | yes | Returns cached dict (no I/O). |
| `True` | no / cache miss | Reads + parses, then caches. |
| `False` | n/a | Always re-reads, never caches. |

`json.JSONDecodeError` and `FileNotFoundError` both yield `None` so the caller can decide the user-facing behaviour. Importantly, *write*-style commands (`add_command`, `remove_command`) call `load_config(..., use_cache=False)` to avoid stale reads.

#### 4.8.3 `shared/dependencies.py`

```python
def install_dependencies(dependencies: list[str], quiet: bool = False) -> bool
def _ensure_pip_available() -> bool
```

The dependency installer. Strategy:

1. If `dependencies` is empty → `True` immediately.
2. Try `uv add <deps...>`. If `uv` isn't on PATH (`FileNotFoundError`) or the command returns non-zero, fall through.
3. Make sure `pip` is available via `_ensure_pip_available()`, which:
   - Calls `python -m pip --version`.
   - If that fails, runs `python -m ensurepip --upgrade`, then re-verifies.
4. Run `python -m pip install <deps...>`.
5. Each branch reports via the printer (`print_info`, `print_success`, `print_error`) unless `quiet=True`.

Subprocess invocations always use `subprocess.run(..., capture_output=True, text=True, check=False)` so failures bubble up via return codes / `result.stderr` rather than raising.

#### 4.8.4 `shared/fastapi_imports.py`

```python
def get_fastapi_imports() -> Optional[dict]
```

Mirrors `printer._get_rich_imports`. Memoizes a dict containing `FastAPI`, `Request`, `JSONResponse`, and `uvicorn` modules. On `ImportError` it returns `None`, signalling to the caller that the user has not installed the FastAPI extras yet (they should run `upsonic install`).

#### 4.8.5 `shared/openapi.py`

Three pure functions that translate the user's input/output schema into a real OpenAPI 3.x schema fragment:

##### `map_inputs_props(inputs_schema)`

Returns `(json_props, multipart_props, required_fields)`. Each input field is mapped into two OpenAPI property definitions because the run server accepts both `application/json` and `multipart/form-data`. The translation table:

| `type` in config | JSON property | Multipart property |
|------------------|---------------|--------------------|
| `files` | `array<string>` | `array<binary>` |
| `file`, `binary`, `string($binary)` | `string` | `string($binary)` |
| `number` | `number` | `number` |
| `integer` | `integer` | `integer` |
| `boolean` / `bool` | `boolean` | `boolean` |
| `list` / `array` | `array<string>` | `array<string>` |
| `json` | `object` | `object` |
| anything else | `string` | `string` |

Defaults (`item["default"]`) are written into `json_props` when present. `required: True` fields are added to `required_fields`.

##### `map_output_props(output_schema)`

A simpler one-direction version that produces only OpenAPI properties for the response model (`JobStatus`). It honours the same type vocabulary plus an `object` synonym for `json`.

##### `modify_openapi_schema(schema, inputs_schema, output_schema_dict, path="/call")`

Surgery on a FastAPI-generated OpenAPI doc:

1. Skip if `path` (default `/call`) isn't in `paths`.
2. Compute json/multipart/required via `map_inputs_props`.
3. Build `RequestModel` (object with `data` -> `properties=json_props`).
4. Add `RequestModel` and `JobStatus` to `components.schemas`. `JobStatus.properties = map_output_props(output_schema_dict)`.
5. Replace `requestBody.content` with a dict that lists `multipart/form-data` *first* and then `application/json` (Swagger UI picks the first by default).
6. Set `responses["200"]` to `$ref` `JobStatus`.
7. Ensure the operation has the `jobs` tag.
8. Return the mutated schema (also mutated in place).

This is invoked from `run_command`'s `custom_openapi()` wrapper.

## 5. Cross-file relationships

```text
                        ┌──────────────────────────────────────────────┐
                        │                upsonic.cli.main              │
                        │  _COMMAND_HANDLERS  →  _handle_*  →  *_command│
                        └──────────────────────────────────────────────┘
                                 │            │            │
                                 ▼            ▼            ▼
                ┌────────────────────────┐    │    ┌───────────────────┐
                │  upsonic.cli.printer   │◀───┘    │ commands/<name>/  │
                │  (Rich, lazy-loaded)   │         │  command.py       │
                └────────────────────────┘         └────────┬──────────┘
                                                            │ shared
                                                            ▼
                                         ┌──────────────────────────────┐
                                         │   commands/shared/           │
                                         │   ├── config.py              │
                                         │   ├── dependencies.py        │
                                         │   ├── fastapi_imports.py     │
                                         │   └── openapi.py             │
                                         └──────────────────────────────┘
```

Concrete relationships per command:

| Command | Shared helpers used | External Upsonic packages touched |
|---------|---------------------|-----------------------------------|
| `init` | – | – (writes a template referencing `upsonic.Task`/`upsonic.Agent`) |
| `add` | `shared/config.py` | – |
| `remove` | `shared/config.py` | – |
| `install` | `shared/config.py`, `shared/dependencies.py` | – |
| `run` | `shared/config.py`, `shared/fastapi_imports.py`, `shared/openapi.py` | `upsonic.interfaces.manager.InterfaceManager` (optional) |
| `zip` | – | – |

Every command lazy-imports printer functions inside its `try` block so that even printer failures don't prevent the basic exit-code semantics.

## 6. Public API / commands exposed

Console exposure (via `pyproject.toml`):

| Shell call | Calls Python function |
|------------|-----------------------|
| `upsonic` | `main(['])` (empty) → prints usage |
| `upsonic --help` / `-h` | `print_help_general()` |
| `upsonic init [--help]` | `init_command()` |
| `upsonic add <lib> <section> [--help]` | `add_command(lib, section)` |
| `upsonic remove <lib> <section> [--help]` | `remove_command(lib, section)` |
| `upsonic install [section] [--help]` | `install_command(section)` |
| `upsonic run [--host H] [--port P] [--help]` | `run_command(host, port)` |
| `upsonic zip [filename] [--help]` | `zip_command(filename)` |

Python-side public API:

```python
from upsonic.cli import main                      # entrypoint
from upsonic.cli.commands import (
    init_command, add_command, remove_command,
    install_command, run_command, zip_command,
)
```

All `*_command` functions return an `int` exit code, never raise to the caller.

## 7. Integration with the rest of Upsonic

| Boundary | Hook |
|----------|------|
| `upsonic.Task`, `upsonic.Agent` | The `init_command` template imports them in the generated `main.py`. |
| `upsonic.interfaces.manager.InterfaceManager` | `run_command` checks for it dynamically; if the user's `main()` returns one, it boots `manager.serve(host, port)` instead of the FastAPI fallback. |
| Project metadata | `pyproject.toml` registers `upsonic = "upsonic.cli.main:main"`. |
| Optional dependencies | The FastAPI/uvicorn group is consumed via `shared/fastapi_imports.py`; missing deps prompt `upsonic install`. The default `init` template seeds the `api` section with `fastapi`, `uvicorn`, etc. |
| Telemetry / env vars | The CLI itself does not honour `UPSONIC_TELEMETRY` directly; that lives in core Upsonic. The CLI is intentionally stdlib-and-Rich until you reach `run`. |

There is no dependency from the CLI back into the agent runtime aside from those two touchpoints, which keeps CLI changes safe and isolated.

## 8. End-to-end flow of a CLI invocation

### 8.1 `upsonic init`

```text
Shell  →  upsonic   (console_script registered by pyproject.toml)
       →  upsonic.cli.main:main(args=[])
       →  args = ["init"]
       →  _COMMAND_HANDLERS["init"]
       →  _handle_init(["init"])
       →  init_command()
              ├── prompt_agent_name()              # Rich prompt
              ├── confirm_overwrite() (if needed)  # Rich confirm
              ├── write main.py                    # template literal
              ├── write upsonic_configs.json       # json.dump indent=4
              └── print_init_success(...)          # Rich panel + table
       →  return 0  →  shell exit 0
```

### 8.2 `upsonic add fastapi==0.115.12 api`

```text
main(args) → _COMMAND_HANDLERS["add"] → _handle_add(args)
  → add_command("fastapi==0.115.12", "api")
       ├── load_config(use_cache=False)
       ├── verify "dependencies" + section "api"
       ├── duplicate check (string equality)
       ├── append + json.dump(indent=4, ensure_ascii=False)
       └── print_dependency_added("fastapi==0.115.12", "api")
```

### 8.3 `upsonic remove fastapi api`

The user types the bare package name. Inside `remove_command`:

1. `target_lib_lower = "fastapi"`, `target_pkg_name = "fastapi"`.
2. Pass 1 (exact): no entry equals `"fastapi"`.
3. Pass 2 (package): finds `"fastapi==0.115.12"` because `get_package_name("fastapi==0.115.12") == "fastapi"`.
4. Removes that entry; writes file; prints the *real* removed string.

### 8.4 `upsonic install all`

```text
install_command("all")
  → load_config(use_cache=True)
  → sections_to_install = list(all_dependencies.keys())
  → flatten lists into one big list
  → install_dependencies(deps)
       ├── try `uv add <deps>`            → success?  print_success.
       ├── _ensure_pip_available()
       └── try `python -m pip install ...`
```

### 8.5 `upsonic run --port 9000`

```text
_handle_run walks args  → host="0.0.0.0", port=9000
run_command("0.0.0.0", 9000)
  ├── load_config(use_cache=True)
  ├── resolve agent_py_path = main.py
  ├── insert agent_dir + project_root into sys.path
  ├── importlib.util.spec_from_file_location("main", path)
  ├── set __package__, __name__
  ├── exec_module()
  │
  ├── _is_interface_mode(path)?            (grep "InterfaceManager")
  │     yes  →  _resolve_interface_manager(main)
  │              └── manager.serve(host, port)        ← Mode A
  │     no   →  fall through                          ← Mode B
  │
  ├── get_fastapi_imports()  → {FastAPI, Request, JSONResponse, uvicorn}
  ├── flatten input_schema dict into list of dicts
  ├── app = FastAPI(title=f"{agent_name} - Upsonic")
  ├── register POST /call (multipart-or-json router)
  ├── monkey-patch app.openapi → custom_openapi → modify_openapi_schema
  └── uvicorn.run(app, host, port, log_level="info")
```

When a request arrives at `/call`:

1. Branch on `Content-Type`.
2. Build `inputs` (dict of strings, files, etc.).
3. `await main(inputs)` if async else `main(inputs)`.
4. Wrap return value in `JSONResponse`.
5. Any exception → 500 with `{"error", "type"}`.

### 8.6 `upsonic zip backup`

```text
zip_command("backup")
  ├── output_file = "backup.zip"
  ├── walk Path.cwd().rglob('*')
  │    skip directories, skip the in-progress zip
  ├── total_size accumulator
  ├── ZipFile(out, "w", ZIP_DEFLATED).write(...) per file
  └── print archive size in MB
```

## 9. Notes for contributors

- **Adding a new command.** Create a new folder `commands/<name>/` with `command.py` exporting `<name>_command(...)`. Re-export it from `commands/__init__.py` and register a handler in `main.py`'s `_COMMAND_HANDLERS` plus a `print_help_<name>` in `printer.py`. Keep imports lazy.
- **Don't `print()` directly.** Always go through `printer.py` so output stays consistent.
- **`load_config`'s cache is process-local.** Don't expect it across `subprocess` calls. Use `use_cache=False` for write paths.
- **`add_command` does string-equality dedup.** That is intentional but means `requests==1` and `requests==2` both happily coexist; pair `add` with `remove` if you want canonical behaviour.
- **`run_command` modifies `sys.path` and `sys.modules`.** It is intended to run once per process. If called twice in tests, expect stale `main` references.
- **`InterfaceManager` detection is a static grep.** If the substring appears inside a comment or docstring, mode A will still be tried; this is acceptable because `_resolve_interface_manager` falls back gracefully.
