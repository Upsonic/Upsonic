# Contributing to Upsonic

Upsonic is an open-source AI Agent Framework. We welcome contributions that align with our standards.

## Development Setup

1. Clone the repository
2. Install `uv` if not already installed:
   ```bash
   pip install uv
   ```
3. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # Unix
   # or
   .venv\Scripts\activate     # Windows
   ```
4. Install the package in editable mode with dev dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

For specific feature development, install the relevant optional dependencies:
```bash
uv pip install -e ".[vectordb,storage,models,embeddings,tools]"
```

## Running Tests

### Unit Tests
```bash
uv run --all-extras pytest tests/unit_tests -v
```

### Smoke Tests (requires Docker)
```bash
make smoke_tests
```

This will automatically start the required Docker services (Redis, PostgreSQL, MongoDB) before running tests.

### Running Specific Tests
```bash
uv run --all-extras pytest tests/unit_tests/tools/test_common_tools_duckduckgo.py -v
uv run --all-extras pytest tests/smoke_tests/memory -v
```

## Code Standards

### Sync/Async Pattern
Every function/method MUST have both synchronous and asynchronous versions:
```python
def process(data: str) -> Result:
    """Synchronous version."""
    ...

async def aprocess(data: str) -> Result:
    """Asynchronous version - prefix with 'a'."""
    ...
```

### Type Annotations
All code MUST have proper type annotations. No `Any` types unless absolutely necessary:
```python
def calculate_score(
    items: list[dict[str, float]],
    threshold: float = 0.5
) -> tuple[float, list[str]]:
    ...
```

### Standalone Functions
Functions must be self-contained and receive all dependencies as parameters:
```python
# ✅ Correct
def process_data(client: HttpClient, config: Config, data: str) -> Result:
    ...

# ❌ Wrong - relies on external state
def process_data(data: str) -> Result:
    client = get_global_client()  # Don't do this
    ...
```

## Extension Points

### Adding a VectorDB Provider

1. Create a new file in `src/upsonic/vectordb/providers/`:
   ```
   src/upsonic/vectordb/providers/your_provider.py
   ```

2. Implement the `VectorDBProvider` interface from `src/upsonic/vectordb/base.py`

3. Add dependencies to `pyproject.toml` under a new optional group:
   ```toml
   [project.optional-dependencies]
   your-provider = [
       "your-client-lib>=x.x.x",
   ]
   ```

4. Update the `vectordb` group to include your vector database

5. Add tests in `tests/smoke_tests/vectordb/`

Reference: `src/upsonic/vectordb/providers/chroma.py`

### Adding a Model Provider

1. Create a new file in `src/upsonic/models/`:
   ```
   src/upsonic/models/your_provider.py
   ```

2. Implement the required interface pattern (see existing providers)

3. Register in `src/upsonic/models/model_registry.py`

4. Add dependencies to `pyproject.toml` under a new optional group:
   ```toml
   [project.optional-dependencies]
   your-provider = [
       "your-client-lib>=x.x.x",
   ]
   ```

5. Update the `models` group to include your model provider

6. Add tests in `tests/unit_tests/` or `tests/smoke_tests/`

Reference: `src/upsonic/models/openai.py`, `src/upsonic/models/anthropic.py`

### Adding a Storage Provider

1. Create a new directory in `src/upsonic/storage/`:
   ```
   src/upsonic/storage/your_storage/
   ├── __init__.py
   ├── your_storage.py      # Sync implementation
   ├── async_your_storage.py # Async implementation
   ├── schemas.py
   └── utils.py
   ```

2. Implement the `BaseStorage` interface from `src/upsonic/storage/base.py`

3. Provide BOTH sync and async implementations

4. Add dependencies to `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   your-storage = [
       "your-client>=x.x.x",
   ]
   ```

5. Update the `storage` group to include your storage

6. Add tests in `tests/smoke_tests/memory/`

Reference: `src/upsonic/storage/postgres/`, `src/upsonic/storage/redis/`

### Adding a Tool

1. For common tools, add to `src/upsonic/tools/common_tools/`:
   ```
   src/upsonic/tools/common_tools/your_tool.py
   ```

2. For custom/integration tools, add to `src/upsonic/tools/custom_tools/`:
   ```
   src/upsonic/tools/custom_tools/your_tool.py
   ```

3. Follow the base tool pattern from `src/upsonic/tools/base.py`

4. Export in the appropriate `__init__.py`

5. Add tests in `tests/unit_tests/tools/`

6. Update the `tools` group to include your tool

Reference: `src/upsonic/tools/common_tools/duckduckgo.py`

## Pull Request Guidelines

1. **Keep PRs focused**: One feature or fix per PR
2. **Include tests**: All new code must have test coverage
3. **Follow type conventions**: Proper annotations
4. **Both sync/async**: Implement both versions for any new function
5. **Run tests locally**: Ensure all tests pass before submitting

## License

This project is licensed under the [MIT License](/LICENCE).
