# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Upsonic is a reliability-focused AI agent framework for building production-ready AI agents and digital workers. The framework provides advanced reliability features, MCP (Model Context Protocol) integration, and supports multiple AI providers (OpenAI, Anthropic, Azure, Bedrock).

## Core Architecture

### Key Components

- **Agent System**: Core agent implementation in `src/upsonic/agent/` with `Direct` class as the main agent interface
- **Task Management**: Task definitions and execution logic in `src/upsonic/tasks/`
- **Tools & MCP Integration**: Tool processing and external tool management in `src/upsonic/tools/`
- **Reliability Layer**: Advanced reliability features in `src/upsonic/reliability_layer/`
- **Safety Engine**: Content filtering and policy enforcement in `src/upsonic/safety_engine/`
- **Storage**: Multi-provider storage system in `src/upsonic/storage/` (In-Memory, JSON, SQLite, Redis, PostgreSQL, MongoDB)
- **Team/Multi-Agent**: Team coordination and delegation in `src/upsonic/team/`
- **Knowledge Base & RAG**: Document processing and retrieval in `src/upsonic/knowledge_base/` and `src/upsonic/rag/`

### Main Entry Points

- `Task`: Task definition and execution (`src/upsonic/tasks/tasks.py`)
- `Agent`/`Direct`: Main agent class (`src/upsonic/agent/agent.py`)
- `Team`: Multi-agent coordination (`src/upsonic/team/team.py`)
- `KnowledgeBase`: RAG and document management (`src/upsonic/knowledge_base/knowledge_base.py`)

## Development Commands

### Environment Setup
```bash
# Install dependencies with uv
uv sync

# Install with optional dependencies
uv sync --extra rag --extra storage
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test directory
uv run pytest tests/rag/

# Run tests with coverage
uv run pytest --cov=src/upsonic
```

### Development Tools
```bash
# Type checking
uv run mypy src/

# Pre-commit hooks (runs automatically on commit)
pre-commit run --all-files

# Lock dependencies
uv lock
```

### Code Quality Standards (Ruff Configuration)

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code quality enforcement. The following rules are configured in `pyproject.toml`:

#### Rule Sets Enabled

```toml
select = [
  "E",  # pycodestyle errors (style)
  "W",  # pycodestyle warnings
  "F",  # pyflakes (undefined names, unused imports)
  "I",  # isort (import order and grouping)
  "B",  # flake8-bugbear (common logic errors)
  "UP", # pyupgrade (modern Python syntax suggestions)
]
```

**Explanation**:
- **E**: pycodestyle errors - Enforces PEP 8 style guidelines
- **W**: pycodestyle warnings - Catches style issues that don't break functionality
- **F**: pyflakes - Detects undefined names, unused imports, and other common errors
- **I**: isort - Ensures consistent import ordering and grouping
- **B**: flake8-bugbear - Identifies common bugs and design problems
- **UP**: pyupgrade - Suggests modern Python syntax improvements

#### Rules Intentionally Ignored

```toml
ignore = [
  "E501",  # Line length (handled by formatter)
  "B008",  # Function default argument warning (too strict for ML/AI use)
]
```

**Explanation**:
- **E501**: Line length is handled by the code formatter, not the linter
- **B008**: Function default argument warnings are disabled because they're too strict for ML/AI use cases where mutable defaults are sometimes necessary

#### Configuration

```toml
# Automatically fix issues where possible
fix = true

# Exclude generated or external files
exclude = [
  ".venv",
  "build",
  "dist",
  "__pycache__",
]
```

**Features**:
- **fix = true**: Automatically fixes issues where possible when running ruff
- **exclude**: Excludes virtual environments, build directories, and cache files from linting

#### Running Ruff

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Check specific directory
ruff check src/upsonic/agent/
```

These rules ensure consistent code quality across the Upsonic framework while maintaining flexibility for ML/AI development patterns.

### Running Examples
```bash
# Run basic agent example
uv run test.py
```


If you get an error about the upsonic is module is not found just try

```python
uv pip uninstall upsonic && uv run 
```

## Model Providers and Configuration

The framework supports multiple AI providers through a unified interface:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable  
- **Azure**: Configure Azure-specific credentials
- **AWS Bedrock**: Configure AWS credentials

Models are specified using the format `provider/model` (e.g., `openai/gpt-4o`, `anthropic/claude-3-sonnet`).

## Key Features to Understand

### Reliability Layer
Advanced reliability features including verifier agents, editor agents, and iterative quality improvement rounds for production-ready outputs.

### MCP Integration
Built-in support for Model Context Protocol tools - can integrate with hundreds of existing MCP servers from the ecosystem.

### Safety Engine
Policy-based content filtering and safety enforcement with configurable rules for sensitive content, adult content, crypto, and social media policies.

### Storage Abstraction
Unified storage interface supporting multiple backends for session management, memory persistence, and user profiles.

## Testing Structure

Tests are organized by functionality:
- `tests/` - Core functionality tests
- `tests/rag/` - RAG and chunking tests  
- `tests/safety_engine/` - Safety policy tests
- `tests/pricing/` - Cost calculation tests

Use pytest for all testing with async support enabled.

## Environment Variables

Key environment variables:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - AI provider credentials
- `UPSONIC_TELEMETRY=False` - Disable telemetry collection
- Database connection strings for storage providers (Redis, PostgreSQL, etc.)

## File Organization

- Source code: `src/upsonic/`
- Tests: `tests/`
- Documentation: `README.md`, inline docstrings
- Configuration: `pyproject.toml`, `.pre-commit-config.yaml`, `pytest.ini`
- Dependencies: Managed by `uv` with `uv.lock`