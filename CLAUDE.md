---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Upsonic is a reliability-focused AI agent framework for building production-ready AI agents and digital workers. The framework provides advanced reliability features, MCP (Model Context Protocol) integration, and supports multiple AI providers (OpenAI, Anthropic, Azure, Bedrock).

## AI Operational Guides

Consult the relevant guide before doing the work — these are operational, not optional.

- **`documents/ai/guides/feature.md`** — Use when adding a feature or non-trivial enhancement: a new public API, a new provider (Vector/Model/Storage/Tool/Embedding/OCR/Loader), a new agent type or prebuilt agent, a new policy in `safety_engine/`, a new RAG component, or a new public method/config knob on an existing class. Defines the four mandatory phases (Understand → Design → Implement → Verify), the eight cross-cutting aspect checklists, the hard gates, and the common anti-patterns.
- **`documents/ai/guides/refactor.md`** — Use when changing internal structure without changing observable behaviour: renaming, extracting, splitting an oversized module, removing dead code, mechanical migrations. Defines the four mandatory phases (Motivate & Scope → Characterize → Transform → Verify Behaviour Preserved), the per-phase hard gates, and anti-patterns specific to refactors. Cross-references `feature.md §4` for shared aspects.
- **`documents/ai/guides/bug-fix.md`** — Use when fixing a reported bug, failing test, traceback, or behaviour that contradicts a stated contract. Defines the four mandatory phases (Reproduce → Diagnose → Fix → Verify), with hard rules around root-cause discipline, regression tests, and minimal-diff scope. Cross-references `feature.md §4` for shared aspects.
- **`documents/ai/guides/testing.md`** — Use when deriving, writing, reviewing, or locking tests for any feature / refactor / bug-fix. Defines the four mandatory phases (Derive Scenarios → Generate RED Tests → Manual Review → Lock & Iterate via Code), with hard rules around user-driven scenario seeding, Serena + memory consultation, the manual review gate, and the immutability of locked tests. Complements `feature.md §4.5` (placement and types).
- **`documents/ai/guides/coding-standards.md`** — Always-on. How code in this repo is named, typed, structured, formatted, tested, and reviewed. Pure Python coding standard; framework-agnostic.
- **`documents/ai/guides/serena.md`** — Always-on. When and how to use Serena MCP for symbol and reference lookups, the read-only constraint on source code, the surface-what-you-found rule, and the optional Serena memory layer (currently unused; gitignored by default).
- **`documents/ai/guides/memory.md`** — Always-on. How Claude Code's auto memory works in this repo, where it lives, what auto-loads vs lazy-loads, when to consult it (recurring corrections, past test mistakes, similar prior requests), and the surface-what-you-found rul
- **`documents/ai/guides/subagents.md`** — Always-on. When to dispatch subagents (heavy reads, separable investigations, long tasks, planning/review), when not to, and how to brief them.
- **`documents/ai/guides/new_prebuilt_agent_adding.md`** — Use when shipping a new prebuilt autonomous agent under `src/upsonic/prebuilt/<your_agent>/`. Canonical reference for the runtime/agent-class/template layering, `AGENT_REPO`/`AGENT_FOLDER` wiring to `PrebuiltAutonomousAgentBase`, and the `new_<X>(...)` high-level API conventions.
- **`documents/ai/guides/commit.md`** — Use before any `git commit`, `git push`, or history-rewriting command. Hard rule: never commit without explicit user approval.

### Default Pre-Work Consultation

Before any non-trivial task, the always-on guides compose into a single pre-work pass — Claude Code memory + Serena code lookup (and Serena memory if active). Surface findings together at the start of the reply so the user can see what informed the response, e.g.:

> *"From memory: prior feedback don't mock the DB. From Serena: existing similar handler at `src/upsonic/X.py:42`."*

Trivial work (single-line typo, comment edit) skips this and says so explicitly: *"Skipping memory / Serena lookup — single-line cosmetic edit."*

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
- **Prebuilt Autonomous Agents**: Ready-to-run agents that bundle a system prompt, first-message template, and skills under `src/upsonic/prebuilt/<agent>/template/`. The shared base class lives in `src/upsonic/prebuilt/prebuilt_agent_base.py`. To add a new prebuilt, follow `documents/ai/guides/new_prebuilt_agent_adding.md`.

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
- Documentation: `README.md`, inline docstrings, and contributor guides under `documents/`
  - `documents/ai/guides/new_prebuilt_agent_adding.md` — how to add a new prebuilt autonomous agent (file layout, base-class wiring, template conventions).
- Configuration: `pyproject.toml`, `.pre-commit-config.yaml`, `pytest.ini`
- Dependencies: Managed by `uv` with `uv.lock`
