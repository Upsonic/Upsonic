# Contributing to Upsonic

First off, thank you for considering contributing to Upsonic! It's people like you that make Upsonic such a great tool for the community.

We welcome contributions of all kinds, from bug reports and documentation improvements to new features and performance enhancements. This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Branch Naming Convention](#branch-naming-convention)
- [Issue and Pull Request Lifecycle](#issue-and-pull-request-lifecycle)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Where do I go from here?

If you've noticed a bug or have a feature request, [create an issue](https://github.com/Upsonic/Upsonic/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or enhancements
- **Documentation**: Improve or add to our documentation
- **Code Contributions**: Fix bugs or implement new features
- **Testing**: Add or improve test coverage

## How to Contribute

### Step 1: Fork & Create a Branch

If this is something you think you can fix, then fork Upsonic and create a branch with a descriptive name.

**Important**: Always create a separate branch for your changes. Never work directly on the `master` branch for local development.

## Branch Naming Convention

We follow a specific naming convention for branches to maintain consistency and clarity:

**Format**: `<type>/<issue-id>/<short-description>`

Where:

- `<type>`: Use conventional commit types like `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `BREAKING_CHANGE`
- `<issue-id>`: The GitHub issue number you're working on
- `<short-description>`: Brief description with words separated by hyphens

**Examples**:

- `feat/388/add-contributing-guidelines`
- `fix/42/memory-leak-agent-cleanup`
- `docs/123/api-documentation-update`
- `chore/456/update-dependencies`
- `refactor/789/context-strategy-pattern-refactor`

## Development Workflow

### Step 2: Get the Code

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/Upsonic.git
cd Upsonic

# Create and checkout your feature branch
git checkout -b feat/388/add-contributing-guidelines
```

### Step 3: Set Up Development Environment

```bash
# Install dependencies using uv (recommended) or pip
uv sync
# or
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt  # if available
```

### Step 4: Make Your Changes

1. **Write Clean Code**: Follow Python best practices and existing code style
2. **Add Documentation**: Update docstrings and README if needed
3. **Write Tests**: Add tests for new functionality
4. **Keep Changes Focused**: One feature/fix per pull request

## Testing

### Step 5: Test Your Changes

It's crucial to test your changes to ensure they don't break existing functionality:

```bash
# Run the full test suite
pytest

# Run tests with coverage
pytest --cov=src/upsonic

# Run specific tests
pytest tests/test_specific_module.py

# Run tests in verbose mode
pytest -v
```

**Testing Guidelines**:

- Write unit tests for new functions and classes
- Write integration tests for complex workflows
- Ensure all tests pass before submitting
- Aim for good test coverage on new code

### Step 6: Commit Your Changes

Use descriptive commit messages following conventional commit format:

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add comprehensive contributing guidelines

- Add detailed development workflow
- Include branch naming conventions
- Add testing and documentation sections
- Improve readability with emojis and structure"
```

**Commit Message Format**:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

### Step 7: Push Your Changes

```bash
# Push your branch to your fork
git push origin feat/388/add-contributing-guidelines
```

### Step 8: Create a Pull Request

1. Go to the [Upsonic repository](https://github.com/Upsonic/Upsonic)
2. Click "New Pull Request"
3. Select your branch from your fork
4. Target the `master` branch of the main repository
5. Fill out the pull request template with:
   - **Clear title**: Summarize your changes
   - **Description**: Explain what you changed and why
   - **Issue reference**: Link to the related issue (e.g., "Closes #388")
   - **Testing**: Describe how you tested your changes

## Issue and Pull Request Lifecycle

### Our Workflow Rules

- **Every new issue should be closed with a pull request**
- **Every pull request should have a separate branch** (never use `master` for local development)
- **Branches must follow our naming convention**: `<type>/<issue-id>/<description>`
- **Please link the issue in the pull request description**
- **All pull requests require review before merging**

### Issue Guidelines

When creating an issue:

- Use a clear and descriptive title
- Provide steps to reproduce (for bugs)
- Include system information when relevant
- Add appropriate labels
- Check if a similar issue already exists

## Documentation

### Updating Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes
- **README**: Update if you add new features or change installation steps
- **API Docs**: Update type hints and documentation for public APIs

### Documentation Style

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value.

    Raises:
        ValueError: Description of when this exception is raised.
    """
    pass
```

## Recognition

Contributors will be:

- Added to our contributors list
- Mentioned in release notes for significant contributions

Thank you for your contribution!
