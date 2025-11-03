AGENTS.md
Code Quality Rules 

Always follow, **Ruff linting rules**, **isort** imports automatically.

---

### 🧠 **agents.md**

````markdown
# 🧩 Agent Configuration: Clean Python Developer

## 🎯 Mission
Write **production-quality Python code** that:
- Passes **Ruff** lint checks (PEP8 + recommended rules).
- Uses **isort** for import ordering.
- Produces code that is **testable**, **readable**, and **extensible**.

---

## ⚙️ Tools & Standards

### 🦮 Ruff Configuration
Follow the same rules as this `pyproject.toml` setup:

```toml
[tool.ruff]
target-version = "py311"
extend-select = [
    "E",  # pycodestyle errors
    "F",  # pyflakes
    "I",  # isort
    "B",  # bugbear
    "UP", # pyupgrade
    "PL", # pylint core rules
]
ignore = [
    "E501",  # line too long (Black handles it)
    "B905",  # zip() without strict=True (optional)
]
fix = true
````


### 🧩 isort

* Automatically organize imports by sections (`standard`, `third party`, `local`).
* Sort alphabetically.

---

## 🧠 Code Writing Instructions

When writing or editing code:

* ✅ Follow all rules above automatically.
* ✅ Comment logic where clarity adds value.
* ✅ Prefer composition over inheritance.
* ✅ Write small, cohesive functions and classes.
* ✅ Ensure your code passes `ruff check .` and `black .` without modifications.

---

## 🧪 Testing Standards

* Always include **pytest-compatible** examples when adding new logic.
* Follow **AAA (Arrange, Act, Assert)** structure.
* Use meaningful test names and parameterization.

---

## 💡 Example

When generating code, always assume the user runs:

```bash
ruff check . && black . && pytest
```

Your code must pass these commands without any errors or formatting changes.

```
