"""Smoke-test configuration.

All smoke tests run against a single env-selected chat model. Upsonic's
``infer_model`` honors the ``LLM_MODEL_KEY`` environment variable as a global
override for every model resolution, so setting it here once routes every smoke
test (and its sub-agents: memory summarizer, reflection evaluator, tools, etc.)
to the same model without touching individual test files.

Selection (see ``_model_selection.smoke_model``):
  * ``OPENAI_API_KEY``    -> ``openai-chat/gpt-5``
  * ``ANTHROPIC_API_KEY`` -> ``anthropic/claude-opus-4-6``

The root ``tests/conftest.py`` has already loaded ``.env`` by the time this
module is imported, so the API keys are visible here.
"""
import os

from tests.smoke_tests._model_selection import smoke_model

# Set the global model override for the whole smoke-test session. Done at import
# (conftest is imported before collection) so every infer_model call sees it.
os.environ["LLM_MODEL_KEY"] = smoke_model()
