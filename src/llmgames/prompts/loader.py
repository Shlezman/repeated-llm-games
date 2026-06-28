"""Load markdown prompt templates as LangChain ``PromptTemplate`` objects.

All prompt wording lives in ``prompts/templates/*.md`` so it can be edited without
touching code. Templates use f-string placeholders and are cached after first load.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import PromptTemplate

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=None)
def load_template(name: str) -> PromptTemplate:
    """Loads a markdown template by name into a LangChain ``PromptTemplate``.

    Args:
        name: Template file stem under ``prompts/templates`` (without ``.md``).

    Returns:
        A cached ``PromptTemplate`` parsed from the markdown file.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = _TEMPLATE_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return PromptTemplate.from_template(path.read_text(encoding="utf-8"), template_format="f-string")
