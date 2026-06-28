"""Load and validate run specifications from YAML, expanding ``${ENV}`` references.

Model identifiers can be written as ``${MODEL_A}`` so the exact model under test is
supplied by the environment, never committed to the config file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .schema import RunSpec

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: object) -> object:
    """Recursively expands ``${VAR}`` references in strings using the environment.

    Args:
        value: A scalar, list, or dict parsed from YAML.

    Returns:
        The value with environment references substituted.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_run_spec(path: str | Path) -> RunSpec:
    """Loads a :class:`RunSpec` from a YAML file.

    Args:
        path: Path to the YAML run configuration.

    Returns:
        The validated run specification.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return RunSpec.model_validate(_expand(raw))
