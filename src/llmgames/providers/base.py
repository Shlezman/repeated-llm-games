"""Provider abstraction: the seam that makes the study model-agnostic.

A :class:`Provider` turns a prompt string into completion text. Concrete adapters
(LiteLLM unified, native Anthropic, native OpenAI) are registered by name and
selected purely from run configuration — no model identifiers are baked into code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenParams:
    """Generation parameters passed to a provider.

    Attributes:
        temperature: Sampling temperature; 0.0 for the deterministic study runs.
        max_tokens: Maximum tokens to generate. Kept small but >1 so modern chat
            models can emit a parseable answer (never assume single-token output).
        top_p: Nucleus sampling parameter.
        seed: Optional provider-side seed for reproducibility where supported.
        logit_bias: Optional token-id -> bias map for constrained decoding.
        extra: Provider-specific passthrough options.
    """

    temperature: float = 0.0
    max_tokens: int = 8
    top_p: float = 1.0
    seed: int | None = None
    logit_bias: dict[str, float] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)

    def canonical(self) -> str:
        """Returns a stable JSON string of the params for cache keying."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@runtime_checkable
class Provider(Protocol):
    """A backend that completes a prompt into text.

    Attributes:
        provider_name: Stable identifier of the backend (e.g. "litellm").
        model: The model identifier this instance was configured with.
    """

    provider_name: str
    model: str

    def complete(self, prompt: str, params: GenParams) -> str:
        """Generates completion text for ``prompt``.

        Args:
            prompt: The fully-rendered prompt string.
            params: Generation parameters.

        Returns:
            The raw completion text returned by the model.
        """
        ...
