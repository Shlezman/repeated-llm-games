"""LiteLLM unified adapter — the default backend that makes any model an input.

LiteLLM routes to OpenAI, Anthropic, Azure, OpenRouter, local/OpenAI-compatible
servers, and many others using a single ``provider/model`` identifier and reads
credentials from the environment. No model name is hardcoded here.
"""

from __future__ import annotations

from .base import GenParams


class LiteLLMProvider:
    """Completes prompts through LiteLLM's unified ``completion`` API.

    Attributes:
        provider_name: Always "litellm".
        model: The LiteLLM model identifier (e.g. "anthropic/<id>", "openai/<id>").
    """

    provider_name = "litellm"

    def __init__(self, model: str) -> None:
        """Stores the model identifier (credentials come from the environment).

        Args:
            model: The LiteLLM-routable model identifier.
        """
        self.model = model

    def complete(self, prompt: str, params: GenParams) -> str:
        """Generates completion text via ``litellm.completion``.

        Args:
            prompt: The fully-rendered prompt.
            params: Generation parameters.

        Returns:
            The model's reply text (empty string if the content is missing).
        """
        import litellm  # lazy import so the package loads without the dependency

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
        }
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.logit_bias:
            kwargs["logit_bias"] = params.logit_bias
        kwargs.update(params.extra)

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""
