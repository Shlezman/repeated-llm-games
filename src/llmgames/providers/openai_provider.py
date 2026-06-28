"""Native OpenAI Chat Completions adapter.

Credentials are read from the ``OPENAI_API_KEY`` environment variable by the SDK;
nothing is hardcoded. The model identifier is supplied by configuration. Supports
``logit_bias`` for constrained decoding where the model allows it.
"""

from __future__ import annotations

from .base import GenParams


class OpenAIProvider:
    """Completes prompts through the OpenAI Chat Completions API.

    Attributes:
        provider_name: Always "openai".
        model: The OpenAI model identifier from configuration.
    """

    provider_name = "openai"

    def __init__(self, model: str) -> None:
        """Creates the OpenAI client (API key sourced from the environment).

        Args:
            model: The OpenAI model identifier.
        """
        from openai import OpenAI  # lazy import

        self.model = model
        self._client = OpenAI(max_retries=5)

    def complete(self, prompt: str, params: GenParams) -> str:
        """Generates completion text via ``chat.completions.create``.

        Args:
            prompt: The fully-rendered prompt.
            params: Generation parameters.

        Returns:
            The reply text (empty string if the content is missing).
        """
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
            kwargs["logit_bias"] = {int(k): v for k, v in params.logit_bias.items()}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
