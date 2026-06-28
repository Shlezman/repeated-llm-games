"""Native Anthropic Messages API adapter.

Credentials are read from the ``ANTHROPIC_API_KEY`` environment variable by the
SDK; nothing is hardcoded. The model identifier is supplied by configuration.
"""

from __future__ import annotations

from .base import GenParams


class AnthropicProvider:
    """Completes prompts through the Anthropic Messages API.

    Attributes:
        provider_name: Always "anthropic".
        model: The Anthropic model identifier from configuration.
    """

    provider_name = "anthropic"

    def __init__(self, model: str) -> None:
        """Creates the Anthropic client (API key sourced from the environment).

        Args:
            model: The Anthropic model identifier.
        """
        from anthropic import Anthropic  # lazy import

        self.model = model
        self._client = Anthropic(max_retries=5)

    def complete(self, prompt: str, params: GenParams) -> str:
        """Generates completion text via ``messages.create``.

        Args:
            prompt: The fully-rendered prompt.
            params: Generation parameters.

        Returns:
            The concatenated text of the reply's content blocks.
        """
        message = self._client.messages.create(
            model=self.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
