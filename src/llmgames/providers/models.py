"""Model-agnostic chat-model construction via LangChain ``init_chat_model``.

A model is selected entirely by configuration: ``provider`` chooses the backend and
``model`` is the identifier passed through to it. No model name is hardcoded. The
"mock" provider returns a deterministic fake chat model for offline/test runs.
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.language_models import BaseChatModel

# Providers whose name we pass straight through to init_chat_model; "auto" lets
# LangChain infer the provider from the model identifier.
_INFER = {"auto", "infer", ""}


def build_chat_model(
    provider: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    mock_responses: Iterable[str] | None = None,
) -> BaseChatModel:
    """Builds a LangChain chat model from provider + model configuration.

    Args:
        provider: Backend selector ("openai", "anthropic", "auto", "mock", ...).
        model: The model identifier passed to the backend (config input).
        temperature: Sampling temperature (0.0 for the deterministic study runs).
        max_tokens: Optional max output tokens.
        mock_responses: Canned replies for the "mock" provider (cycled).

    Returns:
        A ready-to-invoke chat model.
    """
    if provider == "mock":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=list(mock_responses) if mock_responses else ["J"])

    from langchain.chat_models import init_chat_model

    kwargs: dict[str, object] = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    model_provider = None if provider in _INFER else provider
    return init_chat_model(model, model_provider=model_provider, **kwargs)
