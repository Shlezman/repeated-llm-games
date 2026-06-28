"""Pluggable registry mapping provider names to adapter factories.

New backends are added by registering a factory — never by editing call sites.
Providers and cache backends are constructed by name from configuration.
"""

from __future__ import annotations

from typing import Callable

from .anthropic_provider import AnthropicProvider
from .base import Provider
from .cache import CacheBackend, InMemoryCache, PostgresCache
from .litellm_provider import LiteLLMProvider
from .mock_provider import MockProvider
from .openai_provider import OpenAIProvider

ProviderFactory = Callable[[str], Provider]

PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "litellm": lambda model: LiteLLMProvider(model),
    "anthropic": lambda model: AnthropicProvider(model),
    "openai": lambda model: OpenAIProvider(model),
    "mock": lambda model: MockProvider(model),
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Registers a new provider factory under ``name``.

    Args:
        name: The provider name used in configuration.
        factory: A callable ``(model) -> Provider``.
    """
    PROVIDER_FACTORIES[name] = factory


def make_provider(name: str, model: str) -> Provider:
    """Constructs a provider by registered name.

    Args:
        name: A key in :data:`PROVIDER_FACTORIES`.
        model: The model identifier (from configuration).

    Returns:
        The constructed provider.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in PROVIDER_FACTORIES:
        raise KeyError(f"Unknown provider {name!r}. Available: {sorted(PROVIDER_FACTORIES)}")
    return PROVIDER_FACTORIES[name](model)


def make_cache(backend: str, dsn: str | None = None, table: str = "llm_response_cache") -> CacheBackend:
    """Constructs a cache backend by name.

    Args:
        backend: "postgres" (default infra) or "memory".
        dsn: Optional Postgres DSN; falls back to ``$DATABASE_URL``.
        table: Postgres cache table name.

    Returns:
        The constructed cache backend.

    Raises:
        ValueError: If ``backend`` is unknown.
    """
    if backend == "memory":
        return InMemoryCache()
    if backend == "postgres":
        return PostgresCache(dsn=dsn, table=table)
    raise ValueError(f"Unknown cache backend: {backend!r}")
