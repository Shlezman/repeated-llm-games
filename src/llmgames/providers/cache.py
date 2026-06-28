"""Response cache: deterministic, cheap re-runs for temperature=0 experiments.

The cache is keyed on ``(provider, model, params, prompt)`` so identical requests
return identical text without another API call. The default backend is Postgres
(connection string from the ``DATABASE_URL`` environment variable); an in-memory
backend is provided for tests and offline use.

Security: all SQL uses parameterized queries (never string concatenation), and the
database connection string is read from the environment — credentials are never
hardcoded. TLS is requested via ``sslmode`` (default ``require``, overridable).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, runtime_checkable

from .base import GenParams, Provider


def make_cache_key(provider: str, model: str, params: GenParams, prompt: str) -> str:
    """Builds a stable cache key for a completion request.

    Args:
        provider: Provider name (e.g. "litellm").
        model: Model identifier.
        params: Generation parameters.
        prompt: The fully-rendered prompt.

    Returns:
        A hex SHA-256 digest uniquely identifying the request.
    """
    payload = "\x1f".join([provider, model, params.canonical(), prompt])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class CacheBackend(Protocol):
    """A key/value store for cached completion text."""

    def get(self, key: str) -> str | None:
        """Returns the cached value for ``key`` or None if absent."""
        ...

    def set(self, key: str, value: str) -> None:
        """Stores ``value`` under ``key`` (idempotent upsert)."""
        ...


class InMemoryCache:
    """A process-local cache backend for tests and offline runs."""

    def __init__(self) -> None:
        """Initializes an empty in-memory store."""
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Returns the cached value for ``key`` or None."""
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        """Stores ``value`` under ``key``."""
        self._store[key] = value


class PostgresCache:
    """A Postgres-backed completion cache using parameterized queries.

    The table is created on first use. Connection details come from the
    environment, never from code.

    Attributes:
        table: Name of the cache table.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        table: str = "llm_response_cache",
        sslmode: str | None = None,
    ) -> None:
        """Opens a connection and ensures the cache table exists.

        Args:
            dsn: Postgres connection string. Defaults to ``$DATABASE_URL``.
            table: Cache table name (validated as an identifier).
            sslmode: TLS mode. Defaults to ``$PGSSLMODE`` or ``require``.

        Raises:
            RuntimeError: If no DSN is available from args or environment.
            ValueError: If ``table`` is not a safe SQL identifier.
        """
        import psycopg  # imported lazily so tests need no DB driver

        resolved_dsn = dsn or os.environ.get("DATABASE_URL")
        if not resolved_dsn:
            raise RuntimeError(
                "PostgresCache requires a connection string via DATABASE_URL or the dsn argument."
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"Unsafe cache table identifier: {table!r}")

        self.table = table
        resolved_ssl = sslmode or os.environ.get("PGSSLMODE", "require")
        self._conn = psycopg.connect(resolved_dsn, sslmode=resolved_ssl, autocommit=True)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Creates the cache table if it does not already exist."""
        # `table` is validated as a bare identifier in __init__; values are bound.
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(cache_key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )

    def get(self, key: str) -> str | None:
        """Returns the cached value for ``key`` or None."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT value FROM {self.table} WHERE cache_key = %s", (key,))
            row = cur.fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        """Upserts ``value`` under ``key``."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (cache_key, value) VALUES (%s, %s) "
                "ON CONFLICT (cache_key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._conn.close()


class CachingProvider:
    """Decorates any :class:`Provider` with read-through caching.

    Attributes:
        provider_name: Mirrors the wrapped provider's name.
        model: Mirrors the wrapped provider's model.
    """

    def __init__(self, inner: Provider, cache: CacheBackend) -> None:
        """Wraps ``inner`` so completions are served from ``cache`` when present.

        Args:
            inner: The underlying provider performing real API calls.
            cache: The cache backend.
        """
        self._inner = inner
        self._cache = cache
        self.provider_name = inner.provider_name
        self.model = inner.model

    def complete(self, prompt: str, params: GenParams) -> str:
        """Returns cached text if present, otherwise calls the inner provider and caches it.

        Args:
            prompt: The fully-rendered prompt.
            params: Generation parameters.

        Returns:
            The completion text.
        """
        key = make_cache_key(self.provider_name, self.model, params, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        text = self._inner.complete(prompt, params)
        self._cache.set(key, text)
        return text
