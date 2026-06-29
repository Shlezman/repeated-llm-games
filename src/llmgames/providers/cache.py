"""Configure LangChain's global LLM cache for deterministic, cheap re-runs.

At temperature 0 the same prompt yields the same reply; caching makes re-runs
identical and avoids repeat API calls. The default backend is a small Postgres cache
(this module's :class:`PostgresLLMCache`, connection string from ``DATABASE_URL``); an
in-memory backend is used for tests and offline runs.

Security: the connection string comes from the environment (never hardcoded); all SQL
binds values as parameters; the table name is validated as a SQL identifier; and TLS
is requested via ``sslmode`` (default ``require``).
"""

from __future__ import annotations

import os
import re
import warnings

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PostgresLLMCache(BaseCache):
    """A Postgres-backed LangChain LLM cache using parameterized SQL.

    Entries are keyed on ``(prompt, llm_string)`` — LangChain's standard cache key —
    and store the serialized generation list.

    Attributes:
        table: The validated cache table name.
    """

    def __init__(self, url: str, *, table: str = "llm_cache", sslmode: str | None = None) -> None:
        """Opens an engine and ensures the cache table exists.

        Args:
            url: SQLAlchemy/Postgres connection URL.
            table: Cache table name (validated as a SQL identifier).
            sslmode: TLS mode; defaults to ``$PGSSLMODE`` or ``require``.

        Raises:
            ValueError: If ``table`` is not a safe identifier.
        """
        from sqlalchemy import create_engine

        if not _IDENTIFIER.fullmatch(table):
            raise ValueError(f"Unsafe cache table identifier: {table!r}")
        self.table = table
        resolved_ssl = sslmode or os.environ.get("PGSSLMODE", "require")
        connect_args = {"sslmode": resolved_ssl} if url.startswith("postgres") else {}
        self._engine = create_engine(url, connect_args=connect_args)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Creates the cache table if it does not already exist."""
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self.table} "
                    "(prompt TEXT NOT NULL, llm TEXT NOT NULL, val TEXT NOT NULL, "
                    "PRIMARY KEY (prompt, llm))"
                )
            )

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Returns the cached generations for ``(prompt, llm_string)`` or None."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT val FROM {self.table} WHERE prompt = :p AND llm = :l"),
                {"p": prompt, "l": llm_string},
            ).fetchone()
        if not row:
            return None
        # Cache content is trusted — we serialized it ourselves via dumps() in update().
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return loads(row[0])

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Upserts the generations for ``(prompt, llm_string)``."""
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {self.table} (prompt, llm, val) VALUES (:p, :l, :v) "
                    "ON CONFLICT (prompt, llm) DO UPDATE SET val = EXCLUDED.val"
                ),
                {"p": prompt, "l": llm_string, "v": dumps(return_val)},
            )

    def clear(self, **kwargs: object) -> None:
        """Empties the cache table."""
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {self.table}"))


def configure_cache(backend: str, dsn: str | None = None) -> None:
    """Installs the process-global LangChain LLM cache.

    Args:
        backend: "postgres" (default infra) or "memory".
        dsn: Optional Postgres connection string; falls back to ``$DATABASE_URL``.

    Raises:
        RuntimeError: If the Postgres backend is selected without a connection string.
        ValueError: If ``backend`` is unknown.
    """
    from langchain_core.globals import set_llm_cache

    if backend == "memory":
        from langchain_core.caches import InMemoryCache

        set_llm_cache(InMemoryCache())
        return

    if backend == "postgres":
        url = dsn or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "Postgres cache requires a connection string via DATABASE_URL or the dsn argument."
            )
        set_llm_cache(PostgresLLMCache(url))
        return

    raise ValueError(f"Unknown cache backend: {backend!r}")
