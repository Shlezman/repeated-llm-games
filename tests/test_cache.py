"""Tests for the LangChain model factory and global cache configuration."""

from __future__ import annotations

import pytest

from llmgames.engine.families import canonical_pd
from llmgames.engine.game import ACTION_A
from llmgames.players.base import PlayerView
from llmgames.players.llm_player import LLMPlayer
from llmgames.prompts.transforms import build_framing
from llmgames.providers.cache import configure_cache
from llmgames.providers.models import build_chat_model


def test_configure_memory_cache_installs_global_cache():
    """Selecting the memory backend installs a global LLM cache."""
    from langchain_core.globals import get_llm_cache

    configure_cache("memory")
    assert get_llm_cache() is not None


def test_postgres_cache_requires_dsn(monkeypatch):
    """The Postgres backend errors clearly when no connection string is available."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        configure_cache("postgres")


def test_build_mock_model_and_player():
    """The mock provider yields a working offline player."""
    model = build_chat_model("mock", "x", mock_responses=["J"])
    player = LLMPlayer("m", model, build_framing(), scot=False)
    view = PlayerView(game=canonical_pd(), am_row_player=True, round_index=1, num_rounds=10)
    assert player.choose(view) == ACTION_A
