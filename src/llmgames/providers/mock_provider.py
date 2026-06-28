"""Deterministic mock provider for tests and offline (no-API-key) demo runs.

It performs no network calls, so the full pipeline — including ``results.md`` and
figure generation — can be exercised without credentials. Replies cycle through a
configurable list of label strings.
"""

from __future__ import annotations

from itertools import cycle
from typing import Iterable

from .base import GenParams


class MockProvider:
    """Returns canned replies in a fixed cycle.

    Attributes:
        provider_name: Always "mock".
        model: An arbitrary identifier echoed into results.
    """

    provider_name = "mock"

    def __init__(self, model: str = "mock-model", replies: Iterable[str] = ("J",)) -> None:
        """Creates a mock provider.

        Args:
            model: Identifier recorded in results.
            replies: Reply strings to cycle through on each call.
        """
        self.model = model
        self._replies = cycle(list(replies) or ["J"])

    def complete(self, prompt: str, params: GenParams) -> str:
        """Returns the next canned reply, ignoring the prompt and params."""
        return next(self._replies)
