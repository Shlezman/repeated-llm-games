"""Tests for robust action parsing from free-form model replies."""

from __future__ import annotations

import pytest

from llmgames.engine.game import ACTION_A, ACTION_B
from llmgames.players.parser import extract_action

LABELS = {ACTION_A: "J", ACTION_B: "F"}


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("J", ACTION_A),
        ("F", ACTION_B),
        ("Option F", ACTION_B),
        ("I choose Option J.", ACTION_A),
        ("My answer: F", ACTION_B),
        ("j", ACTION_A),
        ("  F\n", ACTION_B),
        ("I would go with J this round", ACTION_A),
    ],
)
def test_extract_valid_replies(reply, expected):
    """Verbose and terse replies resolve to the correct internal action."""
    assert extract_action(reply, LABELS) == expected


def test_unparseable_returns_none():
    """A reply with no valid label returns None (never a guess)."""
    assert extract_action("I refuse to answer", LABELS) is None
    assert extract_action("", LABELS) is None


def test_first_mentioned_label_wins():
    """When both labels appear, the earliest mention is selected."""
    assert extract_action("Between F and J, I pick F", LABELS) == ACTION_B
    assert extract_action("J then maybe F", LABELS) == ACTION_A


def test_custom_labels():
    """Parsing works for non-JF labels (e.g. cooking-competition recipes)."""
    labels = {ACTION_A: "Q", ACTION_B: "X"}
    assert extract_action("Recipe X please", labels) == ACTION_B
    assert extract_action("Q", labels) == ACTION_A
