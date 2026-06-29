"""Robust single-action extraction from free-form model replies.

Modern chat models rarely emit exactly one token, so we never assume that.
Strategy: map display labels back to internal actions, then find the earliest
unambiguous label mention in the reply. Returns ``None`` (unparseable) rather than
guessing when no label is present.
"""

from __future__ import annotations

import re

from ..engine.game import Action


def _label_to_action(action_labels: dict[Action, str]) -> dict[str, Action]:
    """Inverts an action->label map into an uppercased label->action map.

    Args:
        action_labels: Internal-action to display-label mapping.

    Returns:
        A mapping from uppercased display label to internal action.
    """
    return {label.upper(): action for action, label in action_labels.items()}


def extract_action(reply: str, action_labels: dict[Action, str]) -> Action | None:
    """Extracts the chosen internal action from a model reply.

    The earliest label appearing as a standalone token wins (handles "Option F",
    "I choose J.", "F", "My answer: Recipe X"). Labels embedded inside words (the
    "F" in "refuse") are deliberately ignored to avoid false positives; if no
    standalone label is present the reply is treated as unparseable.

    When the reply contains a ``CHOICE:``/``ANSWER:`` marker (reasoning mode, where
    the model explains first then states its pick), only the text after the last
    such marker is scanned, so labels mentioned in the rationale are ignored.

    Args:
        reply: Raw model output text.
        action_labels: Internal-action to display-label mapping for this round.

    Returns:
        The selected internal action, or None if no valid label is present.
    """
    if not reply:
        return None

    lookup = _label_to_action(action_labels)
    upper = reply.upper()

    # Reasoning mode: scan only after the final explicit choice marker.
    marker = max(upper.rfind("CHOICE:"), upper.rfind("ANSWER:"))
    if marker != -1:
        upper = upper[marker:]

    # Choose the label that appears earliest as a standalone token (word boundary).
    best_pos: int | None = None
    best_action: Action | None = None
    for label, action in lookup.items():
        match = re.search(rf"(?<![A-Z0-9]){re.escape(label)}(?![A-Z0-9])", upper)
        if match is None:
            continue
        pos = match.start()
        if best_pos is None or pos < best_pos:
            best_pos, best_action = pos, action

    return best_action
