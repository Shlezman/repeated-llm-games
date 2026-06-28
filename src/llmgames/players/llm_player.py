"""LLM-backed player: model-agnostic, base or Social Chain-of-Thought (SCoT).

The model and provider are injected (never hardcoded). In SCoT mode the player
first asks the model to predict the opponent's move, then asks for its own action
conditioned on that prediction; the prediction is cached for the round so the
theory-of-mind metric can compare it against the opponent's actual move.
"""

from __future__ import annotations

from ..engine.game import ACTION_A, ACTION_B, Action
from ..prompts import builder
from ..prompts.builder import Framing
from ..providers.base import GenParams, Provider
from .base import UNPARSEABLE, PlayerView
from .parser import extract_action


class LLMPlayer:
    """A player whose decisions come from a language model.

    Attributes:
        name: Identifier used in result records.
        scot: Whether Social Chain-of-Thought prompting is enabled.
    """

    def __init__(
        self,
        name: str,
        provider: Provider,
        params: GenParams,
        framing: Framing,
        *,
        scot: bool = False,
    ) -> None:
        """Creates an LLM player.

        Args:
            name: Identifier used in result records.
            provider: The (typically cache-wrapped) provider backend.
            params: Generation parameters.
            framing: Surface presentation (labels, unit word, cover story).
            scot: Enable predict-then-act SCoT prompting.
        """
        self.name = name
        self.scot = scot
        self._provider = provider
        self._params = params
        self._framing = framing
        self._pred_round: int | None = None
        self._pred_action: Action | None = None

    def _ensure_prediction(self, view: PlayerView) -> Action | None:
        """Computes and caches the opponent prediction for the current round (SCoT).

        Args:
            view: The current player view.

        Returns:
            The predicted opponent action, or None if SCoT is off or unparseable.
        """
        if not self.scot:
            return None
        if self._pred_round == view.round_index:
            return self._pred_action

        # SCoT reasoning prompts present options in canonical order (the original SCoT
        # path did not shuffle); only base-mode decisions use the randomized order.
        prompt = builder.prediction_prompt(view, self._framing, (ACTION_A, ACTION_B))
        text = self._provider.complete(prompt, self._params)
        self._pred_action = extract_action(text, self._framing.action_labels)
        self._pred_round = view.round_index
        return self._pred_action

    def choose(self, view: PlayerView) -> Action:
        """Returns the model's chosen action for the current round.

        Args:
            view: The current player view.

        Returns:
            The selected action, or :data:`UNPARSEABLE` if the reply has no label.
        """
        if self.scot:
            prediction = self._ensure_prediction(view)
            if prediction is None:
                # No usable prediction: fall back to the unconditioned decision rather
                # than fabricating a belief the model never expressed.
                prompt = builder.decision_prompt(view, self._framing, (ACTION_A, ACTION_B))
            else:
                prompt = builder.conditioned_decision_prompt(
                    view, self._framing, (ACTION_A, ACTION_B), self._framing.label(prediction)
                )
        else:
            prompt = builder.decision_prompt(view, self._framing, view.option_order)

        text = self._provider.complete(prompt, self._params)
        action = extract_action(text, self._framing.action_labels)
        return action if action is not None else UNPARSEABLE

    def predict_opponent(self, view: PlayerView) -> Action | None:
        """Returns the cached SCoT opponent prediction for the current round."""
        return self._ensure_prediction(view)
