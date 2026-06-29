"""LLM-backed player implemented as a LangGraph per-turn agent.

The turn is a tiny ``StateGraph``: base mode is just a ``decide`` node; Social
Chain-of-Thought (SCoT) is ``predict -> decide``, where ``decide`` is conditioned on
the predicted opponent move. Each node invokes ``prompt | model`` (the prompt comes
from a markdown template), captures the raw reply text (the model's "thinking" in
reasoning mode) and parses out the chosen action. The model is injected
(config-driven); no model identifier is hardcoded.
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from ..engine.game import Action
from ..prompts import render
from ..prompts.loader import load_template
from ..prompts.render import CANONICAL_ORDER
from ..prompts.transforms import Framing
from .base import UNPARSEABLE, PlayerView
from .parser import extract_action

_LOGGER = logging.getLogger(__name__)
# Dedupe noisy per-call failures: log each (player, phase, error type) once.
_LOGGED_ERRORS: set[tuple[str, str, str]] = set()


class _TurnState(TypedDict, total=False):
    """Mutable state threaded through one player's turn graph."""

    view: PlayerView
    prediction: Optional[Action]
    predict_text: str
    action: Action
    decide_text: str


def _message_text(message: object) -> str:
    """Extracts plain text from a chat-model output (AIMessage or str)."""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _template_name(kind: str, reasoning: bool) -> str:
    """Returns the markdown template stem for a step, honouring reasoning mode."""
    base = {"base": "base_decision", "predict": "scot_predict", "decide": "scot_decide"}[kind]
    return f"{base}_reasoned" if reasoning else base


class LLMPlayer:
    """A player whose decisions come from a LangGraph turn agent.

    Attributes:
        name: Identifier used in result records.
        scot: Whether Social Chain-of-Thought prompting is enabled.
        reasoning: Whether prompts elicit a one-sentence rationale before the choice.
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        framing: Framing,
        *,
        scot: bool = False,
        reasoning: bool = False,
    ) -> None:
        """Creates an LLM player and compiles its turn graph.

        Args:
            name: Identifier used in result records.
            model: An injected LangChain chat model (any backend).
            framing: Surface presentation (labels, unit word, cover story).
            scot: Enable predict-then-act SCoT prompting.
            reasoning: Ask the model to explain briefly before answering (captured).
        """
        self.name = name
        self.scot = scot
        self.reasoning = reasoning
        self._framing = framing
        self._base_chain = load_template(_template_name("base", reasoning)) | model
        if scot:
            self._predict_chain = load_template(_template_name("predict", reasoning)) | model
            self._scot_chain = load_template(_template_name("decide", reasoning)) | model
        self._graph = self._build_graph()
        self._last_round: int | None = None
        self._last_prediction: Action | None = None
        self._last_thoughts: dict = {}

    def _build_graph(self):
        """Compiles the base (decide) or SCoT (predict -> decide) turn graph."""
        graph = StateGraph(_TurnState)
        graph.add_node("decide", self._decide_node)
        if self.scot:
            graph.add_node("predict", self._predict_node)
            graph.add_edge(START, "predict")
            graph.add_edge("predict", "decide")
        else:
            graph.add_edge(START, "decide")
        graph.add_edge("decide", END)
        return graph.compile()

    def _safe_invoke(self, chain, variables: dict, phase: str) -> tuple[Action | None, str]:
        """Invokes ``prompt | model``, returning ``(parsed_action, raw_text)``.

        A blocked/malformed/errored gateway response for one call must not crash the
        tournament; it becomes an unparseable round (logged once per kind).

        Args:
            chain: The ``prompt | model`` chain.
            variables: Template variables for this call.
            phase: "predict" or "decide" (for logging).

        Returns:
            ``(action_or_None, reply_text)``; ``(None, "")`` on failure.
        """
        try:
            text = _message_text(chain.invoke(variables))
            return extract_action(text, self._framing.action_labels), text
        except Exception as exc:  # provider error, content filter, null choices, etc.
            key = (self.name, phase, type(exc).__name__)
            if key not in _LOGGED_ERRORS:
                _LOGGED_ERRORS.add(key)
                _LOGGER.warning(
                    "player %s: %s call failed (%s: %s) -> unparseable round",
                    self.name, phase, type(exc).__name__, exc,
                )
            return None, ""

    def _predict_node(self, state: _TurnState) -> dict:
        """SCoT step 1: predict the opponent's move (canonical option order)."""
        prediction, text = self._safe_invoke(
            self._predict_chain, render.predict_vars(state["view"], self._framing), "predict"
        )
        return {"prediction": prediction, "predict_text": text}

    def _decide_node(self, state: _TurnState) -> dict:
        """Chooses an action, conditioned on the prediction in SCoT mode."""
        view = state["view"]
        prediction = state.get("prediction")
        if self.scot and prediction is not None:
            chain = self._scot_chain
            variables = render.scot_decide_vars(view, self._framing, self._framing.label(prediction))
        elif self.scot:
            # No usable prediction: fall back to the unconditioned decision (canonical
            # order) rather than fabricating a belief the model never expressed.
            chain = self._base_chain
            variables = render.decision_vars(view, self._framing, CANONICAL_ORDER)
        else:
            chain = self._base_chain
            variables = render.decision_vars(view, self._framing, view.option_order)
        action, text = self._safe_invoke(chain, variables, "decide")
        return {"action": action if action is not None else UNPARSEABLE, "decide_text": text}

    def choose(self, view: PlayerView) -> Action:
        """Runs the turn graph, captures thoughts, and returns the chosen action.

        Args:
            view: The current player view.

        Returns:
            The selected action, or :data:`UNPARSEABLE` if the reply has no label.
        """
        # Reset before invoking so predict_opponent can never surface a stale belief.
        self._last_round = view.round_index
        self._last_prediction = None
        result = self._graph.invoke({"view": view})
        prediction = result.get("prediction")
        self._last_prediction = prediction
        labels = self._framing.action_labels
        self._last_thoughts = {
            "predicted": labels[prediction] if prediction in labels else None,
            "predict_text": result.get("predict_text", ""),
            "decide_text": result.get("decide_text", ""),
        }
        return result["action"]

    def predict_opponent(self, view: PlayerView) -> Action | None:
        """Returns the SCoT opponent prediction recorded for the current round."""
        if self.scot and self._last_round == view.round_index:
            return self._last_prediction
        return None

    def last_thoughts(self) -> dict:
        """Returns the reasoning capture from the most recent :meth:`choose` call."""
        return dict(self._last_thoughts)
