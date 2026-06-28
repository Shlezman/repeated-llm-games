"""LLM-backed player implemented as a LangGraph per-turn agent.

The turn is a tiny ``StateGraph``: base mode is just a ``decide`` node; Social
Chain-of-Thought (SCoT) is ``predict -> decide``, where ``decide`` is conditioned on
the predicted opponent move. Each node runs an LCEL chain ``prompt | model | parser``
whose prompt comes from a markdown template. The model is injected (config-driven);
no model identifier is hardcoded.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from ..engine.game import Action
from ..prompts import render
from ..prompts.loader import load_template
from ..prompts.render import CANONICAL_ORDER
from ..prompts.transforms import Framing
from .base import UNPARSEABLE, PlayerView
from .parser import extract_action


class _TurnState(TypedDict, total=False):
    """Mutable state threaded through one player's turn graph."""

    view: PlayerView
    prediction: Optional[Action]
    action: Action


def _message_text(message: object) -> str:
    """Extracts plain text from a chat-model output (AIMessage or str)."""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


class LLMPlayer:
    """A player whose decisions come from a LangGraph turn agent.

    Attributes:
        name: Identifier used in result records.
        scot: Whether Social Chain-of-Thought prompting is enabled.
    """

    def __init__(
        self, name: str, model: BaseChatModel, framing: Framing, *, scot: bool = False
    ) -> None:
        """Creates an LLM player and compiles its turn graph.

        Args:
            name: Identifier used in result records.
            model: An injected LangChain chat model (any backend).
            framing: Surface presentation (labels, unit word, cover story).
            scot: Enable predict-then-act SCoT prompting.
        """
        self.name = name
        self.scot = scot
        self._framing = framing
        parser = RunnableLambda(
            lambda message: extract_action(_message_text(message), framing.action_labels)
        )
        self._base_chain = load_template("base_decision") | model | parser
        if scot:
            self._predict_chain = load_template("scot_predict") | model | parser
            self._scot_chain = load_template("scot_decide") | model | parser
        self._graph = self._build_graph()
        self._last_round: int | None = None
        self._last_prediction: Action | None = None

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

    def _predict_node(self, state: _TurnState) -> dict:
        """SCoT step 1: predict the opponent's move (canonical option order)."""
        prediction = self._predict_chain.invoke(render.predict_vars(state["view"], self._framing))
        return {"prediction": prediction}

    def _decide_node(self, state: _TurnState) -> dict:
        """Chooses an action, conditioned on the prediction in SCoT mode."""
        view = state["view"]
        prediction = state.get("prediction")
        if self.scot and prediction is not None:
            action = self._scot_chain.invoke(
                render.scot_decide_vars(view, self._framing, self._framing.label(prediction))
            )
        elif self.scot:
            # No usable prediction: fall back to the unconditioned decision (canonical
            # order) rather than fabricating a belief the model never expressed.
            action = self._base_chain.invoke(
                render.decision_vars(view, self._framing, CANONICAL_ORDER)
            )
        else:
            action = self._base_chain.invoke(
                render.decision_vars(view, self._framing, view.option_order)
            )
        return {"action": action if action is not None else UNPARSEABLE}

    def choose(self, view: PlayerView) -> Action:
        """Runs the turn graph and returns the chosen action.

        Args:
            view: The current player view.

        Returns:
            The selected action, or :data:`UNPARSEABLE` if the reply has no label.
        """
        # Reset before invoking so predict_opponent can never surface a stale belief,
        # even if call ordering changes.
        self._last_round = view.round_index
        self._last_prediction = None
        result = self._graph.invoke({"view": view})
        self._last_prediction = result.get("prediction")
        return result["action"]

    def predict_opponent(self, view: PlayerView) -> Action | None:
        """Returns the SCoT opponent prediction recorded for the current round."""
        if self.scot and self._last_round == view.round_index:
            return self._last_prediction
        return None
