"""Paper-style robustness checks: does a finding survive prompt / payoff / horizon change?

Three arms mirroring the original paper's robustness experiments:
  1. Framing invariance — canonical PD (vs defect_once) + BoS (vs alternate), re-run under
     cover stories, unit-word swap, label swap, and no option-order shuffle.
  2. BoS payoff sweep — the 5 interpolated matrices (JJ 10,7 -> 9,8 -> 8,8 -> 8,9 -> 7,10).
  3. PD ending-probability — indefinite horizon with a per-round end chance
     (0/1/40/60/80%), 20 rounds, prompt-only framing (the loop still runs a fixed 20,
     as in the paper; 0% is the fixed-round baseline).

Writes long-format robustness.csv (arm, variant, model, game, metric, value), consumed by
the "Robustness" tab of game_replay.html. Uses only public engine/player APIs.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .engine.families import canonical_bos, canonical_pd
from .engine.game import ACTION_A, ACTION_B, Game
from .engine.payoff import is_valid_action
from .loop.match import play_match
from .players.llm_player import LLMPlayer
from .prompts.transforms import build_framing, order_sequence
from .players.strategies import make_strategy
from .providers.cache import configure_cache
from .providers.models import build_chat_model

_LOGGER = logging.getLogger(__name__)

# (variant label, build_framing kwargs, randomize_order)
_FRAMINGS = [
    ("base", {}, True),
    ("cooking cover story", {"cover_story": "cooking"}, True),
    ("project cover story", {"cover_story": "project"}, True),
    ("unit=dollars", {"utility_label": "dollars"}, True),
    ("labels=Q/X", {"labels": ("Q", "X")}, True),
    ("no order-shuffle", {}, False),
]
_ENDING_PROBS = [0, 1, 40, 60, 80]


def _bos_sweep() -> list[Game]:
    """The paper's 5 interpolated Battle-of-the-Sexes matrices (off-diagonal 0,0)."""
    games = []
    for jj in [(10, 7), (9, 8), (8, 8), (8, 9), (7, 10)]:
        ff = (jj[1], jj[0])
        games.append(
            Game(
                name=f"JJ={jj[0]},{jj[1]}",
                family="Robustness",
                payoffs={
                    (ACTION_A, ACTION_A): jj,
                    (ACTION_B, ACTION_B): ff,
                    (ACTION_A, ACTION_B): (0, 0),
                    (ACTION_B, ACTION_A): (0, 0),
                },
            )
        )
    return games


def _rates(result) -> tuple[float, float]:
    """Returns (player-1 defection rate, coordination rate) over valid rounds."""
    n = d = c = 0
    for a1, a2 in zip(result.actions_p1, result.actions_p2):
        if not (is_valid_action(a1) and is_valid_action(a2)):
            continue
        n += 1
        d += a1 == ACTION_B
        c += a1 == a2
    return (d / n, c / n) if n else (0.0, 0.0)


def run_robustness(run, output_dir: str | Path) -> Path:
    """Runs the three robustness arms for every model in ``run`` and writes robustness.csv.

    Args:
        run: A :class:`~llmgames.config.schema.RunSpec` (supplies models, gateway, cache, seed).
        output_dir: Directory to write robustness.csv into.

    Returns:
        Path to the written robustness.csv.
    """
    configure_cache(run.cache.backend, run.cache.dsn)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [
        (
            s.id,
            build_chat_model(
                s.provider, s.model, temperature=s.params.temperature,
                max_tokens=s.params.max_tokens, base_url=s.base_url, timeout=s.params.request_timeout,
            ),
        )
        for s in run.models
    ]

    rows: list[dict] = []

    def add(arm, variant, model, game, metric, value):
        rows.append({"arm": arm, "variant": variant, "model": model, "game": game,
                     "metric": metric, "value": round(value, 3)})

    total = (len(_FRAMINGS) + len(_bos_sweep()) + len(_ENDING_PROBS)) * len(models)
    done = 0
    _LOGGER.info(
        "Robustness started: %d models x (%d framing + %d payoff + %d ending) = %d matchsets",
        len(models), len(_FRAMINGS), len(_bos_sweep()), len(_ENDING_PROBS), total,
    )

    # Arm 1 — framing invariance (PD vs defect_once, BoS vs alternate)
    for vname, kw, randomize in _FRAMINGS:
        framing = build_framing(**kw)
        orders = order_sequence(seed=run.seed, num_rounds=10, randomize=randomize)
        for mid, model in models:
            pd_res = play_match(canonical_pd(), LLMPlayer(mid, model, framing),
                                make_strategy("defect_once", seed=run.seed), num_rounds=10, orders=orders)
            bos_res = play_match(canonical_bos(), LLMPlayer(mid, model, framing),
                                 make_strategy("alternate", seed=run.seed), num_rounds=10, orders=orders)
            add("framing", vname, mid, "PD", "defection_rate", _rates(pd_res)[0])
            add("framing", vname, mid, "BoS", "coordination_rate", _rates(bos_res)[1])
            done += 1
            _LOGGER.info("[%d/%d] framing '%s' | %s", done, total, vname, mid)

    # Arm 2 — BoS payoff sweep (vs alternate)
    base_framing = build_framing()
    orders10 = order_sequence(seed=run.seed, num_rounds=10, randomize=True)
    for g in _bos_sweep():
        for mid, model in models:
            res = play_match(g, LLMPlayer(mid, model, base_framing),
                             make_strategy("alternate", seed=run.seed), num_rounds=10, orders=orders10)
            add("payoff_sweep", g.name, mid, "BoS", "coordination_rate", _rates(res)[1])
            done += 1
            _LOGGER.info("[%d/%d] payoff %s | %s", done, total, g.name, mid)

    # Arm 3 — PD ending probability (indefinite horizon, 20 rounds, vs tit_for_tat)
    orders20 = order_sequence(seed=run.seed, num_rounds=20, randomize=True)
    for p in _ENDING_PROBS:
        framing = base_framing if p == 0 else replace(
            base_framing,
            horizon=(f"You will play an indefinite number of rounds with the same player. "
                     f"There is a {p}% probability that the game will end after each round."),
        )
        for mid, model in models:
            res = play_match(canonical_pd(), LLMPlayer(mid, model, framing),
                             make_strategy("tit_for_tat", seed=run.seed), num_rounds=20, orders=orders20)
            add("ending_prob", f"{p}% end/round", mid, "PD", "defection_rate", _rates(res)[0])
            done += 1
            _LOGGER.info("[%d/%d] ending %s%% | %s", done, total, p, mid)

    path = out / "robustness.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _LOGGER.info("Robustness FINISHED: %d rows -> %s", len(rows), path)
    return path
