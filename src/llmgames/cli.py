"""Command-line entrypoint: run a study and regenerate all artifacts.

Usage:
    python -m llmgames.cli run --config config/runs/paper_default.yaml
    python -m llmgames.cli run --config <cfg> --mock   # offline, no API keys
"""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

from .config.loader import load_run_spec
from .config.schema import RunSpec
from .loop.tournament import run_tournament
from .results import generate_results_md
from .viz.comparison import build_comparison
from .viz.replay import generate_replay_html

_PAPER_PD = Path("data/paper/pd.csv")
_PAPER_BOS = Path("data/paper/bos.csv")


def _maybe_comparison(rounds: list[Path]) -> dict | None:
    """Builds the paper-vs-implementation comparison if the paper data is present.

    Args:
        rounds: Round CSVs; the first is treated as the base run, a second as SCoT.

    Returns:
        The comparison dict, or None when paper data is unavailable.
    """
    if not (_PAPER_PD.exists() and _PAPER_BOS.exists()):
        return None
    scot = rounds[1] if len(rounds) > 1 else None
    return build_comparison(_PAPER_PD, _PAPER_BOS, rounds[0], scot)


def _force_mock(run: RunSpec) -> RunSpec:
    """Returns a copy of ``run`` with every model on the mock provider and memory cache.

    Args:
        run: The original run spec.

    Returns:
        A run spec safe to execute without API keys or a database.
    """
    models = [model.model_copy(update={"provider": "mock"}) for model in run.models]
    cache = run.cache.model_copy(update={"backend": "memory"})
    return run.model_copy(update={"models": models, "cache": cache})


def _cmd_run(args: argparse.Namespace) -> None:
    """Executes the ``run`` subcommand."""
    run = load_run_spec(args.config)
    if args.mock:
        run = _force_mock(run)
    result = run_tournament(run)
    logging.getLogger("llmgames.cli").info("Generating results.md, figures, and game_replay.html...")
    results_md = generate_results_md(run, result)
    print(f"Rounds CSV : {result.rounds_csv}")
    print(f"Results doc: {results_md}")
    logging.getLogger("llmgames.cli").info("DONE — open %s (and game_replay.html beside it)", results_md)


def _cmd_replay(args: argparse.Namespace) -> None:
    """Executes the ``replay`` subcommand: (re)generate the HTML replay from CSV(s)."""
    rounds = [Path(r) for r in args.rounds]
    out = Path(args.out) if args.out else rounds[0].with_name("game_replay.html")
    name = args.name or (rounds[0].parent.name if len(rounds) == 1 else "combined")
    path = generate_replay_html(rounds, out, run_name=name, comparison=_maybe_comparison(rounds))
    print(f"Replay: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="llmgames", description="Repeated 2x2 games with LLMs.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a study and write results.md + figures.")
    run_parser.add_argument("--config", required=True, help="Path to a run YAML config.")
    run_parser.add_argument(
        "--mock", action="store_true", help="Use the mock provider + in-memory cache (offline)."
    )
    run_parser.set_defaults(func=_cmd_run)

    replay_parser = sub.add_parser("replay", help="Generate the animated HTML replay from rounds.csv(s).")
    replay_parser.add_argument(
        "--rounds", required=True, nargs="+", help="One or more rounds.csv paths (merged into one HTML)."
    )
    replay_parser.add_argument("--out", help="Output HTML path (default: game_replay.html beside the CSV).")
    replay_parser.add_argument("--name", help="Run name shown in the header.")
    replay_parser.set_defaults(func=_cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
