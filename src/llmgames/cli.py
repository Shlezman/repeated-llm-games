"""Command-line entrypoint: run a study and regenerate all artifacts.

Usage:
    python -m llmgames.cli run --config config/runs/paper_default.yaml
    python -m llmgames.cli run --config <cfg> --mock   # offline, no API keys
"""

from __future__ import annotations

import argparse
from dataclasses import replace as _dc_replace  # noqa: F401  (kept for future use)

from .config.loader import load_run_spec
from .config.schema import RunSpec
from .loop.tournament import run_tournament
from .results import generate_results_md


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
    results_md = generate_results_md(run, result)
    print(f"Rounds CSV : {result.rounds_csv}")
    print(f"Results doc: {results_md}")


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
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
