"""End-to-end pipeline tests using the offline mock provider (no API keys)."""

from __future__ import annotations

from llmgames.config.schema import CacheSpec, GameSelector, ModelSpec, RunSpec
from llmgames.loop.tournament import run_tournament
from llmgames.results import generate_results_md


def _mock_run(tmp_path, mode: str = "base", reasoning: bool = False) -> RunSpec:
    """Builds a fully offline run spec writing into ``tmp_path``."""
    return RunSpec(
        name="test_run",
        rounds=4,
        seed=1,
        mode=mode,
        reasoning=reasoning,
        models=[ModelSpec(id="mock_a", provider="mock", model="mock-model")],
        opponents=["tit_for_tat", "always_defect"],
        self_play=True,
        games=GameSelector(canonical=["pd", "bos"], all_144=False),
        cache=CacheSpec(backend="memory"),
        output_dir=str(tmp_path),
    )


def test_run_tournament_and_results(tmp_path):
    """A mock run produces a rounds CSV, a results.md, and figures."""
    run = _mock_run(tmp_path)
    result = run_tournament(run)
    assert result.rounds_csv.exists()
    assert result.results

    results_md = generate_results_md(run, result)
    assert results_md.exists()
    content = results_md.read_text(encoding="utf-8")
    assert "Table 1" in content and "Payoff matrices" in content
    figures = list((result.output_dir / "figures").glob("*.png"))
    assert figures, "expected at least one rendered figure"


def test_scot_run_records_predictions(tmp_path):
    """SCoT mode records opponent predictions, yielding a prediction-accuracy table."""
    run = _mock_run(tmp_path, mode="scot")
    result = run_tournament(run)
    from llmgames.metrics import prediction

    table = prediction.prediction_accuracy_table(result.results)
    assert not table.empty


def test_scot_reasoning_writes_thoughts(tmp_path):
    """SCoT + reasoning writes a thoughts.csv with prediction + reply text per round."""
    import pandas as pd

    run = _mock_run(tmp_path, mode="scot", reasoning=True)
    result = run_tournament(run)
    assert result.thoughts_csv is not None and result.thoughts_csv.exists()
    tdf = pd.read_csv(result.thoughts_csv)
    assert {"player", "opponent", "round", "predicted", "predict_text", "decide_text"} <= set(tdf.columns)
    assert len(tdf) > 0
