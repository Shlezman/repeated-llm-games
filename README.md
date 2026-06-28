# llmgames — Repeated 2×2 Games with LLMs

A **model-agnostic** reproduction and refactor of:

> Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., & Schulz, E. (2025).
> *Playing repeated games with large language models.* **Nature Human Behaviour.**
> doi:[10.1038/s41562-025-02172-y](https://doi.org/10.1038/s41562-025-02172-y) ·
> arXiv:[2305.16867](https://arxiv.org/abs/2305.16867)

This codebase **ports** the authors' released code
([github.com/eliaka/repeatedgames](https://github.com/eliaka/repeatedgames), MIT) into a
clean, SOLID, provider-agnostic package. The original hardcoded specific models and inline
API keys and relied on single-token completions; here **any** chat/completion model is a
config input behind a provider adapter, secrets come from the environment, responses are
cached in Postgres, and model replies are parsed robustly.

## What it reproduces

- **The six+ game families** of the 144-game ordinal taxonomy (Win-win, Biased, Unfair,
  Cyclic, Second Best, Tragic, PD Family) plus the canonical **Prisoner's Dilemma** and
  **Battle of the Sexes** with the paper's exact payoffs.
- **Hand-coded strategies**: always-cooperate/defect, defect-once, alternate, tit-for-tat
  (+ two-tats, suspicious, reverse, hard), naive probers.
- **Base and SCoT** (Social Chain-of-Thought: predict the opponent, then act).
- **Robustness transforms** (game-preserving): seeded option-order randomization, utility
  label swaps (points/dollars/coins), cover stories (cooking competition / collaborative
  project).
- **Metrics & figures**: Table 1 score ratios, PD defection/score heat maps, BoS
  collaboration / P1-preferred heat maps, round trajectories, SCoT prediction accuracy
  (theory-of-mind), and the Fig 7 human-vs-LLM analysis over the **real** 195-participant
  dataset.

## Install

```bash
python3.10+ -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,litellm]"        # add: anthropic openai postgres as needed
cp .env.example .env                     # fill in keys + model ids (never commit)
```

## Models are inputs — adding one is config-only

No model identifier is hardcoded anywhere. A model is three fields in a run YAML:

```yaml
models:
  - id: model_a
    provider: litellm          # litellm (default, any backend) | anthropic | openai | mock
    model: ${LLMGAMES_MODEL_A} # ← the input; from env, or write the id directly
    params: { temperature: 0.0, max_tokens: 8 }
```

- **`litellm`** routes to OpenAI, Anthropic, Azure, OpenRouter, local/OpenAI-compatible
  servers, etc. — so "any model" is real. Keys are read from the environment by the SDK.
- To add a brand-new backend, register a factory: `register_provider("name", factory)` in
  `providers/registry.py` — no call sites change.

## Run & reproduce

```bash
# Offline smoke test (mock provider, in-memory cache, no keys/DB):
python -m llmgames.cli run --config config/runs/paper_default.yaml --mock

# Real run (set LLMGAMES_MODEL_A/B + keys + DATABASE_URL first):
python -m llmgames.cli run --config config/runs/paper_default.yaml   # canonical PD + BoS
python -m llmgames.cli run --config config/runs/paper_scot.yaml      # SCoT mode
python -m llmgames.cli run --config config/runs/all_144.yaml         # full taxonomy (Table 1)
python -m llmgames.cli run --config config/runs/robustness_cooking.yaml
```

Each run writes to `results/<name>/`:
- `rounds.csv` — one tidy row per round (actions, points, totals, SCoT predictions).
- `figures/*.png` — all heat maps, trajectories, and bar charts.
- `results.md` — a single summary: config, payoff matrices, Table 1, and embedded figures.

`notebooks/reproduce_figures.ipynb` regenerates each artifact interactively.

## Architecture

```
engine/      Game (frozen) · payoff scoring · 144-game taxonomy loader · outcome classifiers
providers/   Provider protocol · LiteLLM/Anthropic/OpenAI/Mock adapters · Postgres+memory cache
players/     Player protocol · LLMPlayer (base/SCoT) · hand-coded strategies · robust parser
prompts/     barebones payoff builder · SCoT predict-then-act · game-preserving transforms
loop/        single-match loop · round-robin tournament (config → results + CSV)
metrics/     score ratio (Table 1) · PD · BoS · prediction accuracy · human (Fig 7)
viz/         heat maps · trajectories · bar charts
config/      pydantic RunSpec + YAML loader (with ${ENV} expansion)
results.py   results.md generator   ·   cli.py   command-line entrypoint
```

## Reproducibility

- **Config-driven**: swap the model list and rerun the whole study on new models.
- **Deterministic**: temperature 0 + a Postgres response cache keyed on
  `(provider, model, params, prompt)` make re-runs identical and cheap. Use `--mock` or
  `cache.backend: memory` for offline/test runs.
- **Seeded** randomized arms (option order, probabilistic strategies).

## Faithful vs adapted

| Faithful to the paper | Adapted (and why) |
|---|---|
| Payoffs, 144-game taxonomy, families | Single-token completion → robust regex parse + retry (chat models don't emit one token) |
| Barebones prompt text, J/F labels, 10 rounds | Models hardcoded → provider adapters + config |
| SCoT predict-then-act wording | Inline API keys → environment variables |
| Strategies (TFT family, defect-once, alternate) | SQLite/no cache → Postgres cache (parameterized SQL, TLS) |
| Robustness transforms (order/labels/cover story) | R analysis → Python metrics + notebook |
| Real 195-participant human data (never fabricated) | Score ratio benchmark made explicit (`best_response` default, `max_cell` optional) |

## Security & data handling

- API keys and the database DSN come from environment variables; nothing is committed.
- Postgres cache uses parameterized queries and requests TLS (`PGSSLMODE`, default `require`).
- The Fig 7 harness **only analyses real participant data** — it never synthesizes humans.
  Without the dataset it skips the section with a clear note.

## License

MIT. This adaptation retains the original authors' copyright; see [LICENSE](LICENSE). The
144-game CSV and human dataset are reproduced from the original MIT release.
