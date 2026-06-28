# llmgames — Repeated 2×2 Games with LLMs

A **model-agnostic** reproduction of:

> Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., & Schulz, E. (2025).
> *Playing repeated games with large language models.* **Nature Human Behaviour.**
> doi:[10.1038/s41562-025-02172-y](https://doi.org/10.1038/s41562-025-02172-y) ·
> arXiv:[2305.16867](https://arxiv.org/abs/2305.16867)

Ports the authors' released code ([github.com/eliaka/repeatedgames](https://github.com/eliaka/repeatedgames),
MIT) into a clean package built on the **LangChain + LangGraph** ecosystem. Players are
LangGraph agents, models are constructed with LangChain `init_chat_model` (so **any**
backend is a config input — no model name hardcoded), prompts live in **markdown
templates**, and responses are cached via LangChain's global LLM cache.

## What it reproduces

- The **144-game ordinal taxonomy** (Win-win, Biased, Unfair, Cyclic, Second Best,
  Tragic, PD Family) plus canonical **Prisoner's Dilemma** and **Battle of the Sexes**
  with the paper's exact payoffs.
- **Hand-coded strategies**: always-cooperate/defect, defect-once, alternate,
  tit-for-tat (+ two-tats, suspicious, reverse, hard), naive probers.
- **Base and SCoT** (Social Chain-of-Thought: predict the opponent, then act) — modelled
  as a LangGraph turn graph.
- **Robustness transforms** (game-preserving): seeded option-order randomization, utility
  label swaps, cover stories (cooking / project), and random labels for the 144-game arm.
- **Metrics & figures**: Table 1 score ratios, PD/BoS heat maps, trajectories, SCoT
  prediction accuracy, and the Fig 7 human-vs-LLM analysis over the **real**
  195-participant dataset.

## Install

```bash
python3.10+ -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,anthropic]"   # add provider extras you need; postgres optional
cp .env.example .env                         # fill in keys + model ids (never commit)
```

## Models are inputs — adding one is config-only

No model identifier is hardcoded. A model is config behind LangChain's `init_chat_model`:

```yaml
models:
  - id: model_a
    provider: ${LLMGAMES_PROVIDER_A}   # openai | anthropic | ... ; empty = infer from model id
    model: ${LLMGAMES_MODEL_A}         # ← the input
    params: { temperature: 0.0, max_tokens: 8 }
```

`init_chat_model` routes to any LangChain-supported backend (OpenAI, Anthropic, Azure,
Bedrock, Ollama/local, …). Keys are read from the environment by the integration package.
Use `provider: mock` (or `--mock`) for offline runs with no keys.

## Players are LangGraph agents

Each player's turn is a compiled `StateGraph` ([players/llm_player.py](src/llmgames/players/llm_player.py)):

- **base**: a single `decide` node.
- **SCoT**: `predict → decide`, where `decide` is conditioned on the predicted opponent
  move (and falls back to the unconditioned decision if the prediction is unparseable).

Each node runs an LCEL chain `prompt | model | parser`, where `prompt` is a markdown
template and `parser` robustly extracts a single action from the reply.

## Prompts live in markdown

All wording is in [src/llmgames/prompts/templates/](src/llmgames/prompts/templates) —
edit prompts without touching code:

| file | role |
|---|---|
| `rules_intro.md`, `rules_outcome.md` | the barebones rules block (all four outcomes) |
| `history_line.md` | one appended round of history |
| `base_decision.md` | base-mode decision question |
| `scot_predict.md`, `scot_decide.md` | the SCoT predict-then-act pair |

Framing (labels, unit word, cover story) fills the placeholders; payoff numbers always
come from the `Game`, so framing can never change the underlying game.

## Run & reproduce

```bash
python -m llmgames.cli run --config config/runs/paper_default.yaml --mock   # offline
python -m llmgames.cli run --config config/runs/paper_default.yaml          # canonical PD+BoS
python -m llmgames.cli run --config config/runs/paper_scot.yaml             # SCoT
python -m llmgames.cli run --config config/runs/all_144.yaml                # full taxonomy
python -m llmgames.cli run --config config/runs/robustness_cooking.yaml
```

Each run writes `results/<name>/`: `rounds.csv` (per-round), `figures/*.png`, and
`results.md` (config + payoff matrices + Table 1 + embedded figures).
`notebooks/reproduce_figures.ipynb` regenerates artifacts interactively.

## Architecture

```
engine/      Game (frozen) · payoff scoring · 144-game taxonomy loader · outcome classifiers
providers/   models.py (init_chat_model factory) · cache.py (Postgres / in-memory LLM cache)
players/     Player protocol · LLMPlayer (LangGraph turn agent) · hand-coded strategies · parser
prompts/     templates/*.md · loader (PromptTemplate) · render (rules/history) · transforms (framing)
loop/        single-match loop · round-robin tournament (config → results + CSV)
metrics/     score ratio (Table 1) · PD · BoS · prediction accuracy · human (Fig 7)
viz/         heat maps · trajectories · bar charts
config/      pydantic RunSpec + YAML loader (${ENV} expansion)
results.py   results.md generator   ·   cli.py   command-line entrypoint
```

## Reproducibility

- **Config-driven**: swap the model list, rerun the whole study on new models.
- **Deterministic**: temperature 0 + LangChain global LLM cache (Postgres via SQLAlchemy /
  in-memory) make re-runs identical and cheap.
- **Seeded** randomized arms (option order, random labels, probabilistic strategies).

## Faithful vs adapted

| Faithful to the paper | Adapted (and why) |
|---|---|
| Payoffs, 144-game taxonomy, families | Single-token completion → robust parse + retry (chat models don't emit one token) |
| Barebones prompt text, J/F labels, 10 rounds | Prompt strings → markdown templates |
| SCoT predict-then-act wording | Hand-rolled players → LangGraph turn agents |
| Strategies (TFT family, defect-once, alternate) | Hardcoded models + inline keys → `init_chat_model` + env |
| Robustness transforms (order/labels/cover story) | No cache → LangChain global LLM cache (Postgres / memory) |
| Real 195-participant human data (never fabricated) | Score-ratio benchmark made explicit; Fig 7 success derived from score |

## Security & data handling

- API keys and the database DSN come from the environment; nothing is committed.
- The Postgres cache uses parameterized SQL, a validated table identifier, and requests
  TLS (`PGSSLMODE`, default `require`).
- The Fig 7 harness **only analyses real participant data** — it never synthesizes humans.

## License

MIT. Retains the original authors' copyright; see [LICENSE](LICENSE). The 144-game CSV and
human dataset are reproduced from the original MIT release.
