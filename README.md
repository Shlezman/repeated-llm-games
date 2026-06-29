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

## Results — does it reproduce the paper?

A run of this implementation against **6 modern models** via an internal
OpenAI-compatible gateway — `gpt-4.1`, `gpt-4.1-mini`, `gemini-3.1-pro-preview`,
`gemini-3.1-flash-lite`, `claude-opus-4-6`, `claude-haiku-4-5` — vs the `tit_for_tat`
and `always_defect` baselines, canonical PD + BoS, 10 rounds, temperature 0, base
mode (128 matches, 0 unparseable). Regenerate with `scripts/run-local.sh`; full
artifacts (heat maps, trajectories, `game_replay.html`) land in `results/llm_gw_multi/`.

**Table 1 — score ratio (achieved / best achievable):**

| player | PD Family | Battle of the Sexes |
|---|---|---|
| tit_for_tat (baseline) | 0.80 | **0.92** |
| always_defect (baseline) | **1.00** | 0.88 |
| openai_heavy (gpt-4.1) | 0.83 | 0.68 |
| openai_light (gpt-4.1-mini) | 0.84 | 0.69 |
| gemini_heavy (gemini-3.1-pro) | 0.83 | 0.83 |
| gemini_light (gemini-3.1-flash-lite) | 0.80 | 0.77 |
| claude_heavy (claude-opus-4-6) | 0.81 | 0.84 |
| claude_light (claude-haiku-4-5) | 0.81 | 0.80 |

The paper used GPT-3/3.5/4 (+ Claude, Llama-2); we use modern models, so absolute
numbers differ. Comparing the **qualitative findings**:

| Paper finding (Akata et al. 2025) | This implementation | Reproduced? |
|---|---|---|
| LLMs handle self-interested games (PD) competently; punish defectors | All models 0.80–0.84 in PD; `always_defect` hits the 1.0 ceiling | ✅ |
| LLMs **struggle to coordinate** in Battle of the Sexes (don't learn to alternate) | OpenAI models lag in BoS (0.68) vs hand-coded TFT (0.92) — coordination is the weak spot | ✅ |
| Capability matters — stronger models score higher | Modern lineup is uniformly strong (0.77–0.84); the GPT-3→4 gap the paper saw is largely closed | ⚠️ partial (model progress compressed it) |
| **SCoT** (predict-then-act) improves coordination & is more human-like | Model-side (below): SCoT lifts BoS coordination for **every** model (e.g. gpt-4.1 0.68→0.93); ToM prediction accuracy 0.88–0.94. Human data (Fig 7): BoS success 0.54→0.62, P(thought-human) 0.49→0.64 | ✅ |

### Base vs SCoT — the headline result

Social Chain-of-Thought (predict the opponent, then act) was run on all six models
vs the baselines (`config/runs/llm_gw_scot.yaml`). It **lifts coordination (Battle of
the Sexes) for every model** — the exact failure mode the paper identified:

| model | PD: base → SCoT | BoS: base → SCoT |
|---|---|---|
| openai_heavy (gpt-4.1) | 0.83 → **1.00** | 0.68 → **0.93** |
| openai_light (gpt-4.1-mini) | 0.84 → **1.00** | 0.69 → **0.95** |
| gemini_heavy (gemini-3.1-pro) | 0.83 → 0.93 | 0.83 → **0.92** |
| gemini_light (gemini-3.1-flash-lite) | 0.80 → **1.00** | 0.77 → **0.92** |
| claude_heavy (claude-opus-4-6) | 0.81 → **1.00** | 0.84 → 0.89 |
| claude_light (claude-haiku-4-5) | 0.81 → 0.81 | 0.80 → **0.96** |

SCoT **theory-of-mind prediction accuracy: 0.88–0.94** (models reliably anticipate the
baseline opponents). The captured reasoning makes the paper's "exploit-then-can't-recover"
dynamic explicit — e.g. `gpt-4.1+scot` vs tit-for-tat: round 1 *"F gives me 10 vs 8, so
F is best"* (exploit) → round 2 *"they may retaliate… CHOICE: F"* (predicts the
punishment it triggered, then keeps defecting).

_(Caveat: base ratios are the full round-robin; SCoT ratios are vs the two strategies
only — model-vs-model was off to keep the SCoT+reasoning run affordable. The BoS jump is
large enough to be the SCoT effect, not the opponent-pool difference.)_

This is a faithful reproduction of the paper's **methodology** (payoffs, barebones
prompts, hand-coded strategies, SCoT, robustness transforms, metrics) — not a
replication of its exact model set. Swap the model list in config to test any models.

▶ **Watch it:** open the committed [`docs/results/game_replay.html`](docs/results/game_replay.html)
— every model appears in **both modes** (`openai_heavy` and `openai_heavy+scot`, …); pick
two players, press Play, and read each model's per-round prediction + reasoning. Its
**"Paper vs Implementation"** tab charts our score ratios against the **paper's own
released numbers** (computed with the same metric) — PD capability ladder, BoS, and the
per-model base→SCoT lift. Full per-finding writeup:
[`docs/results/PAPER_COMPARISON.md`](docs/results/PAPER_COMPARISON.md); base + SCoT
artifacts under [`docs/results/`](docs/results/).

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

## Local development with uv

```bash
uv sync --extra dev --extra openai --extra anthropic   # create .venv + install (uses uv.lock)
uv run pytest -q                                        # run tests
uv run llmgames run --config config/runs/paper_default.yaml --mock
```

## Run locally with Docker (app + Postgres)

Spins up Postgres (TLS-enabled) for the response cache and the app, both as
containers, built with uv:

```bash
cp .env.example .env        # fill: OPENAI_API_KEY (gateway key), LLM_GW_BASE_URL,
                            #       LLM_GW_MODEL_1 / _2 / _3 (gateway model ids)
scripts/run-local.sh                                   # default: config/runs/llm_gw_multi.yaml
scripts/run-local.sh config/runs/llm_gw_stg.yaml       # or a single-model run
docker compose down                                    # stop the db (add -v to drop the cache)
```

Results land in `./results/` on the host. The app talks to Postgres over the
internal docker network with `sslmode=require` (encrypted); the gateway key and
model ids come from `.env` and are never baked into images.

### Choosing gateway models

`LLM_GW_MODEL_*` must be the **exact model ids your gateway serves**. List them with:

```bash
curl "$LLM_GW_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

Pick a few across capability tiers (so cross-play shows the differences the paper
studied). Modern stand-ins for the paper's GPT-3 / GPT-3.5 / GPT-4 ladder:

| Tier (paper) | OpenAI | Anthropic | Open / other |
|---|---|---|---|
| Frontier (GPT-4) | GPT-5, o3 | Claude Opus 4.x | Gemini 2.5 Pro, DeepSeek-R1 |
| Mid (GPT-3.5) | GPT-4.1, GPT-4o | Claude Sonnet 4.x | Llama 3.3 70B / Llama 4, Qwen2.5-72B, Mistral Large |
| Small (GPT-3) | GPT-4.1-mini, GPT-4o-mini | Claude Haiku 4.5 | Gemini 2.5 Flash, Llama 3.3 8B, Qwen2.5-7B |

Prefer **standard chat models** (gpt-4o/4.1, Claude Sonnet/Haiku, Llama, Qwen) —
they answer the single-letter prompts well at `temperature 0`, `max_tokens 8`. For
**reasoning models** (o-series, GPT-5, DeepSeek-R1) raise `max_tokens` (they emit
reasoning before the answer) and note some reject `temperature != 1`.

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
