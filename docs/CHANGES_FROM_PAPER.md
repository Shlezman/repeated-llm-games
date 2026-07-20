# Changes from the paper's released code

This document compares this repository against the code released for
*Playing repeated games with large language models* (Akata et al., 2025, Nature
Human Behaviour) at [eliaka/repeatedgames](https://github.com/eliaka/repeatedgames)
(MIT), and explains every difference and why it exists.

**Method.** Six independent code-reading passes, one per dimension (games/payoffs,
strategies, prompts/SCoT, game loop/protocol, metrics/analysis, engineering),
each reading both codebases side by side. Every claim below cites the file (and
where useful the line) on both sides. `paper:` paths are relative to the
`repeatedgames` repo; `local:` paths to this one.

**Verdict in one line.** The experimental protocol — payoffs, strategies, prompt
wording, 10-round loop, temperature 0 — is reproduced faithfully (much of it
byte-identical); around it, the paper's one-off scripts were consolidated into a
tested, config-driven package, and eleven defects in the released code were
corrected along the way.

---

## 1. Reproduced faithfully

- **144-game taxonomy** — `config/games.csv` is *byte-identical* to
  `all_games/144_games_info.csv` (verified with `diff`).
- **Canonical payoffs** — PD (8,8 / 0,10 / 10,0 / 5,5) and BoS (10,7 / 0,0 /
  0,0 / 7,10): `src/llmgames/engine/families.py` == `pd/query_main.py:150-163`,
  `bos/query_main.py:192-205`.
- **Prompt wording verbatim** — rules intro, all four outcome lines (same JJ/JF/FJ/FF
  order), history line, round cue, SCoT predict and decide questions:
  `prompts/templates/*.md` == the strings in `pd/query_main.py:169-217` and
  `bos/query_main.py:247-249`.
- **Protocol** — 10 rounds, full history re-sent each round, both players queried
  independently with no within-round leakage, shared per-round option order,
  temperature 0.0, round-robin over the full roster product including self-play.
- **Strategies** — every released strategy reproduced move-for-move, *including
  the quirks*: `hard_tft` defects on move 1 (the paper's comment says cooperate,
  its code defects — we follow the code, documented in
  `players/strategies.py:7-8`); the naive probers check the defect probability
  *before* copying, in the paper's order.
- **Invalid-answer sentinel** — `-9999` for both players when either action is
  illegal (`engine/payoff.py` == `all_games/query_play_all_games.py:177-190`),
  still recorded per round like the paper.
- **Robustness arms** — BoS payoff sweep uses the paper's exact five interpolated
  matrices (JJ 10,7 → 9,8 → 8,8 → 8,9 → 7,10, FF mirrored, off-diagonal 0,0);
  PD ending-probability keeps the paper's design of a prompt-only "indefinite
  horizon, p% end chance" sentence over a fixed 20-round loop; cover stories
  (cooking competition, collaborative project), label swap and unit-word swap
  arms all exist.
- **Human-study metrics** — BoS "coordinated" = nonzero score, P(Human) per
  game × opponent: `metrics/human.py` == `repeated_games.Rmd`.

## 2. Paper defects corrected

Each of these is a bug in the *released paper code*, with the exact evidence.

| # | Defect in the paper code | Paper evidence | What this repo does |
|---|---|---|---|
| 1 | **Player 2's rules text contradicts scoring in asymmetric games.** `question_2` swaps only the `_p1/_p2` suffixes without transposing rows/columns, so P2 is told off-diagonal payoffs that the scorer never awards — in most of the 144 games. | `all_games/query_play_all_games.py:238-239` vs the scorer at `:185-188` | `Game.payoff_for(am_row_player=False)` performs the real transpose; prompts and scoring derive from the same object (`engine/game.py:73-77`, `prompts/render.py:35-49`). |
| 2 | **`-9999` poisons cumulative totals.** One unparseable reply drags `total1/total2` down by 9999 for the rest of the match. | `pd/query_main.py:212-213`; `all_games/query_play_all_games.py:266-287` | Totals sum only rounds where both actions are valid; the per-round sentinel is kept for detection (`players/base.py:115-157`). |
| 3 | **Project cover story: shown matrix is the inverse of the scored matrix.** The prompt says R/R→8,8 and H/H→5,5, but `calculate_points` scores H/H→8,8 and R/R→5,5 — GPT-4 was evaluated under the inverse of the rules it was shown. | `pd/variations_robustness/query_robustness_checks_project.py:157-160` vs `:92-95,122-125` | Impossible by construction: prompt text and scores both come from `game.payoff_for` — a framing cannot change payoffs. |
| 4 | **BoS `act_gpt35` returns a method, not a string.** `return response.choices[0].text.strip` (no parentheses) — the answer can never equal `J`/`F`, so every GPT-3.5 BoS round scores −9999. | `bos/query_main.py:155` | One shared parser path for all models (`players/parser.py`); no per-model string handling exists to get wrong. |
| 5 | **Missing data file.** `query_play_all_games.py` opens `144_games.csv`, which is absent from the released repo (only `144_games_info.csv` exists) — the script cannot run as shipped. | `all_games/query_play_all_games.py:199` | Ships and loads the actual CSV (`engine/families.py:63-78`), path configurable. |
| 6 | **Shipped requirements can't run the shipped code.** `requirements.txt` pins `openai==1.47.0` but every script uses the removed v0 API (`openai.ChatCompletion.create`, `openai.error.*`). | `requirements.txt:4` vs `pd/query_main.py:51,59-67` | Coherent locked dependency set (`pyproject.toml` + `uv.lock`) behind a LangChain abstraction. |
| 7 | **API-key placeholders hardcoded in every script** (`api_key="API_KEY"`), inviting real keys to be committed. | `pd/query_main.py:30,44` and 5+ other scripts | Keys come from the environment only; `.env.example` documents them; nothing key-like in source (`providers/models.py:36-37`). |
| 8 | **Human-study Rmd counts joint *defection* as cooperation.** Fig-7-style "P(Cooperation)" uses `score==5`, but the experiment's own JS awards 8 for mutual cooperation and 5 for mutual defection. | `repeated_games.Rmd:299` vs `human_experiment/experiment/js/script.js:428-435` | `pd_mutual_coop` is `score==8` (`metrics/human.py:67`), derived from the experiment's actual payoff mapping. |
| 9 | **Two different SCoT decide prompts for the two seats.** Player 1 gets "best to choose for you", player 2 "best for you to choose" — same condition, different sentence. | `bos/query_main.py:249` vs `:256` | One template (`scot_decide.md`) for every seat. |
| 10 | **Wrong index passed to strategies in the all-games script.** The *game* index `i-1` is passed where the *round* index belongs (round variable is `j`) — any indexed strategy would be driven by game number. Latent (that script's roster has none). | `all_games/query_play_all_games.py:257,261` vs `:249` | `PlayerView.round_index` is explicit; strategies read the view, no positional-int threading. |
| 11 | **Cross-match strategy state leaks.** BoS alternators are module-level infinite generators never reset between pairings (phase depends on match order; benign only because 10 is even); `grim_trigger` needs an external `__dict__.clear()` the driver must remember. | `bos/query_main.py:172-190,233`; `pd/ending_probs/pd_basic_gpt4_tfts.py:60-73,183` | Strategies are pure functions of the per-match view; a fresh player instance per match (`loop/tournament.py:120-134`). |
| 12 | **Seat-selection copy-paste errors in the released R analysis** — e.g. GPT-4's score labelled as Llama-2's (`total1` where `player2=='act_llama2'`), and mixed round filters within one comparison. | `all_games/all_games.r:34,41,82-84` | Not ported; local metrics are computed per (player, seat) from typed match results. |

## 3. Modernizations (same behaviour, better engineering)

- **One engine instead of 8+ script copies.** The paper duplicates its retry
  loop and game loop verbatim across `pd/`, `bos/`, `all_games/`, four
  ending-probability scripts and three robustness scripts. Here a single package
  (`engine`, `loop`, `players`, `prompts`, `providers`, `metrics`, `viz`) covers
  every experiment; variants are YAML data (`config/runs/*.yaml`), not code copies.
- **Config-driven, model-agnostic providers.** The paper hardcodes model IDs in
  five places (and uses Llama-2-7b in PD but 70b in BoS); here provider+model
  come from validated config (`config/schema.py`), built via `init_chat_model` —
  zero model names in code, any OpenAI-compatible gateway via `base_url`.
- **Reproducibility.** The paper's option-order shuffling is unseeded
  (`random.shuffle`, no seed anywhere) — reruns are unreproducible. Here a master
  seed threads through every stochastic choice (`order_sequence(seed=…)`,
  seeded prober RNGs, seeded label draws).
- **Robust answer extraction.** The paper forces `max_tokens=1` and takes
  `.strip()` — anything not exactly one letter becomes an invalid round. Modern
  chat models rarely emit exactly one token; the prompt now asks for exactly one
  letter and a word-boundary parser extracts it (`players/parser.py`), with
  `CHOICE:` markers for the reasoning mode.
- **Fail-soft error handling.** The paper's hand-rolled retry raises after 5
  attempts and kills the whole tournament; an errored call here costs one
  unparseable round, logged once per error type (`llm_player.py:_safe_invoke`),
  with SDK-native retries and per-call timeouts.
- **Typed game representation.** Frozen `Game` dataclass with completeness
  validation; neutral internal actions (A/B) decoupled from display labels —
  which is precisely the design that makes paper defects #1 and #3 impossible.
- **Strategies relabel for free.** The paper re-implements `defect_once` seven
  times (once per letter set) for robustness; here strategies act on abstract
  actions and relabeling lives in the framing layer — one implementation covers
  all letter sets.

## 4. Additions (absent from the paper)

- **Tests** — the paper repo has none; here 59 tests cover payoffs, parsing,
  strategies, prompts, cache, metrics and the full pipeline.
- **LLM response cache** — Postgres (parameterized SQL, TLS-required DSN) or
  in-memory; at temperature 0 reruns are free and deterministic.
- **Docker/Compose stack + CLI** — `llmgames run/replay/robustness`, offline
  `--mock` mode; one-command local runs.
- **Robustness metrics + figures** — the paper released only raw robustness CSVs
  and pre-rendered PNGs, no analysis code; `robustness.py` computes explicit
  defection/coordination rates and the replay HTML renders them as grouped bar
  charts (paper Fig. 4 style) plus player×player heatmaps (Fig. 5 style).
- **Paper-vs-implementation benchmark** — `viz/comparison.py` re-scores the
  paper's released PD/BoS CSVs and this repo's runs with the *same* metric, shown
  side by side in the replay's comparison tab (see
  [results/PAPER_COMPARISON.md](results/PAPER_COMPARISON.md)).
- **SCoT generalized** — the paper's SCoT exists only as a GPT-4-specific
  function inside the BoS script; here it is a mode (predict → decide LangGraph)
  for any model and any game, with prediction-accuracy metrics.
- **Reasoning capture** — optional one-sentence rationale per move
  (`thoughts.csv`, shown in the replay), no equivalent in the paper.
- **Family filtering** — run any subset of the 144-game taxonomy.
- **grim_trigger everywhere** — the paper defines it only in the
  ending-probability scripts; here it is in the general registry.

## 5. Intentional divergences (different by design)

- **Score-ratio denominator.** The paper's Table-1 normalization divides by the
  empirical max per-round payoff × 10 (`all_games.r:10-22`). Default here is the
  stricter *best-response* benchmark (achievable given the opponent's actual
  play); the paper-style `max_cell` method is retained for parity
  (`metrics/score_ratio.py`).
- **Unparseable replies render as `?`** in later history instead of splicing the
  model's raw text into both players' prompts (the paper echoes arbitrary model
  output verbatim into future prompts).
- **SCoT fallback.** If the predict step is unparseable, the decide step falls
  back to the unconditioned prompt instead of injecting garbage as a stated
  belief (the paper splices the raw prediction string unconditionally).
- **Robustness design is one-factor-at-a-time.** The paper crosses 3 cover
  stories × 6 label pairs × 3 unit words (54 cells per game); here each factor is
  varied alone against base (6 variants) — much cheaper, isolates factors, but
  does not reproduce the full grid.
- **Ending-probability set is {0, 40, 60, 80}%.** The 0% arm is the fixed-round
  baseline; the paper's near-indefinite 1% arm is not reproduced. Opponent set
  reduced to tit-for-tat (the paper ran 9 hand-coded opponents).
- **Order shuffling is uniform.** The paper shuffles option order in PD main but
  not in BoS/TFT/ending-prob scripts; here it is one seeded knob applied
  consistently, with an explicit "no order-shuffle" control variant.
- **No inferential statistics.** The paper's t-tests/Bayes factors/χ² are not
  ported (their R script also contains the seat-selection defects noted above);
  this repo reproduces behavioural metrics and figures, not hypothesis tests.
- **Minor prompt paraphrases.** Ending-prob sentence drops "in total"; cover
  stories reuse the generic templates ("The rules of the game are as follows"
  where the paper's cooking script says "The competition rules are as follows",
  and "You will play 10 dishes…" where the paper says "You will prepare…" — the
  paper's own cooking intro also has a typo, "ou are participating"). Semantics
  preserved; token-level layout of the retired `Q:/A: Option` completion
  scaffold intentionally not kept.

## 6. Results comparison

Same-metric comparison against the paper's released data lives in
[results/PAPER_COMPARISON.md](results/PAPER_COMPARISON.md) and the
"Paper vs Implementation" tab of the replay HTML
([results/game_replay.html](results/game_replay.html)) — the qualitative findings
reproduce: near-ceiling PD score ratios and PD-family defection dynamics, BoS
coordination markedly below win-win performance, and SCoT lifting BoS
coordination for every model (paper: GPT-4 0.61 → 0.91).
