# Paper vs. Implementation — findings comparison

Compares the headline findings of **Akata, Schulz, Coda-Forno, Oh, Bethge & Schulz
(2025), *Playing repeated games with large language models*, Nature Human Behaviour**
(doi:10.1038/s41562-025-02172-y; arXiv:2305.16867) against this reproduction.

**Important difference:** the paper evaluated GPT-3 / GPT-3.5 / GPT-4 (+ Claude-2,
Llama-2). This reproduction runs **modern** gateway models — `gpt-4.1`, `gpt-4.1-mini`,
`gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `claude-opus-4-6`, `claude-haiku-4-5`
— at temperature 0, canonical PD + BoS, 10 rounds. So absolute numbers are not
comparable; we test whether the **qualitative findings still hold**.

Metric: **score ratio** = achieved / best-achievable-given-the-opponent (paper's Table 1).

## At a glance — paper's released data, scored with this repo's metric

Computed by running `data/paper/{pd,bos}.csv` through the same `score_ratio` function.

**Prisoner's Dilemma** — paper: Claude-2 0.95 · GPT-4 0.84 · GPT-3.5 0.71 · GPT-3 0.53 ·
Llama-2 0.53 (clear capability ladder). This run (modern): 0.80–0.84 across all six — the
GPT-3→4 gap has closed upward.

**Battle of the Sexes** — paper: **GPT-4+SCoT 0.91** vs GPT-4 base 0.61 · GPT-3 0.68 ·
Llama-2 0.67 · Claude-2 0.49 · GPT-3.5 0.47. This run: base 0.68–0.84 → **SCoT 0.89–0.96**.

The interactive bars + the per-model base→SCoT effect are in the **"Paper vs
Implementation"** tab of [`game_replay.html`](game_replay.html).

---

## 1. LLMs play self-interested games (Prisoner's Dilemma) competently

**Paper:** LLMs — GPT-4 especially — do well in PD; they defect against defectors and
exploit cooperators rather than being exploited.

**This implementation (base):** every model scores 0.80–0.84 in PD; `always_defect`
reaches the 1.0 ceiling. Under SCoT, models best-respond near-optimally (PD mostly
→1.00). Models reliably defect against `always_defect`.

**Verdict: ✅ reproduced.**

---

## 2. LLMs struggle to coordinate (Battle of the Sexes)

**Paper:** LLMs fail at coordination games — they do not learn to alternate between the
two players' preferred equilibria; GPT-4 in particular underperforms here.

**This implementation (base):** BoS is the clear weak spot. The OpenAI models lag
badly — `gpt-4.1` 0.68, `gpt-4.1-mini` 0.69 — versus the hand-coded `tit_for_tat`
(0.92). Claude/Gemini do somewhat better (0.77–0.84) but still below the alternating
baseline.

**Verdict: ✅ reproduced** — coordination is the weak spot, OpenAI models most so.

---

## 3. GPT-4 is "unforgiving" in PD (exploit, then can't recover cooperation)

**Paper:** after a defection, GPT-4 keeps defecting and fails to return to mutual
cooperation.

**This implementation (SCoT reasoning capture):** visible verbatim in the captured
reasoning. `gpt-4.1+scot` vs `tit_for_tat`, Prisoner's Dilemma:
- Round 1: *"F gives me 10 points while J gives 8, so F is best"* → exploits.
- Round 2: *"they may retaliate by choosing F… CHOICE: F"* → predicts the punishment it
  provoked, and keeps defecting (5/5) instead of rebuilding 8/8 cooperation.

**Verdict: ✅ reproduced** — and now legible in the model's own words (`docs/results/scot/`).

---

## 4. Social Chain-of-Thought (SCoT) improves coordination

**Paper:** prompting the model to first predict the opponent's move, then act, markedly
improves coordination behaviour.

**This implementation (base → SCoT, score ratio):**

| model | PD base → SCoT | BoS base → SCoT |
|---|---|---|
| gpt-4.1 | 0.83 → 1.00 | **0.68 → 0.93** |
| gpt-4.1-mini | 0.84 → 1.00 | **0.69 → 0.95** |
| gemini-3.1-pro | 0.83 → 0.93 | 0.83 → 0.92 |
| gemini-3.1-flash-lite | 0.80 → 1.00 | 0.77 → 0.92 |
| claude-opus-4-6 | 0.81 → 1.00 | 0.84 → 0.89 |
| claude-haiku-4-5 | 0.81 → 0.81 | **0.80 → 0.96** |

SCoT lifts BoS coordination for **every** model — the largest gains exactly where the
paper found the weakness (the OpenAI models, +0.25).

**Verdict: ✅ reproduced** (strongest single result).

_Caveat: base ratios are the full round-robin; these SCoT ratios are vs the two
strategies only (model-vs-model SCoT pending — see "Open item")._

---

## 5. Theory of mind — models predict opponent moves

**Paper:** the SCoT prediction step shows LLMs can model the opponent.

**This implementation (SCoT prediction accuracy, Fig 6):** predicted vs actual opponent
move, per model: 0.875–0.938.

| model | accuracy |
|---|---|
| gpt-4.1 | 0.925 |
| gpt-4.1-mini | 0.900 |
| gemini-3.1-pro | 0.938 |
| gemini-3.1-flash-lite | 0.938 |
| claude-opus-4-6 | 0.912 |
| claude-haiku-4-5 | 0.875 |

**Verdict: ✅ reproduced** (predictable strategy opponents make this an upper-ish bound).

---

## 6. Human study (Fig 7) — SCoT raises coordination & "felt human"

**Paper:** humans coordinate better against SCoT-prompted GPT-4 and are more likely to
believe they played a human.

**This implementation (the paper's real 195-participant dataset, analysed here):**

| game | opponent | success / coordination | P(thought human) |
|---|---|---|---|
| BoS | Base GPT-4 | 0.542 | 0.361 |
| BoS | SCoT-Prompted | **0.622** | **0.469** |
| PD | Base GPT-4 | — | 0.485 |
| PD | SCoT-Prompted | — | **0.643** |

**Verdict: ✅ reproduced** (this is the paper's own dataset, re-analysed by our metrics).

---

## 7. Capability scaling

**Paper:** stronger models (GPT-3 → GPT-4) score higher.

**This implementation:** the modern lineup is uniformly strong (base 0.77–0.84); the
large capability gap the paper saw across the GPT-3→4 generations is mostly closed.

**Verdict: ⚠️ partial** — direction not visible because modern models have converged.

---

## Summary

| # | Paper finding | Reproduced? |
|---|---|---|
| 1 | PD competence / punish defectors | ✅ |
| 2 | Poor BoS coordination | ✅ |
| 3 | GPT-4 unforgiving in PD | ✅ (visible in reasoning) |
| 4 | SCoT improves coordination | ✅ (strongest result) |
| 5 | Theory-of-mind prediction | ✅ |
| 6 | Human study: SCoT ↑ coordination & ↑ "felt human" | ✅ |
| 7 | Capability scaling | ⚠️ partial (models converged) |

Six of seven findings reproduce on modern models; the seventh is masked by model
progress, not contradicted.

## Open item

The SCoT run used `model_vs_model: false` (models played only the hand-coded
strategies) to bound cost, so **SCoT model-vs-model matches are not yet evaluated**
(you cannot replay e.g. `openai_heavy+scot` vs `gemini_heavy+scot`). Running
`config/runs/llm_gw_scot.yaml` with `model_vs_model: true` completes the matrix; this
file will be updated with those results.
