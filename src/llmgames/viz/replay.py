"""Self-contained animated HTML replay of model-vs-model games.

Reads a run's ``rounds.csv`` (and ``thoughts.csv`` if present) and renders one
offline HTML file: pick the two players directly from their title dropdowns, then
"play" the match round by round — each player's actual move (J/F), the points won,
the cumulative score, plus the SCoT prediction and reasoning text when captured.
A leaderboard summarises the run. No external/CDN dependencies: inline CSS + JS.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

# Canonical display: internal action A -> "J" (cooperate / P1-preferred), B -> "F".
_LABELS = {"A": "J", "B": "F"}
_COOP_ACTION = "A"

# Canonical payoffs for the legend (P1, P2) keyed by (p1_action, p2_action) as J/F.
_PAYOFFS = {
    "Prisoner's Dilemma": {"JJ": [8, 8], "JF": [0, 10], "FJ": [10, 0], "FF": [5, 5]},
    "Battle of the Sexes": {"JJ": [10, 7], "JF": [0, 0], "FJ": [0, 0], "FF": [7, 10]},
}


def _label(action: str) -> str:
    """Maps an internal action to its display label (unknown -> '?')."""
    return _LABELS.get(action, "?")


def _build_matches(df: pd.DataFrame) -> list[dict]:
    """Builds the per-match replay payload from the tidy rounds DataFrame."""
    matches: list[dict] = []
    for (game, p1, p2), group in df.groupby(["game_name", "player1", "player2"], sort=False):
        ordered = group.sort_values("round")
        rounds = [
            {
                "r": int(row.round),
                "a1": _label(str(row.action1)),
                "a2": _label(str(row.action2)),
                "p1": int(row.points1) if row.points1 >= 0 else 0,
                "p2": int(row.points2) if row.points2 >= 0 else 0,
                "t1": int(row.total1),
                "t2": int(row.total2),
            }
            for row in ordered.itertuples()
        ]
        matches.append(
            {
                "game": str(game),
                "p1": str(p1),
                "p2": str(p2),
                "rounds": rounds,
                "total1": rounds[-1]["t1"] if rounds else 0,
                "total2": rounds[-1]["t2"] if rounds else 0,
            }
        )
    return matches


def _build_leaderboard(df: pd.DataFrame) -> list[dict]:
    """Computes a per-player summary across both seats (avg final score + coop %)."""
    last = df[df["round"] == df["round"].max()]
    rows: dict[str, dict] = {}
    for name_col, total_col, action_col in (
        ("player1", "total1", "action1"),
        ("player2", "total2", "action2"),
    ):
        for row in last.itertuples():
            name = getattr(row, name_col)
            rows.setdefault(name, {"player": name, "finals": [], "coop": 0, "moves": 0})
            rows[name]["finals"].append(int(getattr(row, total_col)))
        for row in df.itertuples():
            name = getattr(row, name_col)
            rows[name]["moves"] += 1
            if getattr(row, action_col) == _COOP_ACTION:
                rows[name]["coop"] += 1

    leaderboard = [
        {
            "player": d["player"],
            "matches": len(d["finals"]),
            "avg_score": round(sum(d["finals"]) / len(d["finals"]), 2) if d["finals"] else 0,
            "coop_pct": round(100 * d["coop"] / d["moves"], 1) if d["moves"] else 0,
        }
        for d in rows.values()
    ]
    return sorted(leaderboard, key=lambda r: r["avg_score"], reverse=True)


def _build_thoughts(tdf: pd.DataFrame | None) -> dict:
    """Builds a ``"game||player||opponent" -> {round: {pd, pt, dt}}`` lookup.

    Args:
        tdf: Merged thoughts DataFrame, or None.

    Returns:
        The nested reasoning lookup (empty if there is no data).
    """
    if tdf is None or tdf.empty:
        return {}
    out: dict[str, dict] = {}
    for row in tdf.itertuples():
        key = f"{row.game_name}||{row.player}||{row.opponent}"
        out.setdefault(key, {})[str(int(row.round))] = {
            "pd": _label(str(row.predicted)) if str(row.predicted) in _LABELS else str(row.predicted),
            "pt": str(row.predict_text),
            "dt": str(row.decide_text),
        }
    return out


def _as_paths(value) -> list[Path]:
    """Normalizes a path or list of paths into a list of :class:`Path`."""
    items = value if isinstance(value, (list, tuple)) else [value]
    return [Path(v) for v in items]


def generate_replay_html(
    rounds_csv,
    out_path: str | Path,
    *,
    run_name: str = "",
    thoughts_csv=None,
) -> Path:
    """Generates the animated HTML replay, merging one or more runs.

    Pass several rounds CSVs (e.g. a base run and a SCoT run) to view every player
    in both modes in one file. Matches are de-duplicated on
    ``(game, player1, player2)`` so identical strategy-vs-strategy pairings are not
    repeated; models keep their distinct names (e.g. ``X`` vs ``X+scot``).

    Args:
        rounds_csv: One path or a list of per-round CSVs.
        out_path: Destination HTML path.
        run_name: Optional run name shown in the header.
        thoughts_csv: One path / list of thoughts CSVs; defaults to the ``thoughts.csv``
            sibling of each rounds CSV.

    Returns:
        The written HTML path.

    Raises:
        FileNotFoundError: If a rounds CSV does not exist.
    """
    round_paths = _as_paths(rounds_csv)
    for path in round_paths:
        if not path.exists():
            raise FileNotFoundError(f"rounds.csv not found: {path}")

    df = pd.concat([pd.read_csv(p) for p in round_paths], ignore_index=True)
    # Dedup at round level: drops a match duplicated across runs (e.g. an identical
    # strategy-vs-strategy pairing) while keeping every round within a single match.
    df = df.drop_duplicates(subset=["game_name", "player1", "player2", "round"], keep="first")

    if thoughts_csv is None:
        thought_paths = [p.with_name("thoughts.csv") for p in round_paths]
    else:
        thought_paths = _as_paths(thoughts_csv)
    thought_paths = [p for p in thought_paths if p.exists()]
    tdf = None
    if thought_paths:
        tdf = pd.concat([pd.read_csv(p) for p in thought_paths], ignore_index=True).fillna("")
        tdf = tdf.drop_duplicates(subset=["game_name", "player", "opponent", "round"], keep="first")

    players = sorted(set(df["player1"]).union(df["player2"]))
    data = {
        "runName": run_name,
        "matches": _build_matches(df),
        "leaderboard": _build_leaderboard(df),
        "players": players,
        "payoffs": _PAYOFFS,
        "games": sorted(df["game_name"].unique().tolist()),
        "thoughts": _build_thoughts(tdf),
    }
    payload = json.dumps(data, separators=(",", ":"))
    document = _HTML_TEMPLATE.replace("/*__DATA__*/", payload).replace(
        "__RUN_NAME__", html.escape(run_name or "run")
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>llmgames — game replay (__RUN_NAME__)</title>
<style>
  :root {
    --bg:#0f1419; --panel:#1a212b; --ink:#e6edf3; --muted:#8b98a5;
    --coop:#2e9e6b; --defect:#d2452f; --accent:#4c8bf5; --line1:#4c8bf5; --line2:#e0a32e;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; padding:24px; }
  h1 { font-size:20px; margin:0 0 2px; } .sub { color:var(--muted); margin:0 0 20px; }
  .panel { background:var(--panel); border:1px solid #2a3440; border-radius:12px; padding:16px; margin-bottom:18px; }
  h2 { font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:6px 10px; border-bottom:1px solid #2a3440; }
  th { color:var(--muted); font-weight:600; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:16px; }
  select,button { background:#0f1722; color:var(--ink); border:1px solid #2a3440; border-radius:8px; padding:7px 10px; font:inherit; cursor:pointer; }
  button:hover,select:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  input[type=range] { accent-color:var(--accent); }
  .stage { display:grid; grid-template-columns:1fr auto 1fr; gap:16px; align-items:start; }
  .player { text-align:center; }
  .psel { width:100%; font-weight:700; text-align:center; margin-bottom:10px; }
  .badge { width:88px; height:88px; line-height:88px; margin:0 auto; border-radius:16px; font-size:44px; font-weight:800; color:#fff; transition:.2s; }
  .badge.J { background:var(--coop); } .badge.F { background:var(--defect); } .badge.q { background:#5a6673; }
  .delta { font-size:13px; color:var(--muted); margin-top:8px; height:18px; }
  .total { font-size:26px; font-weight:800; margin-top:4px; }
  .bar { height:8px; background:#0f1722; border-radius:4px; margin-top:8px; overflow:hidden; }
  .bar > div { height:100%; transition:.25s; }
  .bar1 > div { background:var(--line1); } .bar2 > div { background:var(--line2); }
  .vs { text-align:center; color:var(--muted); padding-top:30px; }
  .outcome { text-align:center; margin:14px 0 6px; min-height:20px; font-size:13px; }
  .dots { display:flex; gap:6px; justify-content:center; margin:10px 0; flex-wrap:wrap; }
  .dot { width:22px; height:22px; border-radius:6px; font-size:10px; line-height:22px; text-align:center; color:#0f1419; font-weight:700; background:#2a3440; opacity:.5; cursor:pointer; }
  .dot.cur { outline:2px solid var(--accent); opacity:1; }
  .dot.done { opacity:1; }
  svg { width:100%; height:160px; display:block; }
  .legend { color:var(--muted); font-size:12px; margin-top:8px; }
  .thoughts { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:14px; }
  .think { background:#0f1722; border:1px solid #2a3440; border-radius:8px; padding:10px 12px; font-size:12.5px; }
  .think h3 { margin:0 0 6px; font-size:12px; color:var(--accent); }
  .think .pred { color:var(--muted); margin-bottom:6px; }
  .think p { margin:4px 0; white-space:pre-wrap; }
  code { background:#0f1722; padding:1px 6px; border-radius:5px; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:700; }
  .pill.J { background:rgba(46,158,107,.2); color:#52c08a; } .pill.F { background:rgba(210,69,47,.2); color:#e0735f; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Repeated 2×2 games — model replay</h1>
  <p class="sub">Run <code>__RUN_NAME__</code> · choose the two players in the title dropdowns, press Play. <span class="pill J">J</span> = cooperate / P1-preferred · <span class="pill F">F</span> = the other option.</p>

  <div class="panel">
    <h2>Leaderboard (avg final score across matches)</h2>
    <table id="leaderboard"><thead><tr><th>#</th><th>Player</th><th>Matches</th><th>Avg final score</th><th>Cooperation (J)</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="panel">
    <h2>Game replay</h2>
    <div class="controls">
      <label>Game <select id="gameSel"></select></label>
      <button id="play" class="primary">▶ Play</button>
      <button id="step">Step ▶</button>
      <button id="reset">⟲ Reset</button>
      <label>Speed <input id="speed" type="range" min="200" max="1600" value="700" step="100"></label>
      <span id="roundLbl"></span>
    </div>
    <input id="scrub" type="range" min="0" max="10" value="0" style="width:100%;margin-bottom:16px;">
    <div class="stage">
      <div class="player">
        <select class="psel" id="n1"></select>
        <div class="badge q" id="b1">·</div>
        <div class="delta" id="d1"></div>
        <div class="total" id="t1">0</div>
        <div class="bar bar1"><div id="bar1" style="width:0%"></div></div>
      </div>
      <div class="vs">vs</div>
      <div class="player">
        <select class="psel" id="n2"></select>
        <div class="badge q" id="b2">·</div>
        <div class="delta" id="d2"></div>
        <div class="total" id="t2">0</div>
        <div class="bar bar2"><div id="bar2" style="width:0%"></div></div>
      </div>
    </div>
    <div class="outcome" id="outcome"></div>
    <div class="dots" id="dots"></div>
    <svg id="chart" viewBox="0 0 600 160" preserveAspectRatio="none"></svg>
    <div class="legend" id="legend"></div>
    <div class="thoughts" id="thoughts"></div>
  </div>
</div>

<script>
const DATA = /*__DATA__*/;
const $ = (id) => document.getElementById(id);

(function(){
  const tb = $("leaderboard").querySelector("tbody");
  DATA.leaderboard.forEach((r,i)=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${i+1}</td><td>${r.player}</td><td>${r.matches}</td><td>${r.avg_score}</td><td>${r.coop_pct}%</td>`;
    tb.appendChild(tr);
  });
})();

let current=null, round=0, timer=null;

function findMatch(){
  const g=$("gameSel").value, a=$("n1").value, b=$("n2").value;
  return DATA.matches.find(m=>m.game===g && m.p1===a && m.p2===b) || null;
}
function thoughtsFor(game, player, opp){ return (DATA.thoughts||{})[`${game}||${player}||${opp}`] || null; }

function fillSelect(sel, items, val){ sel.innerHTML=""; items.forEach(x=>{ const o=document.createElement("option"); o.value=x; o.textContent=x; sel.appendChild(o); }); if(val!==undefined) sel.value=val; }

function loadMatch(){
  stop();
  current = findMatch();
  if(!current){ $("outcome").textContent="No recorded match for this pairing — pick another."; $("chart").innerHTML=""; $("dots").innerHTML=""; $("thoughts").innerHTML=""; return; }
  $("scrub").max=current.rounds.length;
  const pay=DATA.payoffs[current.game]||{};
  $("legend").textContent = Object.entries(pay).map(([k,v])=>`${k[0]}/${k[1]} → ${v[0]}/${v[1]}`).join("   ·   ");
  drawChart(); render(0);
}

function maxTotal(){ return Math.max(1, current.total1, current.total2); }

function render(r){
  if(!current) return;
  round=r; $("scrub").value=r; $("roundLbl").textContent = `Round ${r} / ${current.rounds.length}`;
  if(r===0){
    setBadge("b1","·","q"); setBadge("b2","·","q");
    $("d1").textContent=""; $("d2").textContent=""; $("t1").textContent=0; $("t2").textContent=0;
    $("bar1").style.width="0%"; $("bar2").style.width="0%";
    $("outcome").textContent="Press Play to watch the rounds.";
  } else {
    const x=current.rounds[r-1];
    setBadge("b1",x.a1,x.a1); setBadge("b2",x.a2,x.a2);
    $("d1").textContent=`+${x.p1}`; $("d2").textContent=`+${x.p2}`;
    $("t1").textContent=x.t1; $("t2").textContent=x.t2;
    const mx=maxTotal();
    $("bar1").style.width=(100*x.t1/mx)+"%"; $("bar2").style.width=(100*x.t2/mx)+"%";
    $("outcome").innerHTML=`Round ${x.r}: <b>${current.p1}</b> chose <span class="pill ${x.a1}">${x.a1}</span>, <b>${current.p2}</b> chose <span class="pill ${x.a2}">${x.a2}</span> → <b>${x.p1}</b> vs <b>${x.p2}</b> points`;
  }
  drawDots(); markChart(r); renderThoughts(r);
}

function setBadge(id,txt,cls){ const e=$(id); e.textContent=txt; e.className="badge "+(cls==="q"?"q":cls); }

function drawDots(){
  const d=$("dots"); d.innerHTML="";
  current.rounds.forEach((x,i)=>{
    const el=document.createElement("div"); el.className="dot"+(i<round?" done":"")+(i===round-1?" cur":"");
    el.textContent=`${x.a1}${x.a2}`;
    if(i<round){ el.style.background = (x.a1==="J"&&x.a2==="J")?"var(--coop)":(x.a1==="F"&&x.a2==="F")?"#e0a32e":"var(--defect)"; }
    el.title=`Round ${x.r}: ${x.a1}/${x.a2} → ${x.p1}/${x.p2}`;
    el.onclick=()=>{ stop(); render(i+1); };
    d.appendChild(el);
  });
}

function esc(s){ const d=document.createElement("div"); d.textContent=s||""; return d.innerHTML; }
function thinkBlock(name, who){
  const t = current ? thoughtsFor(current.game, name, who) : null;
  const rec = t ? t[String(round)] : null;
  if(round<1 || !rec) return "";
  const pred = rec.pd ? `<div class="pred">Predicted opponent: <span class="pill ${rec.pd}">${rec.pd}</span></div>` : "";
  const blocks = [];
  if(rec.pt) blocks.push(`<p><b>Prediction reasoning:</b> ${esc(rec.pt)}</p>`);
  if(rec.dt) blocks.push(`<p><b>Decision:</b> ${esc(rec.dt)}</p>`);
  if(!pred && blocks.length===0) return "";
  return `<div class="think"><h3>${esc(name)} — thinking</h3>${pred}${blocks.join("")}</div>`;
}
function renderThoughts(r){
  if(!current){ $("thoughts").innerHTML=""; return; }
  const a=thinkBlock(current.p1, current.p2), b=thinkBlock(current.p2, current.p1);
  $("thoughts").innerHTML = (a||b) ? (a+b) : (round>=1 ? `<div class="think" style="grid-column:1/3;color:var(--muted)">No reasoning captured for this run. Re-run with <code>reasoning: true</code> (or SCoT mode) to see the models' thinking.</div>` : "");
}

function pts(arr,key,W,H){ const n=arr.length, mx=maxTotal(); return arr.map((x,i)=>`${(i/(n-1||1))*W},${H-(x[key]/mx)*H}`).join(" "); }
function drawChart(){
  const W=600,H=150;
  $("chart").innerHTML=`
    <polyline fill="none" stroke="var(--line1)" stroke-width="2.5" points="${pts(current.rounds,"t1",W,H)}"/>
    <polyline fill="none" stroke="var(--line2)" stroke-width="2.5" points="${pts(current.rounds,"t2",W,H)}"/>
    <line id="mk" x1="0" y1="0" x2="0" y2="${H}" stroke="#4c8bf5" stroke-dasharray="3 3" opacity="0"/>`;
}
function markChart(r){
  const W=600, n=current.rounds.length, mk=$("mk"); if(!mk) return;
  if(r<1){ mk.setAttribute("opacity","0"); return; }
  const x=((r-1)/(n-1||1))*W; mk.setAttribute("x1",x); mk.setAttribute("x2",x); mk.setAttribute("opacity","0.7");
}

function step(){ if(!current) return; if(round<current.rounds.length){ render(round+1); } else { stop(); } }
function play(){ if(timer) return stop(); if(!current) return; if(round>=current.rounds.length) render(0);
  $("play").textContent="❚❚ Pause"; timer=setInterval(()=>{ if(round>=current.rounds.length) return stop(); step(); }, +$("speed").value); }
function stop(){ if(timer){ clearInterval(timer); timer=null; } $("play").textContent="▶ Play"; }

$("gameSel").onchange=loadMatch;
$("n1").onchange=loadMatch;
$("n2").onchange=loadMatch;
$("play").onclick=play;
$("step").onclick=()=>{ stop(); step(); };
$("reset").onclick=()=>{ stop(); render(0); };
$("scrub").oninput=(e)=>{ stop(); render(+e.target.value); };

fillSelect($("gameSel"), DATA.games);
fillSelect($("n1"), DATA.players, DATA.players[0]);
fillSelect($("n2"), DATA.players, DATA.players[1] || DATA.players[0]);
loadMatch();
</script>
</body>
</html>
"""
