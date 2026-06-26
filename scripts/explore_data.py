"""
explore_data.py

Generates a standalone HTML data explorer from your local ledger CSVs.
Open the output file in any browser — no server, no dependencies, no
internet connection needed.

Run with: python3 scripts/explore_data.py
Then open: data/explorer.html
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import brier_score, calibration_table
from src.ledger import hr_columns, load_ledger

ROOT = Path(__file__).resolve().parents[1]
HITS_LEDGER = ROOT / "data" / "ledger" / "predictions_log.csv"
HR_LEDGER   = ROOT / "data" / "ledger" / "hr_predictions_log.csv"
OUT         = ROOT / "data" / "explorer.html"

NAIVE_BRIER = 0.235


def daily_stats(df, prob_col, outcome_col):
    df = df[df["graded"] == True].copy()  # noqa: E712
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for d, g in df.groupby(df["date"].dt.date):
        score = float(((g[prob_col] - g[outcome_col]) ** 2).mean())
        rows.append({
            "date": str(d),
            "n": len(g),
            "avg_pred": round(float(g[prob_col].mean()), 3),
            "actual": round(float(g[outcome_col].mean()), 3),
            "brier": round(score, 4),
        })
    return rows


def top_edges(df, prob_col, outcome_col, n=20):
    df = df[df["graded"] == True].copy()  # noqa: E712
    if df.empty:
        return []
    df["edge"] = df[prob_col] - df[outcome_col].astype(float)
    df["correct"] = (df[prob_col] > 0.5) == (df[outcome_col] == 1)
    top = df.nsmallest(n, "edge")  # most overconfident (worst)
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "date": str(r["date"])[:10],
            "player": r["player_name"],
            "team": r["team"],
            "prob": round(float(r[prob_col]), 3),
            "actual": int(r[outcome_col]),
        })
    return rows


def build_payload():
    hits_df = load_ledger(HITS_LEDGER)
    hr_df   = load_ledger(HR_LEDGER, columns=hr_columns())

    hits_graded = hits_df[hits_df["graded"] == True]  # noqa: E712
    hr_graded   = hr_df[hr_df["graded"] == True]      # noqa: E712

    hits_cal = calibration_table(hits_graded) if not hits_graded.empty else pd.DataFrame()
    hr_cal   = calibration_table(hr_graded, prob_col="p_hr", outcome_col="actual_hr") if not hr_graded.empty else pd.DataFrame()

    return {
        "generated": date.today().isoformat(),
        "hits": {
            "brier": round(brier_score(hits_graded), 4) if not hits_graded.empty else None,
            "naive": NAIVE_BRIER,
            "total_graded": len(hits_graded),
            "avg_pred": round(float(hits_graded["p_hit"].mean()), 3) if not hits_graded.empty else None,
            "actual_rate": round(float(hits_graded["actual_hit"].mean()), 3) if not hits_graded.empty else None,
            "daily": daily_stats(hits_df, "p_hit", "actual_hit"),
            "calibration": hits_cal.rename(columns={
                "PredRange": "range", "N": "n",
                "AvgPredicted": "predicted", "ActualFrequency": "actual"
            }).round(3).to_dict(orient="records") if not hits_cal.empty else [],
        },
        "hr": {
            "brier": round(brier_score(hr_graded, prob_col="p_hr", outcome_col="actual_hr"), 4) if not hr_graded.empty else None,
            "naive": NAIVE_BRIER,
            "total_graded": len(hr_graded),
            "avg_pred": round(float(hr_graded["p_hr"].mean()), 3) if not hr_graded.empty else None,
            "actual_rate": round(float(hr_graded["actual_hr"].mean()), 3) if not hr_graded.empty else None,
            "daily": daily_stats(hr_df, "p_hr", "actual_hr"),
            "calibration": hr_cal.rename(columns={
                "PredRange": "range", "N": "n",
                "AvgPredicted": "predicted", "ActualFrequency": "actual"
            }).round(3).to_dict(orient="records") if not hr_cal.empty else [],
        },
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLB Model · Data Explorer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
:root {
  --bg: #0B0F14;
  --surface: #13181F;
  --surface2: #1A2030;
  --border: #1E2840;
  --accent: #4D9FFF;
  --accent2: #00E5A0;
  --warn: #F0A040;
  --text: #D8E0EC;
  --muted: #5A6880;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; }

/* HEADER */
header {
  padding: 24px 40px 20px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: baseline; gap: 16px;
}
.logo { font-family: var(--mono); font-size: 11px; color: var(--accent); letter-spacing: .12em; text-transform: uppercase; }
.logo-sub { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.generated { margin-left: auto; font-family: var(--mono); font-size: 10px; color: var(--muted); }

/* HERO SCORECARD */
.scorecard {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
  background: var(--border); border-bottom: 1px solid var(--border);
}
.scorecard-half { background: var(--bg); padding: 40px; }
.sc-eyebrow { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: .15em; text-transform: uppercase; margin-bottom: 16px; }
.sc-brier { font-family: var(--mono); font-size: 72px; font-weight: 600; line-height: 1; letter-spacing: -.03em; }
.sc-brier.hits { color: var(--accent); }
.sc-brier.hr   { color: var(--accent2); }
.sc-beat { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 10px; }
.sc-beat strong { color: var(--text); }
.sc-stats { display: flex; gap: 32px; margin-top: 24px; }
.sc-stat-label { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
.sc-stat-value { font-family: var(--mono); font-size: 20px; font-weight: 500; color: var(--text); margin-top: 2px; }

/* TABS */
.tabs { padding: 0 40px; border-bottom: 1px solid var(--border); display: flex; }
.tab { padding: 14px 16px; font-family: var(--mono); font-size: 10px; color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent; letter-spacing: .08em; text-transform: uppercase; transition: color .15s, border-color .15s; }
.tab.active.hits { color: var(--accent); border-bottom-color: var(--accent); }
.tab.active.hr   { color: var(--accent2); border-bottom-color: var(--accent2); }
.tab:hover:not(.active) { color: var(--text); }

/* PANELS */
.panels { padding: 32px 40px; }
.panel { display: none; }
.panel.active { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

/* CARD */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.card-header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
.card-title { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; }
.card-meta { font-family: var(--mono); font-size: 10px; color: var(--muted); }

/* DAILY TABLE */
.daily-table { width: 100%; border-collapse: collapse; }
.daily-table th { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; padding: 10px 20px; border-bottom: 1px solid var(--border); text-align: left; }
.daily-table td { font-family: var(--mono); font-size: 12px; padding: 9px 20px; border-bottom: 1px solid var(--border); }
.daily-table tr:last-child td { border-bottom: none; }
.daily-table tr:hover td { background: rgba(255,255,255,.02); }
.brier-cell { display: flex; align-items: center; gap: 10px; }
.brier-bar { height: 2px; background: var(--border); border-radius: 2px; flex: 1; position: relative; }
.brier-fill { height: 100%; border-radius: 2px; position: absolute; left: 0; top: 0; }

/* CALIBRATION TABLE */
.cal-table { width: 100%; border-collapse: collapse; }
.cal-table th { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; padding: 10px 20px; border-bottom: 1px solid var(--border); text-align: left; }
.cal-table td { font-family: var(--mono); font-size: 12px; padding: 10px 20px; border-bottom: 1px solid var(--border); }
.cal-table tr:last-child td { border-bottom: none; }
.delta-good { color: var(--accent2); }
.delta-warn { color: var(--warn); }
.delta-bad  { color: #FF6B6B; }

footer { padding: 24px 40px; border-top: 1px solid var(--border); font-family: var(--mono); font-size: 10px; color: var(--muted); }

@media (max-width: 900px) {
  .scorecard { grid-template-columns: 1fr; }
  .panel.active { grid-template-columns: 1fr; }
  .sc-brier { font-size: 48px; }
}
</style>
</head>
<body>
<header>
  <span class="logo">MLB Prop Model</span>
  <span class="logo-sub">Data Explorer</span>
  <span class="generated" id="gen"></span>
</header>

<section class="scorecard">
  <div class="scorecard-half">
    <div class="sc-eyebrow">Hits Model · Brier Score</div>
    <div class="sc-brier hits" id="hits-brier">—</div>
    <div class="sc-beat" id="hits-beat"></div>
    <div class="sc-stats">
      <div><div class="sc-stat-label">Graded</div><div class="sc-stat-value" id="hits-graded">—</div></div>
      <div><div class="sc-stat-label">Avg Predicted</div><div class="sc-stat-value" id="hits-pred">—</div></div>
      <div><div class="sc-stat-label">Actual Hit Rate</div><div class="sc-stat-value" id="hits-actual">—</div></div>
    </div>
  </div>
  <div class="scorecard-half">
    <div class="sc-eyebrow">HR Model · Brier Score</div>
    <div class="sc-brier hr" id="hr-brier">—</div>
    <div class="sc-beat" id="hr-beat"></div>
    <div class="sc-stats">
      <div><div class="sc-stat-label">Graded</div><div class="sc-stat-value" id="hr-graded">—</div></div>
      <div><div class="sc-stat-label">Avg Predicted</div><div class="sc-stat-value" id="hr-pred">—</div></div>
      <div><div class="sc-stat-label">Actual HR Rate</div><div class="sc-stat-value" id="hr-actual">—</div></div>
    </div>
  </div>
</section>

<nav class="tabs">
  <div class="tab active hits" onclick="switchTab('hits')">Hits Model</div>
  <div class="tab hr" onclick="switchTab('hr')">Home Run Model</div>
</nav>

<main class="panels">
  <div class="panel active" id="panel-hits">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Daily Performance</span>
        <span class="card-meta">Brier score per day</span>
      </div>
      <table class="daily-table" id="hits-daily">
        <thead><tr><th>Date</th><th>N</th><th>Avg Pred</th><th>Actual</th><th>Brier</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Calibration</span>
        <span class="card-meta">All graded predictions</span>
      </div>
      <table class="cal-table" id="hits-cal">
        <thead><tr><th>Range</th><th>N</th><th>Predicted</th><th>Actual</th><th>Δ</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="panel" id="panel-hr">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Daily Performance</span>
        <span class="card-meta">Brier score per day</span>
      </div>
      <table class="daily-table" id="hr-daily">
        <thead><tr><th>Date</th><th>N</th><th>Avg Pred</th><th>Actual</th><th>Brier</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Calibration</span>
        <span class="card-meta">All graded predictions</span>
      </div>
      <table class="cal-table" id="hr-cal">
        <thead><tr><th>Range</th><th>N</th><th>Predicted</th><th>Actual</th><th>Δ</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</main>

<footer id="footer"></footer>

<script>
const DATA = __DATA__;

function pct(v) { return v != null ? (v*100).toFixed(1)+'%' : '—'; }

function deltaClass(d) {
  const a = Math.abs(d);
  if (a < 0.03) return 'delta-good';
  if (a < 0.08) return 'delta-warn';
  return 'delta-bad';
}

function renderHero(d) {
  document.getElementById('gen').textContent = 'Generated ' + d.generated;

  const hb = d.hits.brier, naive = d.hits.naive;
  document.getElementById('hits-brier').textContent = hb != null ? hb.toFixed(3) : '—';
  if (hb != null) {
    const imp = ((naive - hb) / naive * 100).toFixed(1);
    document.getElementById('hits-beat').innerHTML =
      `beats naive baseline <strong>${imp}%</strong> &nbsp;(naive = ${naive})`;
  }
  document.getElementById('hits-graded').textContent = d.hits.total_graded;
  document.getElementById('hits-pred').textContent = pct(d.hits.avg_pred);
  document.getElementById('hits-actual').textContent = pct(d.hits.actual_rate);

  const hrb = d.hr.brier;
  document.getElementById('hr-brier').textContent = hrb != null ? hrb.toFixed(3) : '—';
  if (hrb != null) {
    const imp = ((naive - hrb) / naive * 100).toFixed(1);
    document.getElementById('hr-beat').innerHTML =
      `beats naive baseline <strong>${imp}%</strong> &nbsp;(naive = ${naive})`;
  }
  document.getElementById('hr-graded').textContent = d.hr.total_graded;
  document.getElementById('hr-pred').textContent = pct(d.hr.avg_pred);
  document.getElementById('hr-actual').textContent = pct(d.hr.actual_rate);

  document.getElementById('footer').textContent =
    'Local data explorer · ' + d.generated + ' · not financial or gambling advice';
}

function renderDaily(tableId, rows, accentVar) {
  const tbody = document.querySelector('#' + tableId + ' tbody');
  if (!rows.length) { tbody.innerHTML = '<tr><td colspan="5" style="padding:20px;color:var(--muted)">No graded data yet.</td></tr>'; return; }
  const maxBrier = Math.max(...rows.map(r => r.brier));
  tbody.innerHTML = rows.slice().reverse().map(r => {
    const fillPct = Math.min(100, (r.brier / 0.25) * 100);
    const color = r.brier < 0.23 ? 'var(--accent2)' : r.brier < 0.235 ? 'var(--accent)' : 'var(--warn)';
    return `<tr>
      <td>${r.date}</td>
      <td>${r.n}</td>
      <td>${pct(r.avg_pred)}</td>
      <td>${pct(r.actual)}</td>
      <td>
        <div class="brier-cell">
          <span>${r.brier.toFixed(3)}</span>
          <div class="brier-bar"><div class="brier-fill" style="width:${fillPct}%;background:${color}"></div></div>
        </div>
      </td>
    </tr>`;
  }).join('');
}

function renderCal(tableId, rows) {
  const tbody = document.querySelector('#' + tableId + ' tbody');
  if (!rows.length) { tbody.innerHTML = '<tr><td colspan="5" style="padding:20px;color:var(--muted)">No graded data yet.</td></tr>'; return; }
  tbody.innerHTML = rows.map(r => {
    const delta = (r.actual - r.predicted);
    const sign = delta >= 0 ? '+' : '';
    const cls = deltaClass(Math.abs(delta));
    return `<tr>
      <td>${r.range}</td>
      <td>${r.n}</td>
      <td>${pct(r.predicted)}</td>
      <td>${pct(r.actual)}</td>
      <td class="${cls}">${sign}${delta.toFixed(3)}</td>
    </tr>`;
  }).join('');
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector('.tab.' + name).classList.add('active');
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

renderHero(DATA);
renderDaily('hits-daily', DATA.hits.daily);
renderCal('hits-cal', DATA.hits.calibration);
renderDaily('hr-daily', DATA.hr.daily);
renderCal('hr-cal', DATA.hr.calibration);
</script>
</body>
</html>"""


def main():
    payload = build_payload()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, indent=2))
    OUT.write_text(html)
    print(f"Explorer written to {OUT}")
    print(f"Open it with: open {OUT}")


if __name__ == "__main__":
    main()
