# MLB Hit & Home Run Prop Model

A daily-running prediction system for MLB batter props — hit probability and home run probability — built on empirical-Bayes shrinkage, pitcher/park adjustments, and Poisson modeling. Predictions run automatically each morning via cron, grade themselves each night against real outcomes, and compare against live sportsbook odds to surface edge.

**Live dashboard:** [yuschajp.github.io/mlb_hits_model](https://yuschajp.github.io/mlb_hits_model/)

---

## Real results so far

| Model | Graded predictions | Brier score | Naive baseline | Beats baseline |
|---|---|---|---|---|
| Hits | 414 | **0.223** | 0.235 | ✓ +5.1% |
| Home runs | 234 | **0.110** | 0.235 | ✓ significantly |

Calibration (hits model, 0.6–0.8 probability bucket, N=344): predicted 67%, actual 66%. The bulk of predictions land in this range and are well-calibrated against real outcomes.

These are real numbers from real games — not a backtest.

---

## How it works

### Hits model (`src/hit_model.py`)

For each batter in a confirmed starting lineup:

1. **Blend three shrunk hit-rate signals** — season BA, last-30-days form, and platoon split vs. the opposing pitcher's hand — each pulled toward league average via empirical-Bayes shrinkage so small samples don't dominate (default weights 50/30/20, prior_ab=200)
2. **Adjust for the opposing pitcher** — their batting-average-against, also shrunk, clipped to a ±30% band
3. **Adjust for the ballpark** — park factor multiplier from `data/park_factors.csv`
4. **Convert to per-game probability** — `P(>=1 hit) = 1 - (1 - p)^AB` where AB is the expected at-bats for the batter's lineup slot

### Home run model (`src/hr_model.py`)

Same architecture, two meaningful differences driven by the rarity of home runs (~1 per 35 AB vs. ~1 per 4 for hits):

- **Poisson instead of binomial** — `P(>=1 HR) = 1 - e^(-λ)` where `λ = adjusted_hr_rate × expected_AB`. Mathematically cleaner for rare events: `hr_probability()` and `hr_over_under_probability(line=0.5)` are guaranteed to agree exactly (verified in the test suite), unlike the hits model where AB-rounding creates a small gap
- **Stronger shrinkage** — `prior_ab=400` vs. 200 for hits, because HR rate needs far more data to stabilize
- **Separate HR park factors** (`data/hr_park_factors.csv`) — Fenway is hits-friendly (1.06) but HR-suppressing (0.92) because the Green Monster turns fly balls into doubles; these are genuinely different adjustments

### Odds comparison (`scripts/find_value.py`, `scripts/find_value_hr.py`)

Compares logged predictions against live sportsbook odds via [The Odds API](https://the-odds-api.com/) and flags batters where the model beats the market's implied probability by ≥5% (hits) or ≥3% (HR). Deliberately conservative: uses raw implied probability of the best available price, not a de-vigged fair line, so flagged edge is an understatement rather than an overstatement.

Sample output from a live run:

```
38 batter(s) where the model beats the market by >= 5%:
  Blake Dunn       Cincinnati Reds   model=75.8%  market=64.5%  edge=+11.3%  (-182 @ draftkings)
  JJ Bleday        Cincinnati Reds   model=73.4%  market=65.0%  edge=+8.4%   (-186 @ betrivers)
  Pete Crow-Armstrong  Cubs          model=72.5%  market=66.7%  edge=+5.9%   (-200 @ betmgm)
```

---

## Project structure

```
mlb_hits_model/
├── src/
│   ├── hit_model.py          # Binomial hit probability model
│   ├── hr_model.py           # Poisson home run probability model
│   ├── mlb_api_client.py     # MLB Stats API wrapper (verified live)
│   ├── ledger.py             # CSV prediction ledger with upsert semantics
│   ├── calibration.py        # Brier score + calibration table
│   ├── odds_client.py        # The Odds API wrapper
│   ├── park_factors.py       # Park factor loader
│   └── name_matching.py      # Cross-vendor name normalization
├── scripts/
│   ├── run_daily.py          # Log today's hit predictions
│   ├── run_daily_hr.py       # Log today's HR predictions
│   ├── grade_yesterday.py    # Grade last night's hit results
│   ├── grade_yesterday_hr.py # Grade last night's HR results
│   ├── find_value.py         # Compare hits to live odds
│   ├── find_value_hr.py      # Compare HRs to live odds
│   └── publish_dashboard.py  # Export data to GitHub Pages dashboard
├── tests/                    # 59 tests across 9 files, all network-independent
├── data/
│   ├── park_factors.csv      # Hit-specific park factors (30 teams)
│   ├── hr_park_factors.csv   # HR-specific park factors (30 teams)
│   └── ledger/
│       ├── predictions_log.csv     # Hit prediction history
│       └── hr_predictions_log.csv  # HR prediction history
└── docs/
    ├── index.html            # GitHub Pages dashboard
    └── dashboard_data.json   # Auto-generated daily from ledger CSVs
```

---

## Daily workflow

```bash
# These run automatically via cron (9am grade, 10am predict):
python3 scripts/grade_yesterday.py      # grade last night's results
python3 scripts/grade_yesterday_hr.py
python3 scripts/run_daily.py            # log today's predictions
python3 scripts/run_daily_hr.py

# Run manually after setting ODDS_API_KEY:
export ODDS_API_KEY=your_key            # free tier at the-odds-api.com
python3 scripts/find_value.py
python3 scripts/find_value_hr.py

# Publish updated dashboard to GitHub Pages:
./push_dashboard.sh
```

---

## Test suite

```bash
for f in tests/test_*.py; do python3 "$f"; done
```

59 tests across 9 files. All pure-math or fixture-mocked — no network dependency, no API keys needed.

---

## Honest limitations

**Park factors are illustrative.** `data/park_factors.csv` reflects each park's well-known general character but isn't pulled from a live source. Refresh annually from FanGraphs before relying on these.

**HR model misses Statcast.** Home run rate is driven more by exit velocity, launch angle, and barrel rate than by anything in the basic MLB Stats API hitting line. The Poisson + shrinkage approach here beats a naive baseline, but Statcast data (via Baseball Savant) is the real next step for meaningful improvement.

**Name matching is normalization-only.** `src/name_matching.py` handles accents, casing, periods, and Jr./Sr./II/III suffixes but isn't true fuzzy matching. If unmatched names become a consistent problem, [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) is the natural upgrade.

**Odds comparison is not de-vigged.** Proper de-vigging needs matched same-bookmaker Over/Under pairs; using best-available-price across books is conservative (overstates how good the market is, understates edge) rather than misleading.

---

## Stack

Python · MLB Stats API (free, public) · The Odds API · pandas · empirical-Bayes shrinkage · Poisson distribution · GitHub Pages
