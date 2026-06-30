# Sports Trading Daily Runbook
**Last updated: 2026-06-27**

---

## Terminal Setup

Sports trading and AI Alpha Labs evals should run in **separate terminal tabs** to avoid confusion.

**Open a dedicated sports trading tab:**
```
Cmd+T                          # new tab in Terminal/iTerm2
cd ~/Desktop/mlb_hits_model
export ODDS_API_KEY=4aa9574c600aeea17551361dc706a76a
```

**Keep evals running in a separate tab:**
```
cd ~/Desktop/ai-alpha-labs
# run eval scripts here
```

---

## Master Script (easiest)

```bash
cd ~/Desktop/mlb_hits_model
chmod +x run_sports_daily.sh

./run_sports_daily.sh grade     # 9am  — grade yesterday
./run_sports_daily.sh predict   # 10am — generate today's predictions
./run_sports_daily.sh value     # 11am — find value picks
./run_sports_daily.sh publish   # any  — push dashboard to GitHub Pages
./run_sports_daily.sh           # runs all four in sequence
```

---

## Manual Scripts by Sport

### MLB — Hits Model
```bash
cd ~/Desktop/mlb_hits_model

# 9am: grade yesterday
python3 scripts/grade_yesterday.py

# 10am: generate predictions
python3 scripts/run_daily.py

# 11am: find value (after lineups post)
export ODDS_API_KEY=4aa9574c600aeea17551361dc706a76a
python3 scripts/find_value.py
```

### MLB — Home Run Model
```bash
# 9am
python3 scripts/grade_yesterday_hr.py

# 10am
python3 scripts/run_daily_hr.py

# 11am
python3 scripts/find_value_hr.py
```

### MLB — Strikeout Model
```bash
# 9am
python3 scripts/grade_yesterday_k.py

# 10am (probable pitchers usually post by 10am)
python3 scripts/run_daily_k.py

# value
python3 scripts/find_value_k.py
```

### World Cup — Match Predictions
```bash
# daily (no grading needed until match finishes)
python3 scripts/run_daily_wc.py

# after matches complete
python3 scripts/grade_wc.py

# value picks
python3 scripts/find_value_wc.py
```

### World Cup — Goalscorer Predictions
```bash
# daily
python3 scripts/run_daily_wc_gs.py

# value picks
python3 scripts/find_value_wc_gs.py
```

### Recalibration (every ~200 graded hits predictions)
```bash
python3 scripts/recalibrate.py
```

### Dashboard
```bash
python3 scripts/publish_dashboard.py
chmod +x push_dashboard.sh && ./push_dashboard.sh
```

---

## Formula 1 — Race Weekend Scripts

**Run from `~/Desktop/f1_model/`**

### Friday–Saturday (before qualifying)
```bash
cd ~/Desktop/f1_model

# predict qualifying grid using FP2/FP3 pace data
python3 scripts/predict_quali.py --meeting 1288
```

### Saturday (after qualifying)
```bash
# grade qualifying predictions vs actual grid
python3 scripts/grade_quali.py --meeting 1288 --quali-session 11311

# generate race predictions using real starting grid
python3 scripts/run_prerace.py --meeting 1288 --race-name "Austrian GP" --race-date 2026-06-28
```

### Saturday night — F1 value picks
```bash
# auto scrape DraftKings (may be blocked)
python3 scripts/find_value_race.py --meeting 1288

# manual flags if DK blocks scraper
python3 scripts/find_value_race.py --meeting 1288 \
  --podium "Russell:-500,Leclerc:-175,Hamilton:-175" \
  --h2h "Russell:Antonelli:-175,Leclerc:Hamilton:-130"
```

### Sunday (after race)
```bash
python3 scripts/grade_race.py --year 2026 --round 11
```

### Push F1 data to dashboard
```bash
cd ~/Desktop/mlb_hits_model
python3 scripts/publish_dashboard.py
chmod +x push_dashboard.sh && ./push_dashboard.sh
```

---

## Cron Jobs (auto-runs)

These run automatically — no action needed:

| Time | Script | What it does |
|------|--------|--------------|
| 9:00am | `grade_yesterday.py` | Grades hits |
| 9:00am | `grade_yesterday_hr.py` | Grades HR |
| 10:00am | `run_daily.py` | Hits predictions |
| 10:00am | `run_daily_hr.py` | HR predictions |

---

## Environment Variables

```bash
export ODDS_API_KEY=4aa9574c600aeea17551361dc706a76a
# No other keys needed — WC uses openfootball (free, no key)
```

---

## Repos and Locations

| Repo | Path | GitHub Pages |
|------|------|------|
| MLB + WC dashboard | `~/Desktop/mlb_hits_model/` | yuschajp.github.io/mlb_hits_model/ |
| F1 model | `~/Desktop/f1_model/` | — |
| AI Alpha Labs | `~/Desktop/ai-alpha-labs/` | — |

---

## Dashboard Tabs

| Tab | Data source | Refresh |
|-----|------------|---------|
| Hits Model | `predictions_log.csv` | `publish_dashboard.py` |
| Home Run Model | `hr_predictions_log.csv` | `publish_dashboard.py` |
| Strikeouts | `k_predictions_log.csv` | `publish_dashboard.py` |
| World Cup | `wc_predictions_log.csv` | `publish_dashboard.py` |
| F1 Racing | `f1_model/data/ledger/` | `publish_dashboard.py` |
