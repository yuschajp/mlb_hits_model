#!/bin/bash
# ============================================================
# run_sports_daily.sh
# Master daily sports trading script
# Run from: ~/Desktop/mlb_hits_model/
#
# Usage:
#   chmod +x run_sports_daily.sh
#   ./run_sports_daily.sh          # full morning run
#   ./run_sports_daily.sh grade    # grade only (9am)
#   ./run_sports_daily.sh predict  # predictions only (10am+)
#   ./run_sports_daily.sh value    # value picks only
#   ./run_sports_daily.sh publish  # push dashboard
# ============================================================

set -e
PYTHON="/Users/victoria/Documents/anaconda3/bin/python3"
MLB_DIR="$HOME/Desktop/mlb_hits_model"
F1_DIR="$HOME/Desktop/f1_model"
ODDS_API_KEY="4aa9574c600aeea17551361dc706a76a"

export ODDS_API_KEY

cd "$MLB_DIR"

MODE="${1:-all}"

# ── ANSI colors ──────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

header() { echo -e "\n${BLUE}══════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}══════════════════════════════════════${NC}"; }
ok()     { echo -e "${GREEN}  ✓ $1${NC}"; }
note()   { echo -e "${YELLOW}  → $1${NC}"; }

# ── GRADE YESTERDAY ──────────────────────────────────────────
run_grade() {
    header "GRADING YESTERDAY"

    note "Grading hits..."
    $PYTHON scripts/grade_yesterday.py && ok "Hits graded"

    note "Grading home runs..."
    $PYTHON scripts/grade_yesterday_hr.py && ok "HR graded"

    note "Grading strikeouts..."
    $PYTHON scripts/grade_yesterday_k.py && ok "K graded"

    note "Grading World Cup matches..."
    $PYTHON scripts/grade_wc.py && ok "WC graded"
}

# ── DAILY PREDICTIONS ────────────────────────────────────────
run_predict() {
    header "GENERATING PREDICTIONS"

    note "MLB hits..."
    $PYTHON scripts/run_daily.py && ok "Hits logged"

    note "MLB home runs..."
    $PYTHON scripts/run_daily_hr.py && ok "HR logged"

    note "MLB strikeouts..."
    $PYTHON scripts/run_daily_k.py && ok "K logged"

    note "World Cup matches..."
    $PYTHON scripts/run_daily_wc.py && ok "WC matches logged"

    note "World Cup goalscorers..."
    $PYTHON scripts/run_daily_wc_gs.py && ok "WC goalscorers logged"
}

# ── VALUE PICKS ──────────────────────────────────────────────
run_value() {
    header "VALUE PICKS"

    note "MLB hits value..."
    $PYTHON scripts/find_value.py

    note "MLB HR value..."
    $PYTHON scripts/find_value_hr.py

    note "MLB strikeout value..."
    $PYTHON scripts/find_value_k.py

    note "World Cup match value..."
    $PYTHON scripts/find_value_wc.py

    note "World Cup goalscorer value..."
    $PYTHON scripts/find_value_wc_gs.py
}

# ── PUBLISH DASHBOARD ────────────────────────────────────────
run_publish() {
    header "PUBLISHING DASHBOARD"
    $PYTHON scripts/publish_dashboard.py && ok "JSON generated"
    chmod +x push_dashboard.sh && ./push_dashboard.sh && ok "Pushed to GitHub Pages"
}

# ── MAIN ─────────────────────────────────────────────────────
case "$MODE" in
    grade)   run_grade ;;
    predict) run_predict ;;
    value)   run_value ;;
    publish) run_publish ;;
    all)
        run_grade
        run_predict
        run_value
        run_publish
        ;;
    *)
        echo "Usage: $0 [grade|predict|value|publish|all]"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done. $(date '+%H:%M:%S')${NC}\n"
