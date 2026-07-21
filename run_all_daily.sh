#!/bin/bash

cd ~/Desktop/mlb_hits_model || { echo "Could not cd into mlb_hits_model -- aborting."; exit 1; }

PYTHON=/Users/victoria/Documents/anaconda3/bin/python3
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

declare -a RESULTS

run_step() {
    local label="$1"
    local script="$2"
    local logfile="$3"

    echo ""
    echo "── ${label} ──────────────────────────────────"
    if $PYTHON "$script" >> "$LOG_DIR/$logfile" 2>&1; then
        echo "  OK"
        RESULTS+=("OK    $label")
    else
        echo "  FAILED -- see $LOG_DIR/$logfile for details"
        RESULTS+=("FAIL  $label")
    fi
}

echo "=========================================="
echo " Daily pipeline run: $(date '+%Y-%m-%d %H:%M')"
echo "=========================================="

# ── 1. Grade yesterday's results ──────────────────────────────────────────
run_step "Grade HR"            "scripts/grade_yesterday_hr.py" "grade_hr.log"
run_step "Grade K"             "scripts/grade_yesterday_k.py"  "grade_k.log"

# ── 2. Generate today's predictions ───────────────────────────────────────
run_step "Run HR"              "scripts/run_daily_hr.py"       "run_hr.log"
run_step "Run K"               "scripts/run_daily_k.py"        "run_k.log"

# ── 3. Commit ledger updates ───────────────────────────────────────────────
echo ""
echo "── Commit ledger updates ──────────────────────"
if git diff --quiet data/ledger/ 2>/dev/null; then
    echo "  Nothing to commit."
    RESULTS+=("OK    Commit ledger updates (nothing to commit)")
else
    git add data/ledger/*.csv
    if git commit -m "Update ledger data ($(date '+%Y-%m-%d %H:%M'))" >> "$LOG_DIR/ledger_commit.log" 2>&1; then
        echo "  OK"
        RESULTS+=("OK    Commit ledger updates")
    else
        echo "  FAILED -- see $LOG_DIR/ledger_commit.log for details"
        RESULTS+=("FAIL  Commit ledger updates")
    fi
fi

# ── 4. Publish + push dashboard ───────────────────────────────────────────
echo ""
echo "── Publish dashboard ──────────────────────────"
if $PYTHON scripts/publish_dashboard.py >> "$LOG_DIR/publish.log" 2>&1; then
    echo "  OK"
    RESULTS+=("OK    Publish dashboard")

    echo ""
    echo "── Push dashboard ─────────────────────────────"
    if ./push_dashboard.sh >> "$LOG_DIR/dashboard.log" 2>&1; then
        echo "  OK"
        RESULTS+=("OK    Push dashboard")
    else
        echo "  FAILED -- see $LOG_DIR/dashboard.log for details"
        RESULTS+=("FAIL  Push dashboard")
    fi
else
    echo "  FAILED -- see $LOG_DIR/publish.log for details"
    RESULTS+=("FAIL  Publish dashboard")
    RESULTS+=("SKIP  Push dashboard")
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
n_fail=0
for r in "${RESULTS[@]}"; do
    echo "  $r"
    [[ "$r" == FAIL* ]] && n_fail=$((n_fail + 1))
done
echo "=========================================="

if [ "$n_fail" -gt 0 ]; then
    echo " $n_fail step(s) failed -- check the logs listed above."
    exit 1
else
    echo " All steps completed successfully."
    exit 0
fi
