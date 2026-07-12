#!/bin/bash
#
# run_all_daily.sh
#
# Runs the full daily pipeline: grade yesterday's results, generate
# today's predictions across every sport, then publish and push the
# dashboard. Chains what were previously ~8-10 separate manual commands.
#
# Run with: ./run_all_daily.sh
#
# Design notes:
#   - Does NOT use `set -e`. One script failing (e.g. WC hitting a missing
#     FOOTBALL_DATA_API_KEY, or a rate-limited odds API) should not kill
#     the whole run -- the other sports' predictions are independent and
#     still worth generating. Every step's pass/fail is tracked and
#     printed in a summary at the end instead.
#   - Publish + push still run even if some earlier steps failed, so you
#     get a dashboard update reflecting whatever DID succeed, rather than
#     nothing at all. The summary at the end tells you what to re-run
#     manually if something's missing.
#   - Uses the same explicit anaconda python path as the crontab entries,
#     so behavior is consistent whether this runs interactively or is
#     itself later wired into cron.

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
run_step "Grade hits"          "scripts/grade_yesterday.py"    "grade_hits.log"
run_step "Grade HR"            "scripts/grade_yesterday_hr.py" "grade_hr.log"
run_step "Grade K"             "scripts/grade_yesterday_k.py"  "grade_k.log"
run_step "Grade WC"            "scripts/grade_wc.py"           "grade_wc.log"

# ── 2. Generate today's predictions ───────────────────────────────────────
run_step "Run hits"            "scripts/run_daily.py"          "run_hits.log"
run_step "Run HR"              "scripts/run_daily_hr.py"       "run_hr.log"
run_step "Run K"               "scripts/run_daily_k.py"        "run_k.log"
run_step "Run WC matches"      "scripts/run_daily_wc.py"       "run_wc.log"
run_step "Run WC goalscorers"  "scripts/run_daily_wc_gs.py"    "run_wc_gs.log"
run_step "Run tennis"          "scripts/run_daily_tennis.py"   "run_tennis.log"
run_step "Find tennis value"   "scripts/find_value_tennis.py"  "find_value_tennis.log"

# ── 3. Commit ledger updates ───────────────────────────────────────────────
# The grading/prediction steps above modify data/ledger/*.csv every run.
# If those changes are left uncommitted, push_dashboard.sh's internal
# `git pull --rebase` fails with "You have unstaged changes" and the
# whole publish step gets blocked -- this happened repeatedly before this
# step existed. Commit them here, before publish/push, so that never
# blocks the pipeline again.
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
        echo "  (often a git conflict -- check 'git status' and resolve manually)"
        RESULTS+=("FAIL  Push dashboard")
    fi
else
    echo "  FAILED -- see $LOG_DIR/publish.log for details"
    echo "  Skipping push -- nothing new to push if publish failed."
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
