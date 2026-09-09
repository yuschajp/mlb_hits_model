#!/bin/bash

cd ~/Desktop/mlb_hits_model || { echo "Could not cd into mlb_hits_model -- aborting."; exit 1; }

# Load API keys (CFBD_KEY etc). .env is gitignored; without this the CFB line
# pull works interactively but fails under cron, where the shell has no exports.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

PYTHON=/Users/yuschajp/Desktop/mlb_hits_model/venv/bin/python3
# The CFB/NFL models need pandas. If the venv lacks it, fall back to system
# python3 rather than failing four steps with an import error.
if ! "$PYTHON" -c "import pandas" >/dev/null 2>&1; then
    if python3 -c "import pandas" >/dev/null 2>&1; then
        echo "note: venv has no pandas, using system python3 for football steps"
        FOOTBALL_PY=$(command -v python3)
    else
        echo "WARNING: no python with pandas found -- football steps will skip"
        FOOTBALL_PY=""
    fi
else
    FOOTBALL_PY="$PYTHON"
fi
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

# Football runs only when there is a slate. A missing lines file is a normal
# Tuesday, not an error -- it reports SKIP so the summary stays meaningful.
run_optional() {
    local label="$1"; local guard="$2"; shift 2
    echo ""
    echo "-- ${label} --------------------------------"
    if [ -z "$FOOTBALL_PY" ]; then
        echo "  SKIP (no pandas)"; RESULTS+=("SKIP  $label"); return
    fi
    if [ ! -e "$guard" ]; then
        echo "  SKIP (no $guard)"; RESULTS+=("SKIP  $label"); return
    fi
    if "$FOOTBALL_PY" "$@" >> "$LOG_DIR/football.log" 2>&1; then
        echo "  OK"; RESULTS+=("OK    $label")
    else
        echo "  FAILED -- see $LOG_DIR/football.log"; RESULTS+=("FAIL  $label")
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

# ── 2b. Football: grade what has finished, then build today's boards ────
TODAY=$(date +%Y-%m-%d)

# CFB grading needs no lines file -- it grades every picks_*.csv with a result
# available, so the guard is just the directory.
# Lines are pulled automatically -- nothing to paste. Both writers exit 0 with
# no file when there is no slate, so the board steps below just skip.
run_optional "Pull CFB lines" "data/cfb"                  cfb_lines.py "$TODAY"
run_optional "Pull NFL lines" "data/nfl"                  nfl_lines.py

run_optional "Grade CFB"  "data/cfb"                      cfb_grade.py
run_optional "Run CFB"    "data/cfb/lines_${TODAY}.txt"   cfb_board2.py "$TODAY"

# NFL boards are weekly, not daily. Symlink or copy the current week's file to
# data/nfl/lines_current.txt; on days it does not exist the step skips quietly.
NFL_LINES="${NFL_LINES:-data/nfl/lines_current.txt}"
run_optional "Run NFL props" "$NFL_LINES"                 nfl_td_board.py "$NFL_LINES"
run_optional "Run NFL O/U"   "data/nfl"                   nfl_props_board.py

# ── 3. Commit ledger updates ───────────────────────────────────────────────
echo ""
echo "── Commit ledger updates ──────────────────────"
if git diff --quiet data/ledger/ data/cfb/ data/nfl/ 2>/dev/null; then
    echo "  Nothing to commit."
    RESULTS+=("OK    Commit ledger updates (nothing to commit)")
else
    git add data/ledger/*.csv
    git add data/cfb/*.csv data/nfl/*.csv 2>/dev/null || true
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
