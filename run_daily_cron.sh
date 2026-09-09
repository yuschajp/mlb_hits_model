#!/bin/bash
# Cron entry point. Fixed 2026-09-08: previously pointed at /Users/victoria on
# the old machine, so every scheduled run had been failing silently.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd /Users/yuschajp/Desktop/mlb_hits_model || exit 1
mkdir -p logs
/bin/bash run_all_daily.sh >> logs/cron.log 2>&1
