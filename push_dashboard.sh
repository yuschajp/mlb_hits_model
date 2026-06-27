#!/bin/bash
# push_dashboard.sh
# Exports fresh dashboard data from the local ledger CSVs and pushes to GitHub Pages.
# Run this after find_value.py each day.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/Users/victoria/Documents/anaconda3/bin/python3"

echo "Exporting dashboard data..."
"$PYTHON" "$REPO_DIR/scripts/publish_dashboard.py"

echo "Committing and pushing to GitHub..."
cd "$REPO_DIR"
git add docs/dashboard_data.json
git commit -m "dashboard: auto-update $(date +%Y-%m-%d)"
git push

echo "Done. GitHub Pages will update in ~30 seconds."
echo "View at: https://yuschajp.github.io/mlb_hits_model/"
