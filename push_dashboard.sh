#!/bin/bash
echo "Pulling latest changes first..."
git pull --rebase 2>/dev/null || git checkout --ours docs/dashboard_data.json && git add docs/dashboard_data.json && git rebase --continue 2>/dev/null || true

echo "Exporting dashboard data..."
/Users/victoria/Documents/anaconda3/bin/python3 scripts/publish_dashboard.py

echo "Committing and pushing to GitHub..."
git add docs/dashboard_data.json
git commit -m "dashboard: auto-update $(date +%Y-%m-%d)" || echo "Nothing new to commit."
git push && echo "Done. GitHub Pages will update in ~30 seconds." && echo "View at: https://yuschajp.github.io/mlb_hits_model/"
