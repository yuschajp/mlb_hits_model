#!/bin/bash
cd ~/Desktop/mlb_hits_model

echo "Exporting dashboard data..."
/Users/victoria/Documents/anaconda3/bin/python3 scripts/publish_dashboard.py

echo "Committing and pushing to GitHub..."
git add docs/dashboard_data.json
git commit -m "dashboard: auto-update $(date +%Y-%m-%d)" || echo "Nothing new to commit."
git pull --rebase -X ours 2>/dev/null || true
git push

echo "Done. GitHub Pages will update in ~30 seconds."
echo "View at: https://yuschajp.github.io/mlb_hits_model/"
