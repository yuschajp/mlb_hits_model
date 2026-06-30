#!/bin/bash
# push_dashboard.sh
# Commits and pushes the dashboard JSON to GitHub Pages.
# Pulls first and auto-resolves conflicts on dashboard_data.json by
# always keeping the local (freshly-generated) version -- the JSON is
# fully regenerated from ledgers each run, so there's never a reason
# to merge it; the newest local version is always authoritative.

set -e

echo "Pulling latest changes first..."
git pull --no-rebase --strategy-option=ours -X ours 2>/dev/null || {
    echo "Pull had conflicts -- auto-resolving by keeping local dashboard data..."
    git checkout --ours docs/dashboard_data.json 2>/dev/null || true
    git add docs/dashboard_data.json 2>/dev/null || true
    git commit -m "auto-resolve: keep local dashboard data" 2>/dev/null || true
}

echo "Committing and pushing to GitHub..."
git add docs/dashboard_data.json
git commit -m "dashboard: auto-update $(date +%Y-%m-%d)" || echo "Nothing new to commit."
git push

echo "Done. GitHub Pages will update in ~30 seconds."
echo "View at: https://yuschajp.github.io/mlb_hits_model/"
