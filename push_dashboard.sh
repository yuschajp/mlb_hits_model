#!/bin/bash
set -uo pipefail
cd ~/Desktop/mlb_hits_model

echo "Exporting dashboard data..."
/Users/victoria/Documents/anaconda3/bin/python3 scripts/publish_dashboard.py

echo "Committing and pushing to GitHub..."
git add docs/dashboard_data.json
git commit -m "dashboard: auto-update $(date +%Y-%m-%d)" || echo "Nothing new to commit."

# Pull any remote changes (e.g. from the 11am cron job, or a manual push
# from another session) before pushing. -X ours means: if a rebase
# conflict happens on dashboard_data.json specifically, prefer this run's
# version -- it was just regenerated from the current ledgers, so it's
# the freshest data anyway.
#
# Unlike the previous version of this script, failures here are NOT
# silenced. If the rebase fails for a reason -X ours can't resolve (e.g.
# a conflict in a file other than dashboard_data.json), the script stops
# here instead of blindly attempting a push that's guaranteed to be
# rejected.
if ! git pull --rebase -X ours origin master; then
    echo ""
    echo "!!! git pull --rebase failed -- see the error above. !!!"
    echo "This usually means a conflict outside dashboard_data.json that"
    echo "-X ours couldn't auto-resolve. Run 'git status' to see what's"
    echo "conflicted, resolve it manually, then re-run this script."
    echo "NOT pushing -- your local commit is still safe, just not synced."
    exit 1
fi

if ! git push origin master; then
    echo ""
    echo "!!! git push failed even after a successful rebase. !!!"
    echo "This can happen if someone pushed again in the few seconds"
    echo "since the rebase completed (race with the 11am cron job, or"
    echo "another manual run). Just re-run this script -- the rebase"
    echo "step will pick up the new remote state and retry cleanly."
    exit 1
fi

echo ""
echo "Done. GitHub Pages will update in ~30 seconds."
echo "View at: https://yuschajp.github.io/mlb_hits_model/"
echo "  Pushed to GitHub Pages"
