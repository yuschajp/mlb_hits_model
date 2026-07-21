#!/bin/bash

# Daily scheduler — runs at 5 PM EST

export PATH="/Users/victoria/Documents/anaconda3/bin:$PATH"
LOG_DIR="/Users/victoria/Desktop/mlb_hits_model/logs"
mkdir -p "$LOG_DIR"

echo "📅 Scheduler started: $(date)" >> "$LOG_DIR/scheduler.log"

while true; do
    current_hour=$(date +%H)
    current_min=$(date +%M)
    
    # Check if it's 17:00 (5 PM)
    if [ "$current_hour" = "17" ] && [ "$current_min" = "00" ]; then
        echo "🚀 Running daily script at $(date)" >> "$LOG_DIR/scheduler.log"
        cd /Users/victoria/Desktop/mlb_hits_model
        /bin/bash run_all_daily.sh >> "$LOG_DIR/cron.log" 2>&1
        echo "✅ Daily script completed at $(date)" >> "$LOG_DIR/scheduler.log"
        
        # Sleep 61 seconds so it doesn't run twice in same minute
        sleep 61
    fi
    
    # Check every 30 seconds
    sleep 30
done
