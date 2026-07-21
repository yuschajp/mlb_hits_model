#!/bin/bash
export PATH="/Users/victoria/Documents/anaconda3/bin:$PATH"
cd /Users/victoria/Desktop/mlb_hits_model
/bin/bash run_all_daily.sh >> logs/cron.log 2>&1
