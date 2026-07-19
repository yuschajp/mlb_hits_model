# UFC Integration Guide - Quantified Edge

## Overview
You now have a complete UFC prediction module integrated into your Quantified Edge daily model run. The system includes:

- **ELO-based Fight Prediction** (`ufc_model.py`): Fighter ratings, win probability predictions
- **Prop Betting Analysis** (`ufc_props.py`): KO/TKO, submission, round totals, significant strikes
- **Dashboard Integration** (`ufc_dashboard.py`): Unified HTML dashboard alongside MLB/WC/F1
- **Daily Orchestration** (`run_all_daily.sh`): Updated to run all 4 sports

---

## Files Created

```
├── ufc_model.py              # Main fight prediction engine
├── ufc_props.py              # Prop betting value analysis
├── ufc_dashboard.py          # Dashboard aggregator & HTML gen
├── UFC_INTEGRATION_GUIDE.md  # This file
└── run_all_daily.sh          # Updated master script (now runs 4 sports)
```

---

## Quick Start

### 1. Copy files from Vault to mlb_hits_model
Files are in: `/Users/victoria/Desktop/Joe Resumes/Joe Yuschak Vault/`

```bash
cd ~/Desktop/Joe\ Resumes/Joe\ Yuschak\ Vault
cp ufc_model.py ufc_props.py ufc_dashboard.py UFC_INTEGRATION_GUIDE.md ~/Desktop/mlb_hits_model/
cp run_all_daily.sh ~/Desktop/mlb_hits_model/
```

### 2. Make script executable
```bash
chmod +x ~/Desktop/mlb_hits_model/run_all_daily.sh
chmod +x ~/Desktop/mlb_hits_model/ufc_model.py
```

### 3. Install/verify dependencies
The UFC module uses:
- `requests` (already have for Odds API)
- `pandas` (already have for data handling)
- `json`, `datetime`, `math`, `os` (built-in)

```bash
pip install requests pandas
```

### 4. Run daily
```bash
~/Desktop/mlb_hits_model/run_all_daily.sh
```

Or schedule via cron:
```bash
0 8 * * * /Users/victoria/Desktop/mlb_hits_model/run_all_daily.sh
```

---

## Architecture

### ELO System (`UFCEloRating`)
- **Base Rating**: 1600 (standard Elo)
- **K-Factor**: 32 (adjustable for divisional depth)
- **Bonuses**: +50% rating delta for KO/TKO/Submission wins
- **Floor**: 800 (prevents low-rated fighters from tanking)

**Update mechanism:**
```
rating_delta = K * (result - expected_score)
expected_score = 1 / (1 + 10^((opponent_rating - fighter_rating) / 400))
```

### Data Flow
1. **Fetch Events** → Odds API (upcoming UFC events)
2. **Load Fighter ELOs** → `ufc_elo_ratings.json` (persistent across days)
3. **Predict Fights** → ELO matchup → win probabilities
4. **Analyze Props** → Model probabilities vs. market odds → edge detection
5. **Save Results** → `ufc_predictions_*.json`, `ufc_props_analysis.json`
6. **Dashboard** → HTML aggregation with all 4 sports

---

## UFC Prop Analysis

### Props Supported
1. **KO/TKO** - Odds API market (DraftKings, FanDuel, etc.)
2. **Submission** - Calculated from fighter submission rate history
3. **Round Totals** - Over/Under rounds (fight distance dependent)
4. **Significant Strikes** - Over/Under strike counts
5. **Method of Victory** - Distribution: KO/TKO, Submission, Decision

### Value Detection
**Minimum edge for "Play"**: 350 basis points (3.5%)
**Minimum edge for "Strong Play"**: 600 basis points (6%)

**Recommendation Logic:**
- **Strong Play** → edge ≥ 6% AND EV ≥ +0.15
- **Play** → edge ≥ 3.5% AND EV ≥ +0.05
- **Fade** → edge ≤ -5% (bet opposite side)
- **Pass** → edge too thin

### Example Output
```json
{
  "prop_name": "KO/TKO",
  "fighter": "Fighter A",
  "prediction": "Yes",
  "model_prob": 0.42,           // 42% chance per our model
  "market_odds": 2.10,          // +110 in American
  "implied_prob": 0.4762,       // Market says 47.6%
  "value": 0.0822,              // +8.22% EV
  "edge_percent": 475.0,        // -4.76% edge (actually negative here)
  "recommendation": "Pass",
  "notes": "..."
}
```

---

## Customization

### Adjust ELO Sensitivity
Edit `ufc_model.py`:
```python
class UFCEloRating:
    K_FACTOR = 32  # ← Increase for faster rating swings, decrease for stability
    BASE_RATING = 1600
```

### Change Prop Edge Thresholds
Edit `ufc_props.py`:
```python
class UFCPropAnalyzer:
    MIN_EDGE_BP = 350  # ← 3.5% minimum edge for "Play"
    MIN_ODDS = 1.5     # ← Skip heavy favorites
    MAX_ODDS = 3.5     # ← Skip heavy underdogs
```

### Add Fighter Historical Stats
The skeleton exists in `UFCDataHandler`:
```python
def load_fighter_stats(self, fighter_name: str) -> Dict:
    """Load historical fighter stats (KO%, Sub%, Striking accuracy)"""
    # TODO: Implement cache/API fetch for:
    # - Historical KO/TKO rate
    # - Historical submission rate
    # - Striking accuracy
    # - Takedown defense
    # - Strike distribution (jabs, crosses, kicks, etc.)
```

Add stats JSON file:
```json
{
  "Fighter A": {
    "ko_tko_rate": 0.45,
    "submission_rate": 0.15,
    "decision_rate": 0.40,
    "significant_strikes_avg": 87.3,
    "takedown_accuracy": 0.52,
    "takedown_defense": 0.68
  }
}
```

Then integrate into `predict_method_of_victory()`.

### Fine-tune Odds API Integration
The `UFCDataHandler` currently fetches from Odds API. To add sportsbook filters:

```python
def get_fight_odds(self, event_id: str, sportsbooks: List[str] = None) -> Dict:
    """Get odds from specific sportsbooks only"""
    if sportsbooks is None:
        sportsbooks = ["draftkings", "fanduel", "betmgm"]
    
    params = {
        "apiKey": self.api_key,
        "regions": "us",
        "markets": "h2h,moneyline",
        "oddsFormat": "decimal"
    }
    # Filter results by sportsbook
    ...
```

---

## Output Files

### Daily Generated
- `ufc_predictions_YYYYMMDD_HHMMSS.json` - Fight predictions
- `ufc_props_analysis.json` - Prop analysis (overwritten each run)
- `ufc_dashboard.html` - HTML dashboard (latest)
- `logs/ufc_YYYYMMDD_HHMMSS.log` - Execution log

### Persistent
- `ufc_elo_ratings.json` - Fighter ELO ratings (cumulative)

---

## Troubleshooting

### API Quota Exceeded
You track Odds API quota in your existing model. UFC requests go through same quota.

**Solution**: Reduce frequency of `get_upcoming_events()` calls or upgrade API plan.

### Fighter Not In Elo Ratings
New UFC fighters default to base rating (1600) on first occurrence.
```python
def get_or_create_fighter(self, fighter_name: str) -> UFCEloRating:
    if fighter_name not in self.elo_ratings:
        self.elo_ratings[fighter_name] = UFCEloRating(fighter_name)
    return self.elo_ratings[fighter_name]
```

### No Props Generated
Props only generate for plays with sufficient edge (default 3.5%+).
- Check `edge_percent` in output JSON
- If all are "Pass", market is efficient — good sign your model confidence isn't overblown

### Dashboard Not Rendering
Make sure JSON files exist before running `ufc_dashboard.py`:
```bash
python3 ufc_model.py          # Generates ufc_predictions_*.json
python3 ufc_props.py          # Generates ufc_props_analysis.json
python3 ufc_dashboard.py      # Consumes both
```

---

## Next Steps: Production Enhancements

### 1. Historical Fighter Stats
Integrate with sherdog.com or UFC API (if available) to pull:
- Historical fight outcomes (recent 5–10 fights)
- Strike distribution by opponent type
- Submission success rate vs. specific opponents

### 2. Venue/Altitude Effects
Add slight ELO adjustments for:
- High-altitude events (Denver, Mexico City)
- Home advantage (rarely applies in UFC)
- Crowd size/noise level

### 3. Injury/Recent Activity
- Penalize fighters coming off long layoffs
- Boost recently active fighters (ring rust vs. sharpness)

### 4. Matchup History
- Track head-to-head records
- Add multiplicative ELO adjustment for rematches

### 5. Live Props During Events
- Extend dashboard to show in-game prop updates
- Track calibration as fights unfold

### 6. Backtesting Framework
```python
class UFCBacktester:
    def backtest_predictions(self, historical_fights: List[Dict]) -> Dict:
        """Evaluate model accuracy on past events"""
        # Track prediction vs. actual outcome
        # Calculate Brier score, log loss, calibration
```

---

## Calibration Notes

Like your other models (MLB K%, WC goals), UFC predictions will drift if:
- Fighter pool changes dramatically (retirements, new signings)
- Rule changes affect fight outcomes
- Odds API shifts data provider

**Weekly check:**
```bash
# Review latest predictions vs. actual results
tail -20 logs/ufc_*.log
# Cross-reference against official UFC results
```

---

## Integration with Dashboard Ecosystem

Your existing dashboard (MLB/WC/F1) can be extended to include UFC:

1. **Picks & Parlays** tab → Add UFC legs
2. **Stats** one-pager → UFC stats block
3. **Marketing use** → "4 sports, 1 model" messaging

Example enhancement for Picks & Parlays:
```python
class ParleyBuilder:
    def add_ufc_leg(self, fighter: str, prediction: str, odds: float):
        """Add UFC moneyline or prop to parlay"""
        self.legs.append({
            "sport": "UFC",
            "selection": f"{fighter} {prediction}",
            "odds": odds
        })
```

---

## Questions?

Refer back to:
- **ELO math**: `ufc_model.py` → `UFCEloRating.expected_score()`
- **Prop logic**: `ufc_props.py` → `UFCPropAnalyzer.analyze_*()` methods
- **Data flow**: This guide's "Architecture" section

Good luck! 🥊
