# MLB Hits Model

A daily-running model for the "batter records a hit (1+)" prop, built on
top of MLB's free public Stats API. Designed to run once a day: pull
today's confirmed lineups, score every batter, log the predictions, then
grade yesterday's predictions against actual results once games are final.

## How the model works

For each batter in a confirmed starting lineup:

1. **Blend three hit-rate signals**, each shrunk toward the league average
   via empirical-Bayes shrinkage so small samples don't dominate: season-
   to-date batting average, last-30-days form, and the platoon split vs.
   today's opposing starter's throwing hand (default weights 50/30/20).
2. **Adjust for the opposing pitcher** using their batting-average-against,
   also shrunk, clipped to a +/-30% band so one rough or one dominant
   pitching sample can't swing things too far.
3. **Adjust for the ballpark** via a park factor multiplier (see caveat
   below).
4. **Convert to a per-game probability**: `P(>=1 hit) = 1 - (1 - p)^AB`,
   where AB is the batter's expected at-bats for his lineup spot (leadoff
   hitters get more at-bats than the #9 hitter, all else equal).

All of this is in `src/hit_model.py` and is fully unit-tested with no
network dependency -- run `python3 tests/test_hit_model.py` any time.

## Daily workflow

```bash
pip install -r requirements.txt

# Morning of game day, after most lineups are posted (~1-2 hrs pre-game):
python3 scripts/run_daily.py

# Next morning, after last night's games are final:
python3 scripts/grade_yesterday.py
```

`run_daily.py` prints the day's highest hit-probability batters and logs
every prediction to `data/ledger/predictions_log.csv`. `grade_yesterday.py`
fetches actual results for any ungraded rows, updates the ledger, and
prints rolling calibration stats (Brier score + a predicted-vs-actual
bucket table) -- the same calibration discipline used in the soccer
project, and for the same reason: a model can look good on raw hit rate
while being badly overconfident, and that only shows up if you check.

For automation, add both as daily cron jobs (e.g. `run_daily.py` at 4pm
local, `grade_yesterday.py` the next morning at 9am) once you've confirmed
they work against live data.

## Read this before trusting any of the live data plumbing

I built `src/mlb_api_client.py` against the well-documented community
structure of `statsapi.mlb.com` (the same backend the official MLB apps
use, reverse-engineered by the open-source community -- see the
`toddrob99/MLB-StatsAPI` project on GitHub for the deepest field
reference). **I have no network access in the environment where this was
written, so none of it has been tested against a live response.** Before
trusting the output:

```bash
python3 scripts/run_daily.py --dump-raw
```

This prints the raw schedule JSON for today so you can confirm the field
paths used throughout `mlb_api_client.py` actually match. The two spots
I'm least confident about: the `battingOrder` encoding used to detect
starters vs. substitutes (`get_confirmed_lineup`), and the `sitCodes`
parameter for platoon splits (`get_splits_vs_hand`). Every parsing
function fails soft (returns `(0, 0)` rather than crashing) if a field
isn't where it's expected, so a wrong guess degrades gracefully into "no
data, lean on league average" rather than breaking the whole run -- but
you should still go verify and adjust the field paths if needed.

The parsing *logic itself* (not the live schema assumption) is tested via
mocked fixture data in `tests/test_mlb_api_client.py` and
`tests/test_run_daily_integration.py` -- 23 tests total, all passing, all
network-independent.

## Park factors are illustrative, not authoritative

`data/park_factors.csv` reflects each park's well-known general
character (Coors Field's altitude, marine air at Petco/Oracle Park, the
Green Monster's effect on Fenway, etc.) but isn't pulled from a live,
current-season source. Refresh annually from
[FanGraphs' published guide](https://www.fangraphs.com/guts.aspx?type=pf)
before relying on this -- park factors genuinely shift over time (Camden
Yards moved its left-field fences back in 2022 and meaningfully changed
its profile overnight).

## Known limitations / roadmap

- **Expected at-bats is a fixed point estimate per lineup slot**, not a
  distribution. Refining this with team-specific historical AB-per-slot
  data would tighten things up, especially for teams that run unusually
  long or short games.
- **No market odds integration yet.** This logs and grades the model's
  own probabilities against actual outcomes, which is useful on its own
  (calibration is the real test of whether the model has any skill at
  all), but doesn't yet compare against sportsbook prop prices to flag
  value. That's a natural next step once the model's calibration looks
  solid on real data -- most regulated sportsbooks carry "to record a
  hit" props, though availability and pricing vary by state the same way
  discussed in the soccer-trading-engine README.
- **No injury/lineup-change handling beyond what the boxscore shows at
  call time** -- a late scratch after the script runs won't be caught
  without rerunning closer to first pitch.
- [ ] Verify and fix the live API field paths flagged above
- [ ] Add market odds comparison once a data source is wired in
- [ ] Team-specific expected-AB-by-slot instead of one league-wide table
- [ ] Extend to the over/under hits-total prop using the same building
      blocks (the per-AB rate this already estimates, applied to a
      binomial/Poisson model over expected AB instead of just "any hit")
