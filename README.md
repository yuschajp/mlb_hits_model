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

## Home run model

`src/hr_model.py` is the home run version of the hits model, sharing the
same data client, ledger, calibration, odds, and name-matching plumbing.
Two real modeling differences from the hits model, both because home runs
are a much rarer per-at-bat event (roughly 1 in 30-35 vs. roughly 1 in 4
for any hit):

**Poisson instead of binomial.** `P(>=1 HR) = 1 - e^(-lambda)` where
`lambda = adjusted_hr_rate * expected_AB`. This is both a better statistical
fit for a rare event and mathematically cleaner than the hits model's
approach -- there's no integer-AB-rounding step, so `hr_probability()` and
`hr_over_under_probability(..., line=0.5)` are guaranteed to agree exactly
(checked in the test suite), unlike the analogous hits-model situation
where rounding caused a real, caught-by-testing inconsistency.

**Much stronger shrinkage priors.** A batter who's gone deep twice in his
last 15 at-bats is not actually a 13% HR hitter; HR rate needs far more
at-bats than batting average to mean anything, so `stabilized_hr_rate()`
uses a prior_ab of 400 versus the hits model's 200.

**A separate, HR-specific park factor table** (`data/hr_park_factors.csv`).
Hits-friendly and HR-friendly aren't the same thing -- Fenway Park is
hits-friendly (1.06) but HR-suppressing (0.92), since the Green Monster
turns many would-be home runs into doubles instead. Same illustrative-not-
authoritative caveat as the hits park factors: refresh from a real source
before relying on these.

Daily workflow is identical in shape to the hits model, just with `_hr`
suffixed scripts and a separate ledger file so the two never collide:

```bash
python3 scripts/run_daily_hr.py        # log today's HR predictions
python3 scripts/grade_yesterday_hr.py  # grade yesterday's, build calibration history
python3 scripts/find_value_hr.py       # compare to live "Over 0.5 home runs" odds
```

**Honest gap, flagged deliberately rather than addressed:** home run rate
is driven much more by quality-of-contact (exit velocity, launch angle,
barrel rate, flyball rate) than by anything in the basic MLB Stats API
hitting line this pulls. This v1 will beat a naive baseline using the same
season/recent/platoon-split approach as the hits model, but Statcast data
(via Baseball Savant) is the real next step to materially improve it --
batting-average-style logic captures most of what matters for the hits
model, but captures noticeably less of what actually separates real power
hitters for this one.

## Finding value against real odds

`scripts/find_value.py` compares today's logged predictions to live
"Over 0.5 hits" odds (mathematically identical to the "record a hit" prop)
and flags batters where the model beats the best available price by at
least 5 percentage points.

```bash
export ODDS_API_KEY=your_key_here   # free signup at https://the-odds-api.com/
python3 scripts/run_daily.py        # log today's predictions first
python3 scripts/find_value.py       # then compare to live odds
```

Two things worth knowing before trusting this:

**It's deliberately conservative, not de-vigged.** The comparison uses the
raw implied probability of the best available price across bookmakers,
not a fair/de-vigged line -- proper de-vigging needs the Over and Under
price from the *same* bookmaker, and the best Over price and best Under
price might come from different books. Using the raw (vig-inflated)
implied probability as the bar is a safer overstatement of how good the
market price is, not an understatement -- so flagged value is a
conservative signal, not an inflated one. Pulling matched same-book
Over/Under pairs and de-vigging properly is on the roadmap.

**Name matching across vendors is unverified against live data.**
`src/name_matching.py` normalizes accents, casing, periods, and Jr./Sr./
II/III suffixes so "Andrés Giménez" matches "Andres Gimenez," but it's
exact-after-normalization, not true fuzzy matching. If a meaningful
number of players come back as `[no match]` when you run this for real,
that's the thing to look at first -- either extend the normalization or
swap in a real fuzzy-matching library like `rapidfuzz`.

Also check `the-odds-api.com`'s current pricing page for the live free-tier
quota before running this across a full daily slate -- player-prop calls
cost usage credits separately from the free schedule/odds endpoints, and
exact limits change over time.

## Over/under hits-total prop

The same underlying batter-quality estimate also powers the over/under
market (e.g. "Over 1.5 hits"), via `src/hit_model.py`'s
`over_under_probability()`:

```python
from src.hit_model import hit_probability, over_under_probability

p_hit, adjusted_ba, expected_ab = hit_probability(...)
p_over, p_under = over_under_probability(adjusted_ba, expected_ab, line=1.5)
```

One honest wrinkle: `over_under_probability()` rounds `expected_ab` to an
integer at-bat count to build a proper Binomial count distribution, while
`hit_probability()` itself uses the unrounded fractional value in a
continuous formula. That means `over_under_probability(..., line=0.5)`
will be *close to* but not bit-for-bit identical to `hit_probability()`'s
own output -- the rounding collapses some of the sub-1-at-bat differences
between lineup spots. `hit_probability()` is the one with real calibration
data behind it (see the daily grading history); treat the binomial version
as a reasonable extension for higher lines where there's no existing
baseline to preserve, not as a literal replacement for the "any hit" prop.

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
- [ ] Verify and fix the live API field paths flagged above (the
      `sitCodes` platoon-split param is the remaining unverified one)
- [x] Add market odds comparison (`scripts/find_value.py`)
- [x] Extend to the over/under hits-total prop (`over_under_probability()`)
- [ ] Team-specific expected-AB-by-slot instead of one league-wide table
- [ ] Proper de-vigged odds comparison (matched same-bookmaker Over/Under
      pairs instead of the current conservative raw-implied-probability bar)
- [ ] Verify name_matching.py against live odds data; upgrade to a real
      fuzzy-matching library if normalization alone isn't catching enough
