"""Receptions / receiving yards / rushing yards -- over-under props.

    python3 nfl_props.py

WHY THIS IS NOT A COPY OF THE TD MODEL
    Anytime TD is binary, so a mean is enough: P = 1 - exp(-lambda). An over-under
    needs P(X > line), which depends on the whole DISTRIBUTION, not just its
    centre. A model with a perfect mean and the wrong spread prices every line
    wrong in the same direction, and does it confidently.

DISTRIBUTION: EMPIRICAL, NOT ASSUMED
    Receiving yards are zero-inflated and long-tailed; receptions are a count
    with variance below Poisson given targets but above it once target volume
    varies. Rather than assert gamma or negative-binomial and inherit whatever
    that assumption gets wrong, this uses the EMPIRICAL distribution of the
    ratio actual/predicted, bucketed by usage, learned on the training seasons.
    Zeros, skew and fat tails come along for free.

        mu           = expected opportunities x per-opportunity rate (EB shrunk)
        P(X > line)  = share of training ratios exceeding line/mu

VALIDATION WITHOUT MARKET LINES
    There is no historical prop-line series, so "does it beat a price" cannot be
    asked yet. What CAN be asked is whether the distribution is honest, via the
    probability integral transform: F(actual) should be uniform on [0,1] if the
    model is calibrated. A PIT histogram that piles up at the edges means the
    spread is too narrow -- which is precisely the failure that would make every
    over-under look like value.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
TEST_FROM = 2024
OPP_WINDOW, K_OPP = 4, 0.25
# (stat, opportunity, minimum expected opportunities to be in the sample).
# The minimum is not cherry-picking: books post receiving lines for players with
# real target share, not for a WR5 who might see one look. Validating on players
# who would never be priced tells you nothing about the prices you would take.
MARKETS = {
    "receptions":      ("receptions",      "targets", 2.5),
    "receiving_yards": ("receiving_yards", "targets", 2.5),
    "rushing_yards":   ("rushing_yards",   "carries", 5.0),
}

p = pd.concat([pd.read_csv(f"data/nfl/player_week_{y}.csv", low_memory=False)
               for y in SEASONS], ignore_index=True)
p = p[p.season_type.eq("REG") & p.position_group.isin(["RB", "WR", "TE"])].copy()
for c in ("carries", "targets", "receptions", "receiving_yards", "rushing_yards"):
    p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
p = p.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
gb = p.groupby("player_id", sort=False)

# prior-only totals and recent usage -- no row may see its own outcome
for c in ("carries", "targets", "receptions", "receiving_yards", "rushing_yards"):
    p[f"pr_{c}"] = gb[c].cumsum() - p[c]
for c in ("carries", "targets"):
    p[f"rec_{c}"] = (gb[c].shift(1).groupby(p.player_id)
                     .rolling(OPP_WINDOW, min_periods=1).mean()
                     .reset_index(level=0, drop=True))
p["games_prior"] = gb.cumcount()

pm = p.groupby("position_group")[["carries", "targets"]].transform("mean")
W = np.minimum(p.games_prior, OPP_WINDOW)
p["e_carries"] = (p.rec_carries.fillna(pm.carries) * W + K_OPP * pm.carries) / (W + K_OPP)
p["e_targets"] = (p.rec_targets.fillna(pm.targets) * W + K_OPP * pm.targets) / (W + K_OPP)

print(f"{len(p):,} player-weeks {SEASONS[0]}-{SEASONS[-1]}\n")

def eb_rate(num, den, k):
    mu = num.sum() / max(den.sum(), 1)
    return (num + k * mu) / (den + k), mu

results = {}
for name, (stat, opp, _min) in MARKETS.items():
    k = 25.0 if name == "receptions" else 40.0
    rate, mu_lg = eb_rate(p[f"pr_{stat}"], p[f"pr_{opp}"], k)
    p[f"rate_{name}"] = rate
    p[f"mu_{name}"] = p[f"e_{opp}"] * rate
    print(f"{name:<17} league rate {mu_lg:7.3f} per {opp:<8} shrink k={k:.0f}")

d = p[(p.games_prior >= 4)].copy()
tr = d[d.season < TEST_FROM]
te = d[d.season >= TEST_FROM]
print(f"\ntrain {len(tr):,} rows ({SEASONS[0]}-{TEST_FROM-1})   "
      f"held out {len(te):,} rows ({TEST_FROM}-{SEASONS[-1]})\n")

USAGE_BINS = [0, 2, 4, 6, 9, 99]

def bucket(series):
    return pd.cut(series, USAGE_BINS, labels=False, include_lowest=True)

print("=" * 74)
print("PIT CALIBRATION -- F(actual) should be UNIFORM. Each decile wants ~10.0%")
print("=" * 74)
rng = np.random.default_rng(0)
for name, (stat, opp, minopp) in MARKETS.items():
    trm, tem = tr[tr[f"e_{opp}"] >= minopp], te[te[f"e_{opp}"] >= minopp]
    if len(tem) < 300:
        continue
    trm = trm.assign(b=bucket(trm[f"e_{opp}"]), r=trm[stat] / trm[f"mu_{name}"])
    tem = tem.assign(b=bucket(tem[f"e_{opp}"]))
    ratios = {b: g.r.values for b, g in trm.groupby("b") if len(g) > 200}
    if not ratios:
        continue
    pit, cov = [], {}
    for b, g in tem.groupby("b"):
        R = ratios.get(b)
        if R is None:
            continue
        obs_r = (g[stat] / g[f"mu_{name}"]).values
        Rs = np.sort(R)
        # RANDOMIZED PIT. These outcomes have an atom at zero (and receptions are
        # integer-valued), so plain F(x) piles every tied value onto one point and
        # fakes a calibration failure. Spread each atom uniformly across the
        # probability mass it actually occupies: F(x-) + U*(F(x) - F(x-)).
        lo = np.searchsorted(Rs, obs_r, side="left") / len(Rs)
        hi = np.searchsorted(Rs, obs_r, side="right") / len(Rs)
        pit.append(lo + rng.random(len(obs_r)) * (hi - lo))
    pit = np.concatenate(pit)
    dec = np.clip((pit * 10).astype(int), 0, 9)
    counts = np.bincount(dec, minlength=10) / len(pit) * 100
    worst = np.abs(counts - 10).max()
    print(f"\n{name}   n={len(pit):,}   (expected {opp} >= {minopp})")
    print("  " + " ".join(f"{c:5.1f}" for c in counts))
    print(f"  max deviation from 10.0% : {worst:.1f}pp   "
          f"{'OK' if worst < 2.5 else 'MISCALIBRATED -- spread is wrong'}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        emp = float((pit <= q).mean())
        cov[q] = emp
    print("  coverage  " + "  ".join(f"P<{int(q*100)}={cov[q]*100:.1f}%" for q in cov))
    results[name] = dict(n=int(len(pit)), worst=float(worst),
                         deciles=[float(c) for c in counts])

print("\n" + "=" * 74)
print("Edges piling up above 10% mean the predicted spread is TOO NARROW: real")
print("outcomes land outside the model's range more often than it expects, so")
print("both overs and unders near the tails would look like value when they are")
print("not. That is the failure mode to fix before any of these get priced.")
import json
json.dump(results, open("data/nfl/props_validation.json", "w"), indent=2)
print("\n-> data/nfl/props_validation.json")
