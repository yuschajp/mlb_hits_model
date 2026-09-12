"""How should early-season usage be estimated: last season, this season, or a blend?

The TD model's opportunity term is its biggest early-season weakness -- in week 1
it has zero current-season data and prices everyone on last year's role. The
obvious fix is "use this season once games exist," but how fast to switch is an
empirical question, not a taste one. A single week-1 game is noisy; four 2025
games on a different team are biased. This measures which wins, week by week.

METHOD
  For weeks 1-5 of each season, predict a player's targets (and carries) with:
      PRIOR    last 4 games of the previous season
      CURRENT  this season's games so far (weeks 1..w-1)
      BLEND    weighted average, sweeping the prior's weight
  Score by MAE against what the player actually saw that week.
  Players who CHANGED TEAMS are reported separately -- their prior-season role
  is measured in a different offence and should decay faster.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
MAXW = 5
p = pd.concat([pd.read_csv(f"data/nfl/player_week_{y}.csv", low_memory=False)
               for y in SEASONS], ignore_index=True)
p = p[p.season_type.eq("REG") & p.position_group.isin(["RB", "WR", "TE"])].copy()
for c in ("targets", "carries"):
    p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
p["opp"] = p.targets + p.carries
p = p.sort_values(["player_id", "season", "week"])

rows = []
for season in SEASONS[1:]:
    prev = p[p.season == season - 1]
    cur = p[p.season == season]
    # last 4 games of the prior season, per player
    pr = (prev.sort_values("week").groupby("player_id").tail(4)
          .groupby("player_id").agg(prior_opp=("opp", "mean"),
                                    prior_team=("team", "last")).reset_index())
    for w in range(1, MAXW + 1):
        tgt = cur[cur.week == w]
        hist = cur[cur.week < w]
        cm = (hist.groupby("player_id").agg(cur_opp=("opp", "mean"),
                                            n=("week", "size")).reset_index()
              if len(hist) else pd.DataFrame(columns=["player_id", "cur_opp", "n"]))
        m = tgt.merge(pr, on="player_id", how="inner").merge(cm, on="player_id", how="left")
        m["moved"] = m.team != m.prior_team
        for _, r in m.iterrows():
            rows.append(dict(season=season, week=w, actual=r.opp,
                             prior=r.prior_opp,
                             cur=r.cur_opp if pd.notna(r.get("cur_opp")) else np.nan,
                             n_cur=r.n if pd.notna(r.get("n")) else 0,
                             moved=bool(r.moved)))
d = pd.DataFrame(rows)
d = d[d.prior >= 2]                    # players with a real prior-season role
print(f"{len(d):,} player-weeks, weeks 1-{MAXW}, seasons {SEASONS[1]}-{SEASONS[-1]}\n")

def mae(a, b): return float(np.abs(a - b).mean())

print("MAE of predicted OPPORTUNITIES (targets+carries)")
print(f"{'week':<6}{'n':>7}{'prior only':>12}{'current only':>14}{'best blend':>12}{'prior wt':>10}")
print("-" * 62)
BEST = {}
for w in range(1, MAXW + 1):
    s = d[d.week == w]
    if w == 1:
        print(f"{w:<6}{len(s):>7}{mae(s.prior, s.actual):>12.2f}{'n/a':>14}{'n/a':>12}{'1.00':>10}")
        BEST[w] = 1.0; continue
    s = s[s.cur.notna()]
    if len(s) < 100: continue
    best, bw = 1e9, None
    for wt in np.arange(0, 1.01, 0.05):
        e = mae(wt * s.prior + (1 - wt) * s.cur, s.actual)
        if e < best: best, bw = e, wt
    print(f"{w:<6}{len(s):>7}{mae(s.prior, s.actual):>12.2f}"
          f"{mae(s.cur, s.actual):>14.2f}{best:>12.2f}{bw:>10.2f}")
    BEST[w] = float(bw)

print(f"\nSAME, SPLIT BY WHETHER THE PLAYER CHANGED TEAMS")
print(f"{'week':<6}{'group':<10}{'n':>7}{'prior only':>12}{'current only':>14}{'prior wt':>10}")
print("-" * 60)
for w in range(2, MAXW + 1):
    for lab, s in (("stayed", d[(d.week == w) & ~d.moved & d.cur.notna()]),
                   ("MOVED", d[(d.week == w) & d.moved & d.cur.notna()])):
        if len(s) < 60: continue
        best, bw = 1e9, None
        for wt in np.arange(0, 1.01, 0.05):
            e = mae(wt * s.prior + (1 - wt) * s.cur, s.actual)
            if e < best: best, bw = e, wt
        print(f"{w:<6}{lab:<10}{len(s):>7}{mae(s.prior, s.actual):>12.2f}"
              f"{mae(s.cur, s.actual):>14.2f}{bw:>10.2f}")
import json
json.dump({str(k): v for k, v in BEST.items()}, open("data/nfl/usage_weights.json", "w"), indent=2)
print("\n-> data/nfl/usage_weights.json")
