"""Forward paper record under the RECALIBRATED rules. Flat stakes, no money.

    python3 cfb_paper.py

WHY THIS EXISTS
  On 472 graded games the model went 238-234 (50.4%) and the recalibrated rules
  backtest to 89-92 (49.2%, ROI -6.13%). Both are indistinguishable from a coin
  flip and both are LOSING after vig. Model MAE 14.04 vs market 11.98 -- the
  closing line is a better predictor than these ratings on 59% of games.

  Every filter found so far was fitted on the same data it was measured on. This
  file measures the rules FORWARD ONLY, from the date they were set, so the
  answer cannot be contaminated by the search that produced them.

THE STOPPING RULE -- decided in advance, on purpose
  At n=100 decided picks:
      lower bound of the 95% CI  >  52.38%   -> real edge, keep going
      upper bound                <  52.38%   -> no edge, stop betting these
      CI straddles 52.38%                    -> still unresolved, keep paper only
  Resolving a true 55% edge at 95% confidence needs ~1,400 games. 100 will not
  prove an edge exists; it is enough to rule out a large one and to catch the
  likeliest outcome, which is another ~49%.
"""
import glob, sys
import pandas as pd

# NEW MODEL from 2026-11-12: board switched to 2026 in-season ratings. The old
# model's record (2025 carryover ratings) is frozen; see it with --old.
#   old model final: 42-35 (54.5%), n=77, CI [43.5, 65.2], ROI +4.13%
if "--old" in sys.argv:
    START, END, LABEL = "2026-10-06", "2026-11-11", "OLD model, 2025 ratings -- FROZEN"
else:
    START, END, LABEL = "2026-11-12", None, "NEW model, 2026 in-season ratings"
BE    = 0.5238                # -110 break-even

def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / d
    return c - h, c + h

frames = []
for f in sorted(glob.glob("data/cfb/graded_*.csv")):
    dt = f.split("graded_")[1][:10]
    if dt < START: continue
    if END and dt > END: continue
    try: frames.append(pd.read_csv(f))
    except Exception: pass

if not frames:
    sys.exit(f"no graded slates on or after {START} yet -- nothing to report")

d = pd.concat(frames, ignore_index=True)
d = d[d.get("playable", False) == True]
d = d[d.result.isin(["W", "L"])]
if not len(d):
    sys.exit(f"no decided playable picks since {START}")

W = (d.result == "W").sum(); N = len(d); L = N - W
lo, hi = wilson(W, N)
roi = ((W * 0.909) - L) / N * 100

print("=" * 66)
print(f"PAPER RECORD since {START}   {LABEL}")
print(f"  rules: edge 4-7, blind-spot blocked")
print("=" * 66)
print(f"  {W}-{L}   {W/N*100:.1f}%   n={N}")
print(f"  95% CI [{lo*100:.1f}%, {hi*100:.1f}%]    break-even 52.38%")
print(f"  ROI at -110: {roi:+.2f}%")
print()
if N < 100:
    print(f"  {100-N} more decided picks until the stopping rule applies.")
elif lo > BE:
    print("  -> VERDICT: lower bound clears break-even. Real edge. Keep going.")
elif hi < BE:
    print("  -> VERDICT: upper bound is BELOW break-even. No edge here. Stop.")
else:
    print("  -> VERDICT: unresolved at n=100. Stay on paper; do not bet real money.")
print()
print(f"  {'slate':<14}{'record':>9}{'win%':>8}")
print("  " + "-" * 31)
for dt, s in d.groupby("date"):
    w = (s.result == "W").sum()
    print(f"  {dt:<14}{f'{w}-{len(s)-w}':>9}{w/len(s)*100:>7.1f}%")
print()
v = d.dropna(subset=["model", "market", "actual_margin"])
if len(v):
    me = (v.model - v.actual_margin).abs().mean()
    ke = (v.market - v.actual_margin).abs().mean()
    closer = ((v.model - v.actual_margin).abs() < (v.market - v.actual_margin).abs()).mean()
    print(f"  model MAE {me:.2f}   market MAE {ke:.2f}   model closer on {closer*100:.1f}% of games")
    print("  (the market being more accurate is the finding that matters most --")
    print("   no selection rule fixes a predictor that is worse than the price.)")
