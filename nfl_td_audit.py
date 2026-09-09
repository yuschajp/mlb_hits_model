"""Calibration audit on HELD-OUT seasons. Same instrument that found the park
double-count and the over-aggressive prior in the HR model.

Section 6 is the one that matters most: calibration by implied-team-total
tercile. If high-total games overpredict the way the high-park tercile did
(+28%, z=5.2), the environment term is being applied on top of something that
already contains it.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

TEST_FROM = 2024
d = pd.read_csv("data/nfl/td_features.csv", low_memory=False)
d = d[(d.season >= TEST_FROM) & (d.games_prior >= 4)].dropna(subset=["p_td", "hit"])
p, y, n = d.p_td.values, d.hit.values, len(d)

def wilson(k, m, z=1.96):
    if m == 0: return (0, 0)
    ph = k/m; dd = 1+z*z/m; c = (ph+z*z/(2*m))/dd
    h = z*np.sqrt(ph*(1-ph)/m + z*z/(4*m*m))/dd
    return max(0, c-h), min(1, c+h)

print("="*78)
print(f"HELD-OUT AUDIT   n={n:,}   seasons {int(d.season.min())}-{int(d.season.max())}")
print("  (model fitted on prior rows only; these seasons were never tuned on)")
print("="*78)

lo, hi = wilson(y.sum(), n)
print(f"\n1. OVERALL   predicted {p.mean()*100:5.2f}%   observed {y.mean()*100:5.2f}% "
      f"[95% CI {lo*100:.2f}-{hi*100:.2f}]")
print(f"   ratio pred/obs = {p.mean()/y.mean():.3f}  "
      f"-> model runs {(p.mean()/y.mean()-1)*100:+.1f}% vs reality")

print(f"\n2. CALIBRATION BY BAND")
print(f"   {'band':<14}{'n':>7}{'pred':>9}{'obs':>9}{'95% CI':>18}{'over':>9}")
print("   " + "-"*66)
for a, b in [(0,.05),(.05,.10),(.10,.15),(.15,.20),(.20,.30),(.30,.45),(.45,1.01)]:
    m = (p>=a)&(p<b)
    if m.sum() < 30: continue
    pm, om = p[m].mean(), y[m].mean()
    l, h = wilson(y[m].sum(), m.sum())
    flag = "" if l <= pm <= h else "  MISCALIBRATED"
    print(f"   {a:.2f}-{b:.2f}     {m.sum():>7,}{pm*100:>8.2f}%{om*100:>8.2f}%"
          f"  [{l*100:>5.2f}-{h*100:>5.2f}]{(pm/om-1)*100 if om>0 else np.nan:>8.0f}%{flag}")

brier = np.mean((p-y)**2); bins = np.clip((p*20).astype(int),0,19)
rel = res = 0.0; ybar = y.mean()
for k in range(20):
    m = bins==k
    if m.sum()==0: continue
    w = m.sum()/n
    rel += w*(p[m].mean()-y[m].mean())**2
    res += w*(y[m].mean()-ybar)**2
unc = ybar*(1-ybar)
print(f"\n3. BRIER = {brier:.5f}")
print(f"   reliability (calibration error, lower better) : {rel:.5f}")
print(f"   resolution  (discrimination,   higher better) : {res:.5f}")
print(f"   uncertainty (irreducible)                     : {unc:.5f}")
print(f"   skill score vs always-predicting-base-rate    : {(res-rel)/unc*100:+.2f}%")

order = np.argsort(p); ys = y[order]; ranks = np.arange(1, n+1)
n1, n0 = ys.sum(), n-ys.sum()
auc = (ranks[ys==1].sum() - n1*(n1+1)/2)/(n1*n0)
print(f"\n4. DISCRIMINATION   AUC = {auc:.4f}   (0.50 = none, 0.60+ usable for props)")
dec = pd.qcut(d.p_td, 10, labels=False, duplicates="drop")
top, bot = d.loc[dec==dec.max(),"hit"].mean(), d.loc[dec==dec.min(),"hit"].mean()
print(f"   top decile hits {top*100:.1f}%  vs bottom {bot*100:.1f}%   lift = {top/max(bot,1e-9):.1f}x")

print(f"\n5. BY POSITION")
print(f"   {'pos':<6}{'n':>8}{'pred':>9}{'obs':>9}{'over':>9}")
print("   " + "-"*41)
for pos, s in d.groupby("position_group"):
    if len(s) < 50: continue
    print(f"   {pos:<6}{len(s):>8,}{s.p_td.mean()*100:>8.2f}%{s.hit.mean()*100:>8.2f}%"
          f"{(s.p_td.mean()/s.hit.mean()-1)*100 if s.hit.mean()>0 else np.nan:>8.0f}%")

print(f"\n6. DOUBLE-COUNT CHECK -- calibration by implied team total")
t = pd.qcut(d.implied_pts, 3, labels=["low","mid","high"], duplicates="drop")
print(f"   {'total':<8}{'n':>8}{'pred':>9}{'obs':>9}{'over':>9}")
print("   " + "-"*43)
for lab in ["low","mid","high"]:
    m = t==lab
    if m.sum()<30: continue
    pm, om = d.loc[m,"p_td"].mean(), d.loc[m,"hit"].mean()
    print(f"   {lab:<8}{m.sum():>8,}{pm*100:>8.2f}%{om*100:>8.2f}%"
          f"{(pm/om-1)*100 if om>0 else np.nan:>8.0f}%")
print("   If 'high' overpredicts far more than 'low', the environment term is")
print("   stacked on something that already contains it -- the park-factor error.")

print(f"\n7. BY OPPORTUNITY VOLUME (the expected-AB analogue)")
q = pd.qcut(d.e_carries + d.e_targets, 4, labels=["q1","q2","q3","q4"], duplicates="drop")
print(f"   {'usage':<8}{'n':>8}{'pred':>9}{'obs':>9}{'over':>9}")
print("   " + "-"*43)
for lab in ["q1","q2","q3","q4"]:
    m = q==lab
    if m.sum()<30: continue
    pm, om = d.loc[m,"p_td"].mean(), d.loc[m,"hit"].mean()
    print(f"   {lab:<8}{m.sum():>8,}{pm*100:>8.2f}%{om*100:>8.2f}%"
          f"{(pm/om-1)*100 if om>0 else np.nan:>8.0f}%")
print("\n" + "="*78)
