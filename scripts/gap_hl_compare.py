#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Hardy-Littlewood gap comparison  (gap_hl_compare.py)   -- Paper II litmus test
================================================================================
Decide how much of the "omega-stratified twin-gap distribution" is the
omega-INDEPENDENT known H-L correlation effect, and how much is a genuinely new
signal. Same method as Part I: observed / predicted, look at whether the
residual still has structure in omega.

MATH: the (omega-independent) H-L correlation factor of a gap g
Two twin centres a distance g apart require 6N+-1 and 6(N+g)+-1 all prime.
Relative to "4 independent 6N+-1 points", the 4-tuple correlation factor
(singular series; only depends on g after common constants cancel) is

    C(g) = prod_{q>3 prime}  (1 - nu_q(g)/q) / (1 - 2/q)^2

with nu_q(g) = #{ residues among {-1,+1,6g-1,6g+1} mod q } (deduplicated) <= 4;
when q | 6g some collapse, nu_q drops -> C(g) grows. (q=2,3 absorbed by the
6N+-1 form.) This C(g) is the correlation part of the relative weight of a gap g.
It does NOT depend on omega.

CRITERION
  (1) overall: compare observed P(g) with normalised C(g) (times a geometric
      envelope) to confirm C(g) captures the 30/42/210 peaks.
  (2) stratified residual: for each omega, r(g|omega)=P_obs(g|omega)/P_ref(g),
      where P_ref(g) is the omega-merged observed distribution (the
      omega-independent baseline). Under the H-L null (gap shape independent of
      omega), r should be flat near 1 with no systematic structure in omega.
      Any stable, monotone deviation is a candidate new signal.

INPUT  gap_out/gap_hist_byomega_S{K}.csv  (from twin_gap_stats.py)
OUTPUT hl_Cg_vs_overall_S{K}.csv, hl_residual_byomega_S{K}.csv + console readout
USAGE  K=8 python gap_hl_compare.py   (default reads S8)
Requires: numpy.
================================================================================
"""
import os, sys, csv, math
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    sys.exit("numpy required")

K   = int(os.environ.get("K", 8))
IN  = os.environ.get("IN", "./gap_out")
OUT = os.environ.get("OUT", "./gap_out")
GMAX = 60
hist_path = f"{IN}/gap_hist_byomega_S{K}.csv"
if not os.path.exists(hist_path):
    sys.exit(f"not found: {hist_path}")

counts = defaultdict(lambda: np.zeros(GMAX+1, dtype=np.int64))
with open(hist_path) as f:
    for row in csv.DictReader(f):
        try:
            g = int(row["dN"])
        except ValueError:
            continue                      # skip the ">200" overflow row
        if 1 <= g <= GMAX:
            counts[int(row["omega"])][g] += int(row["count"])

omegas = sorted(counts)
overall = np.zeros(GMAX+1, dtype=np.int64)
for om in omegas:
    overall += counts[om]

def primes_upto(n):
    s=[True]*(n+1); s[0]=s[1]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]:
            for j in range(i*i,n+1,i): s[j]=False
    return [i for i in range(n+1) if s[i]]

QPRIMES = [q for q in primes_upto(2_000_000) if q > 3]

def C_of_g(g):
    """Correct H-L two-endpoint correlation factor C2(g): singular series of the
    4-tuple {-1,+1,6g-1,6g+1} relative to 4 independent 6N+-1 points. Product
    over q>3, taken to a large prime bound to ensure tail convergence."""
    prod = 1.0
    for q in QPRIMES:
        r = {(-1) % q, 1 % q, (6*g-1) % q, (6*g+1) % q}
        nu = len(r)
        if nu == q:
            return 0.0
        prod *= (1.0 - nu/q) / (1.0 - 2.0/q)**2
    return prod

Cg = np.zeros(GMAX+1, dtype=np.float64)
for g in range(1, GMAX+1):
    Cg[g] = C_of_g(g)

# verification (1): C(g) vs merged observed
mask = overall > 0
obs_norm = overall.astype(float).copy()
obs_norm /= obs_norm[1:GMAX+1].sum()
Cg_norm = Cg.copy()
Cg_norm /= Cg_norm[1:GMAX+1].sum()
with open(f"{OUT}/hl_Cg_vs_overall_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["dN","6dN","C(g)_norm","obs_overall_norm","obs/C"])
    for g in range(1,GMAX+1):
        r = (obs_norm[g]/Cg_norm[g]) if Cg_norm[g]>0 else 0
        w.writerow([g,6*g,f"{Cg_norm[g]:.5f}",f"{obs_norm[g]:.5f}",f"{r:.3f}"])

gg = np.arange(1,GMAX+1)
corr = np.corrcoef(Cg_norm[1:GMAX+1], obs_norm[1:GMAX+1])[0,1]
# add a geometric envelope W(g)=exp(-a*g): larger gaps are increasingly
# penalised by the "no-twin-in-between" constraint -> decaying frequency.
# fit a by log-linear regression on obs/C2.
ratio_oc = np.zeros(GMAX+1)
for g in range(1,GMAX+1):
    if Cg[g]>0 and overall[g]>0:
        ratio_oc[g] = overall[g]/Cg[g]
gv=[g for g in range(1,GMAX+1) if ratio_oc[g]>0]
yv=np.log([ratio_oc[g] for g in gv])
a_fit,b_fit=np.polyfit(gv,yv,1)
pred_full=np.array([Cg[g]*math.exp(a_fit*g+b_fit) if g>=1 else 0 for g in range(GMAX+1)])
pn=pred_full.copy(); pn[1:GMAX+1]/=pn[1:GMAX+1].sum()
corr_full=np.corrcoef(pn[1:GMAX+1],obs_norm[1:GMAX+1])[0,1]

# verification (2): stratified residual r(g|omega) = P_obs(g|omega)/P_ref(g)
ref = obs_norm.copy()    # omega-independent baseline = merged observed (absorbs the envelope)
with open(f"{OUT}/hl_residual_byomega_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["omega","dN","6dN","count","P_obs","P_ref","ratio"])
    for om in omegas:
        tot = counts[om][1:GMAX+1].sum()
        if tot < 1000:    # too few samples for a residual (avoid noise)
            continue
        P = counts[om].astype(float)/tot
        for g in range(1,GMAX+1):
            if ref[g] > 0 and counts[om][g] >= 30:
                w.writerow([om,g,6*g,int(counts[om][g]),
                            f"{P[g]:.5f}",f"{ref[g]:.5f}",f"{P[g]/ref[g]:.3f}"])

# console readout
print("="*64)
print(f"S{K}  Hardy-Littlewood gap comparison")
print("="*64)
print(f"[check 1] corr(bare C2(g), merged observed)          = {corr:.4f}")
print(f"          corr(C2(g)*envelope exp({a_fit:.3f}*g), obs) = {corr_full:.4f}")
print(f"          (the latter near 1 means gap shape = known H-L correlation x omega-independent decay)")
top_obs = sorted(range(1,GMAX+1), key=lambda g: -overall[g])[:6]
print(f"          strongest observed gaps 6dN: {[6*g for g in top_obs]}")
top_C   = sorted(range(1,GMAX+1), key=lambda g: -Cg[g])[:6]
print(f"          strongest C(g) gaps 6dN:     {[6*g for g in top_C]}")
print()
print("[check 2] stratified residual r(g|omega) at the key gaps, vs omega:")
valid_om = [om for om in omegas if counts[om][1:GMAX+1].sum()>=1000]
hdr = "  6dN \\ omega |" + "".join(f"{om:>8}" for om in valid_om)
print(hdr)
for g in [5,7,35]:    # 6dN = 30,42,210
    line = f"   {6*g:>4}      |"
    for om in valid_om:
        tot = counts[om][1:GMAX+1].sum()
        P = counts[om][g]/tot if tot else 0
        r = P/ref[g] if ref[g]>0 else 0
        line += f"{r:>8.3f}"
    print(line)
print()
print("Reading: if a row's r is monotone in omega and clearly off 1 (stable on")
print("  large-sample omega), it is a candidate new signal; if r jitters around 1")
print("  with no pattern, the gap shape is omega-independent (the known H-L effect).")
print(f"\n[ok] written {OUT}/hl_Cg_vs_overall_S{K}.csv and hl_residual_byomega_S{K}.csv")
