#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Size-effect control for the gap trend  (gap_size_control.py)  -- Paper II
================================================================================
Question: the previous step found that "the preference for gap 42 rises
monotonically with omega while 210 falls". But high-omega centres N are larger
on average, with larger local mean gaps, which could indirectly reshape the gap
shares through the geometric envelope exp(-a*g). This script subtracts that
SIZE effect and decides whether 42-up / 210-down is:
  (A) a genuine new signal -- a shape change driven by omega itself,
      independent of the stratum mean gap; or
  (B) a size effect -- merely "high omega => large N => large mean gap" via the
      envelope.

METHOD: gap model  P(g|omega) ~ C2(g) * exp(-a_omega * g)
  - C2(g): H-L two-endpoint correlation factor (omega-independent).
  - exp(-a_omega*g): geometric envelope whose decay rate a_omega is fitted from
    THAT stratum, so it absorbs the stratum's mean gap (size) information.
  Criteria:
  (1) is a_omega monotone in omega? (if so, a size effect exists and varies with omega)
  (2) using the per-stratum prediction C2(g)*exp(-a_omega*g) as denominator,
      recompute the residual r*(g|omega). If 42/210 residual VANISHES after
      removing the stratum's own decay rate -> (B) size effect; if it STILL
      varies monotonically with omega -> (A) new signal.
  (3) cross-check: rescale each stratum's gaps by its mean gap; see whether the
      42/210 preferences align onto one curve (aligned = pure size; not = an
      intrinsic omega effect).

INPUT  gap_out/gap_hist_byomega_S{K}.csv
OUTPUT size_control_S{K}.csv + console readout
USAGE  K=8 python gap_size_control.py
Requires: numpy.
================================================================================
"""
import os, sys, csv, math
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    sys.exit("numpy required")

K=int(os.environ.get("K",8)); IN=os.environ.get("IN","./gap_out"); OUT=os.environ.get("OUT","./gap_out")
GMAX=60
path=f"{IN}/gap_hist_byomega_S{K}.csv"
if not os.path.exists(path): sys.exit(f"not found: {path}")

counts=defaultdict(lambda: np.zeros(GMAX+1,dtype=np.int64))
with open(path) as f:
    for row in csv.DictReader(f):
        try: g=int(row["dN"])
        except ValueError: continue
        if 1<=g<=GMAX: counts[int(row["omega"])][g]+=int(row["count"])
omegas=[om for om in sorted(counts) if counts[om][1:GMAX+1].sum()>=1000]
overall=np.zeros(GMAX+1,dtype=np.int64)
for om in counts: overall+=counts[om]

def primes_upto(n):
    s=np.ones(n+1,bool); s[:2]=False
    for i in range(2,int(math.isqrt(n))+1):
        if s[i]: s[i*i::i]=False
    return np.nonzero(s)[0].astype(np.int64)
QP=[q for q in primes_upto(2_000_000) if q>3]
def C2(g):
    p=1.0
    for q in QP:
        r={(-1)%q,1%q,(6*g-1)%q,(6*g+1)%q}; nu=len(r)
        if nu==q: return 0.0
        p*=(1.0-nu/q)/(1.0-2.0/q)**2
    return p
Cg=np.array([0.0]+[C2(g) for g in range(1,GMAX+1)])

def fit_a(cnt):
    """Fit P(g) ~ C2(g) exp(-a g): linear regression of log(cnt/C2), slope = -a."""
    gv=[g for g in range(1,GMAX+1) if cnt[g]>0 and Cg[g]>0]
    y=np.log([cnt[g]/Cg[g] for g in gv])
    slope,inter=np.polyfit(gv,y,1)
    return -slope, inter

def mean_gap(cnt):
    g=np.arange(1,GMAX+1); c=cnt[1:GMAX+1]
    return (g*c).sum()/c.sum()

print("="*70); print(f"S{K} size-effect control"); print("="*70)
print(f"{'omega':>5}{'n_gaps':>9}{'mean_dN':>9}{'a_omega':>10}{'a_global':>10}")
a_glob,_=fit_a(overall); mg_all=mean_gap(overall)
rows_a=[]
for om in omegas:
    a_om,_=fit_a(counts[om]); mg=mean_gap(counts[om])
    rows_a.append((om,counts[om][1:GMAX+1].sum(),mg,a_om))
    print(f"{om:>5}{counts[om][1:GMAX+1].sum():>9}{mg:>9.2f}{a_om:>10.4f}{a_glob:>10.4f}")

def pred_norm(a):
    p=np.array([Cg[g]*math.exp(-a*g) if g>=1 else 0 for g in range(GMAX+1)])
    s=p[1:GMAX+1].sum(); p[1:GMAX+1]/=s; return p

print("\n[criterion 2] residual r*(6dN|omega) using each stratum's OWN decay rate a_omega:")
print("        (if the size effect were the whole story, this residual should flatten to ~1)")
keys=[5,7,35]  # 30,42,210
hdr="  6dN \\ omega |"+"".join(f"{om:>8}" for om in omegas); print(hdr)
for g in keys:
    line=f"   {6*g:>4}      |"
    for om in omegas:
        a_om,_=fit_a(counts[om]); pn=pred_norm(a_om)
        tot=counts[om][1:GMAX+1].sum(); P=counts[om][g]/tot
        r=P/pn[g] if pn[g]>0 else 0
        line+=f"{r:>8.3f}"
    print(line)

print("\n[reference] residual r(6dN|omega) using the GLOBAL decay rate:")
print(hdr)
pn_g=pred_norm(a_glob)
for g in keys:
    line=f"   {6*g:>4}      |"
    for om in omegas:
        tot=counts[om][1:GMAX+1].sum(); P=counts[om][g]/tot
        r=P/pn_g[g] if pn_g[g]>0 else 0
        line+=f"{r:>8.3f}"
    print(line)

def span(g,use_self):
    vals=[]
    for om in omegas:
        if use_self: a_om,_=fit_a(counts[om]); pn=pred_norm(a_om)
        else: pn=pn_g
        tot=counts[om][1:GMAX+1].sum(); vals.append(counts[om][g]/tot/pn[g] if pn[g]>0 else 0)
    return max(vals)-min(vals), vals

print("\n[criterion 3] residual span over omega (max-min):")
print(f"{'6dN':>6}{'global decay':>14}{'per-stratum decay':>18}{'size-absorbed %':>16}")
for g in keys:
    sp_g,_=span(g,False); sp_s,_=span(g,True)
    absorbed=100*(sp_g-sp_s)/sp_g if sp_g>0 else 0
    print(f"{6*g:>6}{sp_g:>14.3f}{sp_s:>18.3f}{absorbed:>16.0f}")

print("""
Reading:
  - "size-absorbed %" = how much the residual span shrinks after using the
    per-stratum decay rate (which absorbs that stratum's mean gap).
  - near 100% -> the trend is almost entirely a size effect -> Paper II should be
    positioned as "the apparent omega-dependence of gap shape is explained by the
    stratum size effect" (still an honest note).
  - clearly < 100% (span still large) -> a size-independent intrinsic omega
    signal exists -> Paper II stands; one may claim "a gap-shape omega-dependence
    not explained by size".
""")
with open(f"{OUT}/size_control_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["omega","n_gaps","mean_dN","a_omega","a_global"])
    for om,n,mg,a_om in rows_a: w.writerow([om,n,f"{mg:.3f}",f"{a_om:.4f}",f"{a_glob:.4f}"])
print(f"[ok] written {OUT}/size_control_S{K}.csv")
