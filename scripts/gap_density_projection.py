#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Attribution of the gap omega-trend  (gap_density_projection.py)  -- Paper II
================================================================================
Question: is the 42-up / 210-down omega-trend (A) a genuinely new phenomenon, or
(B) a projection of the Part-I E(N) density enrichment (high omega => high twin
density => smaller gaps) onto the gap distribution?

LOGIC
If purely (B), omega acts only through one mediator -- the local twin density
rho. Density sets a "renormalisation scale": the gap-shape in g should depend
only on rho (via a rescaling of g) and not on omega. Two equivalent tests:

  Test I (density rescaling collapse):
    each omega stratum has its own mean twin rate rho_omega (= twins/nodes, from
    Part-I data). Under (B), rescaling gaps by g~ = g*(rho_omega/rho_ref) should
    collapse the strata (esp. the 42/210 relative heights) onto one curve.
    collapse => (B) projection; no collapse => (A) intrinsic effect.

  Test II (density pairing):
    find two strata with different omega but similar rho and compare 42/210
    preferences. same rho => same 42/210 => (B); same rho yet different 42/210
    with omega => (A).

MECHANISM density -> gap shape
H-L correlation C2(g) does not depend on density; density enters only via the
"no twin among the g-1 intervening centres" gap factor (1-rho)^(g-1) ~
exp(-rho*g). So the pure-density model predicts P(g|omega) ~ C2(g)*exp(-rho_omega*g),
i.e. the decay rate a_omega should equal rho_omega, with no other omega-dependence.
==> Test III (decay rate = density?): compare fitted a_omega to observed rho_omega.
    If a_omega ~ rho_omega and the 42/210 residual is explained by
    C2(g)exp(-rho_omega g) => (B).

INPUT
  gap_out/gap_hist_byomega_S{K}.csv         (gap distribution)
  Part-I table3 (conditional_prob)          (rho_omega per omega), via env RHOCSV
OUTPUT density_projection_S{K}.csv + console readout
USAGE  RHOCSV=<path to Part-I table3 csv>  K=8  python gap_density_projection.py
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
RHOCSV=os.environ.get("RHOCSV","")
GMAX=60
hp=f"{IN}/gap_hist_byomega_S{K}.csv"
if not os.path.exists(hp): sys.exit(f"not found: {hp}")

counts=defaultdict(lambda: np.zeros(GMAX+1,dtype=np.int64))
with open(hp) as f:
    for r in csv.DictReader(f):
        try: g=int(r["dN"])
        except ValueError: continue
        if 1<=g<=GMAX: counts[int(r["omega"])][g]+=int(r["count"])
omegas=[om for om in sorted(counts) if counts[om][1:GMAX+1].sum()>=1000]

# read Part-I rho_omega (twin conditional probability)
rho={}
if RHOCSV and os.path.exists(RHOCSV):
    with open(RHOCSV) as f:
        for r in csv.DictReader(f):
            try:
                om=int(r.get("omega_big", r.get("omega")))
                if "cond_prob_%" in r: rho[om]=float(r["cond_prob_%"])/100.0
                else:
                    nn=float(r.get("prime_bearing_nodes", r.get("nodes")))
                    tt=float(r.get("twin_pairs", r.get("twins")))
                    rho[om]=tt/nn if nn else 0
            except (ValueError,TypeError,KeyError): continue
else:
    print("[warn] no RHOCSV (Part-I table3) given; using an internal proxy for rho.")
    for om in omegas:
        g=np.arange(1,GMAX+1); c=counts[om][1:GMAX+1]
        rho[om]=1.0/((g*c).sum()/c.sum())

def primes_upto(n):
    s=np.ones(n+1,bool); s[:2]=False
    for i in range(2,int(math.isqrt(n))+1):
        if s[i]: s[i*i::i]=False
    return np.nonzero(s)[0].astype(np.int64)
QP=[q for q in primes_upto(2_000_000) if q>3]
def C2(g):
    p=1.0
    for q in QP:
        nu=len({(-1)%q,1%q,(6*g-1)%q,(6*g+1)%q})
        if nu==q: return 0.0
        p*=(1.0-nu/q)/(1.0-2.0/q)**2
    return p
Cg=np.array([0.0]+[C2(g) for g in range(1,GMAX+1)])

def fit_a(cnt):
    gv=[g for g in range(1,GMAX+1) if cnt[g]>0 and Cg[g]>0]
    y=np.log([cnt[g]/Cg[g] for g in gv]); sl,_=np.polyfit(gv,y,1); return -sl

print("="*72); print(f"S{K} attribution: is the gap omega-trend a projection of Part-I density?"); print("="*72)

print(f"\n[Test III] is the fitted decay rate a_omega equal to the density rho_omega?")
print(f"{'omega':>5}{'rho_omega':>12}{'a_omega':>12}{'a/rho':>10}")
rows=[]
for om in omegas:
    a_om=fit_a(counts[om]); ro=rho.get(om,float('nan'))
    ratio=a_om/ro if ro else float('nan')
    rows.append((om,ro,a_om,ratio))
    print(f"{om:>5}{ro:>12.5f}{a_om:>12.5f}{ratio:>10.2f}")
print("  Note: a pure-density model predicts a_omega ~ rho_omega (a/rho ~ 1). But the")
print("        rho scale need not equal the a scale under the 6N step, so it is more")
print("        reliable to check whether a/rho is CONSTANT in omega.")

print(f"\n[Test I] after controlling for density, do 42/210 preferences still vary with omega?")
print(f"{'omega':>5}{'rho':>10}{'P(42)/Pbar':>12}{'P(210)/Pbar':>13}{'P42/P210':>11}")
def predn(a):
    p=np.array([Cg[g]*math.exp(-a*g) if g>=1 else 0 for g in range(GMAX+1)])
    p[1:GMAX+1]/=p[1:GMAX+1].sum(); return p
rec=[]
for om in omegas:
    a_om=fit_a(counts[om]); pn=predn(a_om)
    tot=counts[om][1:GMAX+1].sum()
    P42=counts[om][7]/tot; P210=counts[om][35]/tot
    r42=P42/pn[7] if pn[7]>0 else 0
    r210=P210/pn[35] if pn[35]>0 else 0
    ratio=P42/P210 if P210>0 else 0
    rec.append((om,rho.get(om,0),r42,r210,ratio))
    print(f"{om:>5}{rho.get(om,0):>10.5f}{r42:>12.3f}{r210:>13.3f}{ratio:>11.3f}")

r42s=[x[2] for x in rec]; r210s=[x[3] for x in rec]
def monotone_span(v): return max(v)-min(v)
sp42=monotone_span(r42s); sp210=monotone_span(r210s)
def is_monotone(v):
    inc=all(v[i]<=v[i+1]+1e-9 for i in range(len(v)-1))
    dec=all(v[i]>=v[i+1]-1e-9 for i in range(len(v)-1))
    return inc or dec
print(f"\n[criterion] after normalising by each stratum's OWN density (a_omega):")
print(f"   r(42)  residual span = {sp42:.3f}  monotone={is_monotone(r42s)}")
print(f"   r(210) residual span = {sp210:.3f}  monotone={is_monotone(r210s)}")
print("""
Reading:
  - if, after normalising by each stratum's own density (a_omega), r(42) and
    r(210) are flattened (span -> 0, no longer monotone) => omega acts only
    through density => (B) projection of the Part-I enrichment onto the gaps.
    Paper II should be positioned as "a geometric corollary of the Part-I local
    density enrichment on the twin-gap distribution".
  - if after normalising r(42), r(210) still vary strongly and monotonically with
    omega (and a/rho is not constant) => a density-independent intrinsic omega
    effect => (A) a new phenomenon; Paper II stands on its own.
""")
with open(f"{OUT}/density_projection_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["omega","rho","a_omega","a/rho","r42_selfnorm","r210_selfnorm","P42/P210"])
    for (om,ro,a_om,ar),(_,_,r42,r210,ratio) in zip(rows,rec):
        w.writerow([om,f"{ro:.6f}",f"{a_om:.6f}",f"{ar:.3f}",f"{r42:.3f}",f"{r210:.3f}",f"{ratio:.3f}"])
print(f"[ok] written {OUT}/density_projection_S{K}.csv")
