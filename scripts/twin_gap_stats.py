#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Twin-prime gap statistics  (twin_gap_stats.py)   --  Paper II data generator
================================================================================
Reuses the verified Part-I engine (complete factorisation + deterministic
interval-sieve primality) to compute, cleanly and independently, the
distribution of gaps between twin-prime centres, stratified by the number of
distinct prime factors (>3) of the LEFT centre.

DEFINITIONS (state these verbatim in the paper; required for reproducibility)
  - Twin centre: a positive integer N such that 6N-1 and 6N+1 are both prime.
  - Sort the twin centres within a shell S_K in increasing order N_1<N_2<...
    The gap is   dN = N_{k+1} - N_k   (in centre-step units). The corresponding
    "centre-coordinate distance" is 6*dN (the literature's 30,42,210 are 6*dN).
  - The gap dN is attributed to the stratum omega_>3(N_k) of the LEFT centre.
    (A convention; the right endpoint gives near-identical results: ATTACH=right.)

CORRECTNESS GUARANTEES
  - No external "centre list" is read. Twin status is decided on the fly by a
    deterministic interval sieve.
  - omega is computed by complete factorisation (segmented sieve), no truncation.
  - Gaps spanning a shell boundary are discarded so that log(6N) is effectively
    frozen within a shell.

OUTPUT (CSV)
  gap_hist_byomega_S{K}.csv    count for each (omega, dN), plus per-omega total
  gap_summary_byomega_S{K}.csv per omega: gap count, mean dN, mode dN(6*dN), top-3
  gap_overall_S{K}.csv         unstratified overall gap histogram (first 60 dN)

USAGE
  python twin_gap_stats.py            # default S8 (minutes; verify first)
  MAXK=9  python twin_gap_stats.py    # S9
  MAXK=10 python twin_gap_stats.py    # S10 (hours)
  ATTACH=right python twin_gap_stats.py   # attribute gap to the right endpoint
  VERIFY=1 python twin_gap_stats.py       # self-check against sympy

Requires: numpy (mandatory); sympy (only for VERIFY).
================================================================================
"""
import os, sys, math, time, csv
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required:  pip install numpy")

MAXK   = int(os.environ.get("MAXK", 8))
SEG    = int(os.environ.get("SEG", 3_000_000))
ATTACH = os.environ.get("ATTACH", "left").lower()   # left | right
OUT    = os.environ.get("OUT", "./gap_out")
VERIFY = bool(int(os.environ.get("VERIFY", 0)))
OW_MAX = 14
DN_MAX = 200          # gap-histogram upper bound (dN above this -> overflow bin)
os.makedirs(OUT, exist_ok=True)

LO = 10**(MAXK-1) // 6 + 1     # lower N bound so that 6N >= 10^{K-1}
HI = 10**MAXK // 6             # upper N bound so that 6N < 10^K
PB = int(math.isqrt(10**MAXK)) + 1

def primes_upto(n):
    s = np.ones(n+1, bool); s[:2] = False
    for i in range(2, int(math.isqrt(n))+1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0].astype(np.int64)

print(f"[setup] shell S{MAXK}: N in [{LO:,}, {HI:,}]  (6N in [10^{MAXK-1},10^{MAXK}))")
BP = primes_upto(PB)
print(f"[setup] base primes <= {PB}: {len(BP):,};  gap attribution endpoint = {ATTACH}")

def omega_segment(n_lo, n_hi):
    """omega_>3 for each N in [n_lo,n_hi) (complete factorisation, int16)."""
    size = n_hi - n_lo
    rem = np.arange(n_lo, n_hi, dtype=np.int64)
    ob  = np.zeros(size, dtype=np.int16)
    for p in BP:
        if p*p > n_hi-1: break
        first = ((n_lo + p - 1)//p)*p
        if first >= n_hi: continue
        idx = np.arange(first-n_lo, size, p)
        if idx.size == 0: continue
        sub = rem[idx]; m = (sub % p) == 0
        while m.any(): sub[m] //= p; m = (sub % p) == 0
        rem[idx] = sub
        if p > 3: ob[idx] += 1
    ob[rem > 1] += 1          # remaining large prime (> sqrt, necessarily > 3)
    return ob

def twin_mask(n_lo, n_hi):
    """Boolean twin mask for N in [n_lo,n_hi): 6N-1 and 6N+1 both prime."""
    vlo = 6*n_lo - 1; vhi = 6*(n_hi-1) + 1; span = vhi - vlo + 1
    comp = np.zeros(span, bool); sq = int(math.isqrt(vhi)) + 1
    for p in BP:
        if p > sq: break
        st = max(p*p, ((vlo + p - 1)//p)*p)
        if st > vhi: continue
        comp[st-vlo : span : p] = True
    N = np.arange(n_lo, n_hi, dtype=np.int64)
    pm = ~comp[(6*N-1) - vlo]
    pp = ~comp[(6*N+1) - vlo]
    return pm & pp

def verify():
    try:
        from sympy import isprime, factorint
    except ImportError:
        print("[verify] sympy not installed, skipping"); return
    n_lo = LO; n_hi = LO + 50000
    ob = omega_segment(n_lo, n_hi); tw = twin_mask(n_lo, n_hi)
    bad = 0
    for i, N in enumerate(range(n_lo, n_hi)):
        ob_t = sum(1 for q in factorint(N) if q > 3)
        tw_t = isprime(6*N-1) and isprime(6*N+1)
        if ob[i] != ob_t or bool(tw[i]) != tw_t:
            bad += 1
            if bad <= 5: print(f"   mismatch N={N}: ob {ob[i]}/{ob_t}  tw {bool(tw[i])}/{tw_t}")
    print(f"[verify] done, mismatches = {bad} (should be 0)")

def main():
    if VERIFY: verify()
    hist = defaultdict(lambda: np.zeros(DN_MAX+2, dtype=np.int64))  # idx DN_MAX+1 = overflow
    sum_dn = defaultdict(int); cnt_dn = defaultdict(int)
    overall = np.zeros(DN_MAX+2, dtype=np.int64)
    sum_all = 0; cnt_all = 0
    prev_N = None; prev_ob = None
    t0 = time.time(); n = LO; seg = 0; total_twins = 0
    while n <= HI:
        nh = min(n + SEG, HI + 1)
        ob = omega_segment(n, nh)
        tw = twin_mask(n, nh)
        N = np.arange(n, nh, dtype=np.int64)
        tw_idx = np.nonzero(tw)[0]
        if tw_idx.size:
            tw_N  = N[tw_idx]; tw_ob = ob[tw_idx]
            total_twins += tw_idx.size
            if prev_N is not None:
                allN  = np.concatenate(([prev_N], tw_N))
                allob = np.concatenate(([prev_ob], tw_ob))
            else:
                allN, allob = tw_N, tw_ob
            d = np.diff(allN)
            attach_ob = allob[1:] if ATTACH == "right" else allob[:-1]
            for dn, om in zip(d.tolist(), attach_ob.tolist()):
                om = min(om, OW_MAX)
                sum_dn[om] += dn; cnt_dn[om] += 1
                sum_all += dn; cnt_all += 1
                b = dn if dn <= DN_MAX else DN_MAX+1
                hist[om][b] += 1; overall[b] += 1
            prev_N = int(tw_N[-1]); prev_ob = int(tw_ob[-1])
        seg += 1
        if seg % 20 == 0 or nh > HI:
            print(f"  N={nh-1:,} ({100*(nh-LO)/(HI-LO):5.1f}%)  twins={total_twins:,}  {time.time()-t0:.0f}s")
        n = nh
    print(f"[done] twin centres {total_twins:,}, gap samples {cnt_all:,}, {time.time()-t0:.0f}s")

    def mode_info(arr):
        core = arr[:DN_MAX+1]
        order = np.argsort(core)[::-1]
        top = [(int(order[i]), int(core[order[i]])) for i in range(min(3, len(order))) if core[order[i]]>0]
        m = top[0][0] if top else 0
        return m, (top[0][1] if top else 0), top

    with open(f"{OUT}/gap_hist_byomega_S{MAXK}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["omega","dN","6dN","count"])
        for om in sorted(hist):
            for dn in range(1, DN_MAX+1):
                c = int(hist[om][dn])
                if c: w.writerow([om, dn, 6*dn, c])
            ov = int(hist[om][DN_MAX+1])
            if ov: w.writerow([om, f">{DN_MAX}", "", ov])

    with open(f"{OUT}/gap_summary_byomega_S{MAXK}.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega","n_gaps","mean_dN","mode_dN","mode_6dN","mode_share_%",
                    "top1_6dN","top2_6dN","top3_6dN"])
        for om in sorted(hist):
            n_g = cnt_dn[om]
            if n_g == 0: continue
            mean = sum_dn[om]/n_g
            m, mc, top = mode_info(hist[om])
            share = 100*mc/n_g if n_g else 0
            t6 = [f"{6*d}({100*c/n_g:.2f}%)" for d,c in top]
            while len(t6) < 3: t6.append("")
            w.writerow([om, n_g, f"{mean:.3f}", m, 6*m, f"{share:.3f}", t6[0], t6[1], t6[2]])

    with open(f"{OUT}/gap_overall_S{MAXK}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["dN","6dN","count","share_%"])
        for dn in range(1, 61):
            c = int(overall[dn])
            w.writerow([dn, 6*dn, c, f"{100*c/cnt_all:.4f}" if cnt_all else "0"])

    print(f"[ok] written to {OUT}/  (gap_hist_byomega / gap_summary_byomega / gap_overall)")
    print("     Note: the shift of mode_6dN with omega is the core observation; always")
    print("     compare against the Hardy-Littlewood k-tuple singular series.")

if __name__ == "__main__":
    main()
