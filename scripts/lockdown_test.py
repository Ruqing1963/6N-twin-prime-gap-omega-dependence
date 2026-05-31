#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live test of the congruence-lockdown mechanism. Using the real S9 twin centres,
apply the congruence locks of each centre and check whether the predicted gap
distribution reproduces the 42-up / 210-down omega trend.

Mechanism (quantified): centre N, gap d. For the right centre N+d to host a
twin, for each q>3 we need: if q|N then N=0 (mod q) => the right endpoint is
fatal iff d == +-6^{-1} (mod q), so a lock at d == +-6^{-1} (mod q) gives weight 0.
A factor-rich centre (high omega) accumulates many such locks; a gap survives to
the next twin centre only if it threads all of them.

For each (omega, d) we predict P(d|omega) ~ mean over the stratum's centres of
the product of per-q lockdown survivals, and compare to the observed r(d|omega).
"""
import numpy as np, math, time
from collections import defaultdict

def primes_upto(n):
    s=np.ones(n+1,bool); s[:2]=False
    for i in range(2,int(math.isqrt(n))+1):
        if s[i]: s[i*i::i]=False
    return np.nonzero(s)[0].astype(np.int64)

LO=10**8//6+1; HI=10**9//6; SEG=4_000_000   # S9
PB=int(math.isqrt(10**9))+1; BP=primes_upto(PB)
SMALLQ=[5,7,11,13,17,19,23,29,31,37,41,43,47]
GMAX=60
DEADRES={q:set([pow(6,-1,q)%q,(-pow(6,-1,q))%q]) for q in SMALLQ}

twOm=[]; twMask=[]
n=LO; t0=time.time()
while n<=HI:
    nh=min(n+SEG,HI+1); sz=nh-n
    rem=np.arange(n,nh,dtype=np.int64); ob=np.zeros(sz,np.int16); mask=np.zeros(sz,np.int32)
    for p in BP:
        if p*p>nh-1: break
        f=((n+p-1)//p)*p
        if f>=nh: continue
        idx=np.arange(f-n,sz,p)
        if idx.size==0: continue
        sub=rem[idx]; m=(sub%p)==0
        while m.any(): sub[m]//=p; m=(sub%p)==0
        rem[idx]=sub
        if p>3:
            ob[idx]+=1
            if p in SMALLQ: mask[idx]|=(1<<SMALLQ.index(int(p)))
    ob[rem>1]+=1
    vlo=6*n-1; vhi=6*(nh-1)+1; span=vhi-vlo+1
    comp=np.zeros(span,bool); sq=int(math.isqrt(vhi))+1
    for p in BP:
        if p>sq: break
        st=max(p*p,((vlo+p-1)//p)*p)
        if st>vhi: continue
        comp[st-vlo:span:p]=True
    Narr=np.arange(n,nh,dtype=np.int64)
    tw=(~comp[(6*Narr-1)-vlo])&(~comp[(6*Narr+1)-vlo])
    pos=np.nonzero(tw)[0]
    twOm.append(ob[pos]); twMask.append(mask[pos])
    n=nh
twOm=np.concatenate(twOm); twMask=np.concatenate(twMask)
print(f"S9 twin centres {len(twOm):,}  scan {time.time()-t0:.0f}s")

# precompute, for d=1..GMAX, which small primes are fatal
d_dead = np.zeros((GMAX+1, len(SMALLQ)), dtype=bool)
for d in range(1,GMAX+1):
    for i,q in enumerate(SMALLQ):
        if (d%q) in DEADRES[q]: d_dead[d,i]=True

omegas=list(range(1,7))
pred=np.zeros((7,GMAX+1))
for om in omegas:
    sel=twMask[twOm==om]
    if len(sel)<5000: continue
    for d in range(1,GMAX+1):
        dead_qs=np.nonzero(d_dead[d])[0]
        if len(dead_qs)==0:
            pred[om,d]=1.0; continue
        deadbits=0
        for i in dead_qs: deadbits|=(1<<i)
        passed=(sel & deadbits)==0
        pred[om,d]=passed.mean()

print("\nModel A (pure congruence lockdown) predicted relative weight r_pred(d|omega):")
base=np.zeros(GMAX+1)
allmask=twMask
for d in range(1,GMAX+1):
    dead_qs=np.nonzero(d_dead[d])[0]
    if len(dead_qs)==0: base[d]=1.0; continue
    deadbits=0
    for i in dead_qs: deadbits|=(1<<i)
    base[d]=((allmask&deadbits)==0).mean()

for d in [5,7,35]:
    print(f"\n  6dN={6*d} (d={d}):")
    print(f"  {'omega':>6}{'pass-rate':>12}{'r_pred':>10}{'r_obs(ref)':>14}")
    obs_ref={5:[0.91,0.96,1.02,1.12,1.19],7:[0.73,0.90,1.06,1.30,1.42],35:[1.14,1.07,0.98,0.82,0.61]}
    for i,om in enumerate(omegas[:5]):
        pn=pred[om,1:GMAX+1]/pred[om,1:GMAX+1].sum()
        bn=base[1:GMAX+1]/base[1:GMAX+1].sum()
        r=pn[d-1]/bn[d-1]
        ref=obs_ref.get(d,[0]*5)[i] if i<5 else 0
        print(f"  {om:>6}{pred[om,d]:>12.4f}{r:>10.3f}{ref:>14.3f}")
