#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate plotting data: congruence-lockdown prediction vs observed, per omega
and per d (complete, no cherry-picking). Reads the S10 histogram and rescans S9
for the centre factor masks; writes lockdown_plot.csv.
"""
import numpy as np, math, time, csv
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

twOm=[]; twMask=[]; twN=[]
n=LO
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
    twOm.append(ob[pos]); twMask.append(mask[pos]); twN.append(Narr[pos])
    n=nh
twOm=np.concatenate(twOm); twMask=np.concatenate(twMask); twN=np.concatenate(twN)

# gaps and the previous centre's attributes
g=np.diff(twN); omL=twOm[:-1]; maskL=twMask[:-1]
keep=(g>=1)&(g<=GMAX)
g=g[keep]; omL=omL[keep]; maskL=maskL[keep]

from collections import defaultdict
obs=defaultdict(lambda: np.zeros(GMAX+1))
sumw=defaultdict(lambda: np.zeros(GMAX+1)); cnt=defaultdict(lambda: np.zeros(GMAX+1))

# observed obs[omega][d]
for d in range(1,GMAX+1):
    pass
for i in range(len(g)):
    gg=int(g[i]); om=int(omL[i])
    obs[om][gg]+=1

# lockdown pass-rate per (omega,d): fraction of stratum centres locking NONE of d's fatal primes
d_dead=np.zeros((GMAX+1,len(SMALLQ)),bool)
for d in range(1,GMAX+1):
    for i,q in enumerate(SMALLQ):
        if (d%q) in DEADRES[q]: d_dead[d,i]=True
def passrate(maskarr,d):
    dq=np.nonzero(d_dead[d])[0]
    if len(dq)==0: return 1.0
    db=0
    for i in dq: db|=(1<<i)
    return ((maskarr&db)==0).mean()

predpass=np.zeros((7,GMAX+1)); pred=np.zeros((7,GMAX+1))
for om in range(1,7):
    sel=twMask[twOm==om]
    if len(sel)<5000: continue
    for d in range(1,GMAX+1): predpass[om,d]=passrate(sel,d)
    pred[om,1:GMAX+1]=predpass[om,1:GMAX+1]/predpass[om,1:GMAX+1].sum()

overall=np.zeros(GMAX+1)
for om in range(1,7): overall+=obs[om]
obasen=overall[1:GMAX+1]/overall[1:GMAX+1].sum()
base=np.zeros(GMAX+1)
for d in range(1,GMAX+1): base[d]=passrate(twMask,d)
basen=base[1:GMAX+1]/base[1:GMAX+1].sum()

for om in range(1,7):
    if obs[om][1:].sum()>0:
        obs[om][1:GMAX+1]/=obs[om][1:GMAX+1].sum()

with open('lockdown_plot.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['d','6d','omega','obs_r','pred_r','pred_passrate'])
    for d in range(1,GMAX+1):
        for om in range(1,7):
            if overall[d]==0: continue
            obs_r=(obs[om][d]/obasen[d-1]) if obasen[d-1]>0 else 0
            pred_r=(pred[om,d]/basen[d-1]) if basen[d-1]>0 else 0
            w.writerow([d,6*d,om,f'{obs_r:.4f}',f'{pred_r:.4f}',f'{predpass[om,d]:.4f}'])
print("plot data written: lockdown_plot.csv")
print(f"S9 twins {len(twN):,}")
