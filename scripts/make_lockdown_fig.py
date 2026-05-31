#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the 3-panel congruence-lockdown figure from lockdown_plot.csv.
Restricted to omega=1..5 (the finite 13-small-prime model degenerates at omega=6).
"""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('lockdown_plot.csv')))
def series(d, ommax=5):   # restrict to omega<=5: the model degenerates at omega=6
    om=[];obs=[];pred=[];pp=[]
    for r in rows:
        if int(r['d'])==d and 1<=int(r['omega'])<=ommax:
            om.append(int(r['omega']));obs.append(float(r['obs_r']))
            pred.append(float(r['pred_r']));pp.append(float(r['pred_passrate']))
    return map(np.array,(om,obs,pred,pp))

fig,axes=plt.subplots(1,3,figsize=(16,4.8))
ax=axes[0]
for d,c,lab in [(7,'#185FA5','6dN=42'),(5,'#2ca25f','6dN=30')]:
    om,obs,pred,pp=series(d)
    ax.plot(om,obs,'o-',color=c,lw=2,ms=7,label=f'{lab} obs')
    ax.plot(om,pred,'s--',color=c,lw=1.6,ms=6,alpha=.75,label=f'{lab} pred')
ax.axhline(1,color='gray',ls=':',lw=1)
ax.set_xlabel(r'$\omega_{>3}(N)$',fontsize=11);ax.set_ylabel(r'relative preference $r(g\mid\omega)$',fontsize=11)
ax.set_title('Short orbits 42 & 30: lockdown predicts the rise',fontsize=11)
ax.legend(fontsize=8.5,loc='upper left');ax.grid(alpha=.25);ax.set_xticks([1,2,3,4,5])

ax=axes[1]
om,obs,pred,pp=series(35)
ax.plot(om,pp,'D-',color='#c0392b',lw=2,ms=7)
ax.set_xlabel(r'$\omega_{>3}(N)$',fontsize=11)
ax.set_ylabel('lockdown survival of $6\\Delta N{=}210$',fontsize=10.5)
ax.set_title('Long orbit 210: absolute survival collapses',fontsize=11)
ax.grid(alpha=.25);ax.set_ylim(0,1.05);ax.set_xticks([1,2,3,4,5])

ax=axes[2]
om,obs,pred,pp=series(35)
ax.plot(om,obs,'o-',color='#c0392b',lw=2,ms=7,label='210 observed')
ax.plot(om,pred,'s--',color='#e8845b',lw=1.6,ms=6,label='210 lockdown pred')
ax.fill_between(om,pred,obs,where=(obs<pred),color='#c0392b',alpha=.15,label='residual (unexplained)')
ax.axhline(1,color='gray',ls=':',lw=1)
ax.set_xlabel(r'$\omega_{>3}(N)$',fontsize=11);ax.set_ylabel(r'$r(210\mid\omega)$',fontsize=11)
ax.set_title('210 normalised: high-$\\omega$ residual = open problem',fontsize=11)
ax.legend(fontsize=9,loc='lower left');ax.grid(alpha=.25);ax.set_xticks([1,2,3,4,5])

plt.figtext(0.5,-0.03,'Congruence-lockdown prediction vs. observation in $S_9$ (strata $\\omega=1$-$5$; '
            'the $\\omega{=}6$ stratum is omitted as the 13-small-prime lockdown model degenerates there).',
            ha='center',fontsize=9,style='italic')
plt.tight_layout()
plt.savefig('fig_paper2_lockdown.pdf',bbox_inches='tight')
plt.savefig('fig_paper2_lockdown.png',dpi=160,bbox_inches='tight')
print("figure saved: fig_paper2_lockdown.{pdf,png}")
