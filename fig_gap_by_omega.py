import csv, math
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 读 S10 直方图
counts=defaultdict(lambda: defaultdict(int)); tot=defaultdict(int)
with open('../data/gap_hist_byomega_S10.csv') as f:
    for r in csv.DictReader(f):
        try: g=int(r['dN'])
        except: continue
        om=int(r['omega']); counts[om][g]+=int(r['count']); tot[om]+=int(r['count'])

GMAX=50
# 只用样本充足的 omega（>=10000）：这里是 1..6
omegas=[om for om in sorted(tot) if tot[om]>=10000]
# 行归一化为百分比
M=np.zeros((len(omegas),GMAX))
for i,om in enumerate(omegas):
    for g in range(1,GMAX+1):
        M[i,g-1]=100*counts[om].get(g,0)/tot[om]

fig,ax=plt.subplots(figsize=(15,5.2))
im=ax.imshow(M,aspect='auto',cmap='magma',origin='lower',
             extent=[0.5,GMAX+0.5,min(omegas)-0.5,max(omegas)+0.5])
cb=fig.colorbar(im,ax=ax,pad=0.01); cb.set_label('relative frequency within stratum (%)',fontsize=10)

# 标注关键间距（只标有真实信号的：30,42,210）
for g,lab,col in [(5,'6ΔN=30','#33d6ff'),(7,'6ΔN=42','#7CFC00'),(35,'6ΔN=210','#ffffff')]:
    ax.axvline(g,color=col,ls='--',lw=1.6,alpha=.7)
    ax.text(g,max(omegas)+0.62,lab,color='black',ha='center',fontsize=9,
            bbox=dict(facecolor=col,alpha=.85,edgecolor='none',boxstyle='round,pad=0.25'))

ax.set_xlabel(r'gap  $\Delta N$   (centre-step; physical distance $=6\,\Delta N$)',fontsize=11)
ax.set_ylabel(r'stratum  $\omega_{>3}(N)$',fontsize=11)
ax.set_yticks(omegas)
# 右侧标样本量
for i,om in enumerate(omegas):
    ax.text(GMAX+2.2,om,f'n={tot[om]:,}',va='center',fontsize=8,color='#444')
ax.set_title(r'Twin-prime gap distribution by $\omega_{>3}$ in $S_{10}$  '
             r'(strata with $\geq10^4$ gaps; $\omega\geq7$ omitted: insufficient sample)',
             fontsize=12,pad=26)
ax.set_xlim(0.5,GMAX+0.5)
plt.tight_layout()
plt.savefig('fig_paper2_gap_by_omega.png',dpi=160,bbox_inches='tight')
plt.savefig('fig_paper2_gap_by_omega.pdf',bbox_inches='tight')
print("honest heatmap saved; omegas used:",omegas)
