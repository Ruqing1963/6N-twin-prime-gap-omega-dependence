#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
间距趋势的尺寸效应扣除  (gap_size_control.py)   —— Paper 2 的最终判据
================================================================================
问题：上一步发现"间距 42 的相对偏好随 omega 单调上升、210 单调下降"。
但高 omega 的中心 N 平均更大，局部平均间距更大，可能通过几何包络 exp(-a*g)
间接改变各间距占比。本脚本扣除这一【尺寸效应】，判断 42↑/210↓ 是：
  (A) 真新信号——独立于地层平均间距的、由 omega 本身驱动的形状变化；还是
  (B) 尺寸效应——仅是"高 omega ⇒ 大 N ⇒ 大平均间距"经包络的二次产物。

【方法】间距分布模型： P(g | omega) ∝ C2(g) · exp(-a_omega * g)
  - C2(g)：H–L 两端关联因子（omega 无关，已在 gap_hl_compare 修正）。
  - exp(-a_omega*g)：几何包络，其衰减率 a_omega 由该地层【自身】拟合，
    因此自动吸收该地层的平均间距（尺寸）信息。
  判据：
  (1) 看 a_omega 是否随 omega 单调变化（若是，尺寸效应确实存在且随 omega 变）。
  (2) 用【地层专属】预测 C2(g)*exp(-a_omega*g) 作分母，重算残差 r*(g|omega)。
      若 42/210 的残差在扣除地层自身衰减率后【消失】→ (B) 尺寸效应；
      若残差【依然随 omega 单调】→ (A) 真新信号。
  (3) 交叉检验：把每个地层的间距分布按其平均间距重标度 g~ = g * (mean_all/mean_omega)，
      看 42/210 偏好是否对齐到同一曲线（对齐=纯尺寸；不对齐=有 omega 内禀效应）。

【输入】 gap_out/gap_hist_byomega_S{K}.csv
【输出】 size_control_S{K}.csv  +  屏幕判读
用法： K=8 python gap_size_control.py
依赖：numpy, scipy(可选，无则用 numpy 线性拟合)
================================================================================
"""
import os, sys, csv, math
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy")

K=int(os.environ.get("K",8)); IN=os.environ.get("IN","./gap_out"); OUT=os.environ.get("OUT","./gap_out")
GMAX=60
path=f"{IN}/gap_hist_byomega_S{K}.csv"
if not os.path.exists(path): sys.exit(f"找不到 {path}")

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
    """拟合 P(g) ∝ C2(g) exp(-a g)：对 log(cnt/C2) 线性回归取斜率 a(>0 表示衰减)。"""
    gv=[g for g in range(1,GMAX+1) if cnt[g]>0 and Cg[g]>0]
    y=np.log([cnt[g]/Cg[g] for g in gv])
    slope,inter=np.polyfit(gv,y,1)
    return -slope, inter   # a = -slope

def mean_gap(cnt):
    g=np.arange(1,GMAX+1); c=cnt[1:GMAX+1]
    return (g*c).sum()/c.sum()

# (1) 各地层衰减率 a_omega 与平均间距
print("="*70); print(f"S{K} 尺寸效应扣除"); print("="*70)
print(f"{'omega':>5}{'n_gaps':>9}{'mean_dN':>9}{'a_omega':>10}{'a_global':>10}")
a_glob,_=fit_a(overall); mg_all=mean_gap(overall)
rows_a=[]
for om in omegas:
    a_om,_=fit_a(counts[om]); mg=mean_gap(counts[om])
    rows_a.append((om,counts[om][1:GMAX+1].sum(),mg,a_om))
    print(f"{om:>5}{counts[om][1:GMAX+1].sum():>9}{mg:>9.2f}{a_om:>10.4f}{a_glob:>10.4f}")

# (2) 地层专属预测残差： r*(g|omega) = P_obs(g|omega) / [C2(g)exp(-a_omega g) 归一]
def pred_norm(a):
    p=np.array([Cg[g]*math.exp(-a*g) if g>=1 else 0 for g in range(GMAX+1)])
    s=p[1:GMAX+1].sum(); p[1:GMAX+1]/=s; return p

print("\n[判据2] 用地层【自身】衰减率 a_omega 作预测后的残差 r*(6dN|omega):")
print("        (若尺寸效应是全部原因，此残差应被压平到≈1)")
keys=[5,7,35]  # 30,42,210
hdr="  6dN \\ omega |"+"".join(f"{om:>8}" for om in omegas); print(hdr)
resid_star={}
for g in keys:
    line=f"   {6*g:>4}      |"
    for om in omegas:
        a_om,_=fit_a(counts[om]); pn=pred_norm(a_om)
        tot=counts[om][1:GMAX+1].sum(); P=counts[om][g]/tot
        r=P/pn[g] if pn[g]>0 else 0; resid_star[(g,om)]=r
        line+=f"{r:>8.3f}"
    print(line)

# 对照：用【全局】衰减率（即上一步验证2的等价物）
print("\n[对照] 用【全局】衰减率 a_global 作预测的残差 r(6dN|omega):")
print(hdr)
pn_g=pred_norm(a_glob)
for g in keys:
    line=f"   {6*g:>4}      |"
    for om in omegas:
        tot=counts[om][1:GMAX+1].sum(); P=counts[om][g]/tot
        r=P/pn_g[g] if pn_g[g]>0 else 0
        line+=f"{r:>8.3f}"
    print(line)

# (3) 量化：地层专属残差的"随 omega 单调跨度"，对比全局残差跨度
def span(g,use_self):
    vals=[]
    for om in omegas:
        if use_self: a_om,_=fit_a(counts[om]); pn=pred_norm(a_om)
        else: pn=pn_g
        tot=counts[om][1:GMAX+1].sum(); vals.append(counts[om][g]/tot/pn[g] if pn[g]>0 else 0)
    return max(vals)-min(vals), vals

print("\n[判据3] 残差随 omega 的跨度 (max-min)：")
print(f"{'6dN':>6}{'全局衰减':>12}{'地层专属衰减':>14}{'尺寸吸收比%':>12}")
for g in keys:
    sp_g,_=span(g,False); sp_s,_=span(g,True)
    absorbed=100*(sp_g-sp_s)/sp_g if sp_g>0 else 0
    print(f"{6*g:>6}{sp_g:>12.3f}{sp_s:>14.3f}{absorbed:>12.0f}")

print("""
判读：
  - "尺寸吸收比" = 用地层专属衰减率(吸收了该层平均间距)后，残差跨度缩小的比例。
  - 接近 100% → 趋势几乎全由尺寸效应解释 → Paper 2 应定位为
      "间距形状随 omega 的表观变化可由地层尺寸效应解释"(仍是诚实的note)。
  - 明显 < 100%(残差跨度仍大) → 存在独立于尺寸的 omega 内禀信号 → Paper 2 成立，
      可主张"间距形状存在不可由尺寸解释的 omega 依赖"。
""")

# 写出明细
with open(f"{OUT}/size_control_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["omega","n_gaps","mean_dN","a_omega","a_global"])
    for om,n,mg,a_om in rows_a: w.writerow([om,n,f"{mg:.3f}",f"{a_om:.4f}",f"{a_glob:.4f}"])
print(f"[ok] 写入 {OUT}/size_control_S{K}.csv")
