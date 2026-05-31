#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
孪生间距的 Hardy–Littlewood 预测对比  (gap_hl_compare.py)   —— Paper 2 试金石
================================================================================
目的：判断"按 omega 分层的孪生间距分布"里，有多少是 omega-无关的【已知 H–L
关联效应】，有多少是真正未被解释的【新信号】。方法与 Paper 1 完全一致：
实测 / 预测，看残差是否随 omega 还有结构。

【数学：间距 g 的 H–L 关联因子（omega-无关）】
两个孪生中心相距 g，需要 6N±1 与 6(N+g)±1 这 4 个数同时为素数。相对于"4 个
独立的 6N±1 单点"，关联修正因子（singular series for the 4-tuple，约去公共
常数后只依赖 g）为：

    C(g) = prod_{q>3 prime}  (1 - nu_q(g)/q) / (1 - 2/q)^2

其中 nu_q(g) = #{ 模 q 下被 {-1,+1,6g-1,6g+1} 命中的非零剩余类数 }
            = |{ (-1) mod q, (+1) mod q, (6g-1) mod q, (6g+1) mod q }| (去重)
            <= 4 ；当 q | 6g 时会发生重合，nu_q 减小 -> C(g) 增大。
（注：q=2,3 已被 6N±1 结构吸收，不计入。乘积对 q>3 收敛。）

这个 C(g) 给出"间距恰好跨越 g 个中心且两端皆孪生"的相对权重的【关联部分】。
它【不依赖 omega】。

【判据】
对每个 omega 地层，取实测间距分布 P_obs(g | omega)。H–L 零假设预测：分布形状
（在 g 上的相对偏好）应当与 omega 无关，正比于 C(g)·(几何衰减)。我们做两件事：
  (1) 整体：把 P_obs(g) 与 归一化的 C(g) 对比，确认 C(g) 抓住了 30/42/210 等峰。
  (2) 分层残差：对每个 omega，计算 r(g|omega) = P_obs(g|omega) / P_ref(g)，
      其中 P_ref(g) 是【所有 omega 合并】的实测分布（即 omega-无关基准）。
      若 H–L 零假设成立（间距形状与 omega 无关），r(g|omega) 应在 1 附近平坦，
      无随 omega 的系统结构。任何稳定的、随 omega 单调的偏离，才是 Paper 2 的
      候选新信号。

【为什么用合并实测作基准而非纯 C(g)】
纯 C(g) 还需乘一个随尺度变化的几何/密度衰减包络；用"合并实测分布"作基准可
自动吸收该包络与边界效应，使分层残差检验更干净（与 Paper 1 用壳层基线 beta_K
吸收 log(6N) 是同一手法）。同时我们仍单独报告 C(g) 与合并实测的吻合，证明
基准本身由已知 H–L 关联主导。

【输入】 gap_out/ 下由 twin_gap_stats.py 生成的：
  gap_hist_byomega_S{K}.csv  (列: omega,dN,6dN,count)
【输出】
  hl_Cg_vs_overall_S{K}.csv      g, C(g)(归一), 合并实测(归一), 比值  —— 验证 (1)
  hl_residual_byomega_S{K}.csv   每个 (omega,g) 的 r(g|omega) 及计数 —— 验证 (2)
  并在屏幕打印"残差是否随 omega 有结构"的快速判读。

【用法】
  python gap_hl_compare.py            # 默认读 S8
  K=9 python gap_hl_compare.py
依赖：numpy。
================================================================================
"""
import os, sys, csv, math
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy")

K   = int(os.environ.get("K", 8))
IN  = os.environ.get("IN", "./gap_out")
OUT = os.environ.get("OUT", "./gap_out")
GMAX = 60          # 对比的最大 dN（g 以"中心步长"计）
hist_path = f"{IN}/gap_hist_byomega_S{K}.csv"
if not os.path.exists(hist_path):
    sys.exit(f"找不到 {hist_path}")

# ---------- 读入实测：counts[omega][g] ----------
counts = defaultdict(lambda: np.zeros(GMAX+1, dtype=np.int64))
with open(hist_path) as f:
    for row in csv.DictReader(f):
        try:
            g = int(row["dN"])
        except ValueError:
            continue                      # 跳过 ">200" 溢出行
        if 1 <= g <= GMAX:
            counts[int(row["omega"])][g] += int(row["count"])

omegas = sorted(counts)
overall = np.zeros(GMAX+1, dtype=np.int64)
for om in omegas:
    overall += counts[om]

# ---------- H–L 关联因子 C(g) ----------
def primes_upto(n):
    s=[True]*(n+1); s[0]=s[1]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]:
            for j in range(i*i,n+1,i): s[j]=False
    return [i for i in range(n+1) if s[i]]

QPRIMES = [q for q in primes_upto(2_000_000) if q > 3]

def C_of_g(g):
    """正确的 H–L 两端关联因子 C2(g)：4-tuple {-1,+1,6g-1,6g+1} 的奇异级数，
    相对 4 个独立 6N±1 点。仅对 q>3 取乘积，乘到大素数界以保证尾部收敛。"""
    prod = 1.0
    for q in QPRIMES:
        r = {(-1) % q, 1 % q, (6*g-1) % q, (6*g+1) % q}
        nu = len(r)
        if nu == q:
            return 0.0
        prod *= (1.0 - nu/q) / (1.0 - 2.0/q)**2
    return prod

# 计算 C(g) for g=1..GMAX（g 必须使 6g±1 与 ±1 不强制含公因子；否则 C=0）
Cg = np.zeros(GMAX+1, dtype=np.float64)
for g in range(1, GMAX+1):
    Cg[g] = C_of_g(g)

# ---------- 验证 (1): C(g) vs 合并实测 ----------
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

# 相关性：C(g) 抓住了多少峰结构
gg = np.arange(1,GMAX+1)
corr = np.corrcoef(Cg_norm[1:GMAX+1], obs_norm[1:GMAX+1])[0,1]
# 加几何包络 W(g)=exp(-a*g)：间距越大，中间无孪生的约束越强 -> 频率衰减
# 用对数线性回归在 obs/C2 上拟合 a
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

# ---------- 验证 (2): 分层残差 r(g|omega) = P_obs(g|omega)/P_ref(g) ----------
ref = obs_norm.copy()    # omega-无关基准 = 合并实测（已吸收尺度包络）
with open(f"{OUT}/hl_residual_byomega_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["omega","dN","6dN","count","P_obs","P_ref","ratio"])
    for om in omegas:
        tot = counts[om][1:GMAX+1].sum()
        if tot < 1000:    # 样本太少不做残差（避免噪声）
            continue
        P = counts[om].astype(float)/tot
        for g in range(1,GMAX+1):
            if ref[g] > 0 and counts[om][g] >= 30:
                w.writerow([om,g,6*g,int(counts[om][g]),
                            f"{P[g]:.5f}",f"{ref[g]:.5f}",f"{P[g]/ref[g]:.3f}"])

# ---------- 屏幕判读：残差是否随 omega 有结构 ----------
print("="*64)
print(f"S{K}  Hardy–Littlewood 间距对比")
print("="*64)
print(f"[验证1] C2(g) 裸关联 与合并实测的相关系数 = {corr:.4f}")
print(f"        C2(g)*几何包络exp({a_fit:.3f}*g) 与实测的相关系数 = {corr_full:.4f}")
print(f"        （后者接近 1 表示间距形状 = 已知H-L关联 × omega无关衰减）")
# 看几个标志性间距的 C(g) 与实测排名
top_obs = sorted(range(1,GMAX+1), key=lambda g: -overall[g])[:6]
print(f"        实测最强 6 个间距 6dN: {[6*g for g in top_obs]}")
top_C   = sorted(range(1,GMAX+1), key=lambda g: -Cg[g])[:6]
print(f"        C(g) 最强 6 个间距 6dN: {[6*g for g in top_C]}")
print()
print("[验证2] 分层残差 r(g|omega) 在关键间距上随 omega 的走向：")
keys = [5,7,35]    # 6dN = 30,42,210
hdr = "  6dN \\ omega |" + "".join(f"{om:>8}" for om in omegas if counts[om][1:GMAX+1].sum()>=1000)
print(hdr)
valid_om = [om for om in omegas if counts[om][1:GMAX+1].sum()>=1000]
for g in keys:
    line = f"   {6*g:>4}      |"
    for om in valid_om:
        tot = counts[om][1:GMAX+1].sum()
        P = counts[om][g]/tot if tot else 0
        r = P/ref[g] if ref[g]>0 else 0
        line += f"{r:>8.3f}"
    print(line)
print()
print("判读：若某行 r 随 omega 单调且明显偏离 1（且在大样本 omega 上稳定），")
print("      才是候选新信号；若 r 在 1 附近无规律抖动，则间距形状与 omega 无关，")
print("      即'已知 H–L 关联效应'，Paper 2 应如实定位为'层圈视角下的干净呈现'。")
print(f"\n[ok] 写入 {OUT}/hl_Cg_vs_overall_S{K}.csv 与 hl_residual_byomega_S{K}.csv")
