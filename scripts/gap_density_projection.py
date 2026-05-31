#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
间距 omega 趋势的归因判定  (gap_density_projection.py)   —— Paper 2 定性的最后一步
================================================================================
问题：42↑ / 210↓ 的 omega 趋势，是 (A) 全新现象，还是 (B) Paper 1 的 E(N) 富集
效应（高 omega ⇒ 孪生密度高 ⇒ 间距偏小）在间距分布上的几何投影？

【判定逻辑】
若纯属 (B)，则 omega 只通过一个中介变量——【局部孪生密度 rho】——影响间距。
密度决定一个"重整化尺度"：间距分布在 g 上的形状应当只依赖 rho（经 g 的重标度），
而与 omega 无关。于是有两个等价检验：

  检验I（密度重标度坍缩）：
    每个 omega 地层有自己的平均孪生率 rho_omega（= twins / nodes，由 Paper 1 数据）。
    若 (B) 成立，把间距按密度重标度 g~ = g * (rho_omega / rho_ref) 后，各 omega 的
    间距分布（尤其 42/210 的相对高度）应坍缩到同一条曲线。
    —— 坍缩 ⇒ (B) 投影；不坍缩 ⇒ (A) 有内禀效应。

  检验II（密度配对）：
    找两个 omega 不同、但 rho 接近的"地层切片"，比较它们的 42/210 偏好。
    若 rho 相同则 42/210 也相同 ⇒ (B)；若 rho 相同而 42/210 仍随 omega 不同 ⇒ (A)。

【关于密度 -> 间距形状的机制】
H–L 关联 C2(g) 不依赖密度；密度只通过"中间 g-1 个中心无孪生"的间隙因子
(1-rho)^(g-1) ≈ exp(-rho*(g-1)) 进入。所以纯密度模型预测：
    P(g|omega) ∝ C2(g) * exp(-rho_omega * g)
即衰减率 a_omega 应当 = rho_omega（密度），别无其他 omega 依赖。
==> 检验III（衰减率=密度？）：比较拟合的 a_omega 与实测 rho_omega。
    若 a_omega ≈ rho_omega 且 42/210 残差被 C2(g)exp(-rho_omega g) 解释 ⇒ (B)。

【输入】
  gap_out/gap_hist_byomega_S{K}.csv         （间距分布）
  PAPER1 table3 (conditional_prob)          （每个 omega 的孪生率 rho_omega）
    —— 通过环境变量 RHOCSV 指定路径；列含 omega_big, prime_bearing_nodes/nodes, twin
【输出】 density_projection_S{K}.csv + 屏幕判读
用法：
  RHOCSV=/path/table3_conditional_prob_S10.csv  K=8  python gap_density_projection.py
================================================================================
"""
import os, sys, csv, math
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy")

K=int(os.environ.get("K",8)); IN=os.environ.get("IN","./gap_out"); OUT=os.environ.get("OUT","./gap_out")
RHOCSV=os.environ.get("RHOCSV","")
GMAX=60
hp=f"{IN}/gap_hist_byomega_S{K}.csv"
if not os.path.exists(hp): sys.exit(f"找不到 {hp}")

counts=defaultdict(lambda: np.zeros(GMAX+1,dtype=np.int64))
with open(hp) as f:
    for r in csv.DictReader(f):
        try: g=int(r["dN"])
        except ValueError: continue
        if 1<=g<=GMAX: counts[int(r["omega"])][g]+=int(r["count"])
omegas=[om for om in sorted(counts) if counts[om][1:GMAX+1].sum()>=1000]

# --- 读 Paper1 的 rho_omega（孪生条件概率）---
rho={}
if RHOCSV and os.path.exists(RHOCSV):
    with open(RHOCSV) as f:
        for r in csv.DictReader(f):
            try:
                om=int(r.get("omega_big", r.get("omega")))
                # 兼容不同列名
                if "cond_prob_%" in r: rho[om]=float(r["cond_prob_%"])/100.0
                else:
                    nn=float(r.get("prime_bearing_nodes", r.get("nodes")))
                    tt=float(r.get("twin_pairs", r.get("twins")))
                    rho[om]=tt/nn if nn else 0
            except (ValueError,TypeError,KeyError): continue
else:
    print("[warn] 未提供 RHOCSV（Paper1 table3）；将仅用间距数据内部估计 rho 的代理。")
    # 代理：用间距均值的倒数近似密度（rho ~ 1/mean_gap），仅作定性
    for om in omegas:
        g=np.arange(1,GMAX+1); c=counts[om][1:GMAX+1]
        rho[om]=1.0/((g*c).sum()/c.sum())

# H–L C2(g)
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

print("="*72); print(f"S{K} 间距 omega 趋势归因：是否 Paper1 密度富集的投影"); print("="*72)

# 检验III：a_omega vs rho_omega
print(f"\n[检验III] 拟合衰减率 a_omega 是否等于密度 rho_omega？")
print(f"{'omega':>5}{'rho_omega':>12}{'a_omega':>12}{'a/rho':>10}")
rows=[]
for om in omegas:
    a_om=fit_a(counts[om]); ro=rho.get(om,float('nan'))
    ratio=a_om/ro if ro else float('nan')
    rows.append((om,ro,a_om,ratio))
    print(f"{om:>5}{ro:>12.5f}{a_om:>12.5f}{ratio:>10.2f}")
print("  说明：纯密度模型预测 a_omega≈rho_omega（a/rho≈1）。但注意 6N 步长下")
print("        rho 的标度未必正好等于 a 的标度，故看 a/rho 是否【随 omega 恒定】更可靠。")

# 检验I：密度重标度坍缩。把每层 42/210 的相对高度，对 rho_omega 作图看是否单调一致
print(f"\n[检验I] 控制密度后，42/210 偏好是否仍随 omega 变（密度配对的近似版）：")
print(f"{'omega':>5}{'rho':>10}{'P(42)/Pbar':>12}{'P(210)/Pbar':>13}{'P42/P210':>11}")
# Pbar：用该层自身 C2*exp(-a_om g) 归一预测作分母（已吸收密度）
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

# 判据：在已用各层自身密度(a_om)归一后，r42、r210 是否仍随 omega 单调
r42s=[x[2] for x in rec]; r210s=[x[3] for x in rec]
def monotone_span(v): return max(v)-min(v)
sp42=monotone_span(r42s); sp210=monotone_span(r210s)
# 单调性检测
def is_monotone(v):
    inc=all(v[i]<=v[i+1]+1e-9 for i in range(len(v)-1))
    dec=all(v[i]>=v[i+1]-1e-9 for i in range(len(v)-1))
    return inc or dec
print(f"\n[判据] 用各 omega 层【自身密度】归一后：")
print(f"   r(42) 残差跨度 = {sp42:.3f}  单调={is_monotone(r42s)}")
print(f"   r(210)残差跨度 = {sp210:.3f}  单调={is_monotone(r210s)}")
print(f"""
判读：
  - 若用各层自身密度(a_omega)归一后，r(42)、r(210) 已被压平（跨度→0、不再单调）
    ⇒ omega 仅通过密度起作用 ⇒ (B) Paper1 富集效应的间距投影。
    Paper 2 应定位为"Paper 1 局部密度富集在孪生间距分布上的几何推论"。
  - 若归一后 r(42)、r(210) 仍大幅、单调地随 omega 变化（且 a/rho 不恒定）
    ⇒ 存在独立于密度的 omega 内禀效应 ⇒ (A) 新现象，Paper 2 可独立成立。
""")
with open(f"{OUT}/density_projection_S{K}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["omega","rho","a_omega","a/rho","r42_selfnorm","r210_selfnorm","P42/P210"])
    for (om,ro,a_om,ar),(_,_,r42,r210,ratio) in zip(rows,rec):
        w.writerow([om,f"{ro:.6f}",f"{a_om:.6f}",f"{ar:.3f}",f"{r42:.3f}",f"{r210:.3f}",f"{ratio:.3f}"])
print(f"[ok] 写入 {OUT}/density_projection_S{K}.csv")
