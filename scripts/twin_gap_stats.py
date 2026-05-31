#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
孪生素数间距统计脚本  (twin_gap_stats.py)   ——  Paper 2 数据生成器
================================================================================
复用 Paper 1 已验证的引擎逻辑（完整分解 + 确定性区间筛），独立、干净地统计
孪生素数中心之间的【间距分布】，并按【前一个孪生中心的 omega】分层。

【定义】（论文中必须原样声明，否则不可复现）
  - 孪生中心：使 6N-1 与 6N+1 同为素数的正整数 N。
  - 把某壳层 S_K 内的孪生中心按大小排序 N_1 < N_2 < ... ，
    相邻间距定义为   dN = N_{k+1} - N_k   （以 N 步长度量）。
    对应的"中心坐标间距"为 6*dN（文献里常说的 30,42,210 即 6*dN）。
  - 该间距 dN 归属于【前一个孪生中心 N_k 的 omega_>3(N_k)】这一地层。
    （这是一个约定；右端点归属会给出几乎相同的结果，可用 --attach=right 检验。）

【关键正确性保证】
  - 不读取任何外部"中心列表"。每个 N 的孪生性由确定性区间筛当场判定。
  - 每个 N 的 omega 由完整分解（分段筛）当场计算，无截断。
  - 跨壳层边界的间距：默认丢弃（只统计两端点同属一个 S_K 的间距），
    以保证"壳层内 log(6N) 近似冻结"这一前提；边界对总体统计影响 < 1/样本数。

【输出】（CSV）
  gap_hist_byomega_S{K}.csv   每个 (omega, dN) 的计数 + 该 omega 的总间距数
  gap_summary_byomega_S{K}.csv 每个 omega 的：间距数、均值 dN、众数 dN(6*dN)、前三众数
  gap_overall_S{K}.csv         不分层的整体间距直方图（前 60 个 dN）

【用法】
  python twin_gap_stats.py            # 默认 S8（几分钟，先验证）
  MAXK=9  python twin_gap_stats.py    # S9
  MAXK=10 python twin_gap_stats.py    # S10（数小时；建议先 S8/S9 验证管线）
  ATTACH=right python twin_gap_stats.py   # 间距归属右端点（稳健性检验）
  VERIFY=1 python twin_gap_stats.py       # 用 sympy 自检前若干孪生中心

依赖：numpy（必需），sympy（仅 VERIFY 时）。
本脚本不"证明"任何猜想；它只产出分解正确、定义清晰、可复现的间距数据。
================================================================================
"""
import os, sys, math, time, csv
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy：  pip install numpy")

MAXK   = int(os.environ.get("MAXK", 8))
SEG    = int(os.environ.get("SEG", 3_000_000))
ATTACH = os.environ.get("ATTACH", "left").lower()   # left | right
OUT    = os.environ.get("OUT", "./gap_out")
VERIFY = bool(int(os.environ.get("VERIFY", 0)))
OW_MAX = 14
DN_MAX = 200          # 间距直方图上界（dN 超过此值归入溢出桶）
os.makedirs(OUT, exist_ok=True)

LO = 10**(MAXK-1) // 6 + 1     # N 下界，使 6N >= 10^{K-1}
HI = 10**MAXK // 6             # N 上界，使 6N < 10^K
PB = int(math.isqrt(10**MAXK)) + 1

def primes_upto(n):
    s = np.ones(n+1, bool); s[:2] = False
    for i in range(2, int(math.isqrt(n))+1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0].astype(np.int64)

print(f"[setup] 壳层 S{MAXK}: N in [{LO:,}, {HI:,}]  (6N in [10^{MAXK-1},10^{MAXK}))")
BP = primes_upto(PB)
print(f"[setup] base primes <= {PB}: {len(BP):,};  间距归属端点 = {ATTACH}")

def omega_segment(n_lo, n_hi):
    """返回 N in [n_lo,n_hi) 每个 N 的 omega_>3（完整分解，int16）。"""
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
    ob[rem > 1] += 1          # 剩余大素数（>sqrt，必 >3）
    return ob

def twin_mask(n_lo, n_hi):
    """返回 N in [n_lo,n_hi) 的孪生布尔掩码（6N-1,6N+1 同素，确定性区间筛）。"""
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
        print("[verify] 无 sympy，跳过"); return
    n_lo = LO; n_hi = LO + 50000
    ob = omega_segment(n_lo, n_hi); tw = twin_mask(n_lo, n_hi)
    bad = 0
    for i, N in enumerate(range(n_lo, n_hi)):
        ob_t = sum(1 for q in factorint(N) if q > 3)
        tw_t = isprime(6*N-1) and isprime(6*N+1)
        if ob[i] != ob_t or bool(tw[i]) != tw_t:
            bad += 1
            if bad <= 5: print(f"   不一致 N={N}: ob {ob[i]}/{ob_t}  tw {bool(tw[i])}/{tw_t}")
    print(f"[verify] 完成，不一致 = {bad}（应为 0）")

def main():
    if VERIFY: verify()

    # 直方图： hist[omega][dN] = 计数；dN>DN_MAX 记入 overflow
    hist = defaultdict(lambda: np.zeros(DN_MAX+2, dtype=np.int64))  # index DN_MAX+1 = overflow
    # 为算均值，另存 sum(dN) 与 count（含 overflow 真值）
    sum_dn = defaultdict(int); cnt_dn = defaultdict(int)
    overall = np.zeros(DN_MAX+2, dtype=np.int64)
    sum_all = 0; cnt_all = 0

    prev_N = None; prev_ob = None     # 上一个孪生中心及其 omega
    t0 = time.time(); n = LO; seg = 0; total_twins = 0

    while n <= HI:
        nh = min(n + SEG, HI + 1)
        ob = omega_segment(n, nh)
        tw = twin_mask(n, nh)
        N = np.arange(n, nh, dtype=np.int64)
        tw_idx = np.nonzero(tw)[0]
        if tw_idx.size:
            tw_N  = N[tw_idx]
            tw_ob = ob[tw_idx]
            total_twins += tw_idx.size
            # 串接上一段尾部的孪生中心，计算相邻间距
            if prev_N is not None:
                allN  = np.concatenate(([prev_N], tw_N))
                allob = np.concatenate(([prev_ob], tw_ob))
            else:
                allN, allob = tw_N, tw_ob
            d = np.diff(allN)                          # 相邻间距 dN
            # 归属端点的 omega
            if ATTACH == "right":
                attach_ob = allob[1:]
            else:
                attach_ob = allob[:-1]
            for dn, om in zip(d.tolist(), attach_ob.tolist()):
                om = min(om, OW_MAX)
                sum_dn[om] += dn; cnt_dn[om] += 1
                sum_all += dn; cnt_all += 1
                b = dn if dn <= DN_MAX else DN_MAX+1
                hist[om][b] += 1
                overall[b] += 1
            prev_N = int(tw_N[-1]); prev_ob = int(tw_ob[-1])
        seg += 1
        if seg % 20 == 0 or nh > HI:
            print(f"  N={nh-1:,} ({100*(nh-LO)/(HI-LO):5.1f}%)  twins={total_twins:,}  {time.time()-t0:.0f}s")
        n = nh

    print(f"[done] 孪生中心 {total_twins:,}，间距样本 {cnt_all:,}，用时 {time.time()-t0:.0f}s")

    def mode_info(arr):
        # 返回 (众数 dN, 该 dN 计数, 前三 dN 列表) ，忽略 overflow 桶
        core = arr[:DN_MAX+1]
        order = np.argsort(core)[::-1]
        top = [(int(order[i]), int(core[order[i]])) for i in range(min(3, len(order))) if core[order[i]]>0]
        m = top[0][0] if top else 0
        return m, (top[0][1] if top else 0), top

    # 输出 1：按 omega 的 (dN) 直方图（长表）
    with open(f"{OUT}/gap_hist_byomega_S{MAXK}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["omega","dN","6dN","count"])
        for om in sorted(hist):
            for dn in range(1, DN_MAX+1):
                c = int(hist[om][dn])
                if c: w.writerow([om, dn, 6*dn, c])
            ov = int(hist[om][DN_MAX+1])
            if ov: w.writerow([om, f">{DN_MAX}", "", ov])

    # 输出 2：按 omega 的摘要
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
            w.writerow([om, n_g, f"{mean:.3f}", m, 6*m, f"{share:.3f}",
                        t6[0], t6[1], t6[2]])

    # 输出 3：整体直方图（前 60 个 dN）
    with open(f"{OUT}/gap_overall_S{MAXK}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["dN","6dN","count","share_%"])
        for dn in range(1, 61):
            c = int(overall[dn])
            w.writerow([dn, 6*dn, c, f"{100*c/cnt_all:.4f}" if cnt_all else "0"])

    print(f"[ok] 写入 {OUT}/  (gap_hist_byomega / gap_summary_byomega / gap_overall)")
    print("     注意：摘要里 mode_6dN 随 omega 的迁移即 Paper 2 的核心观测；")
    print("     务必在论文中与 Hardy–Littlewood k-元组奇异级数预测对比，区分新信号与已知效应。")

if __name__ == "__main__":
    main()
