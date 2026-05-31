# 实弹测试 Gemini 的"同余锁死"机制：用 S9 真实孪生中心，逐中心施加同余枷锁，
# 看预测的间距分布能否重现 42↑/210↓ 的 omega 趋势。
#
# 机制（严格量化）：中心 N，间距 d。右端 N+d 要能孪生，需对每个 q>3 满足
#   若 q|N：  N≡0 => 右端致命 <=> d≡±6^{-1}(mod q)。 故 d≡±6^{-1}(mod q) 则该 (N,d) 权重=0
#   若 q∤N：  右端是否致命取决于 (N+d) mod q，对给定真实 N 可直接判定
# 但"锁死"机制的核心主张是 q|N 那部分（高omega => 更多枷锁）。我们分两个模型测：
#   模型A(纯锁死)：只看 q|N 的枷锁。W_A(N,d)=∏_{q|N,q>3} [d≢±6^{-1}(mod q)]  (0或1)
#   模型B(锁死+右端真实)：W_B(N,d)=W_A × [6(N+d)±1 真的没有小素因子<=某界]
# 我们要的是：对每个 omega 层，预测 P(d|omega) ∝ mean_N[W(N,d)]，看 42/210 趋势。
import numpy as np, math, time
def primes_upto(n):
    s=np.ones(n+1,bool); s[:2]=False
    for i in range(2,int(math.isqrt(n))+1):
        if s[i]: s[i*i::i]=False
    return np.nonzero(s)[0].astype(np.int64)
LO=10**8//6+1; HI=10**9//6; SEG=4_000_000
PB=int(math.isqrt(10**9))+1; BP=primes_upto(PB)
SMALLQ=[5,7,11,13,17,19,23,29,31,37,41,43,47]
GMAX=60

# 预算每个 q 的致命剩余类 ±6^{-1} mod q
DEADRES={q:set([pow(6,-1,q)%q, (-pow(6,-1,q))%q]) for q in SMALLQ}

# 扫描孪生中心，记录其对每个 SMALLQ 的整除性(bitmask)与 omega
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
print(f"S9 孪生中心 {len(twOm):,}  扫描 {time.time()-t0:.0f}s")

# 模型A：对每个 (omega, d)，预测权重 = 该omega层中心里, 通过 d 的全部 q|N 枷锁的比例
# 对中心 mask, d 通过枷锁 <=> 对所有 set 位 q, (d mod q) not in DEADRES[q]
# 预计算 d=1..GMAX 对每个 q 是否致命
d_dead = np.zeros((GMAX+1, len(SMALLQ)), dtype=bool)
for d in range(1,GMAX+1):
    for i,q in enumerate(SMALLQ):
        if (d%q) in DEADRES[q]: d_dead[d,i]=True

omegas=list(range(1,7))
# 对每个 omega 层，取该层所有中心的 mask，算每个 d 的通过率
pred=np.zeros((7,GMAX+1))
for om in omegas:
    sel=twMask[twOm==om]
    if len(sel)<5000: continue
    for d in range(1,GMAX+1):
        dead_qs=np.nonzero(d_dead[d])[0]  # 哪些 q 对此 d 致命
        if len(dead_qs)==0:
            pred[om,d]=1.0; continue
        # 中心通过 <=> 不含任何致命 q (即 mask 在这些位上全0)
        deadbits=0
        for i in dead_qs: deadbits|=(1<<i)
        passed=(sel & deadbits)==0
        pred[om,d]=passed.mean()

# 归一并看 42(d=7),210(d=35) 趋势
print("\n模型A(纯同余锁死) 预测的相对权重 r_pred(d|omega) = P_pred(d|om)/P_pred(d|baseline):")
# baseline=全体平均
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
    print(f"  {'omega':>6}{'pred通过率':>12}{'r_pred':>10}{'实测r(对照)':>14}")
    obs_ref={5:[0.91,0.96,1.02,1.12,1.19],7:[0.73,0.90,1.06,1.30,1.42],35:[1.14,1.07,0.98,0.82,0.61]}
    for i,om in enumerate(omegas[:5]):
        # 归一：pred分布在d上归一后比baseline
        pn=pred[om,1:GMAX+1]/pred[om,1:GMAX+1].sum()
        bn=base[1:GMAX+1]/base[1:GMAX+1].sum()
        r=pn[d-1]/bn[d-1]
        ref=obs_ref.get(d,[0]*5)[i] if i<5 else 0
        print(f"  {om:>6}{pred[om,d]:>12.4f}{r:>10.3f}{ref:>14.3f}")
