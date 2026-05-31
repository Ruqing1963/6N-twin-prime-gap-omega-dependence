# 6N Twin-Prime Gap: ω-Dependence

An exhaustive study of the **gap distribution between consecutive twin-prime
centres** on the 6N ± 1 skeleton, stratified by the number of distinct prime
factors of the centre. Part II of the 6N twin-prime project.

**Main finding.** Aggregated over all centres, the twin-gap distribution is the
known Hardy–Littlewood shape (peaks at the highly divisible gaps 6ΔN = 30, 42,
210). But the *shape* shifts systematically with ω₍>3₎(N), the number of distinct
prime factors > 3 of the centre N: the relative preference for the short gap
6ΔN = 42 **rises** monotonically with ω (0.73 → 1.55 across ω = 1…6 in S₁₀),
while that for 6ΔN = 210 **falls** (1.15 → 0.41). The trend reproduces across
S₈, S₉, S₁₀ — a ~64× range of sample size. This contradicts the *unconditional*
Hardy–Littlewood prediction, in which gap preference depends only on the gap.

**Mechanism (first order).** *Congruence lockdown*: when a prime q divides the
centre N, the right centre N + d can host a twin only if
**d ≢ ±6⁻¹ (mod q)**, equivalently **q ∤ (6d ± 1)**. Each prime factor of the
centre forbids two gap residues. This reproduces the rise of 42 and 30 and the
collapse of the absolute survival of 210 (which is flanked by 209 = 11×19, so it
is locked by the *common* small primes 11, 19; whereas 42 is flanked by the
*primes* 41, 43 and so only rarely locked). A residual at high ω remains and is
posed, with the closed-form conditional singular series 𝔖_ω(d), as an open
problem.

> **Scope.** This is experimental / computational number theory. Nothing here
> bears on the infinitude of twin primes. The aggregate gap shape is the known
> jumping-champion effect (Odlyzko–Rubinstein–Wolf 1999; Goldston–Ledoan 2011);
> empirical twin-gap modelling is due to Kelly–Pilling (2001); almost-prime gap
> bounds to Goldston–Graham–Pintz–Yıldırım and Sono. The contribution here is the
> conditional, ω-resolved layer and the congruence-lockdown mechanism. See the
> paper's introduction for the precise demarcation from prior work.

This is **Part II**. Part I (the conditional *density* model, factor
∏(q−1)/(q−3)) is at Zenodo doi:10.5281/zenodo.20470367.

---

## Layout

```
.
├── README.md
├── LICENSE                 (MIT)
├── CITATION.cff
├── data/                   gap statistics for S8, S9, S10
│   ├── gap_hist_byomega_S{8,9,10}.csv      (omega, dN, 6dN, count)
│   ├── gap_summary_byomega_S{8,9,10}.csv   (per-omega: n_gaps, mean, mode, top-3)
│   └── gap_overall_S{8,9,10}.csv           (aggregate gap histogram)
├── code/
│   ├── twin_gap_stats.py        gap statistics, default S8 (reuses the verified
│   │                            complete-factorisation + interval-sieve engine)
│   ├── twin_gap_stats_S9.py     same, hard-wired to S9
│   ├── twin_gap_stats_S10.py    same, hard-wired to S10
│   ├── gap_hl_compare.py        Hardy–Littlewood C2(d) comparison + stratified residual
│   ├── gap_size_control.py      control 1: subtract the per-stratum size effect
│   ├── gap_density_projection.py control 2: test vs the Part-I scalar density E(N)
│   ├── lockdown_test.py         congruence-lockdown model: prediction vs observed
│   ├── lockdown_fig_data.py     emits the lockdown plotting table
│   ├── make_lockdown_fig.py     builds the 3-panel lockdown figure
│   └── fig_gap_by_omega.py      builds the by-omega heatmap
├── figures/                fig_paper2_gap_by_omega.{pdf,png}, fig_paper2_lockdown.{pdf,png}
└── paper/                  Chen_6N_Paper2_TwinGaps.{tex,pdf} + section-3 body + figures
```

---

## Reproducing

Requirements: Python 3.8+, `numpy`. Optional: `sympy` (self-check),
`matplotlib` (figures).

```bash
pip install numpy sympy matplotlib

# 1. Generate the gap statistics. Output goes to ./gap_out/ ; the shell index is
#    in the filenames so runs do not overwrite each other.
python code/twin_gap_stats.py        # S8  (minutes)
python code/twin_gap_stats_S9.py     # S9
python code/twin_gap_stats_S10.py    # S10 (≈ 20 min)
# Optional integrity self-check vs sympy on the first 50k twin centres
# (should report 0 discrepancies): set VERIFY=1 (or use the S8 launcher).

# 2. Analyses (read the CSVs in data/; pass the shell with K=…):
K=10 python code/gap_hl_compare.py        # aggregate fit + stratified residual
K=10 python code/gap_size_control.py      # size-effect control
RHOCSV=<Part-I table3 csv> K=10 python code/gap_density_projection.py

# 3. Figures:
python code/lockdown_fig_data.py && python code/make_lockdown_fig.py
python code/fig_gap_by_omega.py
```

### Definitions (state these when citing the data)

- **Twin centre:** N with 6N−1 and 6N+1 both prime.
- **Gap:** ΔN = N_{i+1} − N_i between consecutive twin centres (centre-step units);
  the physical distance between the pairs is 6ΔN. The literature's "30, 42, 210"
  are values of 6ΔN.
- **Stratum:** ΔN is attributed to ω₍>3₎ of the *left* centre N_i
  (`ATTACH=right` flips this as a robustness check).
- Cross-shell gaps are discarded so that log(6N) is constant within a stratum.
- **ω ≥ 7 is not analysed:** in S₁₀ there are only ~1,200 gaps at ω = 7 and
  negligibly many at ω = 8; that sample is Poisson-noise-dominated. The
  lockdown-model figure is shown for ω = 1…5, where the finite 13-small-prime
  model is valid (it degenerates at ω = 6).

### Data integrity

The twin-centre scan reuses the Part-I engine (complete segmented-sieve
factorisation + deterministic interval-sieve primality), self-verified against
`sympy` with zero discrepancies. The S₁₀ twin count is 23,988,173, matching
Part I to the integer. (An earlier exploratory script that read a precomputed
"centre list" was found to be contaminated and was discarded; all results here
re-derive twin centres from scratch.)

---

## License

MIT — see `LICENSE`. Please also cite the prior work the paper builds on
(jumping champions, Kelly–Pilling, GGPY/Sono, Dolgikh, Puszkarz) as listed in
the manuscript bibliography.
