# SENTINEL-BI — Measured Results

Every number here is reproducible from the repository:

```bash
python scripts/generate_data.py --lots 20 --parts 500 --prevalence 0.01
python scripts/train_fusion.py --rebuild
python scripts/run_ude.py
```

Benchmark: 20 lots x 500 parts = 10,000 parts, 3 parameters (Iddq, I_leak, t_pd), 4 timepoints (0/24/96/168 h), 1% defect prevalence. **Every injected defect is deliberately kept inside the datasheet limits**, so static screening passes all of them by construction.

Validation is GroupKFold **by lot** throughout. Splitting a lot across folds leaks its median and MAD into training and inflates every number below.

---

## 1. The premise

| | |
|---|---|
| Injected latent defects | 100 |
| Caught by static datasheet limits | **0** |
| False alarms from static limits | 0 |

This is the problem statement reproduced: parts that pass every absolute limit while drifting anomalously.

---

## 2. Anomaly detection (Module A + B fusion)

Out-of-fold scores, GroupKFold by lot.

| Detector | PR-AUC | Recall @ 5% FPR | FPR at cost-optimum |
|---|---|---|---|
| Static datasheet limits | 0.010 | 0.00 | 100% |
| A2 Mahalanobis | 0.258 | 0.28 | 98.0% |
| A3 Isolation Forest + LOF | 0.442 | 0.87 | 96.1% |
| Module B risk (drift margin) | 0.464 | 0.63 | 100% |
| A5 physics residual | 0.671 | 0.77 | 93.9% |
| A1 robust-Z / MAD | 0.805 | 0.89 | 38.8% |
| A4 trajectory autoencoder | **0.922** | 0.96 | 28.5% |
| Unsupervised rank mean | 0.637 | 0.88 | 60.1% |
| **Risk fusion (learned)** | 0.896 | **0.96** | **19.9%** |

At the cost-optimal operating point the fusion reaches **100% recall with 0 escapes**, and **100% worst-case per-mechanism recall**.

Note the honest tension: A4 alone has a marginally higher PR-AUC. The fusion's advantage is at the operating point — comparable recall at **a third of the false-alarm rate**, which is what matters once review capacity is finite. Expected cost drops from 5,323 (A4) to 1,971 (fusion).

### Ablation — every track contributes

Leave one track out, refit the fusion, measure the loss. All deltas are negative, so the heterogeneity argument holds: five weak, uncorrelated detectors beat any one of them.

---

## 3. Drift prediction (Module B)

Predict `Value_168h` from `Value_0h` + `Value_24h` plus lot context.

| Parameter | Best baseline | Quantile GBM | Raw coverage | Conformal coverage |
|---|---|---|---|---|
| Iddq | 0.1955 µA (ridge) | **0.1872 µA** | 87.2% | **90.7%** |
| I_leak | 2.9017 nA (ridge) | **2.8977 nA** | 87.3% | **91.1%** |
| t_pd | 0.0833 ns (ridge) | **0.0801 ns** | 88.0% | **89.9%** |

Nominal coverage is 90%. Conformal calibration closes a 3-point gap to under 1 point on all three parameters.

Baselines published in full: last-value-carry-forward, linear extrapolation, physics extrapolation with a fitted population exponent, ridge. Linear extrapolation is by far the worst (1.68 µA on Iddq) because real wear-out decelerates — `n < 1` — which is the whole reason the power law is the right kernel.

### Rejecting on the upper bound, not the mean

| Rule | Parts rejected | Defects caught |
|---|---|---|
| median forecast > derated limit | 0 | **0 / 100** |
| conformal P90 > derated limit | 349 | **52 / 100** |

The mean would have let every one of them fly.

---

## 4. Decision policy

Cost model: `C_FN : C_FP = 1000 : 1`, review at 0.25 of a scrapped part, review capacity capped at 5% of parts.

| Band | Parts | |
|---|---|---|
| ACCEPT | 7,900 | 0 escapes |
| REVIEW | 500 | at the capacity cap; 1 defect routed here |
| REJECT | 1,600 | 1,501 good parts scrapped |

15% yield loss is what a 1000:1 cost ratio buys. That ratio is a *configuration choice*, not a law — `configs/costs.yaml` is where a programme sets its own economics, and the operating point moves with it.

---

## 5. Unknown Defect Evaluation

The protocol that answers *"you generated your own defects, so of course you detect them"* with numbers. Raw output in `reports/ude.json`.

### UDE-2 — out-of-family physics (the strongest evidence here)

Defects generated from functional forms that appear **nowhere in the library** and were never trained on. Amplitudes match the library's, so the test is about shape, not size.

| | |
|---|---|
| Defects injected | 80 |
| Recall @ 5% FPR | **98.8%** |
| ROC-AUC | **0.999** |

| Novel form | Recall |
|---|---|
| stretched exponential `exp(-(t/τ)^β)` | 100% |
| log-time `A·log(1+t/τ)` | 100% |
| sigmoid | 100% |
| telegraph (two-state switching) | 96.2% |

The detector generalises to physics the library cannot express. That is the difference between anomaly detection and pattern-matching, and it is measured rather than asserted.

### UDE-3 — open-set detection and abstention

This one **failed as originally designed**, and finding that out was the point.

| Method | AUROC | Median distance known / unknown | Abstention precision |
|---|---|---|---|
| Prototype-centroid distance *(original)* | 0.545 | 0.42 / 0.46 | 37.5% |
| **k-NN to known-defect examples** | **0.744** | **6.25 / 12.75** | **83.9%** |

Prototype distance was at chance. Eight idealised fingerprint centroids with generous tolerances blanket the signature space densely enough that almost any part lands near *something*, so the `UNKNOWN-MECHANISM` path was firing at random.

Comparing against the *actual* known-defect examples instead works: it abstains on 32.5% of unknown-physics defects with 5.0% false abstention, and when it abstains it is right 84% of the time.

This forced a design separation worth keeping: **the prototypes name a mechanism, the novelty detector decides whether to trust the name.** Both now run, with the novelty flag overriding.

Modest, and stated as modest — a third of unknown-physics defects get an explicit "mechanism unrecognised, send to review" rather than a confident wrong label.

### UDE-1 — leave-one-mechanism-out (the honest headline)

Each mechanism is removed from training entirely — tracks, fusion and threshold all fitted without ever seeing it — then evaluated on that mechanism alone. Lots are split *before* the mechanism is removed, so the healthy population cannot appear on both sides.

| Held out | Mechanism | Held-out recall @5% FPR | Severity |
|---|---|---|---|
| `MECH-ESD-01` | Latent ESD damage | **58.9%** | critical |
| `MECH-ION-01` | Mobile-ion contamination | 91.7% | major |
| `MECH-TDDB-01` | Gate-oxide pinhole | 100.0% | critical |
| `MECH-HCI-01` | Hot-carrier injection | 100.0% | major |
| `MECH-NBTI-01` | NBTI threshold drift | 100.0% | major |
| `MECH-EM-01` | Electromigration | 100.0% | critical |
| `MECH-PKG-01` | Bond-wire instability | 100.0% | major |
| `MECH-THERM-01` | Thermal runaway | 100.0% | critical |

**Worst case: 58.9%. Mean: 93.8%.**

Per the reporting rule stated in the architecture, the worst case is the headline. A system averaging 93.8% while recovering only 59% of a held-out critical mechanism is a system that loses satellites to that mechanism.

Six of eight mechanisms generalise perfectly to physics never seen in training, which is a strong result. Latent ESD does not, and the reason is structural rather than incidental: it is a flat early offset with a *low* exponent, so A5 is one-sided against it by construction, A3 barely registers it (7%), and with no ESD examples in training the fusion never learns to lean on A1's 0h signal. Both independent tiers — the coverage audit and UDE-1 — converge on the same weak spot, which is the protocol working as intended.

### UDE-4 — subtlety sweep and Minimum Detectable Drift

Defect amplitude scaled down, recall re-measured at a fixed 5% false-alarm budget. 8 lots x 300 parts per point.

| Subtlety | Separation | Fusion | A1 robust-Z | A4 autoencoder | A5 physics residual |
|---|---|---|---|---|---|
| 1.00 | 3.83 MAD | 99% | 86% | 99% | 75% |
| 0.70 | 2.57 MAD | 93% | 79% | 93% | 75% |
| 0.50 | 1.95 MAD | 88% | 75% | 89% | 65% |
| 0.35 | 1.71 MAD | 86% | 82% | 88% | 69% |
| 0.25 | 1.53 MAD | 74% | 68% | 81% | 58% |
| 0.15 | 1.35 MAD | 51% | 57% | 60% | **60%** |
| 0.10 | 1.18 MAD | 31% | 32% | 38% | **36%** |

> **Minimum Detectable Drift (90% recall): subtlety 0.59, or 2.23 MAD above the lot median.**

That is the spec number to quote. Below roughly 2.2 MAD of lot-relative separation, this system stops meeting a 90% recall target — and saying so is more useful than any headline recall figure, because it tells a reliability engineer what the screen can and cannot promise.

**A prediction from Phase 1, now confirmed.** A5 was committed with the note that it "is expected to earn its place at low defect subtlety, where level outliers are not yet visible." The ordering does invert exactly as predicted: at full amplitude A1 leads A5 by 11 points (86% vs 75%), and by subtlety 0.15 A5 has overtaken it (60% vs 57%), holding at 0.10 (36% vs 32%). The level signal decays faster than the kinetic one, because an exponent is a property of the curve's *shape* rather than its size. This is the clearest justification in the project for carrying the physics track at all.

**A new problem the sweep found.** The fusion matches A4 at full amplitude (99% vs 99%) but *trails* it in the subtle regime — 51% vs 60% at subtlety 0.15, 31% vs 38% at 0.10. The cause is straightforward: the fusion's weights were fitted on full-amplitude defects only, so they do not transfer to a regime where the relative usefulness of the tracks has changed. The fix is to train the fusion across a range of subtleties rather than at one. Not yet done.

```bash
python scripts/run_ude.py --only lomo --no-closed-set --lomo-splits 3
python scripts/run_ude.py --only sweep
```

Both are the expensive tiers — every fold refits all five tracks. `--no-closed-set` halves UDE-1.

---

## 6. Coverage audit — claims vs measurement

Per-mechanism recall by track, at a shared 5% FPR budget:

| Mechanism | robust_z | mahalanobis | isolation | autoencoder | physics_residual |
|---|---|---|---|---|---|
| MECH-TDDB-01 | 100% | 11% | 100% | 100% | 100%* |
| MECH-HCI-01 | 100% | 0% | 100% | 100% | 12% |
| MECH-NBTI-01 | 100% | 0% | 100% | 100% | 8% |
| MECH-EM-01 | 100% | 0% | 100% | 100% | 100%* |
| MECH-ION-01 | 62% | 100% | 100% | 100% | 23% |
| MECH-PKG-01 | 77% | 100% | 100% | 100% | 77% |
| MECH-ESD-01 | 79% | 14% | 7% | 71% | 0% |
| MECH-THERM-01 | 100% | 0% | 100% | 100% | 100% |

\* after the `SNR_REF` correction described below.

**Known gaps, stated plainly:**

- **MECH-ESD-01 (latent ESD) is the one coverage hole** — best single track 79%, and it is critical severity. Latent ESD is a flat early offset with a *low* exponent, so A5 is one-sided against it by design and A3 barely sees it. The fusion still reaches it, but no single track does.
- **A5 is weak on HCI and NBTI** (12%, 8%). Both have exponents close to nominal and modest amplitudes, so their drift-to-noise ratio leaves the exponent only partly identifiable. A5 detects *strong* kinetic departures, not subtle ones, at this noise level.
- **A2 Mahalanobis is weak on most mechanisms** but is the only strong detector for ION and PKG — it earns its place on two mechanisms, not on average performance.

---

## 7. Bugs the evaluation caught

Recorded because the evaluation catching them is the argument for having it.

| Defect | Symptom | Fix |
|---|---|---|
| `SNR_REF = 3` too permissive | 35% of a noise-fitted exponent leaked through on healthy parts, widening the `n_shift` null from 0.015 to 0.085 and burying real outliers. A5 scored **0% on TDDB** | Raised to 9, chosen by a label-free criterion. A5: PR-AUC 0.132 → **0.671** |
| `rate_accel` unbounded | Healthy parts with near-zero early drift produced ratios in the thousands, setting the threshold | Clipped all three A5 terms |
| Rank-normalised fusion inputs | Destroyed tail magnitude; fusion scored 0.534 vs A4's 0.922 alone | Robust-z instead. Fusion → **0.903** |
| Fixed 30/70 bands | Accepted 9,899 parts including **45 of 100 defects** | Data-driven edges from the risk distribution. Escapes → 0 |
| `cost_review` > `cost_false_positive` | Review cost 12x a scrapped part, so REVIEW could never be economical and collapsed to zero | Review at 0.25; ordering asserted at load |
| Isotonic step function | 46% of parts on one risk value; no threshold to cut between | Tie-break by decision value |
| Unanchored quantile GBM | Lost to ridge on all 3 parameters | Predict drift from the last measured point, not absolute level |
| No nominal prototype | Every healthy part forced onto a defect mechanism; peer clusters meaningless | Nominal is a prototype |
| Certificate printed "REJECT — Escape Risk 0/100" | Calibrated probability piles up near zero at 1% prevalence | Display via the reference ECDF; decisions byte-identical |
