# SENTINEL-BI

**SIH26170 — AI-Driven Anomaly Detection in Component Burn-In & Screening**

Latent defects — parts that pass every static parametric limit but drift anomalously during burn-in — are what escape screening and fail in orbit. SENTINEL-BI catches them by scoring the **shape of each part's degradation curve against its own lot**, not its value against a datasheet.

On the benchmark, static datasheet limits catch **0 of 100** injected latent defects. SENTINEL-BI catches **100 of 100**, with zero escapes and 100% worst-case per-mechanism recall.

---

## The core idea

Burn-in is not a random walk. It is accelerated wear-out with known kinetics:

```
dV(t) = A * (t / t_ref) ** n          n ~ 0.18 for healthy silicon
```

So the object we score is not a *value*, it is a **Degradation Signature** `[A, n, R², SNR, ...]` fitted per part. A part reading 18 µA inside a 50 µA limit but with `n = 0.42` against its lot's `0.18` is a gate-oxide pinhole heading for runaway. Static screening passes it. We do not.

Four ideas carry the system, and each targets one scored metric:

| | Attacks |
|---|---|
| Physics-signature space + 5 heterogeneous detector tracks | Anomaly detection score |
| Quantile + **conformal** forecasting — reject on the upper bound, not the mean | False negatives (guaranteed coverage) |
| **Defect Mechanism Library** read in both directions, plus a physics defect injector | Labels, and explainability from one source of truth |
| **Unknown Defect Evaluation** — leave-one-mechanism-out, out-of-family, open-set | The *"you invented your own defects"* objection |

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_data.py --lots 20 --parts 500 --prevalence 0.01
python scripts/train_fusion.py --rebuild
python scripts/explain_parts.py --n 3
python scripts/run_ude.py
```

Serve it:

```bash
uvicorn src.api.main:app --reload          # REST API on :8000
streamlit run dashboard/app.py             # QA dashboard on :8501
docker compose up                          # both
```

---

## Headline results

Full detail and reproduction commands in [docs/METRICS.md](docs/METRICS.md).

| Detector | PR-AUC | Recall @ 5% FPR | FPR at cost-optimum |
|---|---|---|---|
| Static datasheet limits | 0.010 | 0.00 | 100% |
| A1 robust-Z / MAD vs lot | 0.805 | 0.89 | 38.8% |
| A4 trajectory autoencoder | **0.922** | 0.96 | 28.5% |
| A5 physics residual | 0.671 | 0.77 | 93.9% |
| **Risk fusion (learned)** | 0.896 | **0.96** | **19.9%** |

Drift prediction beats every baseline on all three parameters, and conformal calibration closes the coverage gap from ~3 points to under 1:

| Parameter | Best baseline MAE | Quantile GBM MAE | Raw → conformal coverage |
|---|---|---|---|
| Iddq | 0.1955 µA | **0.1872 µA** | 87.2% → **90.7%** |
| I_leak | 2.9017 nA | **2.8977 nA** | 87.3% → **91.1%** |
| t_pd | 0.0833 ns | **0.0801 ns** | 88.0% → **89.9%** |

Rejecting on the median catches **0/100** defects. Rejecting on the conformal upper bound catches **52/100**. The mean would have let them fly.

### Unknown Defect Evaluation

| Tier | Result |
|---|---|
| **UDE-1** leave-one-mechanism-out | worst case **58.9%** (latent ESD, critical), mean 93.8%, six of eight at 100% |
| **UDE-2** out-of-family physics | **98.8%** recall, ROC-AUC 0.999 on forms absent from the library |
| **UDE-3** open-set abstention | AUROC 0.744, abstention precision 83.9% (prototype baseline: 0.545) |

UDE-1 worst-case is the honest headline, not the mean. A system averaging 93.8% while recovering 59% of a held-out critical mechanism is a system that loses satellites to that mechanism.

---

## What a decision looks like

```
Part LOT000-W0-D0100 - REJECT - Escape Risk 100/100
Lot LOT000

t_pd measured 8.3 ns at 168 h, sitting 0.1 MAD below its lot median of 8.36 ns,
though inside the datasheet maximum. Its fitted degradation exponent is n = 0.08
against a lot median of 0.18 (R2 = 0.94). The signature is consistent with
**Bond-wire / package instability** (MECH-PKG-01, 1.1 fingerprint-widths,
severity major; MIL-STD-883 TM2011 bond strength).

2 lot-mate(s) share this signature, suggesting a localised process excursion.

Decision drivers : robust_z 51% | autoencoder 35% | physics_residual 10%
Counterfactual   : t_pd total drift <= 6.7% of initial (this part: 38.3%) and
                   t_pd measurement scatter <= 3.4x expected (this part: 16x)
Audit            : model sha256:391d1948de6e | library v1 sha256:6da6748c87e3 |
                   data sha256:44cbb29d000c | 2026-09-30T19:36:34Z
```

Every rejection names a physical mechanism, cites a standard, states what would have had to be different to pass, and carries hashes that make it replayable months later.

---

## Architecture

```
LK  Defect Mechanism Library ── forward ──> defect injector (labels)
                             └─ inverse ──> mechanism attribution (explanations)

L0  ingestion        STDF / CSV -> canonical long format
L1  feature fabric   physics kernel · robust lot stats · shape · rank mobility
L2  Module A         A1 robust-Z · A2 Mahalanobis · A3 isolation · A4 autoencoder · A5 physics residual
    Module B         quantile GBM -> conformal interval -> safety slope
L3  risk fusion      calibrated stack · asymmetric cost · ACCEPT / REVIEW / REJECT
L4  explainability   attribution · SHAP · counterfactual · audit trail
L5  delivery         FastAPI · dashboard · adaptive burn-in

LE  Unknown Defect Evaluation ── validates everything above
```

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full design and evaluation protocol
- [docs/PLAN.md](docs/PLAN.md) — build order, cut list, risk register, demo script
- [docs/METRICS.md](docs/METRICS.md) — every measured number, and the bugs evaluation caught

---

## Honest limitations

Stated here rather than waiting for someone to find them.

- **Attribution is circular on synthetic data.** The same library generates and diagnoses, so its accuracy on generated defects is not evidence. It is only meaningful against held-out mechanisms (UDE-1) and real data.
- **MECH-ESD-01 is the real weak spot**, confirmed independently by the coverage audit (best single track 79%) and by UDE-1 (58.9% held-out recall on a critical-severity mechanism). Latent ESD is a flat early offset with a low exponent, so A5 is one-sided against it by construction and A3 barely registers it. Closing this is the top of the backlog.
- **A5 is weak on HCI and NBTI** (12%, 8%). It detects strong kinetic departures, not subtle ones, at this measurement noise.
- **The 1000:1 cost ratio drives a 15% yield loss.** That ratio is a configuration choice in `configs/costs.yaml`, not a law.
- **All results are on synthetic data.** The physics is drawn from JEDEC/MIL-STD models, but no real burn-in dataset has been used.

## Status

| Phase | State |
|---|---|
| 0 · Mechanism library, ingestion, defect injector | done |
| 1 · Feature fabric, tracks A1 + A5, eval harness | done |
| 2 · Module B drift predictor + conformal | done |
| 3 · Tracks A2–A4, risk fusion, cost decision | done |
| 4 · Mechanism attribution, certificates, audit | done |
| 4.5 · Unknown defect evaluation, coverage audit | done |
| 5 · API, dashboard, adaptive burn-in, Docker | done |
