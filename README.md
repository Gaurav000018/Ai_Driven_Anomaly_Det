# SENTINEL-BI

**SIH26170 — AI-Driven Anomaly Detection in Component Burn-In & Screening**

Latent defects — parts that pass every static parametric limit but drift anomalously during burn-in — are what escape screening and fail in orbit. SENTINEL-BI catches them by scoring the **shape of each part's degradation curve** against its own lot, not its value against a datasheet.

- **Module A** — a five-track dynamic outlier engine (robust-Z/MAD, Mahalanobis, Isolation Forest, trajectory autoencoder, and a physics-kinetics residual) fused under an asymmetric cost model where a false negative is weighted 1000:1.
- **Module B** — a quantile + conformal drift predictor that forecasts `Value_168h` with a guaranteed-coverage upper bound, and rejects on that bound rather than on the mean.
- **Explainability** — every decision ships a one-page certificate naming the likely physical failure mechanism, a counterfactual pass condition, and an immutable model/data audit hash.

## The core idea

Burn-in is not a random walk. It is accelerated wear-out with known kinetics:

```
dV(t) = A * (t / t_ref) ** n        n ~ 0.18 for healthy silicon
```

So the object we score is not a *value*, it is a **Degradation Signature** `[A, n, R2, ...]` fitted per part. A part reading 18 µA inside a 50 µA limit but with `n = 0.42` against its lot's `0.18` is a gate-oxide pinhole heading for runaway. Static screening passes it. We do not.

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_data.py --lots 20 --parts 500 --prevalence 0.01
python scripts/eval_module_a.py
```

## Results so far

Synthetic benchmark, 20 lots x 500 parts, 1% prevalence, every injected defect deliberately kept **inside** the datasheet limits.

| Detector | PR-AUC | Recall @ 5% FPR |
|---|---|---|
| Static datasheet limits | 0.010 | 0.00 |
| A1 robust-Z / MAD vs lot | **0.805** | **0.89** |
| A5 physics residual | 0.132 | 0.32 |

Static screening catches **0 of 100** injected latent defects at 0 false alarms — which is the premise of the problem statement, reproduced.

## Docs

- [Architecture](docs/ARCHITECTURE.md) — full system design and evaluation protocol
- [Plan](docs/PLAN.md) — phased build order, cut list, risk register, demo script

## Status

| Phase | State |
|---|---|
| 0 · Mechanism library, ingestion, defect injector | done |
| 1 · Feature fabric, tracks A1 + A5, eval harness | done |
| 2 · Module B drift predictor | in progress |
| 3 · Tracks A2–A4, risk fusion | pending |
| 4 · Explainability and certificates | pending |
| 4.5 · Unknown defect evaluation | pending |
| 5 · API, dashboard, what-if console | pending |
