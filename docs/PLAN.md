# SENTINEL-BI — Execution Plan

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Build order is chosen so that **you have a demoable, scoreable system at the end of every phase** — never a half-built pipeline.

---

## Guiding rule: the vertical slice

Do **not** build L0 fully, then L1 fully, then L2. Build one thin vertical slice end-to-end first (one parameter, one lot, one detector, one regressor, one explanation), then widen it. If the clock runs out, you still have a working product rather than three excellent subsystems that have never spoken to each other.

---

## Phase 0 — Foundations (Day 1, first half)
**Goal: data exists, and it has labels.**

| # | Task | Output |
|---|---|---|
| 0.1 | **Defect Mechanism Library** — YAML schema + 8 seed mechanisms + loader | `configs/mechanisms/`, `src/knowledge/` |
| 0.2 | Canonical schema + CSV loader + validator | `src/ingest/` |
| 0.3 | **Physics defect injector + synthetic lot generator**, driven by the DML | `src/simulate/` |
| 0.4 | Generate 20 lots x 500 parts x 3 params x 4 timepoints, 1 % defect prevalence, all defects inside static limits | `data/synthetic/` |
| 0.5 | EDA notebook — prove static limits miss the injected defects | `notebooks/01_eda.ipynb` |

**Do 0.1 → 0.3 before any model.** The library is the unlock: it gives the injector its physics, it gives the explainer its vocabulary, and it gives the evaluator its LOMO folds. Write it once, and three later phases become configuration instead of code.

The EDA plot from 0.5 — *"static screening catches 0 of 47 injected latent defects"* — is your opening slide.

**Exit criterion:** a labelled dataframe on disk and a plot showing static limits failing.

---

## Phase 1 — Module A, thin slice (Day 1, second half)
**Goal: catch the brief's own example — 45 µA in a 10 µA lot.**

| # | Task | Output |
|---|---|---|
| 1.1 | Trajectory builder | `src/features/trajectory.py` |
| 1.2 | Robust lot stats: median / MAD / robust-Z (track A1) | `src/features/robust_stats.py` |
| 1.3 | **Physics kernel fit** — A, n, R2 (the differentiator) | `src/features/physics_kernel.py` |
| 1.4 | Physics residual detector (track A5) | `src/module_a/physics_residual.py` |
| 1.5 | Evaluation harness: PR-AUC, recall@FPR, F-beta(5), cost curve, GroupKFold by lot | `src/eval/harness.py` |

**Exit criterion:** A1 + A5 together beat static limits on recall at equal FPR, measured, with a number you can say out loud.

---

## Phase 2 — Module B (Day 2, first half)
**Goal: forecast V168h with an honest upper bound.**

| # | Task | Output |
|---|---|---|
| 2.1 | Baselines — LVCF, linear extrapolation, ridge | `src/module_b/baselines.py` |
| 2.2 | Quantile LightGBM, P10/P50/P90, contextual features | `src/module_b/quantile_gbm.py` |
| 2.3 | Conformal calibration (MAPIE) | `src/module_b/conformal.py` |
| 2.4 | Safety-slope comparator + derating rule | `src/module_b/safety_slope.py` |

**Exit criterion:** MAE beats every baseline, **and** empirical interval coverage lands within 2 points of the nominal 1-alpha.

---

## Phase 3 — Widen Module A + Fusion (Day 2, second half)
| # | Task | Output |
|---|---|---|
| 3.1 | Tracks A2 (Mahalanobis), A3 (IsolationForest + LOF) | `src/module_a/` |
| 3.2 | Track A4 (TCN autoencoder) — *drop this first if time is short* | `src/module_a/tcn_ae.py` |
| 3.3 | Calibrated meta-learner over the 5 tracks + B margin | `src/fusion/meta_learner.py` |
| 3.4 | Asymmetric cost decision layer, 3-band output | `src/fusion/cost_decision.py` |

**Exit criterion:** fused Escape Risk score outperforms the best single track on the cost curve.

---

## Phase 4 — Explainability (Day 3, first half) — **do not compress this**
It is a third of the score and the cheapest marginal point in the whole project.

| # | Task | Output |
|---|---|---|
| 4.0 | **Mechanism attribution** — signature vs DML prototypes, with `UNKNOWN` abstention forcing REVIEW | `src/knowledge/prototypes.py` |
| 4.1 | SHAP over the fusion layer | `src/explain/shap_explainer.py` |
| 4.2 | Counterfactual generator — "pass requires V24h <= X" | `src/explain/counterfactual.py` |
| 4.3 | **Physics narrative templater** — names the mechanism | `src/explain/narrative.py` |
| 4.4 | Audit trail: model hash + data hash + timestamp + config | `src/explain/audit.py` |
| 4.5 | Faithfulness (deletion test) + stability (bootstrap SHAP) metrics | `src/eval/explain_metrics.py` |

**Exit criterion:** a one-page certificate per part that a non-ML quality engineer can read and act on without asking a question.

---

## Phase 4.5 — Unknown Defect Evaluation (Day 3, overlapping) — **the credibility firewall**
Run this the moment fusion (Phase 3) is stable. It will find weaknesses, and you want to find them before a judge does.

| # | Task | Output |
|---|---|---|
| 4.5.1 | UDE-1 leave-one-mechanism-out harness → LOMO table | `src/eval/unknown_defect.py` |
| 4.5.2 | UDE-2 out-of-family generators (stretched exp, log-time, sigmoid, telegraph noise) | `src/simulate/out_of_family.py` |
| 4.5.3 | UDE-3 open-set AUROC + abstention rate/precision | `src/eval/unknown_defect.py` |
| 4.5.4 | UDE-4 subtlety sweep → Minimum Detectable Drift curve | `src/eval/subtlety_sweep.py` |
| 4.5.5 | Coverage-audit matrix from DML `detectability` fields | `src/knowledge/coverage_audit.py` |

**Exit criterion:** worst-case LOMO recall is reported as the headline number, and UDE-2 recall stays meaningfully above chance — proving the system detects anomalies rather than recognising its own parametric form.

**If a mechanism fails LOMO badly, that is a finding, not a failure.** Report it, name the gap, and say which track would close it. Judges trust a team that shows its blind spots far more than one claiming 99 % on everything.

## Phase 5 — Delivery (Day 3, second half)
| # | Task | Output |
|---|---|---|
| 5.1 | FastAPI: score, predict, explain, report | `src/api/main.py` |
| 5.2 | Dashboard: lot heatmap, trajectory spaghetti, wafer map, certificate view | `dashboard/app.py` |
| 5.3 | **Digital Twin / What-If + Adaptive Burn-In optimiser** | `dashboard/whatif.py` |
| 5.4 | Docker Compose, README, demo dataset, smoke tests | root |

---

## Phase 6 — Hardening & pitch (Day 4)
- Robustness: missing timepoints, single-part lots, unit mix-ups, a lot with *no* defects (false-alarm sanity), an all-defect lot.
- Ablation table: each track's marginal contribution to recall. Judges love an ablation.
- Freeze models, tag the release, record a 3-minute backup demo video in case live demo dies.
- Rehearse the pitch against the three scored metrics, in that order.

---

## Scope discipline — cut list, in cut order
If time compresses, drop in this exact sequence. Everything above the line survives.

1. TCN autoencoder (A4) — A1/A3/A5 already cover the space
2. STDF parser — CSV alone is fine for the demo
3. Next.js dashboard — Streamlit is enough
4. Wafer spatial/peer term — needs `(x, y)` that may not exist in the data
5. MLflow registry — a pickle plus a hash will do
6. UDE-4 subtlety sweep — nice-to-have; UDE-1 and UDE-2 are not
7. DML entries beyond the core 5 mechanisms — breadth is cheap to add later
8. ——— *never cut below this line* ———
9. Defect Mechanism Library (core) · physics kernel · injector · conformal bounds · **UDE-1 + UDE-2** · narrative certificates

Item 9 is the project. Everything else is packaging.

Note that the library and UDE sit below the line together, and deliberately so: the DML is what makes the certificates credible, and UDE is what makes the DML credible. Cut either and the other loses its evidence.

---

## Risk register
| Risk | Mitigation |
|---|---|
| No real labelled data released | Injector gives ground truth from day one; retune on real data in hours if it arrives |
| Real data has different timepoints than 0/24/96/168 | Physics kernel is fitted on arbitrary `t` — interpolation is free |
| Tiny lots (n < 30) break MAD statistics | Hierarchical shrinkage toward the parameter-level population prior |
| Overfitting to our own injector | **UDE-1 (leave-one-mechanism-out) + UDE-2 (out-of-family forms)**; headline the worst-case mechanism |
| Attribution looks accurate only because one library both generates and diagnoses | Stated openly in the architecture; validated only via UDE-1 and real data; `UNKNOWN` abstention path is mandatory |
| A real defect physics is absent from the library | Open-set threshold routes it to `UNKNOWN-MECHANISM` -> REVIEW band; it is never silently accepted |
| Library grows stale as new mechanisms appear | Versioned YAML, additive schema; a new mechanism is a config commit plus a LOMO rerun |
| Lot leakage inflates CV scores | GroupKFold by `lot_id`, enforced in the eval harness, not by convention |
| Judges see a black box | Certificate is a first-class output of the API, not a notebook afterthought |

---

## Demo script (5 minutes, in scoring order)
1. **The problem, made real (45 s)** — a lot of 500 parts. Static screening: all pass. Reveal injected ground truth: 5 latent defects, every one inside the limits. *"These five fly."*
2. **Module A (90 s)** — trajectory spaghetti plot; the 5 light up. Show that robust-Z alone catches 3, and the **physics exponent** catches all 5. Name the mechanism for one of them.
3. **Module B (60 s)** — forecast V168h for a borderline part. Point prediction says 40 µA, pass. Conformal P90 says 62 µA, **reject**. *"The mean would have let it fly."*
4. **Explainability (60 s)** — open one certificate, read it aloud. It *names the mechanism*. Show the counterfactual and the audit hash.
5. **"But you made up your own defects" (45 s)** — pre-empt it. Put up the LOMO table: *"we removed electromigration from training entirely and still caught it at X % recall"*, then UDE-2: *"these defect shapes use physics that isn't in our library at all."* Show the UNKNOWN abstention path. **This is the slide that wins the room** — answer the objection before it is raised.
6. **The product (45 s)** — What-If console: adaptive burn-in ends 82 % of parts at 96 h, same escape rate, 38 % less oven time. Close on the cost curve.

---

## Immediate next step
Start with **Phase 0.1, the Defect Mechanism Library** — `configs/mechanisms/mechanisms.v1.yaml` plus a loader. It is one YAML file and an afternoon, and the injector, the explainer, and the evaluator all read from it. Building it first is what keeps those three from drifting apart.
