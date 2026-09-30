"""Tests for the properties that would fail silently if broken.

Deliberately not a coverage exercise. Each test here guards an invariant that,
if violated, produces plausible-looking numbers that are wrong - which is the
dangerous failure mode in a screening system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.physics_kernel import SNR_REF, fit_power_law
from src.features.robust_stats import mad, robust_z
from src.fusion.cost_decision import CostModel
from src.ingest.validate import validate
from src.knowledge.library import MechanismLibrary
from src.module_b.conformal import ConformalCalibrator, coverage
from src.module_b.dataset import make_dataset
from src.simulate.lot_generator import LotGenerator, static_screen


@pytest.fixture(scope="module")
def library():
    return MechanismLibrary.load()


@pytest.fixture(scope="module")
def small_lots(library):
    return LotGenerator(library, seed=11).generate(n_lots=4, parts_per_lot=120, prevalence=0.05)


# --------------------------------------------------------------------- library


def test_library_loads_and_hashes(library):
    assert library.version >= 1
    assert len(library.source_hash) == 64
    assert len(library.defects()) >= 5


def test_lomo_folds_never_drop_nominal(library):
    for held, reduced in library.lomo_folds():
        assert held not in reduced.defect_ids()
        assert reduced.nominal.id == library.nominal.id


def test_cannot_hold_out_nominal(library):
    with pytest.raises(ValueError):
        library.without(library.nominal.id)


# ------------------------------------------------------------------- physics


def test_power_law_recovers_known_exponent():
    hours = np.array([0.0, 24.0, 96.0, 168.0])
    A, n, v0 = 5.0, 0.42, 10.0
    values = v0 + A * (hours / 168.0) ** n
    # High SNR so shrinkage leaves the fitted exponent essentially untouched.
    fit = fit_power_law(hours, values, noise_sigma=A / 100.0)
    assert fit.n == pytest.approx(n, abs=0.02)
    assert fit.r2 > 0.99


def test_low_snr_exponent_is_shrunk_to_prior():
    """The fix that took A5 from useless to useful; it must not regress."""
    hours = np.array([0.0, 24.0, 96.0, 168.0])
    v0 = 10.0
    values = v0 + 0.02 * (hours / 168.0) ** 0.9   # drift far below the noise
    fit = fit_power_law(hours, values, noise_sigma=0.5)
    assert fit.identifiable < 0.05
    assert fit.n == pytest.approx(0.18, abs=0.02)


# ------------------------------------------------------------- robust stats


def test_mad_is_not_moved_by_a_single_outlier():
    """The reason Module A uses MAD and not sigma at all."""
    clean = np.concatenate([np.full(99, 10.0), [10.5]])
    spiked = np.concatenate([np.full(99, 10.0), [45.0]])
    assert mad(spiked) == pytest.approx(mad(clean), rel=0.2)
    assert np.std(spiked) > 3 * np.std(clean)


def test_robust_z_flags_the_brief_example():
    """A lot at 10 uA, one part at 45 uA, datasheet max 50 uA."""
    values = pd.Series(np.concatenate([np.random.default_rng(0).normal(10, 1, 199), [45.0]]))
    lots = pd.Series(["LOT1"] * 200)
    z = robust_z(values, lots)
    assert z.iloc[-1] > 10
    assert z.iloc[:-1].abs().max() < 6


def test_small_lots_are_shrunk_toward_the_population():
    rng = np.random.default_rng(3)
    big = pd.Series(rng.normal(10, 1, 400))
    tiny = pd.Series(rng.normal(10, 1, 6))
    values = pd.concat([big, tiny], ignore_index=True)
    lots = pd.Series(["BIG"] * 400 + ["TINY"] * 6)
    z = robust_z(values, lots)
    # Without shrinkage a 6-part lot produces wild z-scores from noise alone.
    assert z.iloc[400:].abs().max() < 6


# ------------------------------------------------------------------ simulate


def test_injected_defects_pass_static_screening(small_lots):
    """The premise of the whole project. If this fails, nothing else matters."""
    screen = static_screen(small_lots)
    caught = int((screen["is_defect"] & screen["static_fail"]).sum())
    assert caught == 0, f"{caught} injected latent defects breached the datasheet limit"


def test_generated_data_passes_its_own_contract(small_lots):
    errors = [i for i in validate(small_lots) if i.level == "error"]
    assert not errors, [str(e) for e in errors]


def test_defects_are_actually_injected(small_lots):
    labels = small_lots.drop_duplicates("part_id")
    assert labels["is_defect"].sum() > 0
    assert labels["mechanism_id"].nunique() > 2


# ------------------------------------------------------------------ module B


def test_dataset_refuses_to_leak_the_target(library, small_lots):
    with pytest.raises(ValueError, match="leaks the target"):
        make_dataset(small_lots, library, parameter="Iddq", input_hours=(0.0, 168.0), horizon=168.0)


def test_conformal_achieves_nominal_coverage():
    rng = np.random.default_rng(5)
    y_cal = rng.normal(0, 1, 2000)
    # Deliberately over-confident interval; conformal must widen it.
    lo, hi = np.full(2000, -0.5), np.full(2000, 0.5)
    cal = ConformalCalibrator(alpha=0.1).calibrate(lo, hi, y_cal)

    y_test = rng.normal(0, 1, 2000)
    interval = cal.apply(np.full(2000, -0.5), np.zeros(2000), np.full(2000, 0.5))
    assert coverage(y_test, interval.lower, interval.upper) == pytest.approx(0.9, abs=0.03)


# --------------------------------------------------------------------- costs


def test_review_must_be_cheaper_than_scrapping():
    """Guards the config error that silently collapsed the REVIEW band."""
    with pytest.raises(ValueError, match="cost_review"):
        CostModel(cost_false_negative=1000.0, cost_false_positive=1.0, cost_review=12.0)


def test_shipped_cost_model_is_coherent():
    costs = CostModel.load()
    assert costs.cost_review < costs.cost_false_positive < costs.cost_false_negative
    assert 0 < costs.max_review_fraction <= 1.0


# ------------------------------------------------------------------ open set


def test_novelty_detector_excludes_self():
    """Guards the leak that first read AUROC 1.000 with median distance 0.00.

    Scoring the fitting population without excluding self makes every
    reference defect its own nearest neighbour at distance zero, which looks
    like a perfect open-set detector and is nothing of the kind.
    """
    from src.knowledge.prototypes import NoveltyDetector

    library = MechanismLibrary.load()
    lots = LotGenerator(library, seed=21).generate(n_lots=3, parts_per_lot=150, prevalence=0.08)
    from src.features.build import build_features

    feats = build_features(lots, library)
    y = feats["is_defect"].astype(int).to_numpy()

    det = NoveltyDetector(library, k=1).fit(feats, y)
    defects = feats[feats["is_defect"].astype(bool)]

    leaked = det.distance(defects, exclude_self=False)
    honest = det.distance(defects, exclude_self=True)

    assert leaked.median() == pytest.approx(0.0, abs=1e-9)
    assert honest.median() > 0.0


def test_novelty_threshold_uses_known_defects_only():
    """tau must be choosable before any unknown mechanism has been seen."""
    from src.knowledge.prototypes import NoveltyDetector
    from src.features.build import build_features

    library = MechanismLibrary.load()
    lots = LotGenerator(library, seed=22).generate(n_lots=3, parts_per_lot=150, prevalence=0.08)
    feats = build_features(lots, library)
    y = feats["is_defect"].astype(int).to_numpy()

    det = NoveltyDetector(library, k=1).fit(feats, y, keep=0.95)
    assert np.isfinite(det.threshold_) and det.threshold_ > 0

    # Roughly 5% of known defects should fall outside their own threshold.
    flagged = det.is_novel(feats[feats["is_defect"].astype(bool)], exclude_self=True)
    assert flagged.mean() < 0.25


# ------------------------------------------------------- robustness (Phase 6)


def test_lot_too_small_is_refused_not_guessed(library):
    """Borrowing another lot's reference population would be silently wrong."""
    from src.api.service import MIN_LOT_FOR_SCORING

    tiny = LotGenerator(library, seed=31).generate(n_lots=1, parts_per_lot=8, prevalence=0.0)
    assert tiny["part_id"].nunique() < MIN_LOT_FOR_SCORING
    issues = validate(tiny, min_lot_size=MIN_LOT_FOR_SCORING)
    assert any(i.code == "small_lot" for i in issues)


def test_mixed_units_are_an_error_not_a_warning(library, small_lots):
    """A parameter carrying both uA and nA destroys every lot statistic."""
    broken = small_lots.copy()
    mask = (broken["param_name"] == "Iddq") & (broken["hours"] == 96.0)
    broken.loc[mask, "unit"] = "nA"
    issues = validate(broken)
    assert any(i.code == "mixed_units" and i.level == "error" for i in issues)


def test_duplicate_measurement_is_an_error(small_lots):
    dup = pd.concat([small_lots, small_lots.head(5)], ignore_index=True)
    issues = validate(dup)
    assert any(i.code == "duplicate_measurement" and i.level == "error" for i in issues)


def test_missing_timepoints_are_tolerated(library, small_lots):
    """Ragged grids are a warning, not a failure - the kernel fits arbitrary t."""
    from src.features.build import build_features

    ragged = small_lots[~((small_lots["hours"] == 96.0)
                          & (small_lots["part_id"].str.endswith("0")))].copy()
    issues = validate(ragged)
    assert not [i for i in issues if i.level == "error"]
    feats = build_features(ragged, library)
    assert len(feats) > 0 and feats["Iddq__n"].notna().all()


def test_lot_with_no_defects_produces_no_panic(library):
    """A clean lot must not have parts invented for it to flag."""
    from src.features.build import build_features
    from src.module_a.robust_z import RobustZTrack

    clean = LotGenerator(library, seed=32).generate(n_lots=2, parts_per_lot=150, prevalence=0.0)
    feats = build_features(clean, library)
    assert feats["is_defect"].sum() == 0
    s = RobustZTrack().fit_score(feats)
    assert np.isfinite(s).all()


def test_all_defect_lot_does_not_break_lot_statistics(library):
    """The degenerate case: if everything drifts, nothing is an outlier."""
    from src.features.build import build_features
    from src.module_a.robust_z import RobustZTrack

    allbad = LotGenerator(library, seed=33).generate(n_lots=1, parts_per_lot=120, prevalence=1.0)
    feats = build_features(allbad, library)
    s = RobustZTrack().fit_score(feats)
    assert np.isfinite(s).all()
    # Lot-relative scoring cannot flag a uniformly bad lot, and must not pretend
    # to. This is a real limitation of lot-relative screening, not a bug.
    assert s.median() < 10


# ------------------------------------------------------- API (integration)


@pytest.fixture(scope="module")
def client():
    """Skip rather than fail if the model has not been trained in this checkout."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    if not (ROOT / "models" / "fusion.pkl").exists():
        pytest.skip("run scripts/train_fusion.py first")
    from src.api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def one_lot(library):
    from src.ingest.csv_reader import read_csv

    path = ROOT / "data" / "synthetic" / "burnin.csv"
    if not path.exists():
        pytest.skip("run scripts/generate_data.py first")
    df = read_csv(path, library=library)
    lot = df[df["lot_id"].astype(str) == sorted(df["lot_id"].astype(str).unique())[0]]
    cols = ["part_id", "lot_id", "param_name", "unit", "hours", "value",
            "wafer_id", "x", "y", "limit_hi", "temp_C"]
    return {"measurements": lot[cols].to_dict(orient="records")}


def test_health_and_mechanisms(client):
    assert client.get("/health").status_code == 200
    body = client.get("/mechanisms").json()
    assert len(body["mechanisms"]) >= 5


def test_score_lot_then_explain(client, one_lot):
    """The path a QA engineer actually walks: score a lot, question a part."""
    r = client.post("/score/lot", json=one_lot)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["n_parts"] > 0

    flagged = [d for d in body["decisions"] if d["decision"] != "ACCEPT"]
    assert flagged, "expected at least one flagged part in a lot with injected defects"

    cert = client.get(f"/explain/{flagged[0]['part_id']}")
    assert cert.status_code == 200
    text = cert.json()["text"]
    assert "Escape Risk" in text and "Counterfactual" in text and "Audit" in text


def test_predict_drift_uses_prefitted_model(client, one_lot):
    """Regression: fitting per request left an empty training split on one lot."""
    r = client.post("/predict/drift", json={**one_lot, "parameter": "Iddq"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["forecasts"]) > 0
    f = body["forecasts"][0]
    assert f["lower"] <= f["median"] <= f["upper"]


def test_single_lot_cannot_calibrate_conformal(library, small_lots):
    """The bug the API test exposed, pinned at its source."""
    from src.module_b.dataset import make_dataset
    from src.module_b.pipeline import split_train_calib

    one = small_lots[small_lots["lot_id"] == small_lots["lot_id"].iloc[0]]
    ds = make_dataset(one, library, parameter="Iddq")
    with pytest.raises(ValueError, match="at least 2 lots"):
        split_train_calib(ds, np.arange(len(ds.y)))


def test_prediction_only_dataset_without_target(library, small_lots):
    """Forecasting a part still in the oven: the 168h column does not exist yet."""
    from src.module_b.dataset import make_dataset

    early = small_lots[small_lots["hours"] <= 96.0]
    with pytest.raises(ValueError, match="require_target=False"):
        make_dataset(early, library, parameter="Iddq")

    ds = make_dataset(early, library, parameter="Iddq", require_target=False)
    assert len(ds.X) > 0 and ds.y.isna().all()


def test_units_are_harmonised_on_read(library, tmp_path):
    """A file of nA readings must not be silently treated as uA."""
    from src.ingest.csv_reader import read_csv

    rows = []
    for part in range(40):
        for h in (0.0, 24.0, 96.0, 168.0):
            rows.append({"part_id": f"P{part:03d}", "lot_id": "L1", "param_name": "Iddq",
                         "unit": "nA", "hours": h, "value": 10_000.0 + h})
    path = tmp_path / "wrong_units.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    raw = read_csv(path)
    fixed = read_csv(path, library=library)
    assert set(raw["unit"]) == {"nA"} and raw["value"].median() > 1000
    assert set(fixed["unit"]) == {"uA"}
    assert fixed["value"].median() == pytest.approx(raw["value"].median() / 1000.0, rel=1e-6)


# --------------------------------------------- KNOWN DEFECT: fixed-fraction flagging


def test_flag_count_is_currently_independent_of_lot_quality(library):
    """Documents a real, unfixed flaw found by running the dashboard.

    Nearly every feature the tracks read is lot-relative - robust-z per lot,
    rank mobility within lot, n_shift against the lot median. That makes each
    lot's internal score distribution identical by construction, so fixed
    global thresholds cut the same quantiles in every lot and the system flags
    a constant fraction whatever the lot actually contains.

    Consequence: a clean lot loses the same number of good parts to scrap as a
    badly contaminated one, and the system cannot say "this lot is fine".

    This test pins the CURRENT behaviour so the fix is visible when it lands.
    It asserts the flaw, not the desired behaviour - when an absolute anchor is
    added to the feature set, this test should start failing and be replaced.
    """
    pytest.importorskip("sklearn")
    from src.api.service import ScoringService

    if not (ROOT / "models" / "fusion.pkl").exists():
        pytest.skip("run scripts/train_fusion.py first")

    svc = ScoringService()
    counts = {}
    for prevalence in (0.0, 0.10):
        df = LotGenerator(library, seed=int(prevalence * 1000) + 77).generate(
            n_lots=1, parts_per_lot=500, prevalence=prevalence, lot_prefix="QC"
        )
        result = svc.score_lot(df)
        counts[prevalence] = int((result.decisions["decision"] != "ACCEPT").sum())

    clean, contaminated = counts[0.0], counts[0.10]
    assert clean == contaminated, (
        "flag count now varies with lot quality - the fixed-fraction flaw appears "
        "to be fixed, so replace this test with an assertion of the correct behaviour"
    )
    assert clean > 0, "a clean lot is still losing parts to scrap"
