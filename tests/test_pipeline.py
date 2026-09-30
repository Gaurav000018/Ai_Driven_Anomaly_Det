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
