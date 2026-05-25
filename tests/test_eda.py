# ================================================================
# tests/test_eda.py — Unit Tests for MedCity Healthcare Billing EDA
# ================================================================
# HOW TO RUN:
#   python tests/test_eda.py
#   pytest tests/
# ================================================================

import sys, pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from src.eda_engine               import EDAEngine, HealthcareEDAEngine
from src.anomaly_detector         import AnomalyDetector
from src.patient_segment_profiler import PatientSegmentProfiler


# ── TEST DATA FACTORY ─────────────────────────────────────────────

def make_sample_df(rows: int = 60) -> pd.DataFrame:
    """Small, predictable healthcare billing DataFrame for unit tests."""
    np.random.seed(42)
    insurance_types = ["Private", "Medicare", "Medicaid", "Uninsured", "Corporate"]
    genders         = ["Male", "Female", "Non-binary"]
    cities          = ["Toronto", "Houston", "London", "Sydney", "Dubai", "Singapore"]

    lifetime_billed    = np.random.uniform(500, 50000, rows).round(2)
    paid_by_patient    = (lifetime_billed * np.random.uniform(0.1, 0.6, rows)).round(2)
    paid_by_insurance  = (lifetime_billed * np.random.uniform(0.0, 0.5, rows)).round(2)
    outstanding_bal    = np.maximum(lifetime_billed - paid_by_patient - paid_by_insurance, 0).round(2)
    recovery_rate      = np.clip(
        (paid_by_patient + paid_by_insurance) / lifetime_billed * 100, 0, 100
    ).round(2)
    collection_rate    = np.random.uniform(0, 100, rows).round(2)

    return pd.DataFrame({
        "patient_id":           range(1, rows + 1),
        "age_from_dob":         np.random.randint(18, 85, rows),
        "gender":               [genders[i % len(genders)] for i in range(rows)],
        "city":                 [cities[i % len(cities)] for i in range(rows)],
        "insurance_type":       [insurance_types[i % len(insurance_types)] for i in range(rows)],
        "num_bills":            np.random.uniform(1, 15, rows).round(0),
        "lifetime_billed":      lifetime_billed,
        "paid_by_patient":      paid_by_patient,
        "paid_by_insurance":    paid_by_insurance,
        "recovery_rate_pct":    recovery_rate,
        "outstanding_balance":  outstanding_bal,
        "bill_collection_rate": collection_rate,
    })


def _setup_engine(rows: int = 60) -> HealthcareEDAEngine:
    """Return a HealthcareEDAEngine with df/num_cols/cat_cols pre-loaded."""
    engine = HealthcareEDAEngine()
    engine.df = make_sample_df(rows)
    id_like   = {"patient_id"}
    engine.num_cols = [
        c for c in engine.df.select_dtypes(include=["number"]).columns
        if c not in id_like
    ]
    engine.cat_cols = engine.df.select_dtypes(include=["object"]).columns.tolist()
    engine._status  = "loaded"
    return engine


# ── PROFILE TESTS ─────────────────────────────────────────────────

def test_profile_row_count():
    engine = _setup_engine(60)
    engine.profile()
    assert engine.results["profile"]["rows"] == 60
    print("  PASS: test_profile_row_count")


def test_profile_null_count_clean_data():
    engine = _setup_engine()
    engine.profile()
    assert engine.results["profile"]["total_nulls"] == 0
    print("  PASS: test_profile_null_count_clean_data")


def test_profile_stores_descriptive_stats():
    engine = _setup_engine()
    engine.profile()
    assert "descriptive_stats" in engine.results["profile"]
    assert "mean" in engine.results["profile"]["descriptive_stats"]
    print("  PASS: test_profile_stores_descriptive_stats")


def test_profile_memory_is_positive():
    engine = _setup_engine()
    engine.profile()
    assert engine.results["profile"]["memory_mb"] > 0
    print("  PASS: test_profile_memory_is_positive")


# ── GROUP ANALYSIS TESTS ──────────────────────────────────────────

def test_group_analysis_produces_results():
    engine = _setup_engine()
    engine.group_analysis()
    assert "group_analysis" in engine.results
    assert len(engine.results["group_analysis"]) > 0
    print("  PASS: test_group_analysis_produces_results")


def test_group_analysis_rank_starts_at_one():
    engine = _setup_engine()
    engine.group_analysis()
    first_cat = list(engine.results["group_analysis"].keys())[0]
    first_num = list(engine.results["group_analysis"][first_cat].keys())[0]
    top_row   = engine.results["group_analysis"][first_cat][first_num][0]
    assert top_row["rank"] == 1
    print("  PASS: test_group_analysis_rank_starts_at_one")


def test_method_chaining_returns_self():
    engine = _setup_engine()
    result = engine.profile().group_analysis().correlation()
    assert result is engine
    print("  PASS: test_method_chaining_returns_self")


# ── CHI-SQUARE TESTS ──────────────────────────────────────────────

def test_chi_square_produces_result():
    engine = _setup_engine()
    engine.chi_square_test("insurance_type", "bill_collection_rate")
    assert "chi_square" in engine.results
    assert len(engine.results["chi_square"]) > 0
    print("  PASS: test_chi_square_produces_result")


def test_chi_square_has_p_value_and_significant():
    engine = _setup_engine()
    engine.chi_square_test("insurance_type", "bill_collection_rate")
    for key, res in engine.results["chi_square"].items():
        assert "chi2"        in res
        assert "p_value"     in res
        assert "significant" in res
        assert "dof"         in res
        assert "verdict"     in res
    print("  PASS: test_chi_square_has_p_value_and_significant")


def test_chi_square_p_value_between_0_and_1():
    engine = _setup_engine()
    engine.chi_square_test("insurance_type", "bill_collection_rate")
    for key, res in engine.results["chi_square"].items():
        assert 0.0 <= res["p_value"] <= 1.0
    print("  PASS: test_chi_square_p_value_between_0_and_1")


# ── RECOVERY BY GROUP TESTS ───────────────────────────────────────

def test_recovery_by_group_produces_result():
    engine = _setup_engine()
    engine.recovery_by_group("insurance_type")
    assert "recovery_by_group" in engine.results
    assert "insurance_type"    in engine.results["recovery_by_group"]
    print("  PASS: test_recovery_by_group_produces_result")


def test_recovery_by_group_has_all_insurance_types():
    engine = _setup_engine()
    engine.recovery_by_group("insurance_type")
    rows = engine.results["recovery_by_group"]["insurance_type"]
    found_types = {r["insurance_type"] for r in rows}
    expected    = {"Private", "Medicare", "Medicaid", "Uninsured", "Corporate"}
    assert expected == found_types
    print("  PASS: test_recovery_by_group_has_all_insurance_types")


def test_recovery_by_group_has_expected_keys():
    engine = _setup_engine()
    engine.recovery_by_group("insurance_type")
    row = engine.results["recovery_by_group"]["insurance_type"][0]
    assert "recovery_rate_pct"   in row
    assert "bill_collection_rate" in row
    assert "rank"                 in row
    print("  PASS: test_recovery_by_group_has_expected_keys")


# ── AGE CORRELATION TESTS ─────────────────────────────────────────

def test_age_correlation_returns_r_and_p():
    engine = _setup_engine()
    engine.age_correlation("bill_collection_rate")
    assert "age_correlation" in engine.results
    res = engine.results["age_correlation"]["bill_collection_rate"]
    assert "r"           in res
    assert "p_value"     in res
    assert "significant" in res
    print("  PASS: test_age_correlation_returns_r_and_p")


def test_age_correlation_r_in_valid_range():
    engine = _setup_engine()
    engine.age_correlation("lifetime_billed")
    r = engine.results["age_correlation"]["lifetime_billed"]["r"]
    assert -1.0 <= r <= 1.0
    print("  PASS: test_age_correlation_r_in_valid_range")


# ── PATIENT SEGMENT PROFILER TESTS ────────────────────────────────

def test_profiler_profile_all_insurance_types():
    df       = make_sample_df()
    profiler = PatientSegmentProfiler(df)
    profiler.profile_all()
    assert not profiler.profile.empty
    assert "insurance_type" in profiler.profile.columns
    assert len(profiler.profile) == 5   # 5 insurance types
    print("  PASS: test_profiler_profile_all_insurance_types")


def test_profiler_profile_has_required_metrics():
    df       = make_sample_df()
    profiler = PatientSegmentProfiler(df)
    profiler.profile_all()
    for col_prefix in ["lifetime_billed", "recovery_rate_pct", "bill_collection_rate"]:
        assert f"{col_prefix}_mean"   in profiler.profile.columns
        assert f"{col_prefix}_median" in profiler.profile.columns
    print("  PASS: test_profiler_profile_has_required_metrics")


def test_profiler_rank_segments_sorted():
    df       = make_sample_df()
    profiler = PatientSegmentProfiler(df)
    profiler.profile_all()
    ranked   = profiler.rank_segments("recovery_rate_pct")
    assert ranked.iloc[0]["rank"] == 1
    assert list(ranked["rank"]) == sorted(ranked["rank"].tolist())
    print("  PASS: test_profiler_rank_segments_sorted")


# ── ANOMALY DETECTOR TESTS ────────────────────────────────────────

def test_anomaly_detector_flags_obvious_outlier():
    df = make_sample_df(30)
    df.loc[0, "lifetime_billed"] = 999_999.0
    detector = AnomalyDetector(df)
    detector.run(columns=["lifetime_billed"])
    assert 0 in detector.confirmed.index
    print("  PASS: test_anomaly_detector_flags_obvious_outlier")


def test_anomaly_detector_clean_data_few_anomalies():
    np.random.seed(42)
    df = pd.DataFrame({"value": np.random.normal(50, 5, 200)})
    detector = AnomalyDetector(df)
    detector.run(columns=["value"])
    assert len(detector.confirmed) / len(df) * 100 < 2.0
    print("  PASS: test_anomaly_detector_clean_data_few_anomalies")


def test_anomaly_detector_summary_columns():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.run()
    summary = detector.summary()
    for col in ["column", "iqr_flagged", "zscore_flagged", "confirmed_anomalies", "anomaly_pct"]:
        assert col in summary.columns
    print("  PASS: test_anomaly_detector_summary_columns")


def test_anomaly_detector_does_not_modify_original():
    df  = make_sample_df()
    val = df.loc[0, "lifetime_billed"]
    AnomalyDetector(df).run()
    assert df.loc[0, "lifetime_billed"] == val
    print("  PASS: test_anomaly_detector_does_not_modify_original")


def test_flag_low_recovery_produces_result():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.flag_low_recovery()
    assert "low_recovery" in detector.results
    lr = detector.results["low_recovery"]
    assert "flagged_count" in lr
    assert "lower_fence"   in lr
    assert "flagged_pct"   in lr
    print("  PASS: test_flag_low_recovery_produces_result")


def test_flag_low_recovery_flagged_below_fence():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.flag_low_recovery()
    fence = detector.results["low_recovery"]["lower_fence"]
    count = detector.results["low_recovery"]["flagged_count"]
    actual_below = (df["bill_collection_rate"] < fence).sum()
    assert count == actual_below
    print("  PASS: test_flag_low_recovery_flagged_below_fence")


def test_summary_skips_low_recovery_key():
    df = make_sample_df()
    detector = AnomalyDetector(df)
    detector.run(columns=["lifetime_billed"])
    detector.flag_low_recovery()
    summary = detector.summary()
    assert "low_recovery" not in summary["column"].values
    print("  PASS: test_summary_skips_low_recovery_key")


# ── TEST RUNNER ───────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  MEDCITY — HEALTHCARE BILLING EDA UNIT TESTS")
    print("=" * 60)

    print("\n  Profile tests:")
    test_profile_row_count()
    test_profile_null_count_clean_data()
    test_profile_stores_descriptive_stats()
    test_profile_memory_is_positive()

    print("\n  Group analysis tests:")
    test_group_analysis_produces_results()
    test_group_analysis_rank_starts_at_one()
    test_method_chaining_returns_self()

    print("\n  Chi-square tests:")
    test_chi_square_produces_result()
    test_chi_square_has_p_value_and_significant()
    test_chi_square_p_value_between_0_and_1()

    print("\n  Recovery by group tests:")
    test_recovery_by_group_produces_result()
    test_recovery_by_group_has_all_insurance_types()
    test_recovery_by_group_has_expected_keys()

    print("\n  Age correlation tests:")
    test_age_correlation_returns_r_and_p()
    test_age_correlation_r_in_valid_range()

    print("\n  PatientSegmentProfiler tests:")
    test_profiler_profile_all_insurance_types()
    test_profiler_profile_has_required_metrics()
    test_profiler_rank_segments_sorted()

    print("\n  AnomalyDetector tests:")
    test_anomaly_detector_flags_obvious_outlier()
    test_anomaly_detector_clean_data_few_anomalies()
    test_anomaly_detector_summary_columns()
    test_anomaly_detector_does_not_modify_original()
    test_flag_low_recovery_produces_result()
    test_flag_low_recovery_flagged_below_fence()
    test_summary_skips_low_recovery_key()

    print()
    print("=" * 60)
    print("  All tests passed ✓")
    print("=" * 60)
