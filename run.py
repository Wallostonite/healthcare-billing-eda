# ================================================================
# run.py — MedCity Analytics Healthcare Billing EDA Entry Point
# ================================================================
# HOW TO RUN:
#   python run.py
#
# WHAT HAPPENS:
#   1. Load processed-data.csv
#   2. Profile the dataset
#   3. Group analysis (billing metrics by insurance type, city, gender)
#   4. Correlation (numeric feature relationships)
#   5. Time trends (monthly billing volume via registered_at)
#   6. Chi-square test (insurance_type vs bill_collection_rate)
#   7. Recovery by group (all categorical columns)
#   8. Age correlation (age_from_dob vs billing outcomes)
#   9. Save analysis_report.txt
#  10. Patient segment profiling (per insurance type)
#  11. Anomaly detection (IQR + Z-score) + flag_low_recovery
#  12. Save anomalies.csv and segment_profile.csv
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import INDUSTRY, logger
from src.eda_engine           import HealthcareEDAEngine
from src.patient_segment_profiler import PatientSegmentProfiler
from src.anomaly_detector     import AnomalyDetector


def main() -> None:
    logger.info("=" * 60)
    logger.info("  MEDCITY ANALYTICS — HEALTHCARE BILLING EDA")
    logger.info(f"  Industry: {INDUSTRY}")
    logger.info("=" * 60)

    # ── PART 1: EDA ENGINE ────────────────────────────────────────
    engine = HealthcareEDAEngine()
    logger.info(f"Created: {engine}")

    (
        engine
        .load()
        .profile()
        .group_analysis()
        .correlation()
        .time_trends()
        .chi_square_test("insurance_type", "bill_collection_rate")
        .recovery_by_group("insurance_type")
        .recovery_by_group("gender")
        .recovery_by_group("city")
        .age_correlation("lifetime_billed")
        .age_correlation("bill_collection_rate")
        .report(save=True)
    )

    logger.info(f"EDA complete: {engine}")

    # ── PART 2: PATIENT SEGMENT PROFILER ─────────────────────────
    if engine.df is not None:
        logger.info("")
        logger.info("[PROFILER] Profiling patient segments...")

        profiler = PatientSegmentProfiler(engine.df)
        profiler.profile_all().export_csv()

        print()
        print("  SEGMENT PROFILE (by insurance type):")
        print("  " + "-" * 50)
        ranked = profiler.rank_segments("recovery_rate_pct")
        print(ranked.to_string(index=False))

        print()
        print("  RANKED BY BILL COLLECTION RATE:")
        print("  " + "-" * 50)
        print(profiler.rank_segments("bill_collection_rate").to_string(index=False))

        logger.info(f"Segment profiling complete: {profiler}")

        # ── PART 3: ANOMALY DETECTION ─────────────────────────────
        logger.info("")
        logger.info("[ANOMALY] Starting anomaly detection...")

        detector = AnomalyDetector(engine.df)
        detector.run(columns=["lifetime_billed", "outstanding_balance", "bill_collection_rate"])
        detector.flag_low_recovery()
        detector.save_anomalies()

        print()
        print("  ANOMALY DETECTION SUMMARY:")
        print("  " + "-" * 50)
        print(detector.summary().to_string(index=False))
        print()
        print(f"  Confirmed anomaly rows: {len(detector.confirmed):,}")
        print(f"  Saved to: reports/anomalies.csv")

        if "low_recovery" in detector.results:
            lr = detector.results["low_recovery"]
            print()
            print("  LOW-RECOVERY PATIENTS:")
            print("  " + "-" * 50)
            print(f"  IQR lower fence:  {lr['lower_fence']:.2f}%")
            print(f"  Flagged patients: {lr['flagged_count']:,} ({lr['flagged_pct']}%)")
            print(f"  Mean rate (flagged): {lr['mean_rate']:.2f}%")

        logger.info(f"Anomaly detection complete: {detector}")

        print()
        print("  OUTPUTS:")
        print("    reports/analysis_report.txt")
        print("    reports/segment_profile.csv")
        print("    reports/anomalies.csv")
        # ── PART 4: CHARTS ────────────────────────────────────────
        try:
            from src.visualiser import Visualiser
            logger.info("")
            logger.info("[PLOT] Generating charts...")
            vis   = Visualiser(engine.df, engine.results, detector.results)
            paths = vis.run_all()
            print()
            print("  CHARTS SAVED:")
            print("  " + "-" * 50)
            for p in paths:
                print(f"    {p}")
        except Exception as exc:
            logger.warning(f"[PLOT] Chart generation failed: {exc}")

        print()
        print("  NEXT: Open notebooks/healthcare_deep_dive.ipynb")
        print("        for interactive visualisation and Q1–Q4 board analysis")


if __name__ == "__main__":
    main()
