# ================================================================
# src/patient_segment_profiler.py — Billing breakdown by insurance type
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np

from config import REPORTS_DIR, logger

SEGMENT_COL = "insurance_type"
SEGMENT_METRICS = [
    "lifetime_billed",
    "paid_by_patient",
    "paid_by_insurance",
    "recovery_rate_pct",
    "bill_collection_rate",
]


class PatientSegmentProfiler:
    """
    Profiles billing outcomes broken down by insurance_type.

    Answers the CFO question: which insurance segments recover the most
    and which are at risk of write-off?

    Attributes
    ──────────
    df       pd.DataFrame   the patient billing DataFrame
    profile  pd.DataFrame   summary stats per insurance_type (set by profile_all())
    """

    def __init__(self, df: pd.DataFrame):
        self.df      = df.copy()
        self.profile = pd.DataFrame()
        logger.info(f"PatientSegmentProfiler initialised — {len(df):,} patients")

    def profile_all(self) -> "PatientSegmentProfiler":
        """
        Compute mean, median, and std of billing metrics per insurance_type.

        Stores a flat summary DataFrame in self.profile with one row per
        insurance type and columns for each metric × statistic combination.

        Returns self.
        """
        if SEGMENT_COL not in self.df.columns:
            logger.warning(f"[PROFILER] '{SEGMENT_COL}' column not found — skipping")
            return self

        metrics = [m for m in SEGMENT_METRICS if m in self.df.columns]
        rows = []

        for seg in sorted(self.df[SEGMENT_COL].dropna().unique()):
            grp = self.df[self.df[SEGMENT_COL] == seg]
            row = {"insurance_type": seg, "patient_count": len(grp)}
            for metric in metrics:
                row[f"{metric}_mean"]   = round(float(grp[metric].mean()),   2)
                row[f"{metric}_median"] = round(float(grp[metric].median()), 2)
                row[f"{metric}_std"]    = round(float(grp[metric].std()),    2)
            rows.append(row)

        self.profile = pd.DataFrame(rows)
        logger.info(
            f"[PROFILER] Profiled {len(rows)} insurance segments "
            f"across {len(metrics)} metrics"
        )
        return self

    def rank_segments(self, metric: str = "recovery_rate_pct") -> pd.DataFrame:
        """
        Rank insurance types by a given metric (mean value, descending).

        Args:
            metric   name of the metric column in the original df
                     (e.g. "recovery_rate_pct", "bill_collection_rate")

        Returns:
            pd.DataFrame with columns: insurance_type, mean_{metric},
            median_{metric}, patient_count, rank
        """
        if self.profile.empty:
            self.profile_all()

        mean_col = f"{metric}_mean"
        if mean_col not in self.profile.columns:
            logger.warning(f"[PROFILER] Metric '{metric}' not in profile — run profile_all() first")
            return pd.DataFrame()

        ranked = self.profile[["insurance_type", "patient_count", mean_col, f"{metric}_median"]].copy()
        ranked["rank"] = ranked[mean_col].rank(ascending=False).astype(int)
        return ranked.sort_values("rank").reset_index(drop=True)

    def export_csv(self) -> "PatientSegmentProfiler":
        """
        Save the segment profile to reports/segment_profile.csv.

        Returns self.
        """
        if self.profile.empty:
            self.profile_all()

        path = REPORTS_DIR / "segment_profile.csv"
        self.profile.to_csv(path, index=False)
        logger.info(f"[PROFILER] Segment profile saved: {path}")
        return self

    def __str__(self) -> str:
        n = len(self.profile) if not self.profile.empty else 0
        return f"PatientSegmentProfiler({n} segments profiled)"
