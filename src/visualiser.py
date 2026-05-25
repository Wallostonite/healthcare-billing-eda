# ================================================================
# src/visualiser.py — Chart generator for MedCity Healthcare Billing EDA
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import REPORTS_DIR, logger

PLOTS_DIR = REPORTS_DIR / "plots"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


class Visualiser:
    """
    Generates the 5 charts for the Healthcare Billing EDA:
      1 — Recovery rate by insurance type (bar + box)
      2 — Lifetime billed distribution by insurance type (box plot)
      3 — Age vs billing outcomes (scatter)
      4 — Chi-square: collection rate by insurance type (stacked bar + violin)
      5 — Low-recovery outliers (histogram + breakdown bar)
    """

    def __init__(self, df: pd.DataFrame, engine_results: dict = None,
                 detector_results: dict = None):
        self.df               = df
        self.engine_results   = engine_results or {}
        self.detector_results = detector_results or {}
        PLOTS_DIR.mkdir(exist_ok=True)

    # ── CHART 1 ──────────────────────────────────────────────────────

    def recovery_by_insurance(self) -> pathlib.Path:
        """Bar chart of mean recovery_rate_pct ranked by insurance type."""
        rows = self.engine_results.get("recovery_by_group", {}).get("insurance_type", [])
        if not rows:
            rec_df = (
                self.df.groupby("insurance_type")["recovery_rate_pct"]
                .mean().sort_values(ascending=True).reset_index()
            )
            rec_df.columns = ["insurance_type", "recovery_rate_pct"]
        else:
            rec_df = pd.DataFrame(rows).sort_values("recovery_rate_pct", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = sns.color_palette("Blues_r", len(rec_df))
        bars = axes[0].barh(
            rec_df["insurance_type"], rec_df["recovery_rate_pct"],
            color=colors, edgecolor="white"
        )
        for bar, val in zip(bars, rec_df["recovery_rate_pct"]):
            axes[0].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}%", va="center", fontsize=10)
        axes[0].set_xlim(93, 95)
        axes[0].set_xlabel("Mean Recovery Rate (%)", fontsize=12)
        axes[0].set_title("Q1 — Recovery Rate by Insurance Type",
                          fontsize=12, fontweight="bold")

        order = rec_df["insurance_type"].tolist()[::-1]
        sns.boxplot(data=self.df, x="insurance_type", y="bill_collection_rate",
                    order=order, palette="Blues", ax=axes[1])
        axes[1].set_xlabel("Insurance Type", fontsize=12)
        axes[1].set_ylabel("Bill Collection Rate (%)", fontsize=12)
        axes[1].set_title("Q1 — Collection Rate Distribution by Insurance Type",
                          fontsize=12, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        path = PLOTS_DIR / "01_recovery_by_insurance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 2 ──────────────────────────────────────────────────────

    def lifetime_billed_by_insurance(self) -> pathlib.Path:
        """Box plot: lifetime_billed distribution by insurance type."""
        order = (
            self.df.groupby("insurance_type")["lifetime_billed"]
            .median().sort_values(ascending=False).index
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=self.df, x="insurance_type", y="lifetime_billed",
                    order=order, palette="Reds_r", ax=ax)
        ax.axhline(self.df["lifetime_billed"].mean(), color="#2ecc71",
                   linestyle="--", linewidth=1.5,
                   label=f"Overall mean ${self.df['lifetime_billed'].mean():,.0f}")
        ax.set_xlabel("Insurance Type", fontsize=12)
        ax.set_ylabel("Lifetime Billed (USD)", fontsize=12)
        ax.set_title("Q1 — Lifetime Billed by Insurance Type",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = PLOTS_DIR / "02_lifetime_billed_by_insurance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 3 ──────────────────────────────────────────────────────

    def age_vs_billing(self) -> pathlib.Path:
        """Scatter: age_from_dob vs lifetime_billed and bill_collection_rate."""
        sample = self.df.sample(min(2000, len(self.df)), random_state=42)
        ins_types = sorted(self.df["insurance_type"].dropna().unique())
        palette   = dict(zip(ins_types, sns.color_palette("Set2", len(ins_types))))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ins, grp in sample.groupby("insurance_type"):
            axes[0].scatter(grp["age_from_dob"], grp["lifetime_billed"],
                            alpha=0.3, s=10, color=palette.get(ins, "#95a5a6"), label=ins)
        axes[0].set_xlabel("Patient Age", fontsize=12)
        axes[0].set_ylabel("Lifetime Billed (USD)", fontsize=12)
        axes[0].set_title("Q2 — Age vs Lifetime Billed", fontsize=12, fontweight="bold")

        r_billed = self.engine_results.get("age_correlation", {}).get("lifetime_billed", {})
        if r_billed:
            axes[0].annotate(
                f"r = {r_billed['r']:.4f}",
                xy=(0.05, 0.93), xycoords="axes fraction", fontsize=11,
                color="#e74c3c",
            )
        axes[0].legend(fontsize=8, markerscale=2)

        for ins, grp in sample.groupby("insurance_type"):
            axes[1].scatter(grp["age_from_dob"], grp["bill_collection_rate"],
                            alpha=0.3, s=10, color=palette.get(ins, "#95a5a6"), label=ins)
        axes[1].set_xlabel("Patient Age", fontsize=12)
        axes[1].set_ylabel("Bill Collection Rate (%)", fontsize=12)
        axes[1].set_title("Q2 — Age vs Bill Collection Rate", fontsize=12, fontweight="bold")

        r_col = self.engine_results.get("age_correlation", {}).get("bill_collection_rate", {})
        if r_col:
            axes[1].annotate(
                f"r = {r_col['r']:.4f}",
                xy=(0.05, 0.93), xycoords="axes fraction", fontsize=11,
                color="#e74c3c",
            )

        plt.tight_layout()
        path = PLOTS_DIR / "03_age_vs_billing.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 4 ──────────────────────────────────────────────────────

    def chi_square_breakdown(self) -> pathlib.Path:
        """Stacked bar of High/Low collection rate by insurance type + violin."""
        median_val = self.df["bill_collection_rate"].median()
        df_plot    = self.df.copy()
        df_plot["collection_bucket"] = df_plot["bill_collection_rate"].apply(
            lambda x: "High" if x >= median_val else "Low"
        )

        ct = (
            pd.crosstab(df_plot["insurance_type"], df_plot["collection_bucket"],
                        normalize="index") * 100
        )
        # Ensure High/Low column order
        for col in ["High", "Low"]:
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[["High", "Low"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ct.plot(kind="bar", stacked=True, ax=axes[0],
                color=["#3498db", "#e74c3c"], edgecolor="white", width=0.6)
        axes[0].set_xlabel("Insurance Type", fontsize=12)
        axes[0].set_ylabel("Share of Patients (%)", fontsize=12)
        axes[0].set_title("Q3 — Collection Rate (High/Low) by Insurance Type",
                          fontsize=12, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend(title="Collection Rate", fontsize=10)

        chi2_res = self.engine_results.get("chi_square", {}).get(
            "insurance_type ~ bill_collection_rate", {}
        )
        if chi2_res:
            sig_label = "SIGNIFICANT" if chi2_res["significant"] else "not significant"
            axes[0].set_title(
                f"Q3 — Chi-square: p={chi2_res['p_value']:.4f} [{sig_label}]",
                fontsize=11, fontweight="bold"
            )

        order_ins = (
            self.df.groupby("insurance_type")["bill_collection_rate"]
            .mean().sort_values().index
        )
        sns.violinplot(data=self.df, x="insurance_type", y="bill_collection_rate",
                       order=order_ins, palette="Set2", ax=axes[1], cut=0)
        axes[1].set_xlabel("Insurance Type", fontsize=12)
        axes[1].set_ylabel("Bill Collection Rate (%)", fontsize=12)
        axes[1].set_title("Q3 — Collection Rate Distribution",
                          fontsize=12, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        path = PLOTS_DIR / "04_chi_square_breakdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── CHART 5 ──────────────────────────────────────────────────────

    def low_recovery_outliers(self) -> pathlib.Path:
        """Histogram of bill_collection_rate with IQR fence + breakdown by insurance type."""
        lr    = self.detector_results.get("low_recovery", {})
        fence = lr.get("lower_fence", None)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(self.df["bill_collection_rate"], bins=40, kde=True,
                     color="#3498db", ax=axes[0])
        if fence is not None and fence > 0:
            axes[0].axvline(fence, color="#e74c3c", linewidth=2, linestyle="--",
                            label=f"IQR lower fence ({fence:.1f}%)")
            axes[0].legend(fontsize=10)
        axes[0].set_xlabel("Bill Collection Rate (%)", fontsize=12)
        axes[0].set_ylabel("Number of Patients", fontsize=12)
        flagged_pct = lr.get("flagged_pct", 0)
        axes[0].set_title(
            f"Q4 — Collection Rate Distribution  ({flagged_pct}% of patients below fence)",
            fontsize=11, fontweight="bold"
        )

        threshold = fence if (fence is not None and fence > 0) \
            else self.df["bill_collection_rate"].quantile(0.25)
        low_patients = self.df[self.df["bill_collection_rate"] < threshold]
        low_counts = (
            low_patients["insurance_type"].value_counts().sort_values(ascending=True)
        )
        palette = sns.color_palette("Reds_r", len(low_counts))
        bars = axes[1].barh(low_counts.index, low_counts.values,
                            color=palette, edgecolor="white")
        for bar, val in zip(bars, low_counts.values):
            axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{val}", va="center", fontsize=10)
        axes[1].set_xlabel("Number of Low-Recovery Patients", fontsize=12)
        axes[1].set_title("Q4 — Low-Recovery Patients by Insurance Type",
                          fontsize=12, fontweight="bold")

        plt.tight_layout()
        path = PLOTS_DIR / "05_low_recovery_outliers.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Saved: {path.name}")
        return path

    # ── RUN ALL ───────────────────────────────────────────────────────

    def run_all(self) -> list:
        paths = [
            self.recovery_by_insurance(),
            self.lifetime_billed_by_insurance(),
            self.age_vs_billing(),
            self.chi_square_breakdown(),
            self.low_recovery_outliers(),
        ]
        logger.info(f"[PLOT] All {len(paths)} charts saved → {PLOTS_DIR}")
        return paths
