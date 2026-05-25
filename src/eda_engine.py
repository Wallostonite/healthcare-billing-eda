# ================================================================
# src/eda_engine.py
# ================================================================
# CONTEXT:
#   We have processed-data.csv — clean, typed, enriched by Module 05.
#   Now we need to UNDERSTAND what is in it.
#
# THE BUSINESS QUESTION:
#   The VP of People wants to know:
#     - How are salaries distributed across departments?
#     - Do experience and salary correlate as expected?
#     - Are there time trends in our hiring or performance data?
#
# THE ANALOGY:
#   Imagine you just received a report from every department in the company.
#   Before presenting to the board, you need to read it, find the patterns,
#   and summarise the key findings.
#   EDAEngine reads the data report, finds the patterns, and summarises them.
#
# WHY A CLASS AND NOT JUST FUNCTIONS?
#   Because we need to run 4 different types of analysis and keep ALL results.
#   A class stores everything in self.results so any other module can access:
#     engine.results["group_analysis"]  → group stats
#     engine.results["correlation"]     → correlation pairs
#   Functions would run and throw away results. The class remembers.
#
# DESIGN PRINCIPLE: READ-ONLY
#   EDAEngine never modifies the DataFrame. It only reads and summarises.
#   (Same as DataValidator in Module 05 — analysts inspect, they do not edit.)
# ================================================================

# ── IMPORTS ───────────────────────────────────────────────────────
import sys        # sys: for manipulating Python's module search path
import pathlib    # pathlib: cross-platform file paths

# Walk up from this file's directory until we find config.py
# This makes the import work whether the file is run from any directory
_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd    # pandas: the core Python data library
import numpy as np     # numpy: numerical operations used for correlation matrix

# Import our settings from config.py
from config import (
    INDUSTRY,              # which industry schema ("bootcamp_data")
    DATA_PATH,             # where processed-data.csv lives
    REPORTS_DIR,           # where to save the report
    TOP_N_GROUPS,          # how many top groups to show (8)
    CORRELATION_THRESHOLD, # minimum r to include (0.3)
    logger                 # shared logger
)


class EDAEngine:
    """
    Runs Exploratory Data Analysis on the processed industry dataset.

    WHAT IS EDA?
    ─────────────
    EDA (Exploratory Data Analysis) is the process of examining a dataset
    to discover patterns, relationships, and anomalies before building models.
    It was formalised by statistician John Tukey in 1977 and is now standard
    practice at every data-driven company.

    Every data scientist and analyst runs EDA as their FIRST step after
    receiving clean data. It answers the question: "What is in here?"

    WHAT THIS CLASS DOES:
    ──────────────────────
    Five methods, each answering a different business question:
      1. load()           → How many rows/columns? What types?
      2. profile()        → What are the distributions and completeness?
      3. group_analysis() → How do key metrics vary by category?
      4. correlation()    → Which numeric variables move together?
      5. time_trends()    → How do metrics change over time?

    METHOD CHAIN PATTERN:
    ──────────────────────
    engine.load().profile().group_analysis().correlation().time_trends().report()

    Each method returns self so they can be chained like this.
    This is the same pattern we used in Module 05 ETL.

    Attributes
    ──────────
    df         pd.DataFrame  the loaded processed data
    results    dict          all analysis outputs (keyed by analysis name)
    num_cols   list[str]     numeric column names (set by load())
    cat_cols   list[str]     categorical column names (set by load())
    _status    str           lifecycle state
    """

    def __init__(self):
        """
        Initialise the EDA engine.

        We do NOT load data here — that is load()'s job.
        This separation allows:
          - Object creation without any I/O
          - Testing without needing a real CSV file
          - Clear lifecycle: ready → loaded → analysed → reported
        """
        self.df       = None    # will hold the DataFrame after load()
        self.results  = {}      # will hold all analysis outputs
        self.num_cols = []      # numeric columns (identified in load())
        self.cat_cols = []      # categorical columns (identified in load())
        self._status  = "ready"

        logger.info(f"EDAEngine initialised — industry: {INDUSTRY}")

    def load(self) -> "EDAEngine":
        """
        Load processed-data.csv and identify column types.

        WHY DO WE REMOVE METADATA COLUMNS?
        ─────────────────────────────────────
        Module 05 added three columns starting with _:
          _industry, _processed_at, _pipeline_version
        These describe the pipeline run — NOT the business data.
        Including them in groupby or correlation analysis would add noise.
        We exclude them for analysis but keep the full DataFrame for saving.

        WHY select_dtypes?
        ─────────────────
        select_dtypes(include=["number"]) returns a subset of the DataFrame
        containing ONLY numeric columns (int64, float64).
        select_dtypes(include=["object"]) returns only text/categorical columns.
        This is how pandas separates column types automatically.

        Returns self for method chaining.
        """
        # Verify the input file exists before trying to open it
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"processed-data.csv not found at: {DATA_PATH}\n"
                "Run Module 05 first:\n"
                "  python module-05.../run.py\n"
                "Then copy processed-data.csv to data/"
            )

        logger.info(f"[EDA] Loading: {DATA_PATH.name}")

        # pd.read_csv() loads a CSV file from disk into a pandas DataFrame
        # low_memory=False reads the entire file before inferring column types
        # (avoids mixed-type columns on large files)
        self.df = pd.read_csv(DATA_PATH, low_memory=False)

        logger.info(f"[EDA] Loaded {len(self.df):,} rows × {self.df.shape[1]} columns")

        # Remove pipeline metadata columns (start with _) from analysis
        # errors="ignore" means: if a column does not exist, just skip it
        analysis_df = self.df.drop(
            columns=[c for c in self.df.columns if c.startswith("_")],
            errors="ignore"
        )

        # Identify column types using pandas type detection
        # Exclude ID-like columns from numeric analysis — they are identifiers,
        # not measurements, so including them in groupby or correlation adds noise
        id_like = {
            c for c in analysis_df.columns
            if c.lower().endswith("_id") or c.lower() == "id"
        }
        self.num_cols = [
            c for c in analysis_df.select_dtypes(include=["number"]).columns
            if c not in id_like
        ]
        self.cat_cols = analysis_df.select_dtypes(include=["object"]).columns.tolist()

        self._status = "loaded"

        logger.info(
            f"[EDA] Column types: "
            f"{len(self.num_cols)} numeric, "
            f"{len(self.cat_cols)} categorical"
        )

        return self   # return self enables chaining: .load().profile()

    def profile(self) -> "EDAEngine":
        """
        Compute a complete statistical profile of the dataset.

        WHY PROFILE FIRST?
        ────────────────────
        Before asking "which department has the highest salary?" you need to
        know: "Do we have salary data for all employees, or is 30% missing?"
        The profile gives you confidence in — or warnings about — the data
        before you draw any conclusions.

        WHAT pd.DataFrame.describe() DOES:
        ─────────────────────────────────
        For each numeric column, it computes:
          count  → how many non-null values
          mean   → arithmetic average
          std    → standard deviation (how spread out values are)
          min    → smallest value
          25%    → 25th percentile (first quartile)
          50%    → median (middle value)
          75%    → 75th percentile (third quartile)
          max    → largest value

        WHY IS MEDIAN (50%) OFTEN MORE USEFUL THAN MEAN?
        ──────────────────────────────────────────────────
        The mean is dragged by extreme values.
        Example: salaries [50k, 60k, 70k, 80k, 2,500k]
          mean   = 552k — distorted by the extreme
          median = 70k  — the genuine middle salary

        Returns self.
        """
        logger.info("[EDA] Computing dataset profile...")

        profile = {
            "rows":             len(self.df),
            "columns":          len(self.df.columns),
            "numeric_cols":     len(self.num_cols),
            "categorical_cols": len(self.cat_cols),

            # Grand total of null cells across the entire DataFrame
            # .isna() creates a boolean DataFrame (True=null, False=not null)
            # .sum() counts True values per column → .sum() again totals all columns
            "total_nulls":      int(self.df.isna().sum().sum()),

            # Null as a percentage of all cells
            # self.df.size = rows × columns (total cell count)
            "null_pct":         round(
                                    self.df.isna().sum().sum() / self.df.size * 100, 2
                                ),

            # Memory consumed by this DataFrame in megabytes
            # deep=True accurately counts object (text) column memory
            "memory_mb":        round(
                                    self.df.memory_usage(deep=True).sum() / 1024**2, 2
                                ),

            # Rows that are exact copies of another row
            "duplicates":       int(self.df.duplicated().sum()),
        }

        # Descriptive statistics for numeric columns
        # .describe() returns a DataFrame with statistics as rows and columns as columns
        # .round(3) limits decimal places so numbers stay readable
        # .to_dict() converts the result to a nested dictionary for easy access
        if self.num_cols:
            desc = self.df[self.num_cols].describe().round(3)
            # orient="index" → {stat: {col: val}} so report() can do
            # p["descriptive_stats"].get("mean", {}).keys()
            profile["descriptive_stats"] = desc.to_dict(orient="index")

        # Value counts for categorical columns (top values and their frequencies)
        cat_profiles = {}
        for col in self.cat_cols[:8]:   # cap at 8 to keep the report readable
            # .value_counts() counts occurrences of each unique value, sorted descending
            vc = self.df[col].value_counts()
            cat_profiles[col] = {
                "unique_count": int(self.df[col].nunique()),   # .nunique() = unique value count
                "top_5":        vc.head(5).to_dict(),          # 5 most common values
                "null_count":   int(self.df[col].isna().sum()),# nulls in this column
            }
        profile["categorical_profiles"] = cat_profiles

        self.results["profile"] = profile   # store for later use by report() and other modules

        logger.info(
            f"[EDA] Profile complete — "
            f"{profile['rows']:,} rows | "
            f"{profile['null_pct']}% nulls"
        )

        return self

    def group_analysis(self) -> "EDAEngine":
        """
        Group numeric metrics by categorical columns and compute aggregates.

        WHY THIS IS THE MOST IMPORTANT EDA STEP:
        ──────────────────────────────────────────
        "What is the average salary?" is a weak question.
        "What is the average salary per department?" is a strong question.

        Groupby analysis is how we move from data facts to business insights.
        Every business stakeholder cares about group differences:
          HR:      salary equity across departments
          Sales:   revenue performance by region
          Finance: cost per product category
          Ops:     efficiency by shift or plant

        HOW pd.DataFrame.groupby() WORKS:
        ───────────────────────────────────
        df.groupby("department")["salary"].agg(["mean", "median", "std"])
          → splits the DataFrame into one group per unique department
          → takes the salary column from each group
          → computes mean, median, std for each group
          → returns a new DataFrame with one row per department

        We then rank groups by their mean value so the top-performing
        group is rank 1. This is what goes into the executive report.

        Returns self.
        """
        logger.info("[EDA] Running group analysis...")

        group_results = {}

        # Analyse first 2 categorical columns as grouping variables
        # First 3 numeric columns as the metrics to aggregate
        for cat in self.cat_cols[:2]:
            col_results = {}
            for num in self.num_cols[:3]:

                # groupby(cat)[num]: split by category, take the numeric column
                # .agg([...]): compute multiple aggregations at once
                # .reset_index(): move group labels from index to a regular column
                # .round(2): 2 decimal places for readability
                agg = (
                    self.df.groupby(cat)[num]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .round(2)
                    .reset_index()
                )

                # Add rank: which group has the highest mean?
                # rank(ascending=False) gives rank 1 to the highest mean
                # .astype(int) converts float ranks to integers (cleaner display)
                agg["rank"] = agg["mean"].rank(ascending=False).astype(int)

                # .sort_values("rank") puts rank 1 (best group) at the top
                # .to_dict(orient="records") converts each row to a dict
                col_results[num] = (
                    agg.sort_values("rank")
                       .head(TOP_N_GROUPS)   # only show top N groups
                       .to_dict(orient="records")
                )

            group_results[cat] = col_results

        self.results["group_analysis"] = group_results

        logger.info(f"[EDA] Group analysis: {len(group_results)} grouping variables")

        return self

    def correlation(self) -> "EDAEngine":
        """
        Compute Pearson correlation between all numeric column pairs.

        WHAT IS PEARSON CORRELATION?
        ─────────────────────────────
        The Pearson correlation coefficient (r) measures the LINEAR relationship
        between two numeric variables.

        r ranges from -1 to +1:
          +1.0 → perfect positive relationship
                 (when experience goes up, salary goes up exactly)
           0.0 → no linear relationship
          -1.0 → perfect negative relationship
                 (when one goes up, the other goes down exactly)

        BUSINESS INTERPRETATION:
        ─────────────────────────
          |r| > 0.7 → STRONG   (very likely to be meaningful)
          |r| > 0.5 → MODERATE (probably meaningful, worth investigating)
          |r| > 0.3 → WEAK     (small relationship, note but do not over-interpret)
          |r| < 0.3 → NEGLIGIBLE (likely noise, excluded from the report)

        WHY CORRELATION MATTERS FOR ML (Module 09):
        ─────────────────────────────────────────────
        Features highly correlated with the TARGET → likely good predictors
        Features highly correlated with EACH OTHER → one can be removed
          (redundant features waste computation and can confuse linear models)

        HOW .corr() WORKS:
        ───────────────────
        df[numeric_cols].corr() computes pairwise Pearson correlation for all
        numeric column combinations and returns a symmetric matrix where
        position [i, j] = correlation between column i and column j.
        The diagonal is always 1.0 (a variable perfectly correlates with itself).

        Returns self.
        """
        if len(self.num_cols) < 2:
            logger.warning("[EDA] Not enough numeric columns for correlation")
            return self

        logger.info("[EDA] Computing correlation matrix...")

        # .corr() computes pairwise Pearson correlation coefficients
        # numeric_only=True skips any non-numeric columns silently
        corr_matrix = self.df[self.num_cols].corr(numeric_only=True).round(3)

        # Find all pairs with meaningful correlation
        # We check each unique pair (i,j) once — skip self-pairs and duplicates
        corr_pairs = []
        for i, col_a in enumerate(self.num_cols):
            for j, col_b in enumerate(self.num_cols):
                if i >= j:
                    continue   # skip: self-correlations (i==j) and already-seen pairs (i>j)

                val = corr_matrix.loc[col_a, col_b]   # get the correlation value

                # Skip NaN (happens when a column has zero variance)
                if pd.isna(val):
                    continue

                if abs(val) < CORRELATION_THRESHOLD:
                    continue   # below threshold — not meaningful enough to report

                # Classify strength based on absolute value of r
                if abs(val) > 0.7:
                    strength = "STRONG"
                elif abs(val) > 0.5:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"

                corr_pairs.append({
                    "col_a":       col_a,
                    "col_b":       col_b,
                    "correlation": float(val),
                    "strength":    strength,
                    "direction":   "positive" if val > 0 else "negative",
                })

        # Sort by absolute correlation — strongest first
        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        self.results["correlation"] = {
            "matrix":       corr_matrix.to_dict(),
            "strong_pairs": corr_pairs[:10],   # top 10 most correlated pairs
        }

        logger.info(
            f"[EDA] Correlation: {len(corr_pairs)} meaningful pairs "
            f"(threshold |r| > {CORRELATION_THRESHOLD})"
        )

        return self

    def time_trends(self) -> "EDAEngine":
        """
        Detect time/period columns and compute month-over-month trends.

        WHY TIME TRENDS MATTER:
        ────────────────────────
        A single average is a snapshot. A time trend is a movie.
        "Average salary is £92k" — is that growing or shrinking?
        Time trends answer the question that static averages cannot.

        METRICS WE COMPUTE FOR EACH TIME PERIOD:
        ──────────────────────────────────────────
          sum          → total value for this period
          mean         → average value for this period
          mom_change   → mean minus previous period mean (absolute change)
          mom_pct      → percentage change from previous period
          rolling_3m   → 3-period rolling average (smooths short-term noise)

        HOW .diff() WORKS:
        ───────────────────
        Series.diff() subtracts the previous row from the current row.
        Applied to the monthly mean column:
          January mean  = 92,000
          February mean = 94,500
          February diff = 94,500 - 92,000 = +2,500

        HOW .pct_change() WORKS:
        ─────────────────────────
        Series.pct_change() computes (current - previous) / previous.
        Returns a fraction. We multiply by 100 to get a percentage.

        HOW .rolling(3).mean() WORKS:
        ──────────────────────────────
        Creates a 3-row sliding window and computes the mean inside it.
        For row N: rolling mean = (row N-2 + row N-1 + row N) / 3
        This smooths out one-off spikes to reveal the underlying trend.

        Returns self.
        """
        # Look for time/period columns by checking column name keywords.
        # Priority 1: non-date period columns (month, quarter, year).
        # Priority 2: full date columns — parse and aggregate to monthly.
        period_keywords = ["month", "period", "quarter", "year"]
        date_keywords   = ["date", "registered_at", "last_claim"]

        period_cols = [
            c for c in self.df.columns
            if any(kw in c.lower() for kw in period_keywords)
        ]
        date_cols = [
            c for c in self.df.columns
            if any(kw in c.lower() for kw in date_keywords)
        ]

        work_df  = self.df.copy()
        group_col = None

        if period_cols:
            time_col = period_cols[0]
            if work_df[time_col].nunique() <= 200:
                group_col = time_col
        elif date_cols:
            time_col = date_cols[0]
            parsed = pd.to_datetime(work_df[time_col], errors="coerce")
            if parsed.notna().sum() > 0:
                work_df["_month"] = parsed.dt.to_period("M").astype(str)
                group_col = "_month"

        if group_col is None:
            logger.info("[EDA] No usable time column found — skipping time trends")
            return self

        logger.info(f"[EDA] Computing time trends on '{group_col}'...")

        trend_results = {}
        for num_col in self.num_cols[:2]:
            if num_col not in work_df.columns:
                continue
            monthly = (
                work_df.groupby(group_col)[num_col]
                .agg(["count", "sum", "mean"])
                .round(2)
                .reset_index()
                .sort_values(group_col)
            )
            monthly["mom_change"] = monthly["mean"].diff()
            monthly["mom_pct"]    = monthly["mean"].pct_change().mul(100).round(1)
            monthly["rolling_3m"] = monthly["mean"].rolling(3).mean().round(2)
            trend_results[num_col] = monthly.to_dict(orient="records")

        self.results["time_trends"] = trend_results
        logger.info(f"[EDA] Time trends: {len(trend_results)} metrics analysed")

        return self

    def report(self, save: bool = True) -> None:
        """
        Print and optionally save the structured analysis report.

        This is the final step — it turns numbers into language the VP can read.
        A good EDA report:
          - States findings as sentences, not just tables
          - Ranks items (highest to lowest) so the most important comes first
          - Flags anomalies and exceptions explicitly
          - Recommends next steps

        Args:
            save   if True, saves the report to reports/analysis_report.txt
        """
        lines = []

        lines += [
            "═" * 65,
            f"  MODULE 06 — EDA REPORT  |  INDUSTRY: {INDUSTRY.upper()}",
            "═" * 65,
        ]

        # ── Dataset profile ────────────────────────────────────────────
        if "profile" in self.results:
            p = self.results["profile"]
            lines += [
                "",
                "  DATASET OVERVIEW",
                f"    Records:              {p['rows']:,}",
                f"    Columns:              {p['columns']}",
                f"    Numeric columns:      {p['numeric_cols']}",
                f"    Categorical columns:  {p['categorical_cols']}",
                f"    Missing values:       {p['total_nulls']:,} ({p['null_pct']}%)",
                f"    Duplicate rows:       {p['duplicates']:,}",
                f"    Memory usage:         {p['memory_mb']} MB",
            ]

            # Show descriptive stats for numeric columns
            if "descriptive_stats" in p:
                lines += ["", "  DESCRIPTIVE STATISTICS"]
                lines.append(f"    {'Metric':<30} {'Mean':>12} {'Median':>12} {'Std Dev':>10}")
                lines.append("    " + "-" * 65)
                for col in list(p["descriptive_stats"].get("mean", {}).keys())[:6]:
                    mean = p["descriptive_stats"].get("mean",  {}).get(col)
                    med  = p["descriptive_stats"].get("50%",   {}).get(col)
                    std  = p["descriptive_stats"].get("std",   {}).get(col)
                    m_s  = f"{mean:>12,.2f}" if isinstance(mean, float) else f"{mean:>12}"
                    md_s = f"{med:>12,.2f}"  if isinstance(med,  float) else f"{med:>12}"
                    st_s = f"{std:>10,.2f}"  if isinstance(std,  float) else f"{std:>10}"
                    lines.append(f"    {col:<30} {m_s} {md_s} {st_s}")

        # ── Group analysis ─────────────────────────────────────────────
        if "group_analysis" in self.results:
            lines += ["", "  GROUP ANALYSIS"]
            for cat_col, metrics in self.results["group_analysis"].items():
                for num_col, rows in metrics.items():
                    lines += [
                        "",
                        f"    {num_col.upper()} BY {cat_col.upper()}",
                        f"    {'Group':<28} {'Mean':>12} {'Median':>12} {'Count':>8} {'Rank':>6}",
                        "    " + "-" * 68,
                    ]
                    for row in rows[:TOP_N_GROUPS]:
                        g  = str(row.get(cat_col, ""))[:27]
                        m  = row.get("mean",   0)
                        md = row.get("median", 0)
                        c  = row.get("count",  0)
                        r  = row.get("rank",   0)
                        lines.append(
                            f"    {g:<28} {m:>12,.2f} {md:>12,.2f} {c:>8,} {r:>6}"
                        )

        # ── Correlation ────────────────────────────────────────────────
        if "correlation" in self.results:
            pairs = self.results["correlation"]["strong_pairs"]
            if pairs:
                lines += [
                    "",
                    f"  TOP CORRELATIONS (|r| > {CORRELATION_THRESHOLD})",
                    f"    {'Column A':<28} {'Column B':<28} {'r':>8}  Strength",
                    "    " + "-" * 70,
                ]
                for p in pairs[:8]:
                    lines.append(
                        f"    {p['col_a']:<28} {p['col_b']:<28} "
                        f"{p['correlation']:>8.3f}  {p['strength']} {p['direction']}"
                    )

        # ── Time trends ────────────────────────────────────────────────
        if "time_trends" in self.results:
            lines += ["", "  TIME TRENDS (last 6 periods)"]
            for metric, rows in self.results["time_trends"].items():
                if not rows:
                    continue
                period_key = list(rows[0].keys())[0]
                lines += [
                    "",
                    f"    {metric.upper()}",
                    f"    {'Period':<15} {'Mean':>12} {'MoM %':>9} {'Rolling 3':>12}",
                    "    " + "-" * 52,
                ]
                for row in rows[-6:]:
                    period  = str(row.get(period_key, ""))
                    mean_v  = row.get("mean", 0)
                    mom_p   = row.get("mom_pct")  or 0.0
                    rolling = row.get("rolling_3m") or 0.0
                    lines.append(
                        f"    {period:<15} {mean_v:>12,.2f} {mom_p:>8.1f}% {rolling:>12,.2f}"
                    )

        # ── Next steps ─────────────────────────────────────────────────
        lines += [
            "",
            "  NEXT STEPS:",
            "    → Use correlation findings for ML feature selection (Module 09)",
            "    → Pass profile stats to Claude for AI insights (Module 11)",
            "    → Use profile as drift detection baseline (Module 14)",
            "",
            "═" * 65,
        ]

        report_text = "\n".join(lines)
        print(report_text)

        if save:
            report_path = REPORTS_DIR / "analysis_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            logger.info(f"[EDA] Report saved: {report_path}")

    def __str__(self) -> str:
        """Human-readable summary — shown by print(engine)."""
        rows = len(self.df) if self.df is not None else 0
        return (
            f"EDAEngine("
            f"industry={INDUSTRY!r}, "
            f"rows={rows:,}, "
            f"analyses={list(self.results.keys())})"
        )

    def __repr__(self) -> str:
        """Developer representation — shown in debugger."""
        return (
            f"EDAEngine("
            f"industry={INDUSTRY!r}, "
            f"status={self._status!r})"
        )


# ================================================================
# HealthcareEDAEngine — extends EDAEngine with healthcare-specific analysis
# ================================================================
# Three new methods required by the README:
#   chi_square_test(col_a, col_b) — independence test
#   recovery_by_group(group_col)  — mean recovery metrics by any category
#   age_correlation(value_col)    — Pearson r between age_from_dob and outcome
# ================================================================

from scipy import stats as scipy_stats   # statistical tests


class HealthcareEDAEngine(EDAEngine):
    """
    Extends EDAEngine with healthcare billing analysis.

    Added methods
    ─────────────
    chi_square_test()    — tests whether insurance_type is independent
                           of billing outcome (chi-squared test)
    recovery_by_group()  — mean recovery_rate_pct and bill_collection_rate
                           grouped by any categorical column
    age_correlation()    — Pearson correlation between age_from_dob
                           and a numeric billing outcome
    """

    def group_analysis(
        self,
        group_col: str = None,
        value_col: str = None,
    ) -> "HealthcareEDAEngine":
        """
        Override: uses healthcare-relevant grouping columns by default.

        Groups by insurance_type, gender, and city (skipping PII columns like
        first_name, last_name, email) and aggregates billing metrics.
        """
        # Healthcare-relevant categorical columns (skip PII and free-text)
        pii_skip = {"first_name", "last_name", "email", "phone", "date_of_birth",
                    "registered_at", "last_claim_date", "extracted_date",
                    "source_schema", "city"}
        group_cols = (
            [group_col] if group_col
            else [c for c in self.cat_cols if c not in pii_skip][:3]
        )
        value_cols = (
            [value_col] if value_col
            else [
                c for c in ["lifetime_billed", "recovery_rate_pct",
                             "bill_collection_rate", "outstanding_balance"]
                if c in self.df.columns
            ][:3]
        )

        if not group_cols or not value_cols:
            logger.warning("[EDA] No suitable group or value columns — skipping group_analysis")
            return self

        logger.info("[EDA] Running group analysis...")
        group_results = {}

        for cat in group_cols:
            if cat not in self.df.columns:
                continue
            col_results = {}
            for num in value_cols:
                if num not in self.df.columns:
                    continue
                agg = (
                    self.df.groupby(cat)[num]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .round(2)
                    .reset_index()
                )
                agg["rank"] = agg["mean"].rank(ascending=False).astype(int)
                col_results[num] = (
                    agg.sort_values("rank")
                       .head(TOP_N_GROUPS)
                       .to_dict(orient="records")
                )
            group_results[cat] = col_results

        self.results["group_analysis"] = group_results
        logger.info(f"[EDA] Group analysis: {len(group_results)} grouping variables")
        return self

    def chi_square_test(
        self,
        col_a: str = "insurance_type",
        col_b: str = "bill_collection_rate",
    ) -> "HealthcareEDAEngine":
        """
        Chi-squared test of independence between two variables.

        HOW THE CHI-SQUARED TEST WORKS:
        ─────────────────────────────────
        The chi-squared test asks: "Are these two categorical variables
        independent, or does knowing one tell you something about the other?"

        If col_b is numeric, it is automatically bucketed at its median
        into "High" (≥ median) and "Low" (< median) before the test.

        H0 (null hypothesis):   col_a and col_b are independent
        H1 (alternative):       col_a and col_b are associated

        If p < 0.05 we reject H0 — the insurance type genuinely predicts
        whether a patient's bill gets collected.

        Args:
            col_a   first variable — default: "insurance_type"
            col_b   second variable — default: "bill_collection_rate"
                    (auto-bucketed into High/Low if numeric)

        Returns self.
        """
        if col_a not in self.df.columns or col_b not in self.df.columns:
            logger.warning(f"[CHI2] Column not found: {col_a} or {col_b}")
            return self

        work = self.df[[col_a, col_b]].dropna().copy()

        # Auto-bucket numeric col_b at median
        if pd.api.types.is_numeric_dtype(work[col_b]):
            median_val = work[col_b].median()
            bucket_col = f"{col_b}_bucket"
            work[bucket_col] = work[col_b].apply(
                lambda x: "High" if x >= median_val else "Low"
            )
            col_b_used = bucket_col
        else:
            col_b_used = col_b

        contingency = pd.crosstab(work[col_a], work[col_b_used])
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)

        key = f"{col_a} ~ {col_b}"
        if "chi_square" not in self.results:
            self.results["chi_square"] = {}

        self.results["chi_square"][key] = {
            "chi2":        round(float(chi2), 4),
            "p_value":     round(float(p_value), 4),
            "dof":         int(dof),
            "significant": bool(p_value < 0.05),
            "verdict":     (
                "SIGNIFICANT — insurance type predicts billing outcome"
                if p_value < 0.05
                else "not significant — insurance type does not predict billing outcome"
            ),
        }

        logger.info(
            f"[CHI2] {key}: chi2={chi2:.3f}  p={p_value:.4f}  "
            f"dof={dof}  "
            f"[{'SIGNIFICANT' if p_value < 0.05 else 'not significant'}]"
        )
        return self

    def recovery_by_group(
        self, group_col: str = "insurance_type"
    ) -> "HealthcareEDAEngine":
        """
        Mean recovery_rate_pct and bill_collection_rate grouped by any category.

        Ranks groups by mean recovery_rate_pct (highest = rank 1).

        Args:
            group_col   categorical column to group by

        Returns self.
        """
        if group_col not in self.df.columns:
            logger.warning(f"[RECOVERY] Column '{group_col}' not found")
            return self

        metrics = [
            m for m in ["recovery_rate_pct", "bill_collection_rate", "outstanding_balance"]
            if m in self.df.columns
        ]

        agg = (
            self.df.groupby(group_col)[metrics]
            .mean()
            .round(3)
            .reset_index()
        )
        agg["count"] = (
            self.df.groupby(group_col).size().values
        )
        agg["rank"] = (
            agg["recovery_rate_pct"].rank(ascending=False).astype(int)
        )
        agg = agg.sort_values("rank").reset_index(drop=True)

        if "recovery_by_group" not in self.results:
            self.results["recovery_by_group"] = {}

        self.results["recovery_by_group"][group_col] = agg.to_dict(orient="records")

        logger.info(
            f"[RECOVERY] Grouped by '{group_col}': "
            f"{len(agg)} groups ranked by recovery_rate_pct"
        )
        return self

    def age_correlation(
        self, value_col: str = "bill_collection_rate"
    ) -> "HealthcareEDAEngine":
        """
        Pearson correlation between age_from_dob and a billing outcome column.

        Older patients may accumulate more bills, but do they recover at a
        different rate? This method quantifies that relationship.

        Args:
            value_col   numeric outcome column to correlate with age

        Returns self.
        """
        age_col = "age_from_dob"
        if age_col not in self.df.columns or value_col not in self.df.columns:
            logger.warning(f"[AGE CORR] Column not found: {age_col} or {value_col}")
            return self

        clean = self.df[[age_col, value_col]].dropna()
        r, p  = scipy_stats.pearsonr(clean[age_col], clean[value_col])

        if "age_correlation" not in self.results:
            self.results["age_correlation"] = {}

        self.results["age_correlation"][value_col] = {
            "r":           round(float(r), 4),
            "p_value":     round(float(p), 4),
            "significant": bool(p < 0.05),
            "n":           int(len(clean)),
        }

        logger.info(
            f"[AGE CORR] age_from_dob ~ {value_col}: "
            f"r={r:.4f}  p={p:.4f}  n={len(clean):,}"
        )
        return self

    def report(self, save: bool = True) -> None:
        """
        Extends the base report with chi-square, recovery, and age-correlation sections.
        """
        super().report(save=False)   # print base report without saving

        lines = []

        # ── Chi-square results ─────────────────────────────────────
        if "chi_square" in self.results:
            lines += ["", "  CHI-SQUARE TESTS (independence)"]
            for key, res in self.results["chi_square"].items():
                sig = "SIGNIFICANT" if res["significant"] else "not significant"
                lines.append(
                    f"    {key}:  chi2={res['chi2']:.3f}  "
                    f"p={res['p_value']:.4f}  dof={res['dof']}  [{sig}]"
                )
                lines.append(f"    Verdict: {res['verdict']}")

        # ── Recovery by group ──────────────────────────────────────
        if "recovery_by_group" in self.results:
            lines += ["", "  RECOVERY RATE BY GROUP"]
            for group_col, rows in self.results["recovery_by_group"].items():
                lines += [
                    "",
                    f"    BY {group_col.upper()}",
                    f"    {'Group':<20} {'Recovery %':>12} {'Collection %':>14} {'Count':>8} {'Rank':>6}",
                    "    " + "-" * 64,
                ]
                for row in rows:
                    g   = str(row.get(group_col, ""))[:19]
                    rec = row.get("recovery_rate_pct",   0)
                    col = row.get("bill_collection_rate", 0)
                    cnt = row.get("count", 0)
                    rnk = row.get("rank",  0)
                    lines.append(
                        f"    {g:<20} {rec:>12.3f} {col:>14.3f} {cnt:>8,} {rnk:>6}"
                    )

        # ── Age correlation ────────────────────────────────────────
        if "age_correlation" in self.results:
            lines += ["", "  AGE CORRELATION (age_from_dob)"]
            lines.append(
                f"    {'Outcome':<28} {'r':>8} {'p-value':>10} {'Significant':>13}"
            )
            lines.append("    " + "-" * 62)
            for col, res in self.results["age_correlation"].items():
                sig = "YES" if res["significant"] else "no"
                lines.append(
                    f"    {col:<28} {res['r']:>8.4f} {res['p_value']:>10.4f} {sig:>13}"
                )

        # ── Next steps ─────────────────────────────────────────────
        lines += [
            "",
            "  NEXT STEPS:",
            "    → Review segment_profile.csv — focus on lowest-recovery insurance types",
            "    → Prioritise low-recovery patients (anomalies.csv) for billing follow-up",
            "    → If chi-square is significant, use insurance_type as ML feature (Module 09)",
            "",
            "═" * 65,
        ]

        ext_text = "\n".join(lines)
        print(ext_text)

        if save:
            base_path   = REPORTS_DIR / "analysis_report.txt"
            base_text   = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
            report_path = REPORTS_DIR / "analysis_report.txt"
            report_path.write_text(base_text + ext_text, encoding="utf-8")
            logger.info(f"[EDA] Report saved: {report_path}")

    def __str__(self) -> str:
        rows = len(self.df) if self.df is not None else 0
        return (
            f"HealthcareEDAEngine("
            f"industry={INDUSTRY!r}, "
            f"rows={rows:,}, "
            f"analyses={list(self.results.keys())})"
        )
