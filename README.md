# MedCity Analytics — Healthcare Billing EDA

**Company:** MedCity Analytics
**Role:** Data Analyst, Revenue Cycle
**Dataset:** 4,860 patients across 5 insurance types

---

## Project Overview

Bill collection rates were reported to vary widely across patient cohorts and insurance types, but the gap had never been quantified or tested for statistical significance. This project profiles each insurance segment, tests whether insurance type independently predicts billing outcomes, flags low-recovery patients at risk of write-off, and produces a segment-level summary for the Revenue Cycle team.

---

## What Was Built

### `HealthcareEDAEngine` — Extended EDA Engine

| Method | Description |
|---|---|
| `load()` | Loads `data/processed-data.csv`, excludes PII and ID columns from numeric analysis |
| `profile()` | Dataset shape, null rates, descriptive statistics across all billing metrics |
| `group_analysis()` | Groups by insurance_type, gender, and city; ranks by `lifetime_billed`, `recovery_rate_pct`, `bill_collection_rate` |
| `correlation()` | Pearson correlation matrix; skips NaN pairs (zero-variance columns) |
| `time_trends()` | Monthly billing trends derived from `registered_at` date |
| `chi_square_test(col_a, col_b)` | Chi-squared independence test (`scipy.stats.chi2_contingency`); auto-buckets numeric columns at median |
| `recovery_by_group(group_col)` | Mean `recovery_rate_pct` and `bill_collection_rate` per group, ranked |
| `age_correlation(value_col)` | Pearson correlation between `age_from_dob` and a billing outcome |
| `report(save=True)` | Prints and saves the full analysis report including chi-square verdict |

### `PatientSegmentProfiler` — Insurance Type Breakdown

| Method | Description |
|---|---|
| `profile_all()` | Mean, median, and std of all billing metrics per `insurance_type` |
| `rank_segments(metric)` | Ranks all five insurance types by any metric |
| `export_csv()` | Saves segment summary to `reports/segment_profile.csv` |

### `AnomalyDetector` — Outlier Detection with Low-Recovery Flagging

| Method | Description |
|---|---|
| `run(columns)` | IQR + Z-score consensus detection on `lifetime_billed`, `outstanding_balance`, `bill_collection_rate` |
| `flag_low_recovery()` | Flags patients below the IQR lower fence on `bill_collection_rate` — statistically unusual under-recovery |
| `save_anomalies()` | Saves confirmed anomaly rows to `reports/anomalies.csv` |
| `summary()` | Per-column anomaly summary, skipping non-column entries |

### `Visualiser` — Five-Chart Generator

Charts saved to `reports/plots/`:
1. Recovery rate by insurance type — ranked bar + collection rate box plot
2. Lifetime billed by insurance type — box plot sorted by median
3. Age vs billing outcomes — scatter coloured by insurance type with r annotation
4. Chi-square breakdown — High/Low stacked bar + violin, p-value in title
5. Low-recovery outliers — IQR fence on histogram + breakdown by insurance type

### Jupyter Notebook

`notebooks/healthcare_deep_dive.ipynb` — interactive answers to all four CFO questions with inline charts.

---

## Key Findings

| Question | Finding |
|---|---|
| Q1 — Recovery by insurance | Medicare lowest (93.94%); Corporate highest (94.38%); spread is only 0.44 pp — differences are narrow |
| Q1 — Outstanding balance | All outstanding balances = $0 in this dataset — accounts are either fully recovered or written off |
| Q2 — Age and billing | age_from_dob has negligible correlation with lifetime_billed (r=0.013) and bill_collection_rate (r=−0.003) |
| Q3 — Chi-square test | **p = 0.039 — SIGNIFICANT**: insurance type is not independent of billing outcome; it predicts whether bills get collected |
| Q4 — Low-recovery outliers | **384 patients (7.9%)** flagged below the IQR fence of 76.8%; mean collection rate 59% — priority accounts for follow-up |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run.py

# Run unit tests
python tests/test_eda.py
```

---

## Project Structure

```
healthcare-billing-eda/
├── data/
│   └── processed-data.csv          # 4,860 patient billing records
├── reports/
│   ├── analysis_report.txt         # Full EDA text report
│   ├── segment_profile.csv         # Billing profile by insurance type
│   ├── anomalies.csv               # Confirmed statistical outliers
│   └── plots/                      # 5 PNG charts
├── notebooks/
│   └── healthcare_deep_dive.ipynb  # Interactive Q1–Q4 analysis
├── src/
│   ├── eda_engine.py               # EDAEngine + HealthcareEDAEngine
│   ├── anomaly_detector.py         # AnomalyDetector with flag_low_recovery
│   ├── patient_segment_profiler.py # PatientSegmentProfiler
│   └── visualiser.py               # Visualiser — 5 charts
├── tests/
│   └── test_eda.py                 # 25 unit tests
├── config.py
└── run.py                          # Pipeline entry point
```

---

## Dataset Reference

| Column | Type | Description |
|---|---|---|
| `patient_id` | int | Unique patient identifier |
| `date_of_birth` | date | Used to derive `age_from_dob` |
| `gender` | category | Male / Female / Non-binary |
| `city` | category | Patient city |
| `insurance_type` | category | Private / Medicare / Medicaid / Uninsured / Corporate |
| `num_bills` | float | Total number of billing events |
| `lifetime_billed` | float | Total amount billed in USD |
| `paid_by_patient` | float | Amount paid directly by patient |
| `paid_by_insurance` | float | Amount covered by insurer |
| `recovery_rate_pct` | float | Percentage of billed amount recovered (1.7–100) |
| `outstanding_balance` | float | Remaining unpaid balance |
| `bill_collection_rate` | float | Collection efficiency score (0–100) |
| `age_from_dob` | int | Patient age in years |

---

## Tests

25 unit tests covering profile, group analysis, correlation, chi-square, recovery by group, age correlation, PatientSegmentProfiler, and all AnomalyDetector methods including `flag_low_recovery`.

```bash
python tests/test_eda.py
# All tests passed ✓
```

---

## Outputs

| File | Description |
|---|---|
| `reports/analysis_report.txt` | Full structured EDA report with chi-square verdict |
| `reports/segment_profile.csv` | Mean/median/std of billing metrics per insurance type |
| `reports/anomalies.csv` | 115 confirmed statistical outliers |
| `reports/plots/` | 5 PNG charts |
| `notebooks/healthcare_deep_dive.ipynb` | Interactive CFO-level notebook |
