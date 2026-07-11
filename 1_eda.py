"""
=============================================================
FILE 1: EXPLORATORY DATA ANALYSIS (EDA)
Project  : Credit Line Optimisation
Dataset  : UCI Default of Credit Card Clients (30,000 rows)
Author   : Shivansh Shukla | IIT Gandhinagar | ICICI Bank Internship
=============================================================

WHAT THIS FILE DOES:
- Loads and cleans the raw dataset
- Analyses distributions of all key variables
- Checks for missing values, outliers, and class imbalance
- Visualises payment behaviour, utilisation patterns, and default rates
- Saves cleaned data for use in feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("=" * 65)
print("SECTION 1 — LOADING DATA")
print("=" * 65)

df = pd.read_excel(
    "default of credit card clients.xls",
    engine="xlrd",
    header=1          # Row 1 is the real header (row 0 is a label row)
)

# Rename target column for convenience
df.rename(columns={"default payment next month": "DEFAULT"}, inplace=True)

print(f"Dataset shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns         : {list(df.columns)}")


# =============================================================================
# SECTION 2: DATA QUALITY CHECK
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 2 — DATA QUALITY")
print("=" * 65)

# Missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing[missing > 0] if missing.sum() > 0 else 'None — dataset is complete'}")

# Duplicate rows
dupes = df.duplicated().sum()
print(f"\nDuplicate rows  : {dupes}")

# Drop ID — not a feature
df.drop(columns=["ID"], inplace=True)

# ── Clean known data quality issues ──────────────────────────────────────────
# EDUCATION: values 0, 5, 6 are undocumented — map to 'Other' (4)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

# MARRIAGE: value 0 is undocumented — map to 'Other' (3)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("\nAfter cleaning EDUCATION unique values :", sorted(df["EDUCATION"].unique()))
print("After cleaning MARRIAGE  unique values :", sorted(df["MARRIAGE"].unique()))

# ── Basic stats ───────────────────────────────────────────────────────────────
print("\n--- Numeric summary (selected columns) ---")
cols_summary = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
print(df[cols_summary].describe().round(2).to_string())


# =============================================================================
# SECTION 3: TARGET VARIABLE — DEFAULT
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3 — TARGET VARIABLE (DEFAULT)")
print("=" * 65)

default_counts = df["DEFAULT"].value_counts()
default_pct    = df["DEFAULT"].value_counts(normalize=True) * 100

print(f"\nDefault = 0 (No default)  : {default_counts[0]:,}  ({default_pct[0]:.1f}%)")
print(f"Default = 1 (Defaulted)   : {default_counts[1]:,}  ({default_pct[1]:.1f}%)")
print(f"\nClass imbalance ratio     : {default_counts[0]/default_counts[1]:.1f}:1")
print("NOTE: This imbalance must be handled in model training (SMOTE or class weights)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Target Variable — Default Distribution", fontsize=13, fontweight='bold')

axes[0].bar(["No Default (0)", "Default (1)"],
            [default_counts[0], default_counts[1]],
            color=["#2196F3", "#F44336"], edgecolor='white', linewidth=1.2)
axes[0].set_title("Count")
axes[0].set_ylabel("Number of Customers")
for bar, val in zip(axes[0].patches, [default_counts[0], default_counts[1]]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:,}", ha='center', fontsize=10)

axes[1].pie([default_pct[0], default_pct[1]],
            labels=[f"No Default\n{default_pct[0]:.1f}%", f"Default\n{default_pct[1]:.1f}%"],
            colors=["#2196F3", "#F44336"], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png")
plt.close()
print("Plot saved: 01_target_distribution.png")


# =============================================================================
# SECTION 4: CREDIT LIMIT DISTRIBUTION
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4 — CREDIT LIMIT ANALYSIS")
print("=" * 65)

print(f"\nCredit limit stats:")
print(f"  Min     : {df['LIMIT_BAL'].min():>12,.0f}")
print(f"  Median  : {df['LIMIT_BAL'].median():>12,.0f}")
print(f"  Mean    : {df['LIMIT_BAL'].mean():>12,.0f}")
print(f"  Max     : {df['LIMIT_BAL'].max():>12,.0f}")
print(f"  Std Dev : {df['LIMIT_BAL'].std():>12,.0f}")

# Avg limit by default status
avg_limit_by_default = df.groupby("DEFAULT")["LIMIT_BAL"].mean()
print(f"\nAvg credit limit — Non-defaulters : {avg_limit_by_default[0]:,.0f}")
print(f"Avg credit limit — Defaulters     : {avg_limit_by_default[1]:,.0f}")
print("INSIGHT: Defaulters tend to have lower initial limits (higher-risk customers)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Credit Limit Distribution", fontsize=13, fontweight='bold')

axes[0].hist(df["LIMIT_BAL"] / 1000, bins=50, color="#1565C0", edgecolor='white', linewidth=0.5)
axes[0].set_xlabel("Credit Limit (thousands)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Overall Distribution")

for label, color in [(0, "#2196F3"), (1, "#F44336")]:
    subset = df[df["DEFAULT"] == label]["LIMIT_BAL"] / 1000
    axes[1].hist(subset, bins=40, alpha=0.6, color=color,
                 label=f"Default={label}", edgecolor='white', linewidth=0.3)
axes[1].legend()
axes[1].set_xlabel("Credit Limit (thousands)")
axes[1].set_title("By Default Status")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_credit_limit_distribution.png")
plt.close()
print("Plot saved: 02_credit_limit_distribution.png")


# =============================================================================
# SECTION 5: PAYMENT BEHAVIOUR (PAY columns = DPD proxy)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5 — PAYMENT BEHAVIOUR (DPD PROXY)")
print("=" * 65)

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

print("\nPAY column value guide:")
print("  -2 = No consumption that month")
print("  -1 = Paid in full (on time)")
print("   0 = Revolving credit (minimum paid)")
print("   1 = 1-month payment delay  (DPD 1-30)")
print("   2 = 2-month payment delay  (DPD 31-60)")
print("   3+ = Increasingly severe delay")

# Default rate by PAY_0 value (most recent month)
# Merge -2, -1, 0 into one "paid duly" category because they all mean no bad payment delay
def map_pay_status(val):
    if val in (-2, -1, 0):
        return "paid duly"
    if val == 1:
        return "1-month delay"
    if val == 2:
        return "2-month delay"
    return "3-month delay"

pay_group = df["PAY_0"].apply(map_pay_status)
categories = [
    "paid duly",
    "1-month delay",
    "2-month delay",
    "3-month delay",
]

pay_group_cat = pd.Categorical(pay_group, categories=categories, ordered=True)
default_by_pay = df.groupby(pay_group_cat)["DEFAULT"].mean().reset_index()
default_by_pay.columns = ["PAY_0_group", "default_rate"]
print(f"\nDefault rate by PAY_0 group (most recent payment status):")
print(default_by_pay.round(3).to_string(index=False))
print("\nINSIGHT: Default rate rises sharply as delay worsens")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Payment Behaviour Analysis", fontsize=13, fontweight='bold')

axes[0].bar(default_by_pay["PAY_0_group"],
            default_by_pay["default_rate"] * 100,
            color="#E53935", edgecolor='white', linewidth=1)
axes[0].set_xlabel("PAY_0 Group")
axes[0].set_ylabel("Default Rate (%)")
axes[0].set_title("Default Rate by Payment Status")
axes[0].axhline(y=default_pct[1], color='black', linestyle='--', linewidth=1,
                label=f"Overall default rate ({default_pct[1]:.1f}%)")
axes[0].legend()
axes[0].set_xticks(range(len(categories)))
axes[0].set_xticklabels(categories, rotation=45, ha='right')

# Proportion of each pay status group
pay_dist = pay_group_cat.value_counts().reindex(categories).fillna(0)
axes[1].bar(range(len(categories)), pay_dist.values,
            color="#1565C0", edgecolor='white', linewidth=1)
axes[1].set_xlabel("PAY_0 Group")
axes[1].set_ylabel("Number of Customers")
axes[1].set_title("Distribution of PAY_0 Groups (Most Recent)")
axes[1].set_xticks(range(len(categories)))
axes[1].set_xticklabels(categories, rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_payment_behaviour.png")
plt.close()
print("Plot saved: 03_payment_behaviour.png")


# =============================================================================
# SECTION 6: UTILISATION RATIO
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 6 — UTILISATION RATIO")
print("=" * 65)

# Compute utilisation for each month
for i, bill_col in enumerate(["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                               "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"], 1):
    df[f"UTIL_M{i}"] = (df[bill_col] / df["LIMIT_BAL"]).clip(0, 1)

df["UTIL_AVG"] = df[["UTIL_M1", "UTIL_M2", "UTIL_M3",
                      "UTIL_M4", "UTIL_M5", "UTIL_M6"]].mean(axis=1)

util_by_default = df.groupby("DEFAULT")["UTIL_AVG"].describe()
print(f"\nAvg utilisation stats by default status:")
print(util_by_default[["mean", "50%", "75%", "max"]].round(3).to_string())
print("\nINSIGHT: Defaulters have significantly higher average utilisation")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Utilisation Ratio Analysis", fontsize=13, fontweight='bold')

for label, color in [(0, "#2196F3"), (1, "#F44336")]:
    subset = df[df["DEFAULT"] == label]["UTIL_AVG"]
    axes[0].hist(subset, bins=50, alpha=0.6, color=color,
                 label=f"Default={label}", edgecolor='white', linewidth=0.3)
axes[0].legend()
axes[0].set_xlabel("Average Utilisation Ratio (6 months)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution by Default Status")

# Utilisation bands and default rate
bins   = [0, 0.1, 0.3, 0.6, 0.8, 1.0]
labels = ["0–10%", "10–30%", "30–60%", "60–80%", "80–100%"]
df["UTIL_BAND"] = pd.cut(df["UTIL_AVG"], bins=bins, labels=labels, include_lowest=True)
util_default = df.groupby("UTIL_BAND")["DEFAULT"].mean() * 100

axes[1].bar(util_default.index.astype(str), util_default.values,
            color=["#66BB6A", "#26A69A", "#FFA726", "#EF5350", "#B71C1C"],
            edgecolor='white', linewidth=1)
axes[1].set_xlabel("Utilisation Band")
axes[1].set_ylabel("Default Rate (%)")
axes[1].set_title("Default Rate by Utilisation Band")
for i, v in enumerate(util_default.values):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_utilisation_analysis.png")
plt.close()
print("Plot saved: 04_utilisation_analysis.png")


# =============================================================================
# SECTION 7: PAYMENT RATIO (how much of bill is actually paid)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 7 — PAYMENT RATIO (TRANSACTOR vs REVOLVER SIGNAL)")
print("=" * 65)

# Payment ratio = amount paid / bill amount (clipped 0–1)
for i, (pay_col, bill_col) in enumerate(
        zip(["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
            ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]), 1):
    # Avoid division by zero — if bill = 0, ratio = 1 (nothing owed, fully "paid")
    df[f"PAY_RATIO_M{i}"] = np.where(
        df[bill_col] <= 0, 1.0,
        (df[pay_col] / df[bill_col]).clip(0, 1)
    )

df["PAY_RATIO_AVG"] = df[["PAY_RATIO_M1", "PAY_RATIO_M2", "PAY_RATIO_M3",
                           "PAY_RATIO_M4", "PAY_RATIO_M5", "PAY_RATIO_M6"]].mean(axis=1)

pay_ratio_stats = df.groupby("DEFAULT")["PAY_RATIO_AVG"].describe()
print(f"\nAvg payment ratio stats by default status:")
print(pay_ratio_stats[["mean", "50%", "25%"]].round(3).to_string())
print("\nINSIGHT: Non-defaulters pay a much higher fraction of their bill on average")


# =============================================================================
# SECTION 8: DEMOGRAPHIC ANALYSIS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 8 — DEMOGRAPHICS")
print("=" * 65)

edu_map  = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
mar_map  = {1: "Married", 2: "Single", 3: "Other"}
sex_map  = {1: "Male", 2: "Female"}

df["EDU_LABEL"] = df["EDUCATION"].map(edu_map)
df["MAR_LABEL"] = df["MARRIAGE"].map(mar_map)
df["SEX_LABEL"] = df["SEX"].map(sex_map)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Default Rate by Demographics", fontsize=13, fontweight='bold')

for ax, col, title in zip(axes,
                           ["EDU_LABEL", "MAR_LABEL", "SEX_LABEL"],
                           ["Education", "Marital Status", "Gender"]):
    dr = df.groupby(col)["DEFAULT"].mean() * 100
    ax.bar(dr.index, dr.values, color="#1565C0", edgecolor='white', linewidth=1)
    ax.set_ylabel("Default Rate (%)")
    ax.set_title(f"Default Rate by {title}")
    ax.axhline(y=default_pct[1], color='red', linestyle='--', linewidth=1,
               label=f"Overall ({default_pct[1]:.1f}%)")
    ax.legend(fontsize=8)
    for i, v in enumerate(dr.values):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_demographic_analysis.png")
plt.close()
print("Plot saved: 05_demographic_analysis.png")

# Print default rates
for col, label in [("EDU_LABEL", "Education"), ("MAR_LABEL", "Marital"), ("SEX_LABEL", "Gender")]:
    print(f"\nDefault rate by {label}:")
    print((df.groupby(col)["DEFAULT"].mean() * 100).round(2).to_string())


# =============================================================================
# SECTION 9: CORRELATION HEATMAP
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 9 — CORRELATIONS WITH DEFAULT")
print("=" * 65)

key_cols = (["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
              "UTIL_M1", "UTIL_AVG", "PAY_RATIO_AVG", "DEFAULT"])
corr = df[key_cols].corr()["DEFAULT"].drop("DEFAULT").sort_values()

print("\nCorrelation with DEFAULT (sorted):")
print(corr.round(3).to_string())
print("\nINSIGHT: PAY_0 (recent payment status) and utilisation are most correlated with default")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#F44336" if v > 0 else "#2196F3" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, edgecolor='white', linewidth=1)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel("Pearson Correlation with DEFAULT")
ax.set_title("Feature Correlation with Default — Key Variables", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_correlation_with_default.png")
plt.close()
print("Plot saved: 06_correlation_with_default.png")


# =============================================================================
# SECTION 10: SAVE CLEANED DATA
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 10 — SAVING CLEANED DATA")
print("=" * 65)

# Drop temp label columns before saving
df.drop(columns=["EDU_LABEL", "MAR_LABEL", "SEX_LABEL", "UTIL_BAND"], inplace=True)

df.to_csv("cleaned_data.csv", index=False)
print(f"\nCleaned dataset saved  : cleaned_data.csv")
print(f"Rows                   : {df.shape[0]:,}")
print(f"Columns                : {df.shape[1]}")
print(f"\nAdded columns during EDA:")
print("  UTIL_M1 to UTIL_M6  — monthly utilisation ratio")
print("  UTIL_AVG            — 6-month average utilisation")
print("  PAY_RATIO_M1 to M6  — monthly payment ratio")
print("  PAY_RATIO_AVG       — 6-month average payment ratio")

print("\n" + "=" * 65)
print("EDA COMPLETE — All plots saved to outputs/")
print("Next step: Run 2_feature_engineering.py")
print("=" * 65)
