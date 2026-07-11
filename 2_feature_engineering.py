"""
=============================================================
FILE 2: FEATURE ENGINEERING
Project  : Credit Line Optimisation
Dataset  : UCI Default of Credit Card Clients
Author   : Shivansh Shukla | IIT Gandhinagar | ICICI Bank Internship
=============================================================

WHAT THIS FILE DOES:
- Loads cleaned data from EDA step
- Engineers all features needed for the model
- Creates the TARGET VARIABLE: Increase / Decrease / Hold
- Saves final feature matrix ready for model training

TARGET VARIABLE LOGIC:
  INCREASE  — Customer is safe and under-utilising limit
              → reward them, bank earns more interchange + interest
  DECREASE  — Customer showing stress signals
              → reduce exposure before they default (lower EAD)
  HOLD      — Everything else — insufficient signal to act

BUSINESS RULE OVERRIDE:
  If the model says INCREASE but the customer actually defaulted
  (DEFAULT = 1), that decision is WRONG.
  We track this as a business accuracy metric.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: LOAD CLEANED DATA
# =============================================================================
print("=" * 65)
print("SECTION 1 — LOADING CLEANED DATA")
print("=" * 65)

df = pd.read_csv("cleaned_data.csv")
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


# =============================================================================
# SECTION 2: CORE DERIVED FEATURES
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 2 — CORE DERIVED FEATURES")
print("=" * 65)

pay_cols      = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols     = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
payamt_cols   = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
util_cols     = ["UTIL_M1", "UTIL_M2", "UTIL_M3", "UTIL_M4", "UTIL_M5", "UTIL_M6"]
payratio_cols = ["PAY_RATIO_M1", "PAY_RATIO_M2", "PAY_RATIO_M3",
                 "PAY_RATIO_M4", "PAY_RATIO_M5", "PAY_RATIO_M6"]

# ── 2.1 DPD-based features ─────────────────────────────────────────────────
# PAY_n value > 0 means payment delay in months — this is our DPD proxy
# PAY_n = -1: paid on time | 0: revolving | 1+: delay in months

df["DPD_MAX_6M"]        = df[pay_cols].apply(lambda r: max(r.clip(lower=0)), axis=1)
df["DPD_MAX_3M"]        = df[["PAY_0","PAY_2","PAY_3"]].apply(lambda r: max(r.clip(lower=0)), axis=1)
df["DPD_MONTHS_30PLUS"] = df[pay_cols].apply(lambda r: (r >= 1).sum(), axis=1)
df["DPD_MONTHS_60PLUS"] = df[pay_cols].apply(lambda r: (r >= 2).sum(), axis=1)
df["EVER_DPD_90PLUS"]   = (df[pay_cols].apply(lambda r: (r >= 3).sum(), axis=1) > 0).astype(int)
df["DPD_CURRENT"]       = df["PAY_0"].clip(lower=0)

print("DPD features created:")
print(f"  DPD_MAX_6M        : {df['DPD_MAX_6M'].describe()['mean']:.2f} average")
print(f"  EVER_DPD_90PLUS   : {df['EVER_DPD_90PLUS'].mean()*100:.1f}% of customers")

# ── 2.2 Utilisation features ───────────────────────────────────────────────
df["UTIL_CURRENT"]  = df["UTIL_M1"]                    # most recent month
df["UTIL_3M_AVG"]   = df[["UTIL_M1","UTIL_M2","UTIL_M3"]].mean(axis=1)
df["UTIL_6M_AVG"]   = df[util_cols].mean(axis=1)
df["UTIL_MAX_3M"]   = df[["UTIL_M1","UTIL_M2","UTIL_M3"]].max(axis=1)

# Trend: positive = utilisation RISING (bad), negative = FALLING (good)
df["UTIL_TREND"]    = df["UTIL_M1"] - df["UTIL_M3"]    # recent minus 3 months ago
df["UTIL_RISING"]   = (df["UTIL_TREND"] > 0.05).astype(int)  # flag: rising significantly

print("\nUtilisation features created:")
print(f"  UTIL_CURRENT avg  : {df['UTIL_CURRENT'].mean():.2f}")
print(f"  UTIL_RISING flag  : {df['UTIL_RISING'].mean()*100:.1f}% of customers have rising util")

# ── 2.3 Payment ratio features ─────────────────────────────────────────────
df["PAY_RATIO_CURRENT"] = df["PAY_RATIO_M1"]
df["PAY_RATIO_3M_AVG"]  = df[["PAY_RATIO_M1","PAY_RATIO_M2","PAY_RATIO_M3"]].mean(axis=1)
df["PAY_RATIO_6M_AVG"]  = df[payratio_cols].mean(axis=1)

# Min pay streak: consecutive months where payment ratio < 10% (paying minimum only)
def min_pay_streak(row):
    """Count max consecutive months of near-minimum payment"""
    ratios = [row[c] for c in payratio_cols]
    streak, max_streak = 0, 0
    for r in ratios:
        if r < 0.10:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

df["MIN_PAY_STREAK"] = df.apply(min_pay_streak, axis=1)

print("\nPayment ratio features created:")
print(f"  PAY_RATIO_6M_AVG avg      : {df['PAY_RATIO_6M_AVG'].mean():.2f}")
print(f"  MIN_PAY_STREAK (>= 3 mo)  : {(df['MIN_PAY_STREAK'] >= 3).mean()*100:.1f}% of customers")

# ── 2.4 Balance & spend features ──────────────────────────────────────────
df["BALANCE_CURRENT"]   = df["BILL_AMT1"]
df["BALANCE_6M_AVG"]    = df[bill_cols].mean(axis=1)
df["TOTAL_PAY_6M"]      = df[payamt_cols].sum(axis=1)
df["AVG_PAY_6M"]        = df[payamt_cols].mean(axis=1)

# Balance trend: positive = balance growing (potential stress)
df["BALANCE_TREND"]     = df["BILL_AMT1"] - df["BILL_AMT3"]
df["BALANCE_GROWING"]   = (df["BALANCE_TREND"] > 5000).astype(int)

# Cash advance proxy: large negative bill movements suggest cash withdrawal
# (not directly in dataset but approximated by gap between payments and bills)

# ── 2.5 Limit features ────────────────────────────────────────────────────
df["LIMIT_BAND"] = pd.cut(
    df["LIMIT_BAL"],
    bins=[0, 50000, 100000, 200000, 500000, float('inf')],
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

print("\nDemographic features — keeping as-is:")
print("  LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE")


# =============================================================================
# SECTION 3: CREATE TARGET VARIABLE
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3 — CREATING TARGET VARIABLE")
print("=" * 65)

print("""
TARGET VARIABLE: LIMIT_ACTION
  0 = HOLD      — no change in limit
  1 = INCREASE  — customer qualifies for a limit increase
  2 = DECREASE  — customer shows stress, limit should be reduced

BUSINESS RULE: A limit INCREASE recommended for a customer who
subsequently defaulted (DEFAULT=1) is a BAD decision.
The model accuracy check in File 3 will specifically flag this.
""")

def assign_limit_action(row):
    """
    Rule-based target variable creation.

    DECREASE triggers (any one sufficient):
    - DPD >= 2 months in any of last 6 months
    - Utilisation > 80% AND rising trend
    - Min pay streak >= 3 months
    - Balance growing AND DPD > 0

    INCREASE triggers (ALL must be true):
    - No DPD in last 6 months (DPD_MAX = 0)
    - Avg utilisation 10–60% (active but not stressed)
    - Payment ratio avg >= 0.50 (paying at least half)
    - Utilisation not sharply rising

    HOLD: everything else
    """
    # ── DECREASE conditions ─────────────────────────────────────────────
    if row["DPD_MAX_6M"] >= 2:
        return 2
    if row["UTIL_CURRENT"] > 0.80 and row["UTIL_RISING"] == 1:
        return 2
    if row["MIN_PAY_STREAK"] >= 3:
        return 2
    if row["BALANCE_GROWING"] == 1 and row["DPD_CURRENT"] > 0:
        return 2

    # ── INCREASE conditions ─────────────────────────────────────────────
    if (row["DPD_MAX_6M"] == 0
            and 0.10 <= row["UTIL_6M_AVG"] <= 0.60
            and row["PAY_RATIO_6M_AVG"] >= 0.50
            and row["UTIL_RISING"] == 0):
        return 1

    # ── HOLD ─────────────────────────────────────────────────────────────
    return 0

print("Assigning limit actions (this may take ~30 seconds)...")
df["LIMIT_ACTION"] = df.apply(assign_limit_action, axis=1)

action_map   = {0: "HOLD", 1: "INCREASE", 2: "DECREASE"}
action_counts = df["LIMIT_ACTION"].value_counts().sort_index()
action_pct    = df["LIMIT_ACTION"].value_counts(normalize=True).sort_index() * 100

print("\nLimit action distribution:")
for k in [0, 1, 2]:
    print(f"  {action_map[k]:<10}: {action_counts[k]:>6,}  ({action_pct[k]:.1f}%)")


# =============================================================================
# SECTION 4: BUSINESS ACCURACY CHECK ON TARGET VARIABLE
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4 — BUSINESS ACCURACY CHECK")
print("=" * 65)

print("""
KEY BUSINESS QUESTION:
Among customers labelled INCREASE, how many actually defaulted?
This would be a WRONG decision — you raised the limit of someone
who couldn't repay. This increases EAD and is a direct financial loss.
""")

increase_mask = df["LIMIT_ACTION"] == 1
increase_defaulted = (df.loc[increase_mask, "DEFAULT"] == 1).sum()
increase_total     = increase_mask.sum()
increase_safe      = increase_total - increase_defaulted

print(f"  Total INCREASE recommendations : {increase_total:,}")
print(f"  Of these — customer safe (0)   : {increase_safe:,}  ({increase_safe/increase_total*100:.1f}%) ✓ CORRECT")
print(f"  Of these — customer defaulted  : {increase_defaulted:,}  ({increase_defaulted/increase_total*100:.1f}%) ✗ WRONG")

decrease_mask       = df["LIMIT_ACTION"] == 2
decrease_defaulted  = (df.loc[decrease_mask, "DEFAULT"] == 1).sum()
decrease_total      = decrease_mask.sum()
print(f"\n  Total DECREASE recommendations : {decrease_total:,}")
print(f"  Of these — customer defaulted  : {decrease_defaulted:,}  ({decrease_defaulted/decrease_total*100:.1f}%)")
print("  NOTE: High default rate in DECREASE group confirms our labels are correct")

# Cross-tab: action vs actual default
print("\nCross-tab — Limit Action vs Actual Default:")
ct = pd.crosstab(
    df["LIMIT_ACTION"].map(action_map),
    df["DEFAULT"].map({0: "No Default", 1: "Defaulted"}),
    margins=True
)
print(ct.to_string())

# Visual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Target Variable — Limit Action Analysis", fontsize=13, fontweight='bold')

axes[0].bar([action_map[k] for k in [0,1,2]],
            [action_counts[k] for k in [0,1,2]],
            color=["#78909C", "#43A047", "#E53935"], edgecolor='white', linewidth=1)
axes[0].set_ylabel("Number of Customers")
axes[0].set_title("Distribution of Limit Actions")
for i, (k, v) in enumerate([(k, action_counts[k]) for k in [0,1,2]]):
    axes[0].text(i, v + 50, f"{v:,}\n({action_pct[k]:.1f}%)", ha='center', fontsize=9)

# Default rate within each action group
dr_by_action = df.groupby("LIMIT_ACTION")["DEFAULT"].mean() * 100
axes[1].bar([action_map[k] for k in [0,1,2]],
            [dr_by_action[k] for k in [0,1,2]],
            color=["#78909C", "#43A047", "#E53935"], edgecolor='white', linewidth=1)
axes[1].set_ylabel("Actual Default Rate (%)")
axes[1].set_title("Default Rate Within Each Action Group")
axes[1].axhline(y=df["DEFAULT"].mean()*100, color='black', linestyle='--', linewidth=1,
                label=f"Overall ({df['DEFAULT'].mean()*100:.1f}%)")
axes[1].legend()
for i, k in enumerate([0,1,2]):
    axes[1].text(i, dr_by_action[k] + 0.3, f"{dr_by_action[k]:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_target_variable_analysis.png")
plt.close()
print("\nPlot saved: 07_target_variable_analysis.png")


# =============================================================================
# SECTION 5: FINAL FEATURE SET
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5 — FINAL FEATURE SET FOR MODELLING")
print("=" * 65)

FEATURE_COLS = [
    # Demographics
    "AGE", "SEX", "EDUCATION", "MARRIAGE",
    # Credit profile
    "LIMIT_BAL",
    # DPD / payment delay features
    "DPD_CURRENT", "DPD_MAX_6M", "DPD_MAX_3M",
    "DPD_MONTHS_30PLUS", "DPD_MONTHS_60PLUS", "EVER_DPD_90PLUS",
    # Utilisation features
    "UTIL_CURRENT", "UTIL_3M_AVG", "UTIL_6M_AVG",
    "UTIL_MAX_3M", "UTIL_TREND", "UTIL_RISING",
    # Payment ratio features
    "PAY_RATIO_CURRENT", "PAY_RATIO_3M_AVG", "PAY_RATIO_6M_AVG",
    "MIN_PAY_STREAK",
    # Balance features
    "BALANCE_CURRENT", "BALANCE_6M_AVG",
    "TOTAL_PAY_6M", "AVG_PAY_6M",
    "BALANCE_TREND", "BALANCE_GROWING",
]

TARGET_COL   = "LIMIT_ACTION"

print(f"\nTotal features selected : {len(FEATURE_COLS)}")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2}. {col}")

# Check no nulls in final feature set
nulls = df[FEATURE_COLS].isnull().sum()
print(f"\nNull values in features : {nulls.sum()} (should be 0)")

# Save final dataset
final_df = df[FEATURE_COLS + [TARGET_COL, "DEFAULT"]].copy()
final_df.to_csv("features.csv", index=False)

print(f"\nFinal feature dataset saved : features.csv")
print(f"Shape : {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
print(f"  Features  : {len(FEATURE_COLS)}")
print(f"  Target    : {TARGET_COL}")
print(f"  Extra     : DEFAULT (for business accuracy check in model)")

print("\n" + "=" * 65)
print("FEATURE ENGINEERING COMPLETE")
print("Next step: Run 3_model.py")
print("=" * 65)
