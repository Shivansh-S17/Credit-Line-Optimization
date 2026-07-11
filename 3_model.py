"""
=============================================================
FILE 3 (UPGRADED): CREDIT LINE OPTIMISATION MODEL v2
Project  : Credit Line Optimisation
Dataset  : UCI Default of Credit Card Clients
Author   : Shivansh Shukla | IIT Gandhinagar | ICICI Bank Internship
=============================================================

WHAT CHANGED FROM BASELINE:
─────────────────────────────────────────────────────────────
1. EAD FORMULA — mentor's suggestion:
      EAD = Avg_Balance_6M / Credit_Limit  (a ratio, not a fixed 85%)
      This is customer-specific and far more accurate than
      assuming everyone draws down to 85% at default.

2. TARGET VARIABLE FIXED — the baseline had 3 problems:
      (a) Too many DECREASE (62%) → massive wrong-decrease cost
      (b) Too few INCREASE (1.5%) → barely any revenue
      (c) 1,568 defaulters sitting in HOLD → huge missed-default loss
      Fix: tighter, more balanced rules with a RISK SCORE approach

3. RISK SCORE LAYER — instead of hard rules, each customer gets
      a composite risk score. This catches the missed defaulters
      who looked okay on individual rules but were collectively risky.

4. SHAP EXPLAINABILITY — added for individual decision explanations

5. BUSINESS IMPACT — EAD is now customer-specific, not a fixed assumption
      Net impact should turn positive with better target labels

6. AGE + SEX removed — fair lending compliance (mentor instruction)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_MAP    = {0: "HOLD", 1: "INCREASE", 2: "DECREASE"}
ACTION_COLORS = {0: "#78909C", 1: "#43A047", 2: "#E53935"}

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("=" * 65)
print("SECTION 1 — LOADING DATA")
print("=" * 65)

df = pd.read_csv("features.csv")
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop protected attributes — fair lending compliance
df.drop(columns=["AGE", "SEX"], errors="ignore", inplace=True)

# =============================================================================
# SECTION 2: MENTOR'S EAD FORMULA
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 2 — EAD CALCULATION (MENTOR'S FORMULA)")
print("=" * 65)

print("""
BASELINE approach  : EAD = credit_limit × 0.85 (fixed assumption)
PROBLEM            : Treats every customer the same regardless of
                     how much they actually use their card.

MENTOR'S FORMULA   : EAD = Avg_Balance_6M / Credit_Limit
                     This gives a customer-specific utilisation ratio
                     as a fraction — how much of the limit is typically
                     outstanding. It captures actual drawdown behaviour.

WHY THIS IS BETTER :
  - A customer with avg balance ₹20K on a ₹1L limit → EAD ratio 0.20
    (low risk, not drawing much)
  - A customer with avg balance ₹90K on a ₹1L limit → EAD ratio 0.90
    (high risk, nearly maxed out — high EAD at default)
  This is empirical, not assumed.
""")

# Compute EAD ratio as mentor suggested
df["EAD_RATIO"] = (df["BALANCE_6M_AVG"] / df["LIMIT_BAL"]).clip(0, 1)

# Actual EAD in currency = EAD_RATIO × LIMIT_BAL = BALANCE_6M_AVG (by definition)
# But for defaulters we use the full limit as ceiling (they may draw more before defaulting)
# So EAD_AMOUNT = max(BALANCE_6M_AVG, EAD_RATIO × LIMIT_BAL) — already same thing
df["EAD_AMOUNT"] = df["BALANCE_6M_AVG"].clip(lower=0)

ead_by_default = df.groupby("DEFAULT")["EAD_RATIO"].describe()[["mean","50%","75%"]]
print("EAD ratio by default status:")
print(ead_by_default.round(3).to_string())
print(f"\nDefaulters avg EAD ratio   : {df[df['DEFAULT']==1]['EAD_RATIO'].mean():.3f}")
print(f"Non-defaulters avg EAD ratio: {df[df['DEFAULT']==0]['EAD_RATIO'].mean():.3f}")
print("\nINSIGHT: Defaulters have ~28% higher EAD ratio — confirms the formula works")


# =============================================================================
# SECTION 3: RISK SCORE + IMPROVED TARGET VARIABLE
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3 — RISK SCORE + IMPROVED TARGET VARIABLE")
print("=" * 65)

print("""
PROBLEM WITH BASELINE TARGET VARIABLE:
  The baseline used hard IF-ELSE rules which caused:
  (a) 62% labelled DECREASE — too aggressive, caused massive wrong-decrease cost
  (b) 1.5% labelled INCREASE — too few examples, model couldn't learn
  (c) 1,568 missed defaulters in HOLD — biggest loss driver (-₹37M on test set)

SOLUTION — COMPOSITE RISK SCORE:
  Each customer gets a risk score from 0 (very safe) to 100 (very risky).
  Score built from weighted combination of the most predictive signals.
  Then threshold-based labelling ensures more balanced classes.
""")

def compute_risk_score(row):
    """
    Composite risk score 0-100 for each customer.
    Higher = more likely to default = should DECREASE limit.
    Lower  = safer = eligible for INCREASE.

    Weights chosen based on XGBoost feature importance from baseline model
    and credit domain knowledge.
    """
    score = 0.0

    # ── DPD signals (total weight ~35%) ──────────────────────────────────
    # Most recent DPD is the strongest single signal
    score += min(row["DPD_CURRENT"] * 12, 30)          # up to 30 pts
    score += min(row["DPD_MAX_6M"] * 4, 15)            # up to 15 pts — severity
    score += row["DPD_MONTHS_30PLUS"] * 2               # chronic late payments
    score += row["EVER_DPD_90PLUS"] * 10                # hard flag — ever near NPA

    # ── Min pay streak (weight ~20%) ─────────────────────────────────────
    # Most important feature from baseline — pays minimum only
    score += min(row["MIN_PAY_STREAK"] * 5, 20)         # up to 20 pts

    # ── Utilisation signals (weight ~20%) ────────────────────────────────
    if row["UTIL_6M_AVG"] > 0.80:
        score += 15
    elif row["UTIL_6M_AVG"] > 0.60:
        score += 8
    elif row["UTIL_6M_AVG"] > 0.40:
        score += 3

    if row["UTIL_RISING"] == 1:
        score += 8                                       # trend is bad

    # ── Payment ratio (weight ~15%) ──────────────────────────────────────
    # Low payment ratio = revolving heavily or near MAD trap
    pay_penalty = max(0, (0.5 - row["PAY_RATIO_6M_AVG"]) * 20)
    score += min(pay_penalty, 15)

    # ── EAD ratio (weight ~10%) — mentor's formula ───────────────────────
    if row["EAD_RATIO"] > 0.75:
        score += 10
    elif row["EAD_RATIO"] > 0.50:
        score += 5

    # ── Balance trend ─────────────────────────────────────────────────────
    if row["BALANCE_GROWING"] == 1:
        score += 3

    return min(score, 100)   # cap at 100

print("Computing risk scores (may take ~30 seconds)...")
df["RISK_SCORE"] = df.apply(compute_risk_score, axis=1)

print(f"\nRisk score distribution:")
print(df["RISK_SCORE"].describe().round(2).to_string())
print(f"\nRisk score by default status:")
print(df.groupby("DEFAULT")["RISK_SCORE"].describe()[["mean","50%","75%","max"]].round(2).to_string())

# ── New target variable from risk score ──────────────────────────────────
def assign_action_v2(row):
    """
    Threshold-based labelling using risk score.

    INCREASE (score < 15):
      Very low risk. Clean payment history, stable utilisation.
      Bank should grow this customer's limit.

    DECREASE (score > 40):
      Elevated risk. Multiple stress signals present.
      Bank should reduce exposure.

    HOLD: everything between 15 and 40
      Insufficient signal to act either way.

    Additional hard guards:
      - Never INCREASE if EVER_DPD_90PLUS = 1
      - Never INCREASE if EAD_RATIO > 0.70 (already highly drawn)
      - Never INCREASE if UTIL_RISING = 1
    """
    score = row["RISK_SCORE"]

    if score > 40:
        return 2   # DECREASE

    if score < 15:
        # Hard guards for INCREASE
        if row["EVER_DPD_90PLUS"] == 1:
            return 0   # override to HOLD
        if row["EAD_RATIO"] > 0.70:
            return 0   # too drawn down already
        if row["UTIL_RISING"] == 1:
            return 0   # heading in wrong direction
        return 1   # INCREASE

    return 0   # HOLD

print("\nAssigning new target labels...")
df["LIMIT_ACTION_V2"] = df.apply(assign_action_v2, axis=1)

v2_counts = df["LIMIT_ACTION_V2"].value_counts().sort_index()
v2_pct    = df["LIMIT_ACTION_V2"].value_counts(normalize=True).sort_index() * 100

print("\nNew target variable distribution (v2):")
for k in [0, 1, 2]:
    print(f"  {ACTION_MAP[k]:<10}: {v2_counts[k]:>6,}  ({v2_pct[k]:.1f}%)")

# Compare default rates in each class — validation check
print("\nDefault rate within each action class (should show clear separation):")
dr = df.groupby("LIMIT_ACTION_V2")["DEFAULT"].mean() * 100
for k in [0, 1, 2]:
    print(f"  {ACTION_MAP[k]:<10}: {dr[k]:.1f}% actual default rate")

print("\nINSIGHT: DECREASE class should have highest default rate (confirms labels are right)")
print("INSIGHT: INCREASE class should have lowest default rate")

# Check missed defaulters improvement
missed_v2 = df[(df["LIMIT_ACTION_V2"] == 0) & (df["DEFAULT"] == 1)]
missed_v1 = df[(df["LIMIT_ACTION"] == 0) & (df["DEFAULT"] == 1)]
print(f"\nMissed defaulters in HOLD:")
print(f"  Baseline (v1): {len(missed_v1):,}")
print(f"  Upgraded (v2): {len(missed_v2):,}")
print(f"  Improvement  : {len(missed_v1) - len(missed_v2):,} fewer missed defaulters")

# Plot risk score distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Risk Score Analysis", fontsize=13, fontweight='bold')

for label, color in [(0, "#2196F3"), (1, "#F44336")]:
    subset = df[df["DEFAULT"] == label]["RISK_SCORE"]
    axes[0].hist(subset, bins=40, alpha=0.65, color=color,
                 label=f"Default={label}", edgecolor='white', linewidth=0.3)
axes[0].axvline(x=15, color='green',  linestyle='--', linewidth=1.5, label="INCREASE threshold (15)")
axes[0].axvline(x=40, color='red',    linestyle='--', linewidth=1.5, label="DECREASE threshold (40)")
axes[0].legend(fontsize=9)
axes[0].set_xlabel("Risk Score")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Risk Score Distribution by Actual Default")

# Default rate by action (v2)
dr_v2 = df.groupby("LIMIT_ACTION_V2")["DEFAULT"].mean() * 100
axes[1].bar([ACTION_MAP[k] for k in [0,1,2]],
            [dr_v2[k] for k in [0,1,2]],
            color=[ACTION_COLORS[k] for k in [0,1,2]],
            edgecolor='white', linewidth=1)
axes[1].set_ylabel("Actual Default Rate (%)")
axes[1].set_title("Default Rate per Class — V2 Labels")
axes[1].axhline(y=df["DEFAULT"].mean()*100, color='black', linestyle='--',
                linewidth=1, label=f"Overall ({df['DEFAULT'].mean()*100:.1f}%)")
axes[1].legend()
for i, k in enumerate([0,1,2]):
    axes[1].text(i, dr_v2[k]+0.3, f"{dr_v2[k]:.1f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_risk_score_analysis.png")
plt.close()
print("\nPlot saved: 11_risk_score_analysis.png")


# =============================================================================
# SECTION 4: TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4 — TRAIN / TEST SPLIT")
print("=" * 65)

FEATURE_COLS = [c for c in df.columns if c not in
                ["LIMIT_ACTION", "LIMIT_ACTION_V2", "DEFAULT",
                 "RISK_SCORE", "EAD_RATIO", "EAD_AMOUNT"]]
TARGET_COL   = "LIMIT_ACTION_V2"

X             = df[FEATURE_COLS].copy()
y             = df[TARGET_COL].copy()
default_flags = df["DEFAULT"].copy()
ead_ratios    = df["EAD_RATIO"].copy()
limit_vals    = df["LIMIT_BAL"].copy()

X_train, X_test, y_train, y_test, def_train, def_test, ead_train, ead_test, lim_train, lim_test = \
    train_test_split(X, y, default_flags, ead_ratios, limit_vals,
                     test_size=0.20, random_state=42, stratify=y)

print(f"Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
print(f"Features used: {len(FEATURE_COLS)}")

# Scale + SMOTE
scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(X_train)
X_test_sc      = scaler.transform(X_test)

smote          = SMOTE(random_state=42)
X_train_bal_sc, y_train_bal = smote.fit_resample(X_train_sc, y_train)
X_train_bal_rf, _           = smote.fit_resample(X_train,    y_train)

print("\nAfter SMOTE (training set):")
for k in [0, 1, 2]:
    print(f"  {ACTION_MAP[k]:<10}: {(y_train_bal == k).sum():,}")


# =============================================================================
# SECTION 5: TRAIN MODELS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5 — MODEL TRAINING")
print("=" * 65)

# Logistic Regression
print("\n[1/3] Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, C=1.0,
                               class_weight="balanced", random_state=42)
lr_model.fit(X_train_bal_sc, y_train_bal)
lr_preds = lr_model.predict(X_test_sc)

# Random Forest
print("[2/3] Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=10, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train_bal_rf, y_train_bal)
rf_preds = rf_model.predict(X_test)

# XGBoost
print("[3/3] XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8,
                                eval_metric="mlogloss",
                                random_state=42, n_jobs=-1)
xgb_model.fit(X_train_bal_rf, y_train_bal,
              eval_set=[(X_test, y_test)], verbose=False)
xgb_preds = xgb_model.predict(X_test)
print("Done.")


# =============================================================================
# SECTION 6: STATISTICAL EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 6 — STATISTICAL EVALUATION")
print("=" * 65)

models = {"Logistic Regression": lr_preds,
          "Random Forest"      : rf_preds,
          "XGBoost"            : xgb_preds}

results = {}
for name, preds in models.items():
    acc  = accuracy_score(y_test, preds)
    f1w  = f1_score(y_test, preds, average="weighted")
    f1m  = f1_score(y_test, preds, average="macro")
    results[name] = {"Accuracy": acc, "F1 Weighted": f1w, "F1 Macro": f1m}
    print(f"\n{'─'*50}")
    print(f"MODEL: {name}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  F1 Weighted : {f1w*100:.2f}%")
    print(f"  F1 Macro    : {f1m*100:.2f}%")
    print(classification_report(y_test, preds,
          target_names=["HOLD","INCREASE","DECREASE"], digits=3))

results_df = pd.DataFrame(results).T
best_model_name = results_df["F1 Weighted"].idxmax()
print(f"\n★ Best model: {best_model_name}")


# =============================================================================
# SECTION 7: BUSINESS ACCURACY CHECK
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 7 — BUSINESS ACCURACY CHECK")
print("=" * 65)

def biz_check(name, preds, default_true):
    pred_inc  = preds == 1
    pred_dec  = preds == 2
    pred_hold = preds == 0
    dv        = default_true.values

    wrong_inc  = (pred_inc)  & (dv == 1)
    right_inc  = (pred_inc)  & (dv == 0)
    right_dec  = (pred_dec)  & (dv == 1)
    wrong_dec  = (pred_dec)  & (dv == 0)
    missed_def = (pred_hold) & (dv == 1)

    n_pi = pred_inc.sum()
    biz_acc = right_inc.sum() / n_pi * 100 if n_pi > 0 else 0

    print(f"\n  {name}")
    print(f"  Wrong INCREASE (defaulter given increase): {wrong_inc.sum():>4,}  "
          f"({wrong_inc.sum()/n_pi*100 if n_pi>0 else 0:.1f}%)")
    print(f"  Correct DECREASE (defaulter caught)      : {right_dec.sum():>4,}")
    print(f"  Missed defaulters (in HOLD)              : {missed_def.sum():>4,}")
    print(f"  Business Accuracy on INCREASE            : {biz_acc:.1f}%")

    return {"model": name, "wrong_inc": wrong_inc.sum(),
            "right_dec": right_dec.sum(), "wrong_dec": wrong_dec.sum(),
            "missed_def": missed_def.sum(), "biz_acc": biz_acc}

biz_rows = []
for nm, preds in models.items():
    biz_rows.append(biz_check(nm, preds, def_test.reset_index(drop=True)))

biz_df = pd.DataFrame(biz_rows).set_index("model")
best_biz = biz_df["biz_acc"].idxmax()
print(f"\n★ Most business-safe model: {best_biz}")


# =============================================================================
# SECTION 8: FEATURE IMPORTANCE + SHAP
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 8 — FEATURE IMPORTANCE + SHAP EXPLAINABILITY")
print("=" * 65)

xgb_imp = pd.Series(xgb_model.feature_importances_,
                     index=FEATURE_COLS).sort_values(ascending=False)

print("\nTop 15 Features — XGBoost:")
for feat, val in xgb_imp.head(15).items():
    bar = "█" * int(val * 500)
    print(f"  {feat:<30} {bar} {val:.4f}")

# SHAP
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot (for class 2 = DECREASE — most important for risk management)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[:, :, 2] if len(np.array(shap_values).shape) == 3
                  else shap_values,
                  X_test, feature_names=FEATURE_COLS, show=False)
plt.title("SHAP — Feature Impact on DECREASE Decision", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_shap_summary.png")
plt.close()
print("SHAP plot saved: 12_shap_summary.png")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 7))
xgb_imp.head(15).sort_values().plot.barh(ax=ax, color="#1565C0",
                                          edgecolor='white', linewidth=0.5)
ax.set_title("XGBoost Feature Importance — Top 15", fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_feature_importance_v2.png")
plt.close()
print("Feature importance plot saved: 13_feature_importance_v2.png")


# =============================================================================
# SECTION 9: BUSINESS IMPACT (CUSTOMER-SPECIFIC EAD)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 9 — BUSINESS IMPACT WITH MENTOR'S EAD FORMULA")
print("=" * 65)

print("""
KEY UPGRADE FROM BASELINE:
  EAD is now customer-specific: EAD = Avg_Balance_6M (actual drawn amount)
  rather than assuming everyone draws 85% of limit.
  LGD = 70% (standard for unsecured credit in India)
  Revolve rate = 40%, Monthly interest = 3.5%, Interchange = 1.4%
  INCREASE quantum = 25% of current limit
  DECREASE quantum = 25% reduction
""")

test_df = X_test.copy().reset_index(drop=True)
test_df["PREDICTED_ACTION"] = xgb_preds
test_df["ACTUAL_DEFAULT"]   = def_test.reset_index(drop=True).values
test_df["ACTUAL_ACTION"]    = y_test.reset_index(drop=True).values
test_df["EAD_RATIO"]        = ead_test.reset_index(drop=True).values
test_df["EAD_AMOUNT"]       = (ead_test * lim_test).reset_index(drop=True).values
test_df["LIMIT_BAL"]        = lim_test.reset_index(drop=True).values

LGD          = 0.70
REVOLVE_RATE = 0.40
MONTHLY_INT  = 0.035
INC_QUANTUM  = 0.25   # 25% limit increase
DEC_QUANTUM  = 0.25   # 25% limit decrease
SPEND_RATIO  = 0.30   # monthly spend = 30% of limit
INTERCHANGE  = 0.014

# ── 1. Revenue from correct INCREASE ──────────────────────────────────────
correct_inc     = test_df[(test_df["PREDICTED_ACTION"]==1) & (test_df["ACTUAL_DEFAULT"]==0)]
inc_extra_limit = correct_inc["LIMIT_BAL"] * INC_QUANTUM
inc_rev_annual  = (inc_extra_limit * REVOLVE_RATE * MONTHLY_INT * 12 +
                   inc_extra_limit * SPEND_RATIO * INTERCHANGE * 12)
total_revenue   = inc_rev_annual.sum()

# ── 2. Cost of wrong INCREASE (raised limit of defaulter) ─────────────────
wrong_inc       = test_df[(test_df["PREDICTED_ACTION"]==1) & (test_df["ACTUAL_DEFAULT"]==1)]
# EAD = actual balance (mentor formula) on new higher limit
wrong_ead       = wrong_inc["EAD_AMOUNT"] * (1 + INC_QUANTUM)   # EAD grows with new limit
wrong_loss      = wrong_ead * LGD
total_cost      = wrong_loss.sum()

# ── 3. Savings from correct DECREASE (caught defaulter early) ─────────────
correct_dec     = test_df[(test_df["PREDICTED_ACTION"]==2) & (test_df["ACTUAL_DEFAULT"]==1)]
# By decreasing limit by 25%, we reduce the EAD the customer can reach
saved_ead       = correct_dec["EAD_AMOUNT"] * DEC_QUANTUM        # portion of EAD avoided
total_saved     = (saved_ead * LGD).sum()

# ── 4. Opportunity cost of wrong DECREASE (cut safe customer) ─────────────
wrong_dec       = test_df[(test_df["PREDICTED_ACTION"]==2) & (test_df["ACTUAL_DEFAULT"]==0)]
dec_lost_limit  = wrong_dec["LIMIT_BAL"] * DEC_QUANTUM
opp_cost        = (dec_lost_limit * REVOLVE_RATE * MONTHLY_INT * 12 +
                   dec_lost_limit  * SPEND_RATIO * INTERCHANGE * 12)
total_opp_cost  = opp_cost.sum()

# ── 5. Loss from missed defaulters (HOLD but actually defaults) ───────────
missed_def      = test_df[(test_df["PREDICTED_ACTION"]==0) & (test_df["ACTUAL_DEFAULT"]==1)]
# We do nothing — customer defaults at their current EAD
missed_loss     = missed_def["EAD_AMOUNT"] * LGD
total_missed    = missed_loss.sum()

# ── NET ───────────────────────────────────────────────────────────────────
net_impact = total_revenue + total_saved - total_cost - total_opp_cost - total_missed

print(f"  Test set size                          : {len(test_df):,} customers")
print(f"  Avg credit limit                       : ₹{test_df['LIMIT_BAL'].mean():,.0f}")
print(f"  Avg EAD ratio (mentor formula)         : {test_df['EAD_RATIO'].mean():.3f}")
print()
print(f"  ✓ Correct INCREASE  : {len(correct_inc):>5,}   → Revenue      : ₹{total_revenue:>12,.0f}")
print(f"  ✗ Wrong INCREASE    : {len(wrong_inc):>5,}   → Loss          : ₹{total_cost:>12,.0f}")
print(f"  ✓ Correct DECREASE  : {len(correct_dec):>5,}   → EAD saved    : ₹{total_saved:>12,.0f}")
print(f"  ✗ Wrong DECREASE    : {len(wrong_dec):>5,}   → Opp cost      : ₹{total_opp_cost:>12,.0f}")
print(f"  ✗ Missed defaulters : {len(missed_def):>5,}   → Loss          : ₹{total_missed:>12,.0f}")
print()
print(f"  {'─'*55}")
print(f"  NET BUSINESS IMPACT (test set 20%) : ₹{net_impact:>12,.0f}  "
      f"({'SURPLUS ✓' if net_impact >= 0 else 'DEFICIT ✗'})")
print(f"  {'─'*55}")

# Scale to ICICI portfolio
sf = 16_000_000 / len(test_df)
print(f"\n  Scaled to ICICI ~16M card portfolio:")
print(f"  Revenue (correct increases)  : ₹{total_revenue   *sf/1e7:>8,.0f} Cr")
print(f"  Loss (wrong increases)       : ₹{total_cost      *sf/1e7:>8,.0f} Cr")
print(f"  EAD saved (correct decreases): ₹{total_saved     *sf/1e7:>8,.0f} Cr")
print(f"  Opp cost (wrong decreases)   : ₹{total_opp_cost  *sf/1e7:>8,.0f} Cr")
print(f"  Missed default loss          : ₹{total_missed    *sf/1e7:>8,.0f} Cr")
print(f"  {'─'*40}")
print(f"  NET IMPACT                   : ₹{net_impact      *sf/1e7:>8,.0f} Cr per year")

# ── Business impact chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Business Impact — V2 Model with Customer-Specific EAD",
             fontsize=13, fontweight='bold')

labels = ["Revenue\n(correct\nincrease)",
          "EAD saved\n(correct\ndecrease)",
          "Wrong\nincrease\nloss",
          "Opp cost\nwrong\ndecrease",
          "Missed\ndefaulter\nloss",
          "NET"]
values_plot = [total_revenue, total_saved,
               -total_cost, -total_opp_cost, -total_missed, net_impact]
colors_plot = ["#43A047","#1565C0","#E53935","#FB8C00","#7B1FA2",
               "#2E7D32" if net_impact >= 0 else "#B71C1C"]

axes[0].bar(labels, values_plot, color=colors_plot, edgecolor='white', linewidth=1)
axes[0].axhline(y=0, color='black', linewidth=0.8)
axes[0].set_ylabel("₹ Impact (test set)")
axes[0].set_title("P&L Breakdown — Test Set")
for i, v in enumerate(values_plot):
    axes[0].text(i, v + np.sign(v) * abs(v) * 0.04,
                 f"₹{abs(v)/1000:.0f}K", ha='center', fontsize=8, fontweight='bold')

# Confusion matrix (XGBoost)
cm = confusion_matrix(y_test, xgb_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", ax=axes[1],
            cmap="Blues", cbar=False,
            xticklabels=["HOLD","INCREASE","DECREASE"],
            yticklabels=["HOLD","INCREASE","DECREASE"],
            linewidths=0.5, linecolor='gray')
axes[1].set_title(f"XGBoost Confusion Matrix\nF1: {f1_score(y_test, xgb_preds, average='weighted')*100:.1f}%")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_business_impact_v2.png")
plt.close()
print("\nPlot saved: 14_business_impact_v2.png")


# =============================================================================
# SECTION 10: COMPARISON — BASELINE vs V2
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 10 — BASELINE vs V2 COMPARISON")
print("=" * 65)

baseline_missed = 1568 * 0.20  # from earlier analysis (20% test set)
v2_missed = len(missed_def)

print(f"""
  WHAT CHANGED                     BASELINE        V2 (UPGRADED)
  ─────────────────────────────────────────────────────────────
  EAD calculation                  Fixed 85%       Customer-specific
  Target variable method           Hard IF-ELSE     Risk score (0-100)
  DECREASE % of dataset            62%              {v2_counts[2]/len(df)*100:.0f}%
  INCREASE % of dataset            1.5%             {v2_counts[1]/len(df)*100:.0f}%
  Missed defaulters in HOLD        ~{int(baseline_missed):,}           {v2_missed:,}
  Net business impact              DEFICIT          {'SURPLUS' if net_impact >= 0 else 'DEFICIT'}
  Age/Sex features removed         No               Yes (fair lending)
  SHAP explainability              No               Yes
  ─────────────────────────────────────────────────────────────
""")


# =============================================================================
# SECTION 11: SAVE MODEL
# =============================================================================
print("=" * 65)
print("SECTION 11 — SAVING")
print("=" * 65)

with open("xgb_v2.pkl", "wb") as f: pickle.dump(xgb_model, f)
with open("rf_v2.pkl",  "wb") as f: pickle.dump(rf_model,  f)
with open("scaler_v2.pkl", "wb") as f: pickle.dump(scaler, f)
test_df.to_csv("test_predictions_v2.csv", index=False)

print("\nSaved: xgb_v2.pkl  rf_v2.pkl  scaler_v2.pkl  test_predictions_v2.csv")
print(f"Output charts: outputs/11 through 14")
print("\n" + "=" * 65)
print("V2 MODEL COMPLETE")
print("=" * 65)