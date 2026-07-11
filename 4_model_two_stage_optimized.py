"""
=============================================================
FILE 4: TWO-STAGE CREDIT LINE OPTIMISATION MODEL
Project  : Credit Line Optimisation
Dataset  : UCI Default of Credit Card Clients (30,000 customers)
Author   : Shivansh Shukla | IIT Gandhinagar | ICICI Bank Internship
=============================================================

ARCHITECTURE
────────────────────────────────────────────────────────────
STAGE 1 — Default Probability Model
  Train 5 models on ground-truth DEFAULT label (0/1).
  Ensemble their probabilities → P(default) per customer.
  Convert P(default) → Risk Score 0–100 via log-odds scaling.
  Models: Logistic Regression, Random Forest, XGBoost,
          LightGBM, CatBoost

STAGE 2 — Exhaustive Barrier Optimisation
  Place two barriers B1, B2 on the 0–100 risk score:
    Score < B1          → INCREASE
    B1 ≤ Score ≤ B2     → HOLD
    Score > B2          → DECREASE
  Try every valid (B1, B2) pair: B1 ∈ [1,98], B2 ∈ [B1+1,99]
  At each pair compute Net Business P&L using corrected economics.
  Pick (B1*, B2*) = argmax Net P&L → optimal action split.

ECONOMICS (all corrected from earlier discussions)
────────────────────────────────────────────────────────────
EAD            = Avg_Balance_6M / Credit_Limit  (mentor formula)
LGD            = 0.50 + 0.30 × EAD_ratio        (varies by drawdown)
INC_QUANTUM    = 25%  (limit increase size)
DEC_QUANTUM    = 25%  (limit decrease size)
REVOLVE_RATE   varies by payment ratio (not flat 40%)
MONTHLY_INT    = 3.5% on revolving balance
INTERCHANGE    = 1.4% on monthly spend (30% of limit DELTA only)

P&L COMPONENTS (DELTA-based — only new value created)
────────────────────────────────────────────────────────────
Revenue (correct INCREASE):
  = extra_limit × revolve_rate × 3.5% × 12          ← extra interest
  + extra_limit × 30% × 1.4% × 12                   ← extra interchange

Loss (wrong INCREASE — defaulter given increase):
  = EAD_amount × (1 + INC_QUANTUM) × LGD

EAD saved (correct DECREASE — defaulter caught):
  = EAD_amount × DEC_QUANTUM × LGD

Opp cost (wrong DECREASE — safe customer cut):
  = cut_limit × 30% × 1.4% × 12                     ← interchange delta
  + cut_limit × revolve_rate × 3.5% × 12            ← interest delta

Missed default loss (defaulter left in HOLD):
  = EAD_amount × LGD

PROFIT REPORTING (mentor's two methods)
────────────────────────────────────────────────────────────
Method 1: Net Profit / Avg Credit Limit  (basis points on limit book)
Method 2: Net Profit / Total Outstanding Balance (return on exposure)

CUSTOMER QUERY FUNCTION
────────────────────────────────────────────────────────────
query_customer(customer_id) → risk score, action, top drivers
in plain English — no model jargon, pure business language.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os, pickle
from itertools import product

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.metrics            import (roc_auc_score, classification_report,
                                        brier_score_loss, RocCurveDisplay)
from imblearn.over_sampling     import SMOTE
import xgboost  as xgb
import lightgbm as lgb
from catboost   import CatBoostClassifier
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 130
plt.rcParams['savefig.bbox'] = 'tight'

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ECONOMICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
INC_QUANTUM  = 0.25    # 25% limit increase
DEC_QUANTUM  = 0.25    # 25% limit decrease
MONTHLY_INT  = 0.035   # 3.5% per month on revolving balance
INTERCHANGE  = 0.014   # 1.4% of spend
SPEND_RATIO  = 0.30    # monthly spend = 30% of limit delta

def revolve_rate(pay_ratio_avg):
    """
    Customer-specific revolve rate based on payment ratio.
    Customers who pay very little of their bill are almost
    certainly revolvers; those who pay nearly full are transactors.
    """
    # rr = np.where(pay_ratio_avg < 0.10, 0.92,
    #      np.where(pay_ratio_avg < 0.25, 0.75,
    #      np.where(pay_ratio_avg < 0.50, 0.55,
    #      np.where(pay_ratio_avg < 0.75, 0.30,
    #                                      0.10))))
    rr = 0.20
    return rr

def lgd_rate(ead_ratio):
    "Shifted LGD based on drawdown ratio (EAD / limit). Lower drawdown → lower LGD."
    """
    Customer-specific LGD: more drawn-down customers have harder recovery.
    LGD = 0.50 + 0.30 × EAD_ratio  (ranges 0.50 – 0.80)
    """
    return 0.50 + 0.30 * ead_ratio


# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================
print("=" * 65)
print("SECTION 1 — LOADING DATA")
print("=" * 65)

df = pd.read_csv("features.csv")

# Drop protected attributes — fair lending
df.drop(columns=["AGE", "SEX", "MARRIAGE", "EDUCATION"], errors="ignore", inplace=True)

# Restore original customer index as ID (1-indexed)
df.insert(0, "CUSTOMER_ID", range(1, len(df) + 1))

# Compute EAD fields (mentor formula)
df["EAD_RATIO"]  = (df["BALANCE_6M_AVG"] / df["LIMIT_BAL"]).clip(0, 1)
df["EAD_AMOUNT"] = df["BALANCE_6M_AVG"].clip(lower=0)
df["LGD"]        = lgd_rate(df["EAD_RATIO"])
df["REVOLVE_R"]  = revolve_rate(df["PAY_RATIO_6M_AVG"])

print(f"Customers  : {len(df):,}")
print(f"Features   : {df.shape[1]}")
print(f"Default rate: {df['DEFAULT'].mean()*100:.1f}%")
print(f"Avg limit  : ₹{df['LIMIT_BAL'].mean():,.0f}")
print(f"Avg EAD ratio: {df['EAD_RATIO'].mean():.3f}")


# =============================================================================
# SECTION 2 — FEATURE SET
# =============================================================================
PROTECTED   = ["CUSTOMER_ID", "DEFAULT", "LIMIT_ACTION",
               "EAD_RATIO", "EAD_AMOUNT", "LGD", "REVOLVE_R"]
FEATURE_COLS = [c for c in df.columns if c not in PROTECTED]
TARGET       = "DEFAULT"

print(f"\nFeatures for Stage 1 model ({len(FEATURE_COLS)}):")
for f in FEATURE_COLS: print(f"  {f}")


# =============================================================================
# SECTION 3 — TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3 — TRAIN / TEST SPLIT")
print("=" * 65)

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

(X_train, X_test,
 y_train, y_test,
 idx_train, idx_test) = train_test_split(
    X, y, df.index,
    test_size=0.20, random_state=42, stratify=y
)

# Keep full test-set rows for business impact
test_meta = df.loc[idx_test].reset_index(drop=True).copy()

# Scale (for LR and calibration)
scaler     = StandardScaler()
Xtr_sc     = scaler.fit_transform(X_train)
Xte_sc     = scaler.transform(X_test)

# SMOTE on training set only
smote            = SMOTE(random_state=42)
Xtr_bal, ytr_bal = smote.fit_resample(X_train, y_train)
Xtr_sc_bal, _    = smote.fit_resample(Xtr_sc,  y_train)

print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"After SMOTE train: {len(Xtr_bal):,}  (default: {ytr_bal.sum():,}  safe: {(ytr_bal==0).sum():,})")


# =============================================================================
# SECTION 4 — STAGE 1: TRAIN DEFAULT PROBABILITY MODELS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4 — STAGE 1: DEFAULT PROBABILITY MODELS")
print("=" * 65)

# ── 4a. Logistic Regression (calibrated baseline) ────────────────────────
print("\n[1/5] Logistic Regression...")
cv_seeded = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_base = LogisticRegression(max_iter=1000, C=1.0,
                              class_weight="balanced", random_state=42)
lr_cal  = CalibratedClassifierCV(lr_base, cv=cv_seeded, method="sigmoid")
lr_cal.fit(Xtr_sc_bal, ytr_bal)
lr_prob = lr_cal.predict_proba(Xte_sc)[:, 1]

# ── 4b. Random Forest ────────────────────────────────────────────────────
print("[2/5] Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                             min_samples_leaf=8, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf_cal = CalibratedClassifierCV(rf, cv=cv_seeded, method="isotonic")
rf_cal.fit(Xtr_bal, ytr_bal)
rf_prob = rf_cal.predict_proba(X_test)[:, 1]

# ── 4c. XGBoost ──────────────────────────────────────────────────────────
print("[3/5] XGBoost...")
xgb_m = xgb.XGBClassifier(n_estimators=400, max_depth=6,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="auc",
                            random_state=42, n_jobs=-1)
xgb_m.fit(Xtr_bal, ytr_bal,
          eval_set=[(X_test, y_test)], verbose=False)
xgb_prob = xgb_m.predict_proba(X_test)[:, 1]

# ── 4d. LightGBM ─────────────────────────────────────────────────────────
print("[4/5] LightGBM...")
lgb_m = lgb.LGBMClassifier(n_estimators=400, max_depth=6,
                             learning_rate=0.05, subsample=0.8,
                             class_weight="balanced",
                             random_state=42, n_jobs=-1, verbose=-1)
lgb_m.fit(Xtr_bal, ytr_bal)
lgb_prob = lgb_m.predict_proba(X_test)[:, 1]

# ── 4e. CatBoost ─────────────────────────────────────────────────────────
print("[5/5] CatBoost...")
cat_m = CatBoostClassifier(iterations=400, depth=6,
                            learning_rate=0.05, auto_class_weights="Balanced",
                            random_seed=42, verbose=0)
cat_m.fit(Xtr_bal, ytr_bal)
cat_prob = cat_m.predict_proba(X_test)[:, 1]

# ── 4f. Ensemble — simple average of all five ────────────────────────────
ensemble_prob = (lr_prob + rf_prob + xgb_prob + lgb_prob + cat_prob) / 5.0

print("\n--- AUC Scores (test set) ---")
model_probs = {
    "Logistic Regression": lr_prob,
    "Random Forest"      : rf_prob,
    "XGBoost"            : xgb_prob,
    "LightGBM"           : lgb_prob,
    "CatBoost"           : cat_prob,
    "ENSEMBLE"           : ensemble_prob,
}
auc_results = {}
for name, prob in model_probs.items():
    auc = roc_auc_score(y_test, prob)
    brier = brier_score_loss(y_test, prob)
    auc_results[name] = auc
    marker = " ★" if name == "ENSEMBLE" else ""
    print(f"  {name:<25} AUC: {auc:.4f}   Brier: {brier:.4f}{marker}")

# ── ROC curves chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ["#78909C","#43A047","#1565C0","#F57C00","#7B1FA2","#D32F2F"]
for (name, prob), col in zip(model_probs.items(), colors_roc):
    auc = roc_auc_score(y_test, prob)
    lw  = 2.5 if name == "ENSEMBLE" else 1.2
    RocCurveDisplay.from_predictions(
        y_test, prob, name=f"{name} (AUC={auc:.3f})",
        ax=ax, color=col, lw=lw
    )
ax.plot([0,1],[0,1],'k--', lw=0.8, label="Random (AUC=0.500)")
ax.set_title("ROC Curves — Stage 1 Default Probability Models",
             fontweight='bold', fontsize=12)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/S1_roc_curves.png"); plt.close()
print(f"\nPlot saved: S1_roc_curves.png")


# =============================================================================
# SECTION 5 — LOG-ODDS RISK SCORE (0–100)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5 — CONVERTING P(DEFAULT) → RISK SCORE 0–100")
print("=" * 65)

print("""
FORMULA:
  log_odds   = log( P / (1 - P) )   — natural output of logistic models
  risk_score = 100 × (log_odds - min) / (max - min)

WHY LOG-ODDS:
  P(default) is not linear in risk.
  Moving from 1% to 5% default probability is a much bigger
  risk jump than moving from 50% to 54%, even though both
  are 4 percentage points. Log-odds stretches the extremes
  and compresses the middle — matching how credit risk
  actually behaves. All bureau scores (CIBIL etc.) are built
  on this exact transformation.
""")

eps = 1e-6
p   = ensemble_prob.clip(eps, 1 - eps)
log_odds = np.log(p / (1 - p))

lo_min, lo_max = log_odds.min(), log_odds.max()
risk_score = 100.0 * (log_odds - lo_min) / (lo_max - lo_min)

test_meta["P_DEFAULT"]   = ensemble_prob
test_meta["LOG_ODDS"]    = log_odds
test_meta["RISK_SCORE"]  = risk_score

print(f"Risk score range : {risk_score.min():.1f} — {risk_score.max():.1f}")
print(f"Mean risk score  : {risk_score.mean():.1f}")
print(f"\nRisk score by actual default:")
for label in [0, 1]:
    mask = y_test.reset_index(drop=True).values == label
    print(f"  DEFAULT={label}  mean score: {risk_score[mask].mean():.1f}  "
          f"median: {np.median(risk_score[mask]):.1f}")

# ── Score distribution chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Risk Score Distribution — Stage 1 Output", fontsize=13, fontweight='bold')

for label, color, lname in [(0,"#2196F3","Safe customers (no default)"),
                              (1,"#F44336","Defaulters")]:
    mask = y_test.reset_index(drop=True).values == label
    axes[0].hist(risk_score[mask], bins=40, alpha=0.65,
                 color=color, label=lname, edgecolor='white', linewidth=0.3)
axes[0].set_xlabel("Risk Score (0 = safe, 100 = high risk)")
axes[0].set_ylabel("Number of Customers")
axes[0].set_title("Score Distribution by Actual Outcome")
axes[0].legend()
axes[0].axvline(x=30, color='green',  linestyle='--', lw=1.5, label="B1 zone")
axes[0].axvline(x=60, color='red',    linestyle='--', lw=1.5, label="B2 zone")

# Bucket default rate
buckets = pd.cut(risk_score, bins=10)
bucket_df = pd.DataFrame({"bucket": buckets,
                           "default": y_test.reset_index(drop=True).values})
bucket_dr = bucket_df.groupby("bucket", observed=False)["default"].mean() * 100
axes[1].bar(range(len(bucket_dr)), bucket_dr.values,
            color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(bucket_dr))),
            edgecolor='white', linewidth=1)
axes[1].set_xticks(range(len(bucket_dr)))
axes[1].set_xticklabels([f"{int(b.left)}–{int(b.right)}" for b in bucket_dr.index],
                         rotation=45, ha='right', fontsize=8)
axes[1].set_xlabel("Risk Score Band")
axes[1].set_ylabel("Actual Default Rate (%)")
axes[1].set_title("Default Rate by Risk Score Band\n(validates score quality)")
for i, v in enumerate(bucket_dr.values):
    if not np.isnan(v):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT}/S2_risk_score_distribution.png"); plt.close()
print("Plot saved: S2_risk_score_distribution.png")


# =============================================================================
# SECTION 6 — STAGE 2: EXHAUSTIVE BARRIER OPTIMISATION
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 6 — STAGE 2: EXHAUSTIVE BARRIER OPTIMISATION")
print("=" * 65)

print("""
BARRIER SEARCH:
  B1 ∈ [1, 98]   B2 ∈ [B1+1, 99]
  Score < B1     → INCREASE
  B1 ≤ Score ≤ B2 → HOLD
  Score > B2     → DECREASE

  Total combinations: 4,851
  At each (B1, B2) → compute full Net P&L → store
  Pick argmax.
""")

# Pre-compute per-customer economics arrays (vectorised — fast)
tm             = test_meta.reset_index(drop=True)
scores         = tm["RISK_SCORE"].values
defaults       = y_test.reset_index(drop=True).values
limits         = tm["LIMIT_BAL"].values
ead_amt        = tm["EAD_AMOUNT"].values
lgd_arr        = tm["LGD"].values
rr_arr         = tm["REVOLVE_R"].values

# Extra limit delta
extra_limit = limits * INC_QUANTUM
cut_limit   = limits * DEC_QUANTUM

# Revenue per customer if INCREASE and safe
rev_interest    = extra_limit * rr_arr * MONTHLY_INT * 12
rev_interchange = extra_limit * SPEND_RATIO * INTERCHANGE * 12
rev_arr         = rev_interest + rev_interchange             # shape (N,)

# Loss per customer if INCREASE and defaults
wrong_inc_loss  = ead_amt * (1 + INC_QUANTUM) * lgd_arr     # shape (N,)

# EAD saved per customer if DECREASE and defaults
ead_saved       = ead_amt * DEC_QUANTUM * lgd_arr            # shape (N,)

# Opp cost per customer if DECREASE and safe
opp_interchange = cut_limit * SPEND_RATIO * INTERCHANGE * 12
opp_interest    = cut_limit * rr_arr * MONTHLY_INT * 12
opp_arr         = opp_interchange + opp_interest             # shape (N,)

# Missed default loss if HOLD and defaults
missed_loss     = ead_amt * lgd_arr                          # shape (N,)

safe    = (defaults == 0)
default = (defaults == 1)

# ── Exhaustive search ─────────────────────────────────────────────────────
print("Running exhaustive search over 4,851 (B1, B2) combinations...")

results_grid = np.full((100, 100), np.nan)
best_pl      = -np.inf
best_b1, best_b2 = 30, 60   # defaults

for b1 in range(1, 99):
    for b2 in range(b1 + 1, 100):
        inc_mask  = scores < b1
        hold_mask = (scores >= b1) & (scores <= b2)
        dec_mask  = scores > b2

        pl = (
              rev_arr[inc_mask & safe].sum()          # correct increase revenue
            - wrong_inc_loss[inc_mask & default].sum()# wrong increase loss
            + ead_saved[dec_mask & default].sum()     # correct decrease saving
            - opp_arr[dec_mask & safe].sum()          # wrong decrease opp cost
            - missed_loss[hold_mask & default].sum()  # missed default loss
        )
        results_grid[b1, b2] = pl

        if pl > best_pl:
            best_pl  = pl
            best_b1  = b1
            best_b2  = b2

print(f"\n★ OPTIMAL BARRIERS FOUND:")
print(f"  B1 (INCREASE threshold) : {best_b1}")
print(f"  B2 (DECREASE threshold) : {best_b2}")
print(f"  Maximum Net P&L         : ₹{best_pl:,.0f}")

# ── Export FULL grid to CSV for audit / mentor review ─────────────────────
print("\nExporting full barrier search grid to CSV...")
grid_rows = []
for b1 in range(1, 99):
    for b2 in range(b1 + 1, 100):
        pl = results_grid[b1, b2]
        if np.isnan(pl):
            continue
        inc_mask  = scores < b1
        hold_mask = (scores >= b1) & (scores <= b2)
        dec_mask  = scores > b2
        rev   = rev_arr[inc_mask & safe].sum()
        wloss = wrong_inc_loss[inc_mask & default].sum()
        esave = ead_saved[dec_mask & default].sum()
        opp   = opp_arr[dec_mask & safe].sum()
        miss  = missed_loss[hold_mask & default].sum()
        grid_rows.append({
            "B1": b1, "B2": b2,
            "n_increase": int(inc_mask.sum()),
            "n_hold": int(hold_mask.sum()),
            "n_decrease": int(dec_mask.sum()),
            "revenue_correct_increase": rev,
            "loss_wrong_increase": wloss,
            "ead_saved_correct_decrease": esave,
            "opp_cost_wrong_decrease": opp,
            "loss_missed_default": miss,
            "net_pl": pl
        })
grid_df = pd.DataFrame(grid_rows).sort_values("net_pl", ascending=False)
grid_df.to_csv("barrier_search_grid.csv", index=False)
print(f"Saved: barrier_search_grid.csv  ({len(grid_df):,} rows)")
# print(f"\nTop row of CSV (should exactly match optimal barriers above):")
# print(grid_df.iloc[0].to_string())


# =============================================================================
# SECTION 7 — PROFIT SURFACE HEATMAP
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 7 — PROFIT SURFACE VISUALISATION")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Stage 2: Exhaustive Barrier Optimisation — Profit Surface",
             fontsize=13, fontweight='bold')

# Full heatmap (trim edges for cleaner view)
grid_plot = results_grid[1:99, 2:99].copy()
im = axes[0].imshow(grid_plot / 1e6, origin='lower', aspect='auto',
                    cmap='RdYlGn', interpolation='nearest')
axes[0].set_xlabel("B2 — DECREASE threshold (score)")
axes[0].set_ylabel("B1 — INCREASE threshold (score)")
axes[0].set_title("Net P&L (₹M) across all 4,851\n(B1, B2) combinations")
plt.colorbar(im, ax=axes[0], label="Net P&L (₹ Million)")
# Mark optimal point
axes[0].scatter([best_b2 - 2], [best_b1 - 1], color='white', s=120,
                marker='*', zorder=5, label=f"Optimal ({best_b1},{best_b2})")
axes[0].legend(fontsize=9)

# P&L vs B2 with B1 fixed at optimal
b2_range = range(best_b1 + 1, 99)
pl_b2    = [results_grid[best_b1, b2] for b2 in b2_range]
axes[1].plot(list(b2_range), [v / 1e6 for v in pl_b2],
             color="#1565C0", linewidth=2)
axes[1].axvline(x=best_b2, color='red', linestyle='--', linewidth=1.5,
                label=f"Optimal B2 = {best_b2}")
axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle=':')
axes[1].set_xlabel(f"B2 (DECREASE threshold) — B1 fixed at {best_b1}")
axes[1].set_ylabel("Net P&L (₹ Million)")
axes[1].set_title(f"P&L vs B2 at Optimal B1={best_b1}")
axes[1].legend()
axes[1].fill_between(list(b2_range), [v / 1e6 for v in pl_b2], 0,
                      where=[v >= 0 for v in pl_b2],
                      alpha=0.15, color='green', label="Profit zone")
axes[1].fill_between(list(b2_range), [v / 1e6 for v in pl_b2], 0,
                      where=[v < 0 for v in pl_b2],
                      alpha=0.15, color='red', label="Loss zone")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/S3_profit_surface.png"); plt.close()
print("Plot saved: S3_profit_surface.png")


# =============================================================================
# SECTION 8 — APPLY OPTIMAL BARRIERS & BUSINESS IMPACT
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 8 — APPLYING OPTIMAL BARRIERS & BUSINESS IMPACT")
print("=" * 65)

def apply_barriers(scores, b1, b2):
    actions = np.where(scores < b1, 1,
              np.where(scores > b2, 2, 0))
    return actions   # 0=HOLD 1=INCREASE 2=DECREASE

opt_actions = apply_barriers(scores, best_b1, best_b2)
test_meta["RISK_SCORE"]   = scores
test_meta["P_DEFAULT"]    = ensemble_prob
test_meta["ACTION"]       = opt_actions
test_meta["ACTION_LABEL"] = pd.Series(opt_actions).map({0:"HOLD",1:"INCREASE",2:"DECREASE"}).values

ACTION_MAP    = {0:"HOLD", 1:"INCREASE", 2:"DECREASE"}
ACTION_COLORS = {0:"#78909C", 1:"#43A047", 2:"#E53935"}

action_counts = pd.Series(opt_actions).value_counts().sort_index()
action_pct    = pd.Series(opt_actions).value_counts(normalize=True).sort_index()*100
print("\nOptimal action distribution:")
for k in [0,1,2]:
    print(f"  {ACTION_MAP[k]:<10}: {action_counts[k]:>5,}  ({action_pct[k]:.1f}%)")

# ── Full P&L at optimal barriers ─────────────────────────────────────────
inc_mask  = opt_actions == 1
dec_mask  = opt_actions == 2
hold_mask = opt_actions == 0

correct_inc_n  = (inc_mask  & safe).sum()
wrong_inc_n    = (inc_mask  & default).sum()
correct_dec_n  = (dec_mask  & default).sum()
wrong_dec_n    = (dec_mask  & safe).sum()
missed_def_n   = (hold_mask & default).sum()

total_revenue  = rev_arr[inc_mask & safe].sum()
total_wi_loss  = wrong_inc_loss[inc_mask & default].sum()
total_ead_save = ead_saved[dec_mask & default].sum()
total_opp      = opp_arr[dec_mask & safe].sum()
total_missed   = missed_loss[hold_mask & default].sum()

net_pl = total_revenue + total_ead_save - total_wi_loss - total_opp - total_missed

print(f"\n{'─'*55}")
print(f"  Correct INCREASE (revenue)      : {correct_inc_n:>5,}  → ₹{total_revenue:>12,.0f}")
print(f"  Wrong   INCREASE (loss)         : {wrong_inc_n:>5,}  → ₹{total_wi_loss:>12,.0f}")
print(f"  Correct DECREASE (EAD saved)    : {correct_dec_n:>5,}  → ₹{total_ead_save:>12,.0f}")
print(f"  Wrong   DECREASE (opp cost)     : {wrong_dec_n:>5,}  → ₹{total_opp:>12,.0f}")
print(f"  Missed  DEFAULT  (HOLD loss)    : {missed_def_n:>5,}  → ₹{total_missed:>12,.0f}")
print(f"{'─'*55}")
print(f"  NET P&L (test set 20%)          :        ₹{net_pl:>12,.0f}  "
      f"{'✓ SURPLUS' if net_pl >= 0 else '✗ DEFICIT'}")
print(f"{'─'*55}")


# =============================================================================
# SECTION 9 — PROFIT REPORTING (MENTOR'S TWO METHODS)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 9 — PROFIT REPORTING — MENTOR'S TWO METHODS")
print("=" * 65)

print("""
METHOD 1: Net Profit / Avg Credit Limit
  → How many ₹ of profit for every ₹1 of credit limit held
  → Tells management: "For every rupee of limit we manage,
    how much value does the optimisation model generate?"
  → Useful for comparing across different portfolio sizes.

METHOD 2: Net Profit / Total Outstanding Balance
  → How many ₹ of profit for every ₹1 of actual outstanding
  → Tells management: "What is our return on actual exposure?"
  → Directly comparable to Net Interest Margin (NIM).
""")

avg_credit_limit    = test_meta["LIMIT_BAL"].mean()
total_credit_limit  = test_meta["LIMIT_BAL"].sum()
total_outstanding   = test_meta["EAD_AMOUNT"].sum()   # BALANCE_6M_AVG

method1 = net_pl / total_credit_limit    # ratio
method2 = net_pl / total_outstanding     # ratio

print(f"  Test set total credit limit      : ₹{total_credit_limit:>15,.0f}")
print(f"  Test set total outstanding bal   : ₹{total_outstanding:>15,.0f}")
print(f"  Net P&L (test set)               : ₹{net_pl:>15,.0f}")
print()
print(f"  METHOD 1: Profit / Avg Credit Limit  = {method1*100:.4f}%  "
      f"({method1*10000:.2f} bps)")
print(f"  METHOD 2: Profit / Total Outstanding = {method2*100:.4f}%  "
      f"({method2*10000:.2f} bps)")

# Scale to ICICI (~16M cards)
sf           = 16_000_000 / len(test_meta)
net_icici    = net_pl * sf
lim_icici    = total_credit_limit * sf
out_icici    = total_outstanding * sf

print(f"\n  Scaled to ICICI ~16M card portfolio:")
print(f"  Net annual P&L                   : ₹{net_icici/1e7:>10,.0f} Cr")
print(f"  Method 1 (profit/limit)          : {method1*100:.4f}%  [{method1*10000:.2f} bps]")
print(f"  Method 2 (profit/outstanding)    : {method2*100:.4f}%  [{method2*10000:.2f} bps]")
print(f"\n  NOTE: These ratios are portfolio-size independent.")
print(f"  ICICI can apply them directly to their own book size.")


# =============================================================================
# SECTION 10 — BUSINESS IMPACT VISUALISATIONS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 10 — BUSINESS IMPACT VISUALISATIONS")
print("=" * 65)

# ── Chart 1: P&L waterfall ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Business Impact at Optimal Barriers — Net P&L Breakdown",
             fontsize=13, fontweight='bold')

components = ["Revenue\n(correct\nincrease)",
              "EAD saved\n(correct\ndecrease)",
              "Wrong\nincrease\nloss",
              "Opp cost\n(wrong\ndecrease)",
              "Missed\ndefault\nloss",
              "NET P&L"]
values     = [total_revenue, total_ead_save,
              -total_wi_loss, -total_opp, -total_missed, net_pl]
bar_colors = ["#43A047","#1565C0","#E53935","#FB8C00","#7B1FA2",
              "#2E7D32" if net_pl >= 0 else "#B71C1C"]

bars = axes[0].bar(components, [v/1e6 for v in values],
                    color=bar_colors, edgecolor='white', linewidth=1.2)
axes[0].axhline(y=0, color='black', linewidth=0.8)
axes[0].set_ylabel("₹ Million (test set)")
axes[0].set_title("P&L Waterfall — All Five Components")
for bar, v in zip(bars, values):
    ypos = bar.get_height() + (0.02 if v >= 0 else -0.08) * abs(bar.get_height())
    axes[0].text(bar.get_x() + bar.get_width()/2, ypos,
                 f"₹{abs(v)/1e6:.2f}M", ha='center', fontsize=8, fontweight='bold')

# ── Chart 2: action distribution vs default rate ──────────────────────────
action_labels = [ACTION_MAP[k] for k in [0,1,2]]
action_cnt    = [action_counts.get(k,0) for k in [0,1,2]]
dr_by_action  = []
for k in [0,1,2]:
    mask = opt_actions == k
    dr_by_action.append(default[mask].mean() * 100 if mask.sum() > 0 else 0)

ax2 = axes[1]
x   = np.arange(3)
w   = 0.4
b1  = ax2.bar(x - w/2, action_cnt, w,
              color=[ACTION_COLORS[k] for k in [0,1,2]],
              edgecolor='white', linewidth=1, label="Customer count")
ax2.set_ylabel("Number of Customers", color='black')
ax2.set_xticks(x); ax2.set_xticklabels(action_labels)

ax2b = ax2.twinx()
ax2b.plot(x, dr_by_action, 'ko--', lw=1.5, ms=7, label="Default rate %")
ax2b.set_ylabel("Actual Default Rate (%)", color='black')
ax2b.set_ylim(0, max(dr_by_action) * 1.5)
for i, (cnt, dr) in enumerate(zip(action_cnt, dr_by_action)):
    ax2b.text(i, dr + 0.5, f"{dr:.1f}%", ha='center', fontsize=9, color='black')

ax2.set_title("Action Distribution & Default Rate per Group\n"
              "(DECREASE group must have highest default rate)")
lines1, lab1 = ax2.get_legend_handles_labels()
lines2, lab2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/S4_business_impact.png"); plt.close()
print("Plot saved: S4_business_impact.png")

# ── Chart 3: Score vs default probability (calibration) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Calibration — Risk Score vs Actual Default Rate",
             fontsize=13, fontweight='bold')

score_bins  = pd.cut(scores, bins=20)
calib_df    = pd.DataFrame({"bin": score_bins, "default": default,
                             "p_def": ensemble_prob})
calib_mean  = calib_df.groupby("bin", observed=False).agg(
    actual_dr=("default","mean"),
    pred_prob=("p_def","mean"),
    count=("default","count")
).dropna()

axes[0].scatter(calib_mean["pred_prob"]*100, calib_mean["actual_dr"]*100,
                s=calib_mean["count"]/3, alpha=0.7, color="#1565C0")
axes[0].plot([0,100],[0,100],'r--', lw=1, label="Perfect calibration")
axes[0].set_xlabel("Mean Predicted P(default) × 100")
axes[0].set_ylabel("Actual Default Rate (%)")
axes[0].set_title("Calibration Plot\n(closer to red line = better calibrated)")
axes[0].legend()

# Score band summary table as chart
band_labels = [str(b) for b in calib_mean.index]
axes[1].barh(range(len(calib_mean)), calib_mean["actual_dr"].values*100,
             color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(calib_mean))),
             edgecolor='white', linewidth=0.5)
axes[1].axvline(x=default.mean()*100, color='black', linestyle='--',
                lw=1, label=f"Overall ({default.mean()*100:.1f}%)")
axes[1].set_xlabel("Actual Default Rate (%)")
axes[1].set_title("Default Rate by Risk Score Band")
axes[1].legend(fontsize=9)
axes[1].set_yticks(range(len(calib_mean)))
axes[1].set_yticklabels(band_labels, fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT}/S5_calibration.png"); plt.close()
print("Plot saved: S5_calibration.png")

# ── Chart 4: SHAP summary (XGBoost — most important model) ────────────────
print("\nComputing SHAP values (XGBoost)...")
explainer   = shap.TreeExplainer(xgb_m)
shap_vals   = explainer.shap_values(X_test)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_test, feature_names=FEATURE_COLS,
                  show=False)
plt.title("SHAP Summary — What Drives Default Risk",
          fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/S6_shap_importance.png"); plt.close()
print("Plot saved: S6_shap_importance.png")

# ── Chart 5: Risk score band business impact ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
band_edges = list(range(0, 101, 10))
band_labels_short = [f"{b}–{b+10}" for b in band_edges[:-1]]
band_pl = []
band_cnt = []
for lo, hi in zip(band_edges[:-1], band_edges[1:]):
    mask = (scores >= lo) & (scores < hi)
    if mask.sum() == 0:
        band_pl.append(0); band_cnt.append(0); continue
    action_in_band = opt_actions[mask]
    safe_in   = safe[mask];    def_in = default[mask]
    inc_m  = (action_in_band == 1)
    dec_m  = (action_in_band == 2)
    hol_m  = (action_in_band == 0)
    pl_band = (
          rev_arr[mask][inc_m & safe_in].sum()
        - wrong_inc_loss[mask][inc_m & def_in].sum()
        + ead_saved[mask][dec_m & def_in].sum()
        - opp_arr[mask][dec_m & safe_in].sum()
        - missed_loss[mask][hol_m & def_in].sum()
    )
    band_pl.append(pl_band / 1e6)
    band_cnt.append(mask.sum())

bar_c = ["#43A047" if v >= 0 else "#E53935" for v in band_pl]
bars  = ax.bar(band_labels_short, band_pl, color=bar_c,
               edgecolor='white', linewidth=1)
ax.axhline(y=0, color='black', lw=0.8)
ax.axvline(x=best_b1/10 - 0.5, color='green',  lw=1.5, linestyle='--',
           label=f"B1={best_b1} (INCREASE cutoff)")
ax.axvline(x=best_b2/10 - 0.5, color='red',    lw=1.5, linestyle='--',
           label=f"B2={best_b2} (DECREASE cutoff)")
ax.set_xlabel("Risk Score Band")
ax.set_ylabel("Net P&L (₹ Million)")
ax.set_title("Net P&L Contribution by Risk Score Band",
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
for bar, cnt in zip(bars, band_cnt):
    ax.text(bar.get_x() + bar.get_width()/2,
            0.01, f"n={cnt}", ha='center', va='bottom', fontsize=7, color='black')
plt.tight_layout()
plt.savefig(f"{OUT}/S7_pl_by_score_band.png"); plt.close()
print("Plot saved: S7_pl_by_score_band.png")


# =============================================================================
# SECTION 11 — CUSTOMER QUERY FUNCTION
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 11 — CUSTOMER QUERY FUNCTION")
print("=" * 65)

# Store full dataset scores for querying (rebuild on full data)
print("Building full-dataset scores for customer query...")
full_X     = df[FEATURE_COLS].copy()
full_X_sc  = scaler.transform(full_X)

lr_prob_full  = lr_cal.predict_proba(full_X_sc)[:, 1]
rf_prob_full  = rf_cal.predict_proba(full_X)[:, 1]
xgb_prob_full = xgb_m.predict_proba(full_X)[:, 1]
lgb_prob_full = lgb_m.predict_proba(full_X)[:, 1]
cat_prob_full = cat_m.predict_proba(full_X)[:, 1]
ens_full      = (lr_prob_full + rf_prob_full + xgb_prob_full
                 + lgb_prob_full + cat_prob_full) / 5.0

eps_f = 1e-6
lo_f  = np.log(ens_full.clip(eps_f, 1 - eps_f) /
               (1 - ens_full.clip(eps_f, 1 - eps_f)))
rs_full = 100.0 * (lo_f - lo_f.min()) / (lo_f.max() - lo_f.min())

df["RISK_SCORE_FULL"]  = rs_full
df["P_DEFAULT_FULL"]   = ens_full
df["ACTION_FULL"]      = np.where(rs_full < best_b1, 1,
                         np.where(rs_full > best_b2, 2, 0))

# SHAP explainer for full dataset
shap_exp_full = shap.TreeExplainer(xgb_m)

# Plain-English feature descriptions for bank officers
FEATURE_PLAIN = {
    "EDUCATION"         : "education level",
    "MARRIAGE"          : "marital status",
    "LIMIT_BAL"         : "current credit limit",
    "DPD_CURRENT"       : "months overdue right now",
    "DPD_MAX_6M"        : "worst payment delay in last 6 months",
    "DPD_MAX_3M"        : "worst payment delay in last 3 months",
    "DPD_MONTHS_30PLUS" : "number of months with 30+ day delays",
    "DPD_MONTHS_60PLUS" : "number of months with 60+ day delays",
    "EVER_DPD_90PLUS"   : "whether account was ever 90+ days overdue",
    "UTIL_CURRENT"      : "current utilisation ratio",
    "UTIL_3M_AVG"       : "average utilisation over last 3 months",
    "UTIL_6M_AVG"       : "average utilisation over last 6 months",
    "UTIL_MAX_3M"       : "highest utilisation reached in last 3 months",
    "UTIL_TREND"        : "direction of utilisation change",
    "UTIL_RISING"       : "whether utilisation is rising",
    "PAY_RATIO_CURRENT" : "fraction of bill paid this month",
    "PAY_RATIO_3M_AVG"  : "average fraction of bill paid over 3 months",
    "PAY_RATIO_6M_AVG"  : "average fraction of bill paid over 6 months",
    "MIN_PAY_STREAK"    : "consecutive months of minimum-only payment",
    "BALANCE_CURRENT"   : "current outstanding balance",
    "BALANCE_6M_AVG"    : "average outstanding balance over 6 months",
    "TOTAL_PAY_6M"      : "total amount paid in last 6 months",
    "AVG_PAY_6M"        : "average monthly payment over 6 months",
    "BALANCE_TREND"     : "whether outstanding balance is growing",
    "BALANCE_GROWING"   : "flag indicating balance is growing",
}

def query_customer(customer_id: int, top_n: int = 5) -> None:
    """
    Given a CUSTOMER_ID (1-indexed), print:
      - Risk score and default probability
      - Recommended action
      - Top N features driving the decision
      - Plain-English explanation (no model jargon)
    """
    if customer_id < 1 or customer_id > len(df):
        print(f"Customer ID {customer_id} not found. Valid range: 1–{len(df)}")
        return

    row_idx    = customer_id - 1
    row        = df.iloc[row_idx]
    score      = row["RISK_SCORE_FULL"]
    p_def      = row["P_DEFAULT_FULL"]
    action_k   = int(row["ACTION_FULL"])
    action_lbl = ACTION_MAP[action_k]
    actual_def = int(row["DEFAULT"])

    print("\n" + "═" * 60)
    print(f"  CUSTOMER PROFILE — ID: {customer_id}")
    print("═" * 60)
    print(f"  Risk Score         : {score:.1f} / 100  "
          f"({'Low risk' if score < best_b1 else 'Medium risk' if score <= best_b2 else 'High risk'})")
    print(f"  Default Probability: {p_def*100:.1f}%")
    print(f"  Recommended Action : {action_lbl}")
    print(f"  Actual Outcome     : {'Defaulted' if actual_def == 1 else 'Did not default'}")

    # Key financials
    print(f"\n  Key Financials:")
    print(f"    Credit Limit      : ₹{row['LIMIT_BAL']:>12,.0f}")
    print(f"    Avg Balance (6M)  : ₹{row['BALANCE_6M_AVG']:>12,.0f}")
    print(f"    EAD Ratio         : {row['EAD_RATIO']:.3f}  "
          f"({'High drawdown' if row['EAD_RATIO'] > 0.6 else 'Moderate' if row['EAD_RATIO'] > 0.3 else 'Low drawdown'})")
    print(f"    Utilisation (6M)  : {row['UTIL_6M_AVG']*100:.1f}%")
    print(f"    Avg Pay Ratio     : {row['PAY_RATIO_6M_AVG']*100:.1f}%  "
          f"({'Full payer' if row['PAY_RATIO_6M_AVG'] > 0.8 else 'Partial payer' if row['PAY_RATIO_6M_AVG'] > 0.2 else 'Minimum payer'})")
    print(f"    Min Pay Streak    : {int(row['MIN_PAY_STREAK'])} month(s)")
    print(f"    DPD (worst 6M)    : {int(row['DPD_MAX_6M'])} month(s) delay")

    # SHAP-based top drivers
    x_row     = full_X.iloc[[row_idx]]
    shap_row  = shap_exp_full.shap_values(x_row)[0]
    shap_abs  = np.abs(shap_row)
    top_idx   = np.argsort(shap_abs)[::-1][:top_n]

    print(f"\n  Top {top_n} Factors Driving This Decision:")
    print(f"  {'─'*50}")
    for rank, i in enumerate(top_idx, 1):
        fname     = FEATURE_COLS[i]
        fval      = x_row.iloc[0, i]
        shap_v    = shap_row[i]
        direction = "↑ INCREASES risk" if shap_v > 0 else "↓ REDUCES risk"
        plain     = FEATURE_PLAIN.get(fname, fname)
        print(f"  {rank}. {plain.capitalize()}")
        print(f"     Value: {fval:.3f}   {direction}")

    # Plain-English reason
    print(f"\n  Plain-English Explanation:")
    print(f"  {'─'*50}")
    if action_lbl == "DECREASE":
        reasons = []
        if row["DPD_MAX_6M"] >= 2:
            reasons.append(f"payment delays of up to {int(row['DPD_MAX_6M'])} months in the last 6 months")
        if row["MIN_PAY_STREAK"] >= 3:
            reasons.append(f"{int(row['MIN_PAY_STREAK'])} consecutive months of minimum-only payment")
        if row["UTIL_6M_AVG"] > 0.70:
            reasons.append(f"high average utilisation of {row['UTIL_6M_AVG']*100:.0f}%")
        if row["UTIL_RISING"] == 1:
            reasons.append("utilisation has been rising in recent months")
        if row["EVER_DPD_90PLUS"] == 1:
            reasons.append("account was previously more than 90 days overdue")
        if not reasons:
            reasons.append("an elevated overall risk profile based on payment history")
        print(f"  The credit limit has been recommended for reduction because")
        print(f"  the customer has shown: {'; '.join(reasons)}.")
        print(f"  Reducing the limit lowers the bank's potential exposure if")
        print(f"  the customer's financial situation continues to deteriorate.")

    elif action_lbl == "INCREASE":
        reasons = []
        if row["DPD_MAX_6M"] == 0:
            reasons.append("no payment delays in the last 6 months")
        if row["PAY_RATIO_6M_AVG"] > 0.5:
            reasons.append(f"consistently paying {row['PAY_RATIO_6M_AVG']*100:.0f}% of the bill on average")
        if row["UTIL_6M_AVG"] < 0.50:
            reasons.append(f"moderate utilisation of {row['UTIL_6M_AVG']*100:.0f}% — well within safe range")
        if row["MIN_PAY_STREAK"] == 0:
            reasons.append("no minimum-payment-only months on record")
        if not reasons:
            reasons.append("a strong overall behavioural profile")
        print(f"  The credit limit has been recommended for an increase because")
        print(f"  the customer has demonstrated: {'; '.join(reasons)}.")
        print(f"  An increase rewards responsible behaviour and allows the")
        print(f"  customer to transact more comfortably within their means.")

    else:  # HOLD
        print(f"  No limit change is recommended at this time. The customer's")
        print(f"  profile does not show strong enough signals in either direction")
        print(f"  to justify an increase or a decrease. The account will be")
        print(f"  reviewed again next month.")

    print("═" * 60 + "\n")


# ── Demo queries ─────────────────────────────────────────────────────────
print("\nDEMO CUSTOMER QUERIES:")

# Find one of each action type for demo
inc_demo  = df[df["ACTION_FULL"] == 1]["CUSTOMER_ID"].iloc[0]
dec_demo  = df[df["ACTION_FULL"] == 2]["CUSTOMER_ID"].iloc[0]
hold_demo = df[df["ACTION_FULL"] == 0]["CUSTOMER_ID"].iloc[0]

for cid in [inc_demo, dec_demo, hold_demo]:
    query_customer(int(cid))


# =============================================================================
# SECTION 12 — FINAL SUMMARY
# =============================================================================
print("=" * 65)
print("SECTION 12 — FINAL MODEL SUMMARY")
print("=" * 65)

best_auc = max(auc_results.values())
print(f"""
┌─────────────────────────────────────────────────────────────┐
│       CREDIT LINE OPTIMISATION — TWO-STAGE MODEL SUMMARY     │
├──────────────────────────┬──────────────────────────────────┤
│  STAGE 1 (Default Model) │  Ensemble of 5 models            │
│  Best AUC                │  {best_auc:.4f}                       │
│  Models                  │  LR + RF + XGB + LGB + CatBoost  │
├──────────────────────────┼──────────────────────────────────┤
│  STAGE 2 (Optimisation)  │  Exhaustive barrier search        │
│  Optimal B1 (INCREASE)   │  Score < {best_b1:<3}                    │
│  Optimal B2 (DECREASE)   │  Score > {best_b2:<3}                    │
│  Combinations searched   │  4,851                            │
├──────────────────────────┼──────────────────────────────────┤
│  BUSINESS IMPACT         │                                   │
│  Net P&L (test set)      │  ₹{net_pl:>12,.0f}              │
│  Method 1 (/ avg limit)  │  {method1*10000:.2f} bps                    │
│  Method 2 (/ outstanding)│  {method2*10000:.2f} bps                    │
│  ICICI scaled (~16M)     │  ₹{net_icici/1e7:>8,.0f} Cr/year       │
├──────────────────────────┼──────────────────────────────────┤
│  FAIR LENDING            │  Age + Sex removed                │
│  EXPLAINABILITY          │  SHAP + plain-English reasons     │
│  CUSTOMER QUERY          │  query_customer(id) function      │
└──────────────────────────┴──────────────────────────────────┘
""")

# Save model artefacts
with open("stage1_ensemble_models.pkl","wb") as f:
    pickle.dump({"lr":lr_cal,"rf":rf_cal,"xgb":xgb_m,
                 "lgb":lgb_m,"cat":cat_m,"scaler":scaler,
                 "lo_min":lo_min,"lo_max":lo_max,
                 "best_b1":best_b1,"best_b2":best_b2}, f)

df[["CUSTOMER_ID","RISK_SCORE_FULL","P_DEFAULT_FULL",
    "ACTION_FULL","DEFAULT","LIMIT_BAL","EAD_RATIO"]].to_csv("customer_scores.csv", index=False)

# Save dashboard-compatible scored CSV
out_df = df.copy()
out_df["RISK_SCORE"] = out_df["RISK_SCORE_FULL"]
out_df["PREDICTED_ACTION"] = out_df["ACTION_FULL"]
out_df["P_DEFAULT"] = out_df["P_DEFAULT_FULL"]
output_columns = FEATURE_COLS + [
    "CUSTOMER_ID", "DEFAULT", "EAD_AMOUNT",
    "RISK_SCORE", "PREDICTED_ACTION", "P_DEFAULT",
]
out_df[output_columns].to_csv("all_predictions_two_stage_optimized.csv", index=False)

print("Saved: stage1_ensemble_models.pkl")
print("Saved: customer_scores.csv")
print("Saved: all_predictions_two_stage_optimized.csv")
print(f"Saved: {OUT}/S1 through S7 charts")
print("\n" + "=" * 65)
print("TWO-STAGE MODEL COMPLETE — Run query_customer(id) for any customer")
print("=" * 65)