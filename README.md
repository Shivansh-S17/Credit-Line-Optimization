# Credit Line Optimisation: Dynamic Limit Management

## 📌 Project Overview

This repository contains a **Two-Stage Machine Learning Framework** developed during my internship at **ICICI Bank** (Data Science & Analytics Group).

The system systematically reviews credit card portfolios to optimize customer credit limits. Instead of relying on static limits or purely statistical thresholds, this framework dynamically recommends whether to **INCREASE**, **HOLD**, or **DECREASE** a customer's credit limit by directly maximizing the bank's Net Profit & Loss (P&L).

## 🧠 Two-Stage Architecture

### Stage 1: Risk Assessment (Default Probability)
* Engineered 25 financial and behavioral features from 6 months of historical data. Demographic variables (Age, Gender) were excluded for **Fair Lending Compliance**.
* **Champion Model:** Random Forest (Gini: 0.5270, AUC: 0.7635).
* Transformed raw probabilities into a **0–100 Risk Score** using Log-Odds scaling.

### Stage 2: Exhaustive Barrier Optimization
* The risk scores feed into a financial simulator using banking constants (Revolve Rate, Interest, Interchange, EAD, LGD).
* Tested **4,851 combinations** of barrier thresholds to maximize Net P&L. 
* **Optimal Decision Barriers:** B1 = 45 and B2 = 46.

## 📊 Business Impact

On a test set of 6,000 customers:
* **Net Profit Impact:** **+₹0.74 Crore** (₹74.48 Lakhs)
* **Portfolio Metrics:** Extracts **74 bps** return on Total Credit Limit and **272 bps** on Total Outstanding Balance.

## 🖥️ Explainability & Dashboard (Decision Desk)

The project includes a **Flask-based interactive web dashboard**:
* **SHAP values** provide granular feature explainability.
* Integrated with **Google's Gemini AI API** to translate ML metrics into plain-English business rationales.

## 📂 Project Structure

```text
├── 1_eda.py                                # Exploratory analysis
├── 2_feature_engineering.py                # Behavioral feature creation
├── 3_model.py                              # Model training & SHAP
├── 4_model_two_stage_optimized.py          # P&L optimization
├── Dashboard.html                          # Frontend UI
├── static/ & templates/                    # Flask assets
└── requirements.txt                        # Dependencies
