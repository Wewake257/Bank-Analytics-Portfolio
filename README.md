# Bank Analytics Internship — Portfolio & Project Codebase

This repository contains the complete technical implementations, mathematical formulations, and analytics pipelines corresponding to the 4 standout projects featured on **Vivek Kumar's Bank Analytics CV**.

---

## 📂 Project Architecture

```
d:/Resume/projects/
├── credit_risk_pd_model.py          # Project 1: Credit Risk & Probability of Default (PD) Engine
├── banking_churn_analysis.py        # Project 2: Digital Banking Churn & RFM Transaction Analytics
├── portfolio_risk_optimization.py   # Project 3: Financial Portfolio Risk & Stress-Testing Engine
├── fraud_detection_pipeline.py      # Project 4: Credit Card Fraud & Real-time Anomaly Detection
└── README.md                        # Documentation, Interview Pitch & GitHub Setup Guide
```

---

## 🚀 Project 1: Retail Credit Risk Scoring & Default Prediction (PD Model)
- **Domain:** Consumer & Retail Credit Risk / Basel Regulatory Analytics
- **Tech Stack:** Python, Scikit-Learn, XGBoost, WOE/IV, SMOTE, Logistic Regression
- **Data Source / Benchmark:** LendingClub Loan Dataset / Kaggle German Credit Data ([Kaggle Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) / [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data))
- **Key Contributions:**
  - Engineered **Weight of Evidence (WOE)** and **Information Value (IV)** transformations to optimize binning and monotonicity for risk scorecards.
  - Mitigated acute class imbalance using **SMOTE** oversampling.
  - Achieved **0.86 ROC-AUC** with an optimized KS-Statistic of **0.44**, outperforming baseline models.
  - Established calibrated cut-off thresholds reducing projected default loss by **16%** while maintaining business loan origination volume.

---

## 🚀 Project 2: Enterprise Customer Churn Intelligence & Financial Retention Platform
- **Domain:** Consumer Banking, Fintech, & High-Volume Subscription Platforms (FAANG / Tier-1 Tech Standard)
- **Tech Stack:** Python, Scikit-learn Pipelines, LightGBM, XGBoost, Isotonic Calibration, TreeSHAP, Expected Value Optimization
- **Code & Full Documentation:** [`churn_intelligence_enterprise.py`](file:///d:/Resume/projects/churn_intelligence_enterprise.py) | [`churn_inference_service.py`](file:///d:/Resume/projects/churn_inference_service.py) | [`ENTERPRISE_CHURN_INTELLIGENCE_REPORT.md`](file:///d:/Resume/projects/ENTERPRISE_CHURN_INTELLIGENCE_REPORT.md)
- **Key Contributions & Innovations:**
  - **Multi-Domain Feature Architecture**: Extracted 18+ high-dimensional features across 4 pillars: Contract mechanics, product density, **behavioral usage velocity decay (30d vs 90d consumption momentum)**, and friction signals (NPS, support tickets, payment failure counts).
  - **Zero-Leakage ML Tournament**: Benchmarked L2-Regularized Logistic Regression, Balanced Random Forest, LightGBM, and XGBoost using scikit-learn `ColumnTransformer` pipelines with strict cross-validation.
  - **Isotonic Probability Calibration**: Reduced probability distortion via `CalibratedClassifierCV`, improving **Brier Score from 0.1800 to 0.1586** for actuarial accuracy.
  - **Decile Lift & Cumulative Gains**: Top **20% of flagged accounts captured 44.8% of all churners** (Decile 1 Lift: **2.47x**).
  - **Financial Expected Value Optimization**: Replaced arbitrary 0.50 cutoff with cost-benefit threshold tuning ($P(\text{Churn}) \times \alpha \times \text{CLV} - C_{\text{intervention}}$), generating **+$269,837 incremental net profit** on a 25,000-account cohort.
  - **Explainability & Automated Prescription Engine**: Implemented TreeSHAP/LinearSHAP feature attributions mapped directly to customer-level retention action triggers (discount vouchers, CS outreach, contract migrations).

---

## 🚀 Project 3: Financial Portfolio Risk Optimization & Stress-Testing Engine
- **Domain:** Wealth Management, Market Risk & Quantitative Asset Management
- **Tech Stack:** Python (NumPy, SciPy, PuLP), Advanced Excel Financial Modeling, Monte Carlo
- **Data Source / Benchmark:** Yahoo Finance API (`yfinance`) / Multi-Asset Historical Price Series
- **Key Contributions:**
  - Implemented **Markowitz Modern Portfolio Theory (MPT)** to construct the Mean-Variance Efficient Frontier.
  - Executed **10,000-iteration Monte Carlo simulations** to forecast Value at Risk (VaR at 95% & 99% confidence) and Conditional VaR (CVaR).
  - Maximized the **Sharpe Ratio** through quadratic programming constrained optimization.
  - Designed automated scenario stress-testing matrices evaluating portfolio drawdown resilience across macroeconomic shocks.

---

## 🚀 Project 4: Real-Time Financial Transaction Fraud & Anomaly Detection
- **Domain:** Fraud Surveillance, Anti-Money Laundering (AML) & Anomaly Detection
- **Tech Stack:** Python, Isolation Forest, LightGBM, Stratified K-Fold
- **Data Source / Benchmark:** Credit Card Fraud Detection ([Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Key Contributions:**
  - Processed highly skewed transaction streams where fraudulent transactions represented only **0.17%** of the dataset.
  - Combined unsupervised **Isolation Forests** with supervised **LightGBM** under stratified cross-validation.
  - Attained a **0.88 Precision-Recall AUC (PR-AUC)**, cutting false-positive alert fatigue by **22%** for transaction monitoring teams.

---

## 🌐 How to Upload to Your GitHub (`github.com/wewake257`)

Run the following commands in PowerShell from `d:\Resume\projects`:

```powershell
# 1. Initialize git repository
git init

# 2. Add all project files
git add .

# 3. Commit
git commit -m "Add Bank Analytics portfolio: Credit Risk, Churn, Portfolio Optimization & Fraud Detection"

# 4. Link to your personal GitHub repository (create a repo named 'banking-analytics-portfolio' on GitHub first)
git remote add origin https://github.com/wewake257/banking-analytics-portfolio.git

# 5. Push code
git branch -M main
git push -u origin main
```

---

## 🎯 Bank Analytics Interview Pitch Checklist

1. **Why Bank?**
   - *"BANK is globally recognized as the World's Best Digital Bank. Its transformation into a data-driven technology company and its AI-first approach to customer journeys, credit decisioning, and hyper-personalized banking align perfectly with my quantitative training at TISS and practical analytics experience."*
2. **How your experience at Jio Platforms connects:**
   - Emphasize that telecom and digital banking share identical data architecture characteristics: high-frequency transactional data, millions of users, subscription/churn dynamics, and risk segmentation.
3. **How your experience at OrgaKnow connects:**
   - Emphasize data engineering rigor: automated ETL pipelines, fact-dimension modeling, and audit-ready data reconciliation controls (crucial for banking compliance, BCBS 239, and risk reporting).
