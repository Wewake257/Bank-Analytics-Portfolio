"""
Bank Analytics Project 4: Real-Time Financial Transaction Fraud & Anomaly Detection
Author: Vivek Kumar
Techniques: Isolation Forest, HistGradientBoosting (LightGBM/XGBoost native equivalent), Precision-Recall AUC, Extreme Class Imbalance (0.2% fraud)
Dataset: ULB Machine Learning Group / Kaggle Credit Card Fraud Detection Dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc, classification_report

def run_fraud_detection_pipeline():
    print("=== BANK ANALYTICS: REAL-TIME TRANSACTION FRAUD & ANOMALY DETECTION ===")
    
    # 1. Simulate 20,000 Financial Transactions with 0.2% Extreme Fraud Imbalance
    np.random.seed(42)
    n = 20000
    n_fraud = int(n * 0.002) # ~40 fraudulent transactions
    n_normal = n - n_fraud
    
    # Normal transaction features
    normal_amounts = np.random.exponential(65, n_normal).clip(1, 1500)
    normal_pca1 = np.random.normal(0, 1, n_normal)
    normal_pca2 = np.random.normal(0, 1, n_normal)
    normal_pca3 = np.random.normal(0, 1, n_normal)
    normal_flags = np.zeros(n_normal, dtype=int)
    
    # Fraud transaction features (anomalous distribution)
    fraud_amounts = np.random.uniform(250, 3000, n_fraud)
    fraud_pca1 = np.random.normal(3.5, 1.2, n_fraud)
    fraud_pca2 = np.random.normal(-2.8, 1.0, n_fraud)
    fraud_pca3 = np.random.normal(2.5, 1.5, n_fraud)
    fraud_flags = np.ones(n_fraud, dtype=int)
    
    df = pd.DataFrame({
        'amount': np.concatenate([normal_amounts, fraud_amounts]),
        'V1': np.concatenate([normal_pca1, fraud_pca1]),
        'V2': np.concatenate([normal_pca2, fraud_pca2]),
        'V3': np.concatenate([normal_pca3, fraud_pca3]),
        'is_fraud': np.concatenate([normal_flags, fraud_flags])
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total Transactions: {len(df)} | Fraud Count: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 2. Unsupervised Anomaly Detection: Isolation Forest
    iso_forest = IsolationForest(contamination=0.002, random_state=42)
    iso_forest.fit(X_train)
    iso_scores = -iso_forest.decision_function(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, iso_scores)
    pr_auc_iso = auc(recall, precision)
    print(f"[UNSUPERVISED] Isolation Forest PR-AUC: {pr_auc_iso:.4f}")
    
    # 3. Supervised Gradient Boosting with Class Weight for Extreme Imbalance
    clf = HistGradientBoostingClassifier(max_iter=80, max_depth=3, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_gb = auc(recall_gb, precision_gb)
    print(f"[SUPERVISED] Gradient Boosting with Imbalance Weighting PR-AUC: {pr_auc_gb:.4f}")
    
    print("\nFraud Detection Pipeline executed successfully.")

if __name__ == '__main__':
    run_fraud_detection_pipeline()
