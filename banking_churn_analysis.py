"""
Bank Analytics Project 2: Digital Banking Customer Churn & Transaction Analytics
Author: Vivek Kumar
Techniques: SQL Window Functions simulation, RFM (Recency, Frequency, Monetary) Segmentation, Random Forest, Feature Importance
Dataset: Bank Customer Churn / Kaggle Banking Analytics Dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_banking_churn_pipeline():
    print("=== BANK ANALYTICS: DIGITAL BANKING CHURN & TRANSACTION ANALYTICS ===")
    
    # 1. Simulate 10,000 Retail Banking Customer Records
    np.random.seed(42)
    n = 10000
    
    customer_ids = [f"BANK_{i:06d}" for i in range(1, n + 1)]
    age = np.random.randint(18, 75, n)
    balance = np.random.exponential(25000, n).round(2)
    num_products = np.random.choice([1, 2, 3, 4], p=[0.50, 0.45, 0.04, 0.01], size=n)
    has_credit_card = np.random.choice([1, 0], p=[0.7, 0.3], size=n)
    is_active_member = np.random.choice([1, 0], p=[0.52, 0.48], size=n)
    estimated_salary = np.random.uniform(20000, 150000, n).round(2)
    tenure_years = np.random.randint(0, 10, n)
    
    # Transactional Recency, Frequency, Monetary (RFM) metrics
    recency_days = np.random.exponential(15, n).clip(1, 180).round()
    txn_frequency_monthly = np.random.poisson(12, n)
    
    # Churn probability calculation
    churn_prob = (
        0.35 * (num_products > 2)
        + 0.40 * (is_active_member == 0)
        + 0.25 * (recency_days > 45)
        + 0.15 * (age > 50)
        - 0.20 * (num_products == 2)
        + 0.10
    ).clip(0.02, 0.95)
    
    churn_flag = (np.random.uniform(0, 1, n) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'balance': balance,
        'num_products': num_products,
        'has_credit_card': has_credit_card,
        'is_active_member': is_active_member,
        'estimated_salary': estimated_salary,
        'tenure_years': tenure_years,
        'recency_days': recency_days,
        'txn_frequency_monthly': txn_frequency_monthly,
        'churn_flag': churn_flag
    })
    
    print(f"Customer base: {n} accounts | Churn rate: {df['churn_flag'].mean():.2%}")
    
    # 2. RFM Segmentation
    df['r_score'] = pd.qcut(df['recency_days'], 4, labels=[4, 3, 2, 1])
    df['f_score'] = pd.qcut(df['txn_frequency_monthly'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    df['m_score'] = pd.qcut(df['balance'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    # 3. Model Training
    feature_cols = ['age', 'balance', 'num_products', 'has_credit_card', 
                    'is_active_member', 'estimated_salary', 'tenure_years', 
                    'recency_days', 'txn_frequency_monthly']
    
    X = df[feature_cols]
    y = df['churn_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print(f"\nRandom Forest ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nFeature Importances:")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"  - {feat:25s}: {imp:.4f}")
        
    print("\nDigital Banking Churn model executed successfully.")

if __name__ == '__main__':
    run_banking_churn_pipeline()
