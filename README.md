# End-to-End Fraud Detection System

🔗 **Live Demo:** https://end-to-end-fraud-detection.onrender.com

A production-oriented fraud detection pipeline built on a large-scale financial transaction dataset (6.3M+ records, 0.13% fraud rate).


This project demonstrates the complete machine learning lifecycle:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Imbalanced Classification Handling
- Threshold Optimization
- Risk-Based Decision System
- Deployment with Streamlit

---

# 1. Executive Summary

Fraud detection presents a classic imbalanced classification challenge. In this dataset, fraudulent transactions represent only **0.13%** of all records.

A Random Forest model with class balancing and validation-based threshold optimization was implemented to improve fraud detection precision while maintaining strong recall.

### Final Test Performance

- Fraud Precision: **0.94**
- Fraud Recall: **0.74**
- Fraud F1-Score: **0.82**
- ROC-AUC: **0.9989**
- PR-AUC: **0.9041**

The model is deployed via Streamlit and provides real-time fraud probability scoring with operational risk categorization.

---

# 2. Business Problem

Financial institutions process millions of transactions daily. Even a small fraction of fraud can lead to substantial financial losses.

### Key Challenges

- Severe class imbalance (0.13% fraud)
- High cost of false negatives (missed fraud)
- Operational burden from false positives

### Objective

Develop a robust fraud detection system that:
- Maximizes fraud detection accuracy
- Minimizes unnecessary investigation workload
- Supports operational risk-based decision-making

---
# 3. Dataset Overview

Source: Kaggle – Fraud Detection Dataset  
🔗 https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

- Total Transactions: 6,362,620
- Fraudulent Transactions: 8,213
- Fraud Rate: 0.13%
- Transaction Types:
  - CASH_IN
  - CASH_OUT
  - DEBIT
  - PAYMENT
  - TRANSFER

EDA revealed that fraud is heavily concentrated in:
- TRANSFER
- CASH_OUT

Due to GitHub file size limitations, the dataset is not included in this repository.  
To run this project locally, download the dataset from Kaggle and place the CSV file in the project root directory.

# 4. Methodology

## 4.1 Exploratory Data Analysis (EDA)

EDA was conducted to understand fraud behavior patterns and ensure engineered features reflect real-world transaction anomalies.

- Identify class imbalance severity
- Analyze fraud distribution by transaction type
- Detect abnormal balance transitions
- Understand transaction amount distributions (log-transformed)
- Examine correlations among financial variables

---

## 4.2 Feature Engineering

Two behavioral features were engineered:

```
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
balanceDiffDest = newbalanceDest - oldbalanceDest
```

Why?

Fraudulent transactions often exhibit abnormal balance changes:
- Sudden account depletion
- Irregular receiver balance behavior

These engineered features capture transactional behavior rather than relying only on raw balances.

---

## 4.3 Train / Validation / Test Split

A 3-way split was used to prevent data leakage.  
Threshold tuning was performed on the validation set to preserve the integrity of final test performance.

- Train Set → Model training
- Validation Set → Threshold tuning
- Test Set → Final unbiased evaluation

---

## 4.4 Baseline Model

Logistic Regression (class-weight balanced)

Result:

High recall (0.94) but extremely low precision (0.02), resulting in excessive false positives.
This indicated need for a more expressive model.

---

## 4.5 Final Model

Random Forest Classifier with:

- `class_weight="balanced"`
- Scikit-learn Pipeline
- ColumnTransformer (StandardScaler + OneHotEncoder)

Why Random Forest?

- Handles non-linear interactions
- Robust to outliers
- Effective for tabular financial data
- Performs well in imbalanced classification with class weighting

---

## 4.6 Threshold Optimization

Default probability threshold (0.5) is not optimal for imbalanced datasets.

Using the validation set:

```
Best threshold ≈ 0.97
```

Why threshold tuning?

- Improves fraud precision
- Reduces false positives
- Aligns model behavior with business risk tolerance

---

# 5. Model Performance

After threshold optimization:

| Metric    | Fraud Class |
|-----------|-------------|
| Precision | 0.94 |
| Recall    | 0.74 |
| F1-score  | 0.82 |
| PR-AUC    | 0.9041 |


### Interpretation

- High precision reduces unnecessary manual investigations.
- Strong recall ensures majority of fraud cases are detected.
- Balanced trade-off suitable for financial operations.

---

# 6. Risk-Based Categorization System

Transactions are categorized into:

- 🟢 Low Risk
- 🟠 High Risk
- 🔴 Fraud

Why?

Not all risky transactions require blocking.  
This system enables operational prioritization:

- Auto-block high-confidence fraud
- Route high-risk transactions for manual review
- Allow low-risk transactions to proceed

---

# 7. Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

# 8. Skills Demonstrated

- Large-scale data handling (6.3M+ rows)
- Imbalanced classification techniques
- Feature engineering for financial data
- Pipeline-based model architecture
- Validation-based threshold tuning
- Model evaluation (ROC-AUC, PR-AUC, F1)
- Risk-based decision system design
- Production deployment with Streamlit

---

# 9. Future Improvements

- Hyperparameter tuning (RandomizedSearchCV)
- Compare with XGBoost / LightGBM
- SHAP for model interpretability
- Cost-sensitive evaluation using fraud-loss simulation

---

# Conclusion

This project demonstrates the design of a practical fraud detection system that balances statistical rigor with business considerations.

It integrates:
- Data exploration
- Behavioral feature engineering
- Imbalanced learning strategies
- Validation discipline
- Operational risk modeling
- Deployment

The result is a deployable, threshold-aware fraud detection solution suitable for real-world financial environments.
