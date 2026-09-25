# E-Commerce Fraud Detection Analysis

**Academic Project**: MBA Data Science & Data Analytics  
**Subject**: Fraud Detection Analysis  
**Dataset**: Actual IEEE-CIS Fraud Detection Dataset  
**Validation Strategy**: Time-Aware Chronological Split (80% Train, 20% Validation)  

---

## 1. Project Overview & Objective

Digital payment systems process millions of online transactions daily and face significant financial losses due to fraudulent activity. 

The primary objective of this project is to perform an end-to-end, data-driven academic **Fraud Detection Analysis** on the IEEE-CIS Fraud Detection dataset to answer two core analytical questions:
1. **What characteristics and patterns are associated with fraudulent transactions?**
2. **How effectively can these patterns be used to detect fraudulent transactions?**

---

## 2. Project Structure

```text
fraud_detection_analysis/
│
├── data/
│   ├── raw/
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   └── processed/
│       ├── X_train.parquet
│       ├── X_val.parquet
│       ├── y_train.npy
│       └── y_val.npy
│
├── notebooks/
│   ├── 01_data_and_fraud_analysis.ipynb
│   ├── 02_preprocessing_and_features.ipynb
│   └── 03_modeling_and_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── modeling.py
│
├── outputs/
│   ├── figures/
│   │   ├── fraud_distribution.png
│   │   ├── missing_value_analysis.png
│   │   ├── transaction_amount_fraud.png
│   │   ├── fraud_by_product.png
│   │   ├── fraud_by_card_type.png
│   │   ├── fraud_by_email_domain.png
│   │   ├── fraud_by_time.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_decision_tree.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── confusion_matrix_histgradientboosting.png
│   │   ├── roc_curves.png
│   │   ├── precision_recall_curves.png
│   │   ├── threshold_tradeoff.png
│   │   └── feature_importance.png
│   └── tables/
│       ├── dataset_summary.csv
│       ├── fraud_distribution.csv
│       ├── missing_value_summary.csv
│       ├── fraud_rate_product.csv
│       ├── fraud_rate_card4.csv
│       ├── fraud_rate_card6.csv
│       ├── fraud_rate_identity.csv
│       ├── fraud_rate_amt_bands.csv
│       ├── feature_engineering_summary.csv
│       ├── model_comparison.csv
│       └── threshold_analysis.csv
│
├── README.md
├── project_report.md
└── requirements.txt
```

---

## 3. Analytical Methodology & Pipeline

```text
Dataset Understanding 
  → Data Quality Analysis 
  → Target Variable Analysis 
  → Class Imbalance & Accuracy Paradox 
  → Fraud Pattern Analysis 
  → Fraud-Oriented Feature Engineering 
  → Fraud Detection Models 
  → Model Evaluation 
  → False Positive / False Negative Error Analysis 
  → Decision Threshold Analysis 
  → Model Interpretation 
  → Final Fraud Detection Findings
```

### Key Methodology Decisions:
1. **Left Join Strategy**: Merged `train_identity.csv` onto `train_transaction.csv` using `TransactionID` to preserve all **590,540 transaction records**. The merged dataset contains **434 total columns**, including `TransactionID` and `isFraud`. The remaining 432 columns represent candidate transaction/identity variables.
2. **Leakage-Free Temporal Split**: Transactions are sorted chronologically by `TransactionDT` (first 80% time for training = 472,432 rows; final 20% time for validation = 118,108 rows). Standard random K-Fold cross-validation was strictly avoided to prevent look-ahead bias. `TransactionID` and `isFraud` are strictly excluded from feature matrix `X`.
3. **Leakage-Free Preprocessing**: Categorical encoding and missing value imputations were fitted strictly on `X_train` and applied to `X_val`.

---

## 4. Empirical Model Performance Comparison

Evaluated on the out-of-time validation dataset (**118,108 transactions; 4,064 actual frauds**):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Primary Metric) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 96.56% | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0344 | 0 | 0 | 4,064 | 114,044 |
| **HistGradientBoosting** ⭐ | **81.72%** | **0.1137** | **0.6348** | **0.1929** | **0.8037** | **0.2289** | **2,580** | **20,106** | **1,484** | **93,938** |
| **Random Forest** | 85.87% | 0.1315 | 0.5541 | 0.2125 | 0.7938 | 0.2008 | 2,252 | 14,875 | 1,812 | 99,169 |
| **Decision Tree** | 79.37% | 0.0988 | 0.6152 | 0.1702 | 0.7662 | 0.1789 | 2,500 | 22,807 | 1,564 | 91,237 |
| **Logistic Regression** | 75.23% | 0.0809 | 0.5987 | 0.1426 | 0.7329 | 0.1136 | 2,433 | 27,625 | 1,631 | 86,419 |

### Metric Evaluation Rationale:
- **Why Accuracy is Misleading**: The Majority-Class Baseline predicts all transactions as legitimate and achieves **96.56% accuracy**, yet catches **0% of fraud (0 Recall)**.
- **Primary Metric Justification**: Due to extreme class imbalance (3.50% fraud rate), **PR-AUC (Precision-Recall AUC)** is chosen as the primary metric because it evaluates Precision directly against Recall without distortion from True Negatives.
- **Top Classifier**: **HistGradientBoostingClassifier** achieved the highest PR-AUC (**0.2289**) and ROC-AUC (**0.8037**), catching **63.48% of all fraudulent transactions** (2,580 out of 4,064).



---

## 5. Classification Decision Threshold Analysis

Evaluating decision thresholds for **HistGradientBoosting**:

| Decision Threshold | Precision | Recall (Detection Rate) | F1-Score | Analytical Trade-off |
| :---: | :---: | :---: | :---: | :--- |
| **0.30** | 0.0593 | **86.27%** | 0.1109 | High Fraud Interception; Increased False Alarms |
| **0.40** | 0.0795 | 76.43% | 0.1440 | Balanced Fraud Interception |
| **0.50** | 0.1137 | 63.48% | 0.1929 | Standard Classification Baseline |
| **0.60** | 0.1536 | 51.80% | 0.2370 | Reduced False Positives |
| **0.70** | **0.2046** | 41.26% | **0.2736** | High Precision Flagging; Missed Fraud Increases |

*Changing the classification decision threshold shifts the operational balance between catching more fraudulent transactions (higher Recall) and reducing false alerts on legitimate customers (higher Precision).*

---

## 6. Viva Defense Quick Guide (Q&A)

**Q1: What is the core objective of this project?**  
*Answer*: To perform a Fraud Detection Analysis on the IEEE-CIS dataset to identify transaction patterns associated with fraud and evaluate how effectively supervised machine learning algorithms detect fraudulent transactions.

**Q2: Why did you use a Left Join for transaction and identity data?**  
*Answer*: Identity metadata is available for only 24.42% of transactions (144,233 rows). An inner join would drop 75.58% of transactions. A left join preserves all 590,540 transaction records while allowing identity missingness to serve as an explicit feature.

**Q3: Why is standard K-Fold Cross-Validation inappropriate?**  
*Answer*: Transactions have a natural temporal order (`TransactionDT`). Random K-Fold shuffles future transactions into training sets (look-ahead bias). A time-aware chronological split (80% early train, 20% late validation) simulates operational deployment.

**Q4: Why is Accuracy misleading for imbalanced fraud datasets?**  
*Answer*: In a dataset with 3.50% fraud, predicting every transaction as legitimate yields **96.56% accuracy** but detects 0% of fraud. PR-AUC, Recall, and Precision are the necessary metrics.

**Q5: What do model-important features represent?**  
*Answer*: They indicate which variables contributed most strongly to the model's predictions. They indicate statistical association and predictive utility, not direct causes of fraud.

---

## 7. How to Run the Project

### Environment Setup
```bash
git clone <repo_url>
cd fraud_detection_analysis
pip install -r requirements.txt
```

### Execution
Run the notebooks in order:
1. `jupyter notebook notebooks/01_data_and_fraud_analysis.ipynb`
2. `jupyter notebook notebooks/02_preprocessing_and_features.ipynb`
3. `jupyter notebook notebooks/03_modeling_and_evaluation.ipynb`
