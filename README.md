# Classification of Business Bankruptcy
**Machine Learning Assignment**

## 📌 Overview
This repository contains the implementation and analysis of a **binary classification problem** aiming to predict whether a company will **remain healthy or go bankrupt** in the following year.

The project was developed in the context of the **Machine Learning** course and compares the performance of **four supervised learning classifiers** using real financial data from Greek companies (2006–2009).

---

## 🎯 Objectives
- Load and analyze financial data from an Excel dataset
- Compute **descriptive statistics** (min, max, mean) per year
- Normalize features to the range **[0, 1]**
- Train and evaluate multiple classification models
- Compare models using **confusion matrices and evaluation metrics**
- Export all results to a `.csv` file

---

## 🧠 Implemented Classifiers
The following classifiers were implemented using **scikit-learn**:

- Logistic Regression (with class balancing)
- k-Nearest Neighbors (k = 5)
- Gaussian Naive Bayes
- Support Vector Classifier (SVC)

---

## 📊 Evaluation Metrics
Each model is evaluated on both **training** and **test** sets using:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Accuracy
- Precision
- Recall
- F1 Score

Confusion matrices are visualized using **Seaborn heatmaps**.

---

## 🗂 Dataset
- Source: Excel file loaded directly from **Google Sheets**
- Years covered: **2006 – 2009**
- Target variable:
  - `1` → Healthy company
  - `2` → Bankrupt company (next year)

Dataset split:
- **80% Training set**
- **20% Test set**

---

## 🛠 Technologies & Libraries
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## 📁 Repository Structure

├── assignment-1.py # Main Python implementation  
├── assignment-1-report.pdf # Detailed analysis & conclusions  
├── Model Results.csv # Exported evaluation results  
└── README.md # Project documentation

---

## ▶️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/panagiotismisios/Classification-of-Business-Bankruptcy

pip install numpy pandas matplotlib seaborn scikit-learn

python assignment-1.py
```

---

## 📈 Key Findings

- kNN achieved the highest accuracy but poor recall on the test set.

- Gaussian Naive Bayes provided the most balanced performance and the best F1-score on test data.

- Logistic Regression and SVC showed sensitivity to class imbalance.

- Performance differences highlight the importance of representative train/test splits.
