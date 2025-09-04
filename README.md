# Anomaly Detection in Credit Card Transactions

This project implements an **unsupervised anomaly detection** pipeline on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
The goal is to detect fraudulent transactions using a **Gaussian-based probability model**, tuned for high recall (catching as many frauds as possible).

---

## 📌 Overview

1. **Data**
   - Dataset: `creditcard.csv` (284,807 transactions, 492 frauds).
   - Features: `V1..V28` (PCA components), `Time`, `Amount`, and `Class` (target).

2. **Preprocessing & Feature Engineering**
   - Train only on *authentic transactions* (`Class = 0`).
   - Split `Time` into `Day`, `Hour`, `Minute`, `Second`.
   - Log-transform transaction amount:  
     `Amount_transformed = log10(Amount + 0.001)`.
   - Selected features for modeling:  
     `['V4','V11','V12','V14','V16','V17','V18','V19','Hour']`.

3. **Model**
   - Each selected feature is modeled as a **univariate Gaussian** using training data:
     - Compute mean `μ` and std `σ` for each feature.
     - Per-feature density:  

       \[
       f(x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}} \exp\!\Big(-\frac{1}{2}\Big(\frac{x-\mu}{\sigma}\Big)^2\Big)
       \]

   - Joint probability of a record is the **product of all feature densities** (assumes independence).

4. **Threshold Tuning**
   - A transaction is flagged as fraud if joint probability < `ε`.
   - Threshold is chosen via validation set:
     - Search over `α` in `[0.001, 0.05]`.
     - Set `ε = α^n` where `n = number of features`.
     - Select `α` that maximizes **F2-score** (recall weighted more than precision).

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1, F2, MCC.
   - Tools: Confusion matrix + heatmap.
   - Evaluation is done on both validation and test sets.

---

## 🛠 Dependencies

Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
