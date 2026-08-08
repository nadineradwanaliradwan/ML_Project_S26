# Mushroom Edibility Classification — CSE381 Introduction to Machine Learning

Milestone 1 of the CSE381 course project (Summer 2026), Faculty of Engineering, Ain Shams University.

The goal is to predict whether a mushroom is **edible (`e`)** or **poisonous (`p`)** from its physical characteristics — a supervised binary classification task on the UCI Mushroom dataset.

---

## Repository Contents

| File | Description |
|------|-------------|
| `ML_summer_26.ipynb` | Full pipeline: data loading, preprocessing, EDA, baseline model, evaluation |
| `ML_Project.pdf` | Milestone 1 report (methodology, figures, results, AI usage log, references) |

Colab notebook: https://colab.research.google.com/drive/1gHk7dHS_0-hDRb7unL6lWuB5pIbFG4ar?usp=sharing

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Mushroom (id=73)](https://archive.ics.uci.edu/dataset/73/mushroom)
- Loaded programmatically with the `ucimlrepo` package — no manually modified copy is used.
- **Samples:** 8,124
- **Original features:** 22, all categorical
- **Target:** edibility — 4,208 edible (51.8%) / 3,916 poisonous (48.2%)
- **Duplicates:** 0
- **Missing values:** 2,480, all in `stalk-root`

---

## Requirements

```bash
pip install ucimlrepo pandas scikit-learn matplotlib seaborn
```

Python 3.9+. Runs as-is in Google Colab.

---

## How to Run

1. Open `ML_summer_26.ipynb` in Colab or Jupyter.
2. Run all cells top to bottom. The first cell installs `ucimlrepo`; the dataset is fetched at runtime, so no local data file is needed.
3. All figures and metrics are regenerated from the notebook outputs.

A fixed `random_state=42` is used for the split, so results are reproducible.

---

## Pipeline

**1. Feature reduction**
`odor` and `spore-print-color` are dropped. Both are near-perfect single predictors of edibility and would make the task trivial. 20 features remain.

**2. Train–test split**
80/20 stratified split → 6,499 training / 1,625 testing samples. The split happens *before* any value is learned from the data.

**3. Missing-value imputation**
`stalk-root` is filled by mode imputation. The mode is computed **from `X_train` only** and applied to both sets, to avoid leakage from the test set.

**4. Encoding**
One-Hot Encoding via `pd.get_dummies()`, then `align(join='left', fill_value=0)` so train and test share identical columns in identical order. Final shape: **98 encoded features**, 0 missing values in both sets.

**5. EDA**
Four captioned figures: class distribution, `gill-size`, `bruises`, and `habitat` vs. the target. Consistent color coding (green = edible, red = poisonous) and category codes replaced with readable labels.

**6. Baseline model**
`LogisticRegression(max_iter=1000)`, no hyperparameter tuning — this is the reference point, not the final model. Evaluated with 5-fold cross-validation on the training set, then fit on the full training set and scored once on the held-out test set.

---

## Results

The **poisonous** class is treated as the positive class, since a missed poisonous mushroom is the costlier error.

| Metric | Value |
|--------|-------|
| Mean 5-fold CV accuracy | 99.49% |
| Test accuracy | 99.57% |
| Precision (poisonous) | 100.00% |
| Recall (poisonous) | 99.11% |
| F1-score (poisonous) | 99.55% |
| ROC-AUC | 0.99996 |

**Confusion matrix (test set, n = 1,625):**

| True \ Predicted | Edible | Poisonous |
|------------------|--------|-----------|
| **Edible** | 842 | 0 |
| **Poisonous** | 7 | 776 |

All 7 errors are false negatives — poisonous mushrooms predicted as edible, which is the safety-critical direction. The model produced no false alarms. CV accuracy and test accuracy agree closely, indicating a stable estimate rather than a lucky split.

---

## Limitations

- Results describe this UCI dataset only and may not generalize to mushrooms from other environments.
- All features are categorical codes, which simplifies real biological variation; some categories are sparsely represented (e.g. `waste` habitat has 192 samples, all edible — not a safety rule).
- Two highly predictive features were removed by design, so performance here is not comparable to published results on the full feature set.
- Only one model family is evaluated at this stage.


---

## Next Steps (Final Deliverable)

- Compare classifiers across model families (tree-based, SVM, k-NN, ensembles)
- Hyperparameter tuning with consistent evaluation metrics
- Decision-threshold adjustment to prioritize poisonous-class recall
- Inspect the 7 misclassified samples for shared feature patterns

---

## Team

| Member | ID | Role |
|--------|-----|------|
| Nadine Radwan | 22P0214 | Data and Preprocessing Lead |
| Hisham Mohamed | 23P0259 | EDA and Visualization Lead |
| Ahmed Mohamed Elsayed | 23P0035 | Modeling and Evaluation Lead |

**Supervisors:** Dr. Mahmoud Khalil, Eng. Hala Shaheen

---

## References

1. UCI Machine Learning Repository. (1981). *Mushroom* [Dataset]. https://doi.org/10.24432/C5959T
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
3. McKinney, W. (2010). Data structures for statistical computing in Python. *Proc. 9th Python in Science Conf.*, 56–61.
4. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.
5. Waskom, M. L. (2021). Seaborn: Statistical data visualization. *JOSS*, 6(60), 3021.
