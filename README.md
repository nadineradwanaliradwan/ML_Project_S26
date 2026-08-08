# Mushroom Edibility Classification

A supervised machine learning project that predicts whether a mushroom is **edible** or **poisonous** based on its physical characteristics.

This repository contains the first project deliverable, including data acquisition, preprocessing, exploratory data analysis, baseline classification, performance evaluation, error analysis, and the complete project report.

> **Important:** This project is intended for educational purposes only. It must not be used to determine whether real mushrooms are safe for consumption.

---

## Project Objective

The objective of this project is to develop a binary classification solution that predicts mushroom edibility using physical characteristics such as:

- Cap shape and color
- Gill size and color
- Stalk properties
- Ring type
- Population
- Habitat
- Bruising status

The target variable contains two classes:

- `e`: Edible
- `p`: Poisonous

Because the target contains two possible outcomes, this is a **supervised binary classification problem**.

---

## Dataset

The project uses the **Mushroom Dataset** from the UCI Machine Learning Repository.

- **Source:** [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)
- **Dataset ID:** 73
- **Number of samples:** 8,124
- **Original input features:** 22
- **Feature type:** Categorical
- **Target variable:** Mushroom edibility
- **Target classes:** Edible and Poisonous

The dataset was loaded directly into the notebook using the `ucimlrepo` Python package. No manually modified or external version of the dataset was used.

---

## Repository Contents

| File | Description |
|---|---|
| `ML_summer_26.ipynb` | Complete Google Colab notebook containing data loading, preprocessing, EDA, model training, and evaluation |
| `ML Project.pdf` | Full written project report, including methodology, findings, error analysis, contributions, and references |
| `README.md` | Summary of the project workflow and results |

---

## Dataset Inspection

The initial data inspection produced the following findings:

| Data-quality check | Result |
|---|---:|
| Total samples | 8,124 |
| Original features | 22 |
| Duplicate rows | 0 |
| Features containing missing values | 1 |
| Missing values in `stalk-root` | 2,480 |

The `stalk-root` feature was the only feature containing missing values. Removing every incomplete row would have deleted a considerable portion of the dataset, so missing-value imputation was used instead.

---

## Target Distribution

The target classes were approximately balanced:

| Class | Samples | Percentage |
|---|---:|---:|
| Edible | 4,208 | 51.8% |
| Poisonous | 3,916 | 48.2% |

The difference between the classes was only 292 samples. Therefore, oversampling, undersampling, and synthetic balancing techniques were not required.

Stratified sampling was still used to preserve the same class distribution in the training and testing sets.

---

## Data Preprocessing

### 1. Feature Reduction

Two highly predictive features were intentionally removed:

- `odor`
- `spore-print-color`

These features are strongly associated with mushroom edibility and could make the classification problem overly simple.

Removing them required the classifier to learn from combinations of the remaining mushroom characteristics instead of depending on a small number of obvious predictors.

After feature reduction, the dataset contained **20 original categorical input features**.

### 2. Train-Test Split

The dataset was divided using an 80/20 stratified split:

| Dataset portion | Samples | Percentage |
|---|---:|---:|
| Training set | 6,499 | 80% |
| Testing set | 1,625 | 20% |

The split used:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
