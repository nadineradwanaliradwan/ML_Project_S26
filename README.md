# CSE382 Mushroom Edibility Classification

This repository contains the **Role 1: Data and Preprocessing** work for the first deliverable of the CSE382 Introduction to Machine Learning project.

The project uses the [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom) to prepare a supervised binary-classification study that distinguishes between **edible** and **poisonous** mushrooms.

## Notebook

[`Machine_Learning_Project.ipynb`](Machine_Learning_Project.ipynb) covers:

- Automatic dataset acquisition from the UCI repository
- Dataset schema and integrity validation
- Stratified 80/20 train-test splitting
- Detection and treatment of hidden `?` values
- Training-only feature analysis
- Removal of shortcut and zero-variance features
- Feature-reduction sensitivity analysis
- Class-balance assessment
- One-hot encoding preparation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nadineradwanaliradwan/ML_Project_S26/blob/main/Machine_Learning_Project.ipynb)

## Dataset

- **Source:** UCI Machine Learning Repository
- **Instances:** 8,124
- **Input attributes:** 22 categorical attributes
- **Target:** `class`
  - `e`: edible
  - `p`: poisonous
- **Missing-value marker:** `?` in `stalk-root`

The notebook downloads the dataset automatically. If downloading is unavailable in Google Colab, it provides a manual upload fallback.

## Preprocessing Workflow

1. Load the raw headerless dataset and assign the published column names.
2. Verify the number of rows, columns, classes, and missing-value markers.
3. Encode poisonous mushrooms as the positive class (`1`).
4. Create a stratified 80/20 train-test split before data-dependent analysis.
5. Replace unknown `stalk-root` values with an explicit `unknown` category.
6. Evaluate individual features using cross-validated accuracy and mutual information on the training set only.
7. Remove the three strongest shortcut features:
   - `odor`
   - `spore-print-color`
   - `gill-color`
8. Remove `veil-type` because it has zero variance.
9. Retain 18 attributes for later modeling.
10. Prepare nominal categorical attributes using one-hot encoding.

## Requirements

The notebook uses:

- Python 3
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn

Google Colab already provides these packages.

## How to Run

### Google Colab

1. Open the notebook using the **Open in Colab** button above.
2. Select **Runtime → Restart session and run all**.
3. Wait for all cells to finish.
4. Confirm that no errors appear.

### Local Jupyter Environment

```bash
git clone https://github.com/nadineradwanaliradwan/ML_Project_S26.git
cd ML_Project_S26
jupyter notebook Machine_Learning_Project.ipynb
```

Then run all cells from top to bottom.

## Reproducibility

- Random seed: `42`
- Test size: `20%`
- Cross-validation: five-fold stratified cross-validation
- Feature-reduction decisions use training data only
- The notebook has been executed sequentially without errors or warnings

## Scope

This notebook contains the **Data and Preprocessing Lead** contribution only. Exploratory data analysis, final model comparison, and error analysis are handled in the other project deliverables.

