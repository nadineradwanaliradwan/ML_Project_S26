# CSE382 Mushroom Edibility Classification

This repository contains the complete machine learning workflow for the **CSE382 Introduction to Machine Learning** project.

The project uses the [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom) to develop and compare supervised classification models that distinguish between **edible** and **poisonous** mushrooms using their physical characteristics.

## Notebook

[`ML_summer_26_Phase_2.ipynb`](ML_summer_26_Phase_2.ipynb) covers:

- Automatic dataset acquisition from the UCI repository
- Dataset inspection and data-quality validation
- Missing-value handling and feature reduction
- Stratified 80/20 train-test splitting
- One-hot encoding of categorical variables
- Exploratory data analysis with four visualizations
- Logistic Regression baseline development
- Decision Tree and K-Nearest Neighbors implementation
- Five-fold cross-validation
- Model evaluation and comparison
- Confusion matrices and ROC curves
- Misclassification and error analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nadineradwanaliradwan/ML_Project_S26/blob/main/ML_summer_26_Phase_2.ipynb)

## Dataset

- **Source:** UCI Machine Learning Repository
- **Instances:** 8,124
- **Input attributes:** 22 categorical attributes
- **Target:** `class`
  - `e`: edible
  - `p`: poisonous
- **Class distribution:**
  - 4,208 edible samples (51.8%)
  - 3,916 poisonous samples (48.2%)
- **Missing values:** 2,480 values in `stalk-root`
- **Duplicated rows:** 0

The dataset is downloaded automatically using the `ucimlrepo` package.

## Preprocessing Workflow

1. Load the dataset and separate the input features from the target.
2. Inspect the dimensions, data types, missing values, duplicate records, and class distribution.
3. Remove two highly predictive shortcut features:
   - `odor`
   - `spore-print-color`
4. Create a stratified 80/20 train-test split using a fixed random seed.
5. Calculate the `stalk-root` mode using the training set only.
6. Use the training mode to impute missing values in both sets.
7. Convert the remaining categorical attributes using one-hot encoding.
8. Align the training and testing columns to produce 98 encoded features.
9. Confirm that the processed data contains no missing values.

## Exploratory Data Analysis

The notebook presents four focused visualizations:

1. Overall class distribution
2. Gill size and mushroom edibility
3. Bruising and mushroom edibility
4. Habitat and mushroom edibility

The plots show that the remaining attributes still contain meaningful predictive information after removing the two shortcut features. Broad gills and bruising appear more frequently among edible mushrooms, while narrow gills and several habitat categories are more strongly associated with poisonous mushrooms.

## Classification Models

Three classifiers from different model families are compared:

| Model | Classifier Family |
|---|---|
| Logistic Regression | Linear classifier |
| Decision Tree | Tree-based classifier |
| K-Nearest Neighbors | Instance-based classifier |

Logistic Regression is used as the baseline model. Decision Tree and KNN are added during Phase 2 to provide a broader comparison between linear, nonlinear, and instance-based learning approaches.

## Model Evaluation

All three models are trained using the same processed training data and evaluated on the same unseen testing set. Five-fold cross-validation is also applied to the training data for a more reliable performance estimate.

The comparison includes:

- Cross-validation accuracy
- Test accuracy
- Poisonous-class F1-score
- Confusion matrix
- ROC curve
- ROC-AUC score

The poisonous class is treated as the positive class because incorrectly classifying a poisonous mushroom as edible is the most safety-critical error.

## Error Analysis

The notebook identifies the misclassified samples produced by each model and compares their total error counts. Logistic Regression errors are examined in further detail by displaying their actual classes, predicted classes, and encoded feature values.

This analysis helps determine whether strong overall accuracy hides dangerous false-negative predictions.

## Requirements

The notebook uses:

- Python 3
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- ucimlrepo

Google Colab already provides most of these packages. The notebook installs `ucimlrepo` automatically.

## How to Run

### Google Colab

1. Open the notebook using the **Open in Colab** button above.
2. Select **Runtime → Restart session and run all**.
3. Wait for all cells to finish.
4. Confirm that all tables, plots, and model outputs appear without errors.

### Local Jupyter Environment

```bash
git clone https://github.com/nadineradwanaliradwan/ML_Project_S26.git
cd ML_Project_S26
pip install pandas matplotlib seaborn scikit-learn ucimlrepo notebook
jupyter notebook ML_summer_26_Phase_2.ipynb
```

Then run all cells from top to bottom.

## Reproducibility

- Random seed: `42`
- Test size: `20%`
- Split strategy: stratified sampling
- Cross-validation: five folds
- Imputation value calculated from training data only
- Identical testing set used for all model comparisons

## Team Contributions

- **Nadine Radwan:** Data understanding and preprocessing
- **Hisham Mohamed:** Exploratory data analysis and visualization
- **Ahmed Elsayed:** Modeling, evaluation, and error analysis

## Academic Disclaimer

This project is an academic machine learning exercise. Its predictions must not be used to determine whether a real mushroom is safe to eat.
