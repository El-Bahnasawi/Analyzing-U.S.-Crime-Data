[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### Project Overview 📝

This repository contains two separate notebooks focusing on exploratory data analysis (EDA) and regression analysis using machine learning (ML) models.

#### Notebook 1: Exploratory Data Analysis (EDA) 📊
- Addressing questions through visualizations and hypothesis testing.
- File: `Project.ipynb`

#### Notebook 2: Regression Analysis and ML Models 🧠
This notebook is divided into several key steps:

1. **Data Validation and Preparation** 📝
   - Ensuring data availability.
   - Reading the dataset.
   - Handling NaN values by categorizing them into numerical and categorical features. Then using KNN imputer for numerical features, Simple Imputer with most frequent for categorical features.

2. **Feature Engineering** 🔧
   - Identifying categorical features and further categorizing them into boolean, nominal, and ordinal.
   - Encoding nominal features using OneHotEncoder and ordinal features using OrdinalEncoder.

3. **Regression Analysis** 📈
   - Investigating feature correlations via heatmap.
   - Conducting linear regression, assessing the model's performance with R2 score, and visualizing coefficients.

4. **Machine Learning Models** 🤖
   - Checking dataset balance.
   - Establishing a baseline accuracy.
   - Implementing various ML models, evaluating them with training and testing accuracies, confusion matrices, precision, recall, and AUC-ROC score.

**Enhanced Model Building Process** 🛠️

Outlined in `models.py`, the following comprehensive approach is undertaken:

**Model Categories:**
   - **Dimensionality Reduction Models:** These models employ techniques to reduce feature dimensionality while retaining crucial information.
   - **Feature Selection Models:** Focused on selecting pertinent features while discarding redundant or correlated ones.

1. **Dimensionality Reduction plus Classifier:**
   - **Pipeline Setup:** Begins with StandardScaler for feature normalization, ensuring uniformity across features.
   - **Dimensionality Management:** using PCA and UMAP to reduce dimensionality while keeping valuable information
   - **Classifier Integration:** Incorporating Random Forest and Gradient Boosting classifiers to leverage ensemble learning for robust predictions.
   - **Hyperparameter Tuning:** RandomizedSearchCV facilitates efficient hyperparameter optimization, complemented by cross-validation to improve model generalization.

2. **Feature Selection Methods plus Classifier:**
   - **Recursive Feature Elimination (RFE):** Prioritizes feature selection based on importance to model performance, further refining the feature set.
   - **Correlation-based Feature Elimination:** Subsequent utilization of Variance Inflation Factor (VIF) identifies and eliminates correlated features, enhancing model stability and interpretability.
   - **Efficiency and Simplicity:** These models offer lower computational complexities compared to their dimensionality reduction counterparts, ensuring suitability for scenarios prioritizing interpretability and computational efficiency.

### Models Comparison 📊
| Model      | Technique       | features | Estimators/Neighbors | Min Samples Split | Max Depth | Training Accuracy | Test Accuracy |
|------------|-----------------|------------|----------------------|-------------------|-----------|-------------------|---------------|
| Model 1    | PCA with RF     | 20         | 100                  | 5                 | 30        | 99.96%            | 97.12%        |
| Model 2    | PCA with GB     | 20         | 5                    | -                 | 20        | 98.51%            | 92.55%        |
| Model 3    | UMAP with RF    | 20         | -                    | 2                 | 20        | 100%              | 93.28%        |
| Model 4    | RFE, VIF, RF    | 40         | 50                   | -                 | 4         | 99.26%            | 99.45%        |
| Model 5    | RFE, VIF, XGB   | 40         | 12                   | -                 | 1         | 100%              | 100%          |
