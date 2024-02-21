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
   - Handling NaN values by categorizing them into numerical and categorical features. Then using SimpleImputer with mean for numerical features, most frequent for categorical features.

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

5. **Model Building Process** 🛠️
   - Referenced in `models.py`, this includes:
     - Constructing a pipeline starting with StandardScaler to normalize features.
     - Addressing dimensionality issues with PCA and UMAP. As the dataset contains more than `50 features` and PCA returns uncorrelated principal components.
     - Employing Random Forest and Gradient Boosting classifiers.
     - Utilizing RandomizedSearchCV for hyperparameter tuning, enhancing efficiency with cross-validation.

### Models Comparison 📊
| Model      | Technique       | Components | Estimators/Neighbors | Min Samples Split | Max Depth | Training Accuracy | Test Accuracy |
|------------|-----------------|------------|----------------------|-------------------|-----------|-------------------|---------------|
| Model 1    | PCA with RF     | 20         | 100                  | 5                 | 30        | 99.96%            | 97.12%        |
| Model 2    | PCA with GB     | 20         | 5                    | -                 | 20        | 98.51%            | 92.55%        |
| Model 3    | UMAP with RF    | 20         | -                    | 2                 | 20        | 100%              | 93.28%        |