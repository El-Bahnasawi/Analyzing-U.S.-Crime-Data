import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
import xgboost as xgb
from sklearn.model_selection import cross_val_score


def model_1(X_train: pd.DataFrame, y_train: pd.DataFrame):
    # Create a pipeline with PolynomialFeatures and RandomForestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),        
        ('classifier', RandomForestClassifier())
    ])

    # Define the hyperparameter grid including n_components
    param_grid = {
        'pca__n_components': [5, 8, 10, 15, 20, 25],  # Hyperparameter for PCA
        'classifier__n_estimators': [10, 50, 100, 200, 250, 300],
        'classifier__max_depth': [10, 20, 30, 40, 50],
        'classifier__min_samples_split': [2, 5, 10, 15],
    }

    # Create the random search model on the pipeline with polynomial features
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)

    # Fit the random search model on the training data
    random_search.fit(X_train, y_train)

    return random_search

def model_2(X_train: pd.DataFrame, y_train: pd.DataFrame):
    # Convert y_train to 1D array
    y_train = y_train.values.ravel()

    # Create a pipeline with StandardScaler, PCA, and XGBClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),        
        ('classifier', GradientBoostingClassifier())
    ])

    # Define the hyperparameter grid including n_components
    param_grid = {
        'pca__n_components': [5, 8, 10, 15, 20, 25],  # Hyperparameter for PCA
        'classifier__max_depth': [20, 30, 40],
        'classifier__n_estimators': [5, 10, 50, 100],
    }

    # Create the random search model on the pipeline
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=5, cv=5, n_jobs=-1, random_state=42)

    # Fit the random search model on the training data
    random_search.fit(X_train, y_train)

    return random_search

def model_3(X_train: pd.DataFrame, y_train: pd.DataFrame):
    # Create a pipeline with PolynomialFeatures and RandomForestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('umap', UMAP(n_jobs=-1)),        
        ('classifier', RandomForestClassifier())
    ])

    # Define the hyperparameter grid including n_components
    param_grid = {
        'umap__n_components': [10, 15, 20, 25],  
        'umap__n_neighbors': [20, 25, 30],  
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10, 15],
    }

    # Create the random search model on the pipeline
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=5, cv=5, n_jobs=-1, random_state=42)

    # Fit the random search model on the training data
    random_search.fit(X_train, y_train)

    return random_search

def model_4(X_train, X_test, y_train, y_test):
    # Train a model using Random Forest
    clf = RandomForestClassifier(max_depth=4, n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate model performance
    accuracy_train = clf.score(X_train, y_train)
    cv_scores = cross_val_score(clf, X_test, y_test, cv=5)
    return accuracy_train, cv_scores

def model_5(X_train, X_test, y_train, y_test):
    
    xgb_classifier = xgb.XGBClassifier(n_estimators=12, 
                                    objective='binary:logistic', 
                                    tree_method='hist', 
                                    eta=0.15,
                                    max_depth=1,
                                    enable_categorical=True, 
                                    random_state=42,
                                    reg_alpha = 3,
                                    reg_lambda = 3)  

    # Fit the model on the entire training set
    xgb_classifier.fit(X_train, y_train)

    # Evaluate model performance
    accuracy_train = xgb_classifier.score(X_train, y_train)
    cv_scores = cross_val_score(xgb_classifier, X_test, y_test, cv=5)
    return accuracy_train, cv_scores