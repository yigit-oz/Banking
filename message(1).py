import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')
matplotlib inline
df = pd.read_csv('./bank-additional.csv', delimiter=';')
# Define ordinal encoding for education, month, and day_of_week
ordinal_features = ["education", "month", "day_of_week"]
ordinal_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder())
])

# Define one-hot encoding for the the categorical features
categorical_features = ["job", "marital", "default", "housing", "loan", "contact", "poutcome"]
onehot_transformer = Pipeline(steps=[
    ('onehot_encoder', OneHotEncoder())
])

# Define standard scaling for numerical features
numeric_features = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", 
                    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine ordinal and one-hot transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_features),
        ('onehot', onehot_transformer, categorical_features),
        ('numeric', numeric_transformer, numeric_features)
    ])

# Label encode the target variable 'y'
label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.1, test_size=0.1, random_state=42)
models = [
    Pipeline([
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression())
    ]),
    Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier())
    ]),
    Pipeline([
        ('preprocess', preprocessor),
        ('classifier', XGBClassifier())
    ])
]
for model in models:
    model.fit(X_train, y_train)
param_grid_logistic = {
    'classifier__C': [0.1, 1, 10, 100]
}

param_grid_rf = {
    'classifier__n_estimators': [5, 50, 100, 200, 300]
}

param_grid_xgb = {
    'classifier__n_estimators': [50, 100, 200, 300, 500]
}
param_grids = [param_grid_logistic, param_grid_rf, param_grid_xgb]
grid_searches = []
for model, param_grid in zip(models, param_grids):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    grid_searches.append(grid_search)
for idx, grid_search in enumerate(grid_searches, start=1):
    print(f"Best parameters for Model {idx}: {grid_search.best_params_}")
    print(f"Best cross-validation score for Model {idx}: {grid_search.best_score_}")
