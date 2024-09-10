# Install necessary packages
!pip install shap
!pip install xgboost  # Optional, if using XGBoost

# Import libraries
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb  # Optional, depending on your model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)  # Use shap.KernelExplainer for non-tree models

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Visualize SHAP values
# Summary plot
shap.summary_plot(shap_values, X_test)

# For a single prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
