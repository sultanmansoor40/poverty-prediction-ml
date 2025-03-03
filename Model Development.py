import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# READING DATA AND PIVOTTING
numeric_df = pd.read_csv('dataset/preprocessed_data.csv')


# HIGHLY CORRELATED INDICATORS
# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Extract correlations with poverty indicator
poverty_corr = corr_matrix['SI.POV.DDAY']

# Identify highly correlated indicators (|correlation| > 0.7)
high_corr_indicators = poverty_corr[(poverty_corr > 0.7) | (poverty_corr < -0.7)].sort_values(ascending=False)

# Display top correlated indicators
print("Highly correlated indicators with SI.POV.DDAY:" + str(len(high_corr_indicators)))
print(high_corr_indicators)
# HIGHLY CORRELATED INDICATORS

# Visualize correlation with poverty
plt.figure(figsize=(10, 8))
sns.heatmap(high_corr_indicators.to_frame(), annot=True, cmap='coolwarm')
plt.title('Highly Correlated indicators with the Poverty headcount ratio')
plt.show()

# data preparing for model
# Define features (independent variables) and target (dependent variable)
selected_features = high_corr_indicators.index.tolist()  # Select only high correlation features
print(selected_features)
X = numeric_df[selected_features]  # Features
X = numeric_df.drop(columns=['SI.POV.DDAY'])
y = numeric_df['SI.POV.DDAY']  # Target
# data preparing for model

#split train and test
# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display dataset sizes
print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")
#split train and test

# Initialize models
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train each model
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Function to evaluate model
def evaluate_model(model_name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📌 {model_name} Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-Squared (R²): {r2:.4f}")

    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'R2': r2}

# Evaluate Decision Tree
dt_results = evaluate_model("Decision Tree", y_test, dt_predictions)

# Evaluate Random Forest
rf_results = evaluate_model("Random Forest", y_test, rf_predictions)

# Evaluate Gradient Boosting
gb_results = evaluate_model("Gradient Boosting", y_test, gb_predictions)


# Create DataFrame for comparison
results_df = pd.DataFrame([dt_results, rf_results, gb_results])

# Display model performance comparison
print("\n🔍 Model Performance Comparison:")
print(results_df)

# Visualize results
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='R2', data=results_df, palette='viridis')
plt.title("R-Squared (R²) Score Comparison")
plt.ylabel("R² Score")
plt.show()

import pickle

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)



# RANDOM FOREST TUNING
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

# Grid Search
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="r2", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
# RANDOM FOREST TUNING

#GRADIENT BOOST TUNING
from sklearn.ensemble import GradientBoostingRegressor

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

# Grid Search
gb = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=3, scoring="r2", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
#GRADIENT BOOST TUNING

#TEST THE MODEL REFINEMENT
#Get the best model from Grid Search
best_rf = grid_search.best_estimator_  # For Random Forest
best_gb = grid_search.best_estimator_  # For Gradient Boosting

# Train the best models
best_rf.fit(X_train, y_train)
best_gb.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to evaluate model
def evaluate_model(model_name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📌 {model_name} Model Performance (After Refinement):")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-Squared (R²): {r2:.4f}")

    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'R2': r2}

# Evaluate the refined models
rf_results = evaluate_model("Random Forest (Refined)", y_test, rf_predictions)
gb_results = evaluate_model("Gradient Boosting (Refined)", y_test, gb_predictions)



import requests
import json

url = 'http://127.0.0.1:8080/predict'
data = X_train.iloc[0].to_dict()

headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())





