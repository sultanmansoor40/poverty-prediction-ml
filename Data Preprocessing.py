import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# Load the dataset
df = pd.read_csv('dataset/indicators.csv')

numeric_df = numeric_df.astype('float32')
for col in numeric_df.select_dtypes(include=['int64']).columns:
    numeric_df[col] = pd.to_numeric(numeric_df[col], downcast='integer')
    
numeric_df = numeric_df.dropna(subset=["SI.POV.DDAY"])



# Pivot the data
pivot_df = df.pivot(index=['CountryName', 'Year'], columns='IndicatorCode', values='Value').reset_index()

# Drop non-numeric columns
numeric_df = pivot_df.drop(columns=['CountryName', 'Year'])

# Check missing values percentage
missing_percentage = numeric_df.isnull().mean() * 100
print(missing_percentage.sort_values(ascending=False))
#print(missing_percentage.sort_values(ascending=False).to_string())

# MISSING VALUES < 10
for col in numeric_df.columns:
    if missing_percentage[col] < 10:
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())  # Median works better for skewed data

# MISSING VALUES < 10

# MISSING VALUES 10 - 40
knn_imputer = KNNImputer(n_neighbors=5)
columns_to_knn = missing_percentage[(missing_percentage >= 10) & (missing_percentage < 40)].index
#number of existing data
print(len(missing_percentage[(missing_percentage >= 10) & (missing_percentage < 40)]))
numeric_df[columns_to_knn] = knn_imputer.fit_transform(numeric_df[columns_to_knn])
# MISSING VALUES 10 - 40


# MISSING VALUES 40 - 80
iter_imputer = IterativeImputer(
    estimator=HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=20, max_depth=5), 
    max_iter=10, 
    random_state=42
)
columns_to_iter = missing_percentage[(missing_percentage >= 40) & (missing_percentage <= 80)].index
#number of existing data
print(len(columns_to_iter))
numeric_df[columns_to_iter] = iter_imputer.fit_transform(numeric_df[columns_to_iter])
# MISSING VALUES 40 - 80

# (D) Drop Features with >80% Missing Data
numeric_df.drop(columns=missing_percentage[missing_percentage > 80].index, inplace=True)

# Validate that missing values are handled
print("Missing values after imputation")
print(numeric_df.isnull().sum())


numeric_df.to_csv("dataset/preprocessed_data.csv", index=False)

import pickle

# Save the dataset as a pickle file
with open("../dataset/preprocessed_data.pkl", "wb") as f:
    pickle.dump(numeric_df, f)

    
# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Extract correlations with the target variable 'SI.POV.DDAY'
poverty_corr = corr_matrix['SI.POV.DDAY']

print(poverty_corr)

# Filter highly correlated indicators
high_corr_indicators = poverty_corr[(poverty_corr > 0.7) | (poverty_corr < -0.7)]
print("Highly correlated indicators with SI.POV.DDAY:")
high_corr_indicators = high_corr_indicators.sort_values(ascending=False)
print(high_corr_indicators)
print(high_corr_indicators.to_string()) #to get full list


# Check missing values percentage
missing_percentage = numeric_df[high_corr_indicators.index].isnull().mean() * 100
print("Percentage of Missing Values in Highly Correlated Indicators")
print(missing_percentage.sort_values(ascending=False))

print(missing_percentage.describe())


# Create DataFrame to display correlation and missing value percentage
high_corr_summary = pd.DataFrame({
    'Correlation': high_corr_indicators,
    'Missing_Percentage': missing_percentage[high_corr_indicators.index]
})

# Sort by absolute correlation value for better visualization
high_corr_summary = high_corr_summary.reindex(high_corr_summary['Correlation'].abs().sort_values(ascending=False).index)

# Print summary
print(high_corr_summary)

print("\n📌 Highly Correlated Indicators with SI.POV.DDAY (Sorted by Correlation Strength)\n")
for index, row in high_corr_summary.iterrows():
    print(f"{index}: Correlation = {row['Correlation']:.2f}, Missing = {row['Missing_Percentage']:.2f}%")



