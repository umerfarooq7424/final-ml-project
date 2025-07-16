# final-ml-project
This repository contains my Final Machine Learning Project notebook. The project demonstrates the full machine learning pipeline, including:

Data Loading & Preprocessing: Importing datasets, handling missing values, and data normalization.

Exploratory Data Analysis (EDA): Visualizations, feature relationships, and data distribution insights.

Model Building & Training: Implementation of supervised learning algorithms (e.g., regression, classification) with scikit-learn or other frameworks.

Model Evaluation: Performance metrics such as accuracy, precision, recall, or RMSE, plus confusion matrices and ROC curves where relevant.

Hyperparameter Tuning: Use of GridSearchCV or other tuning techniques to optimize model performance.

Insights & Conclusions: Interpretation of results and final remarks on model performance and possible future improvements.
# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# 2. Load and preprocess the data
df = pd.read_csv('Rate.csv', parse_dates=['observation_date'])
df.rename(columns={'observation_date': 'Date', 'Rate': 'Inflation'}, inplace=True)
df.set_index('Date', inplace=True)
df = df.asfreq('MS')  # monthly start frequency

# 3. Visualize the original data
plt.figure(figsize=(10, 4))
plt.plot(df['Inflation'], label='Inflation Rate')
plt.title('US Inflation Rate (1948 - Present)')
plt.xlabel('Year')
plt.ylabel('Inflation (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Train/test split (80/20)
split_index = int(len(df) * 0.8)
train, test = df.iloc[:split_index], df.iloc[split_index:]

# 5. ARIMA (AR/ARMA-like univariate model)
# Using ARIMA with d=0 for ARMA structure
arima_model = ARIMA(train['Inflation'], order=(2, 0, 1)).fit()
arima_forecast = arima_model.forecast(steps=len(test))

# 6. Create lagged features for Random Forest
def create_lagged_features(series, lags=12):
    df_lagged = pd.DataFrame()
    for i in range(1, lags + 1):
        df_lagged[f'lag_{i}'] = series.shift(i)
    df_lagged['target'] = series.values
    return df_lagged.dropna()

lagged_df = create_lagged_features(df['Inflation'], lags=12)
X = lagged_df.drop(columns='target')
y = lagged_df['target']

X_train = X.iloc[:split_index - 12]
X_test = X.iloc[split_index - 12:]
y_train = y.iloc[:split_index - 12]
y_test = y.iloc[split_index_]()_

