import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import Ridge

# Load the dataset
file_path = 'D:/pyprog/QAT_stock/stock_data.csv'
stock_data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set as index
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

# Handle missing values with forward fill and backward fill
stock_data_ffill = stock_data.fillna(method='ffill')
stock_data_bfill = stock_data_ffill.fillna(method='bfill')

# Feature engineering functions (lagged features, rolling statistics, volatility measures, date features)
def create_lagged_features(data, columns, lags):
    for column in columns:
        for lag in lags:
            data[f'{column}_lag_{lag}'] = data[column].shift(lag)
    return data

def create_rolling_statistics(data, columns, windows):
    for column in columns:
        for window in windows:
            data[f'{column}_roll_mean_{window}'] = data[column].rolling(window=window).mean()
            data[f'{column}_roll_std_{window}'] = data[column].rolling(window=window).std()
    return data

def create_volatility_measures(data, columns, windows):
    for column in columns:
        for window in windows:
            data[f'{column}_volatility_{window}'] = data[column].rolling(window=window).std()
    return data

def create_date_features(data):
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['year'] = data.index.year
    return data

# Apply feature engineering
lags = [1, 3, 5, 10]
windows = [5, 10, 20]
columns_to_transform = ['Close', 'Volume', 'PE Ratio', 'EPS', 'PB Ratio', 'Dividend Yield', 'ROE']

stock_data_bfill = create_lagged_features(stock_data_bfill, columns_to_transform, lags)
stock_data_bfill = create_rolling_statistics(stock_data_bfill, columns_to_transform, windows)
stock_data_bfill = create_volatility_measures(stock_data_bfill, columns_to_transform, windows)
stock_data_bfill = create_date_features(stock_data_bfill)

# Fill missing values in engineered features
stock_data_ffill = stock_data_bfill.fillna(method='ffill')
stock_data_bfill = stock_data_ffill.fillna(method='bfill')
stock_data_filled = stock_data_bfill.interpolate(method='linear').fillna(0)

# Identify non-numeric columns
non_numeric_columns = stock_data_filled.select_dtypes(include=['object']).columns

# Define feature columns and target column
feature_columns = [col for col in stock_data_filled.columns if col != 'Close']
target_column = 'Close'

# One-hot encode non-numeric columns and scale the features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), non_numeric_columns)
    ])

# Split the data into train and test sets
X = stock_data_filled.drop(columns=non_numeric_columns)
y = stock_data_filled[target_column]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Apply the transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Define the neural network model
def build_nn_model(optimizer='adam', activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=activation, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse')
    return model

# 1. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# 3. AdaBoost Model
ab_model = AdaBoostRegressor(n_estimators=100, random_state=42)

# 4. Neural Network Model using TensorFlow
nn_model = KerasRegressor(build_fn=build_nn_model, epochs=50, batch_size=32, verbose=0)

# Hyperparameter tuning for Neural Network
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh']
}
grid = GridSearchCV(estimator=nn_model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)
print(f"Best parameters for NN: {grid_result.best_params_}")
print(f"Best score for NN: {grid_result.best_score_}")
best_nn_model = grid_result.best_estimator_

# Ensembling using Stacking
base_models = [
    ('rf', rf_model),
    ('gb', gb_model),
    ('ab', ab_model),
    ('nn', best_nn_model)
]
meta_model = Ridge()
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3)

# Train the stacking regressor
stacking_regressor.fit(X_train, y_train)

# Predict on the test set with individual models and the ensemble
y_pred_rf = rf_model.fit(X_train, y_train).predict(X_test)
y_pred_gb = gb_model.fit(X_train, y_train).predict(X_test)
y_pred_ab = ab_model.fit(X_train, y_train).predict(X_test)
y_pred_nn = best_nn_model.predict(X_test)
y_pred_stack = stacking_regressor.predict(X_test)

# Evaluate the models
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
rmse_ab = mean_squared_error(y_test, y_pred_ab, squared=False)
rmse_nn = mean_squared_error(y_test, y_pred_nn, squared=False)
rmse_stack = mean_squared_error(y_test, y_pred_stack, squared=False)

print(f"RMSE - Random Forest: {rmse_rf}")
print(f"RMSE - Gradient Boosting: {rmse_gb}")
print(f"RMSE - AdaBoost: {rmse_ab}")
print(f"RMSE - Optimized Neural Network: {rmse_nn}")
print(f"RMSE - Stacking Regressor: {rmse_stack}")

# Blending as an alternative ensemble technique
def blend_models_predict(X_train, y_train, X_test):
    blend_train = np.zeros((X_train.shape[0], len(base_models)))
    blend_test = np.zeros((X_test.shape[0], len(base_models)))

    for j, (name, model) in enumerate(base_models):
        model.fit(X_train, y_train)
        blend_train[:, j] = model.predict(X_train)
        blend_test[:, j] = model.predict(X_test)
    
    meta_model.fit(blend_train, y_train)
    blend_predictions = meta_model.predict(blend_test)
    return blend_predictions

y_pred_blend = blend_models_predict(X_train, y_train, X_test)
rmse_blend = mean_squared_error(y_test, y_pred_blend, squared=False)
print(f"RMSE - Blending: {rmse_blend}")
