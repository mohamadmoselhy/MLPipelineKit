# Linear Regression Model with Scikit-Learn

This project provides a complete pipeline for training, evaluating, saving, and making predictions with a Linear Regression model using Scikit-Learn. The pipeline includes data preprocessing, model training, evaluation metrics, model persistence, and user input-based prediction.

## Features

- Splits dataset into training and testing sets
- Normalizes input features using `StandardScaler`
- Trains a `LinearRegression` model
- Evaluates the model using MSE, MAE, and R² score
- Saves the model and scaler with timestamped filenames
- Loads a saved model and scaler to make predictions on user input

## Requirements

- Python 3.x
- `scikit-learn`
- `numpy`
- `joblib`

Install required packages via pip:

```bash
pip install scikit-learn numpy joblib
