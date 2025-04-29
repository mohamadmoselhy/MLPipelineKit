# Machine Learning System Toolkit Documentation

## Table of Contents
1. [MyDataUtilsLib](#mydatautilslib)
2. [MyMachineLearningLib](#mymachinelearninglib)
3. [MyVisualizationLib](#myvisualizationlib)

## MyDataUtilsLib

### Data Loading and Basic Operations

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `load_data(file_path)` | Loads and processes CSV data | `file_path (str)`: Path to CSV file | `pd.DataFrame`: Processed dataset |
| `check_data_for_preprocessing(df, verbose, return_summary, return_text_report, show_plots)` | Analyzes dataset for preprocessing needs | `df`: DataFrame to analyze<br>`verbose`: Print results<br>`return_summary`: Return insights dict<br>`return_text_report`: Return text report<br>`show_plots`: Show distribution plots | `dict/list/None`: Analysis results |
| `standardize_column_headers(df)` | Normalizes column headers | `df`: Input DataFrame | `pd.DataFrame`: DataFrame with standardized headers |

### Data Cleaning and Preprocessing

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `handle_missing_values(df, strategy, fill_value)` | Handles missing values | `df`: Input DataFrame<br>`strategy`: Handling method<br>`fill_value`: Value for filling | `pd.DataFrame`: Cleaned DataFrame |
| `drop_columns(df, columns_to_drop)` | Removes specified columns | `df`: Input DataFrame<br>`columns_to_drop`: List of columns | `pd.DataFrame`: Modified DataFrame |
| `drop_duplicates(df)` | Removes duplicate rows | `df`: Input DataFrame | `pd.DataFrame`: Deduplicated DataFrame |
| `identify_outliers(df)` | Identifies outliers in numeric columns | `df`: Input DataFrame | `pd.DataFrame`: Outlier information |

### Feature Engineering

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `feature_engineering(df, target_column)` | Creates advanced features | `df`: Input DataFrame<br>`target_column`: Target column name | `pd.DataFrame`: Enhanced DataFrame |
| `encode_column(data, column_name, encoding_type)` | Encodes categorical columns | `data`: Input DataFrame<br>`column_name`: Column to encode<br>`encoding_type`: Encoding method | `pd.DataFrame`: Encoded DataFrame |
| `encode_by_ranges(df, column, new_column, bins, labels)` | Creates binned features | `df`: Input DataFrame<br>`column`: Column to bin<br>`new_column`: New column name<br>`bins`: Bin boundaries<br>`labels`: Bin labels | `pd.DataFrame`: Modified DataFrame |
| `create_time_series_features(df, date_column, target_column)` | Creates time series features | `df`: Input DataFrame<br>`date_column`: Date column<br>`target_column`: Target column | `pd.DataFrame`: Enhanced DataFrame |

### Data Validation and Quality

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `validate_data(df, schema)` | Validates data against schema | `df`: Input DataFrame<br>`schema`: Data schema | `dict`: Validation results |
| `detect_anomalies(df, method, threshold)` | Detects anomalies in data | `df`: Input DataFrame<br>`method`: Detection method<br>`threshold`: Anomaly threshold | `pd.DataFrame`: Anomaly flags |

### Data Storage

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `save_to_csv(data, filename)` | Saves data to CSV | `data`: Data to save<br>`filename`: Output filename | `str`: File path |
| `write_to_text_file(data, filename)` | Writes data to text file | `data`: Data to write<br>`filename`: Output filename | `None` |

## MyMachineLearningLib

### Model Creation and Training

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `create_linear_regression_model(x, y, test_size, shuffle, random_state, params)` | Creates linear regression model | `x`: Features<br>`y`: Target<br>`test_size`: Test split size<br>`shuffle`: Shuffle data<br>`random_state`: Random seed<br>`params`: Model parameters | `tuple`: (scaler, model, x_train, x_test, y_train, y_test) |
| `create_svm_model(x, y, test_size, shuffle, random_state, params)` | Creates SVM model | Same as above | Same as above |
| `create_random_forest_model(x, y, test_size, shuffle, random_state, params)` | Creates random forest model | Same as above | Same as above |
| `create_decision_tree_model(x, y, test_size, shuffle, random_state, params)` | Creates decision tree model | Same as above | Same as above |

### Model Evaluation and Optimization

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `evaluate_model_performance(y_true, y_pred, task_type)` | Evaluates model performance | `y_true`: True labels<br>`y_pred`: Predicted labels<br>`task_type`: Task type | `dict`: Performance metrics |
| `optimize_hyperparameters(model, param_grid, X, y, cv, scoring)` | Optimizes model hyperparameters | `model`: Base model<br>`param_grid`: Parameter grid<br>`X`: Features<br>`y`: Target<br>`cv`: CV folds<br>`scoring`: Scoring metric | `dict`: Optimization results |
| `create_ensemble_model(models, X, y, weights)` | Creates ensemble model | `models`: Base models<br>`X`: Features<br>`y`: Target<br>`weights`: Model weights | Ensemble model |

### Model Management

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `save_model_and_scaler(model, scaler, model_name, base_dir)` | Saves model and scaler | `model`: Trained model<br>`scaler`: Fitted scaler<br>`model_name`: Model name<br>`base_dir`: Save directory | `tuple`: (model_path, scaler_path) |
| `load_model_and_scaler(model_filename, scaler_filename)` | Loads model and scaler | `model_filename`: Model file path<br>`scaler_filename`: Scaler file path | `tuple`: (model, scaler) |
| `predict_with_model(model, scaler, x_new)` | Makes predictions | `model`: Trained model<br>`scaler`: Fitted scaler<br>`x_new`: New data | Predictions |

## MyVisualizationLib

### Basic Visualizations

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `plot_boxplots(df, features, save_folder, figsize, palette)` | Creates boxplots | `df`: Input DataFrame<br>`features`: Features to plot<br>`save_folder`: Save directory<br>`figsize`: Figure size<br>`palette`: Color palette | `None` |
| `plot_histograms(data, features, colors, save_folder, figsize)` | Creates histograms | `data`: Input data<br>`features`: Features to plot<br>`colors`: Color scheme<br>`save_folder`: Save directory<br>`figsize`: Figure size | `None` |
| `plot_pairplots(data, features, hue, save_folder, graph_Name)` | Creates pair plots | `data`: Input data<br>`features`: Features to plot<br>`hue`: Color encoding<br>`save_folder`: Save directory<br>`graph_Name`: Output name | `None` |
| `plot_heatmap(data, features, save_folder, graph_Name, cmap, figsize)` | Creates correlation heatmap | `data`: Input data<br>`features`: Features to plot<br>`save_folder`: Save directory<br>`graph_Name`: Output name<br>`cmap`: Color map<br>`figsize`: Figure size | `None` |

### Advanced Visualizations

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `plot_model_performance(y_true, y_pred, task_type, save_folder)` | Creates model performance plots | `y_true`: True labels<br>`y_pred`: Predicted labels<br>`task_type`: Task type<br>`save_folder`: Save directory | `None` |
| `plot_feature_importance(model, feature_names, save_folder)` | Creates feature importance plot | `model`: Trained model<br>`feature_names`: Feature names<br>`save_folder`: Save directory | `None` |
| `plot_correlation_heatmap(df, save_folder, figsize)` | Creates enhanced correlation heatmap | `df`: Input DataFrame<br>`save_folder`: Save directory<br>`figsize`: Figure size | `None` |
| `plot_time_series(data, date_column, value_column, save_folder)` | Creates time series plots | `data`: Input data<br>`date_column`: Date column<br>`value_column`: Value column<br>`save_folder`: Save directory | `None` |

## Usage Examples

### Data Preprocessing
```python
from MyDataUtilsLib import load_data, feature_engineering

# Load and preprocess data
df = load_data('data.csv')
df_engineered = feature_engineering(df, target_column='target')
```

### Model Training
```python
from MyMachineLearningLib import create_random_forest_model, evaluate_model_performance

# Create and evaluate model
scaler, model, x_train, x_test, y_train, y_test = create_random_forest_model(x, y)
metrics = evaluate_model_performance(y_test, model.predict(x_test), task_type='regression')
```

### Visualization
```python
from MyVisualizationLib import plot_model_performance, plot_feature_importance

# Create visualizations
plot_model_performance(y_test, y_pred, task_type='regression')
plot_feature_importance(model, feature_names, save_folder='feature_importance')
```

## Notes
- All visualization functions automatically save plots to specified directories
- Data preprocessing functions maintain data integrity and provide detailed logging
- Model functions include automatic scaling and parameter optimization
- All functions include comprehensive error handling and logging 