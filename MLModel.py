# First, let's add the necessary import for GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameter grids for each model type
def get_param_grid(model_type):
    """
    Returns a parameter grid dictionary for the specified model type for use in GridSearchCV.
    
    Parameters:
    - model_type (str): The type of model to get parameters for
    
    Returns:
    - param_grid (dict): Dictionary of parameters to search
    """
    param_grids = {
        # Regression models
        "linear_regression": {
            'fit_intercept': [True, False],
            'copy_X': [True],
            'positive': [False]
        },
        "ridge_regression": {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        "lasso_regression": {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']
        },
        "elastic_net": {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']
        },
        "decision_tree_regressor": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "random_forest_regressor": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "gradient_boosting_regressor": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        "xgboost_regressor": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        "lightgbm_regressor": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        "svr": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1],
            'epsilon': [0.1, 0.2]
        },
        "knn_regressor": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        "mlp_regressor": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        },

        # Classification models
        "logistic_regression": {
            'C': [0.1, 1, 10],
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'penalty': ['l2', 'none'],
            'max_iter': [100, 200, 300]
        },
        "knn_classifier": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        "decision_tree_classifier": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        "random_forest_classifier": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        "gradient_boosting_classifier": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        "xgboost_classifier": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        "lightgbm_classifier": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        "svm_classifier": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1],
            'probability': [True]
        },
        "gaussian_nb": {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        "multinomial_nb": {
            'alpha': [0.1, 0.5, 1.0],
            'fit_prior': [True, False]
        },
        "mlp_classifier": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    return param_grids.get(model_type, {})

# Add default parameters for each model
def get_default_params(model_type):
    """
    Returns a dictionary of default parameters for the specified model type.
    
    Parameters:
    - model_type (str): The type of model to get default parameters for
    
    Returns:
    - default_params (dict): Dictionary of default parameters
    """
    default_params = {
        # Regression models
        "linear_regression": {
            'fit_intercept': True,
            'copy_X': True,
            'positive': False
        },
        "ridge_regression": {
            'alpha': 1.0,
            'fit_intercept': True,
            'solver': 'auto'
        },
        "lasso_regression": {
            'alpha': 1.0,
            'fit_intercept': True,
            'selection': 'cyclic'
        },
        "elastic_net": {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'fit_intercept': True,
            'selection': 'cyclic'
        },
        "decision_tree_regressor": {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 0
        },
        "random_forest_regressor": {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 0
        },
        "gradient_boosting_regressor": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'random_state': 0
        },
        "xgboost_regressor": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 0
        },
        "lightgbm_regressor": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 0
        },
        "svr": {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'epsilon': 0.1
        },
        "knn_regressor": {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        "mlp_regressor": {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 1000,
            'random_state': 0
        },

        # Classification models
        "logistic_regression": {
            'C': 1.0,
            'solver': 'lbfgs',
            'penalty': 'l2',
            'max_iter': 100,
            'random_state': 0
        },
        "knn_classifier": {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        "decision_tree_classifier": {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini',
            'random_state': 0
        },
        "random_forest_classifier": {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini',
            'random_state': 0
        },
        "gradient_boosting_classifier": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'random_state': 0
        },
        "xgboost_classifier": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 0
        },
        "lightgbm_classifier": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 0
        },
        "svm_classifier": {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 0
        },
        "gaussian_nb": {
            'var_smoothing': 1e-9
        },
        "multinomial_nb": {
            'alpha': 1.0,
            'fit_prior': True
        },
        "mlp_classifier": {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 1000,
            'random_state': 0
        }
    }
    
    return default_params.get(model_type, {})

# Now let's update the prepare_and_train_model function to include default params and param_grid
def prepare_and_train_model(x, y, model_type, test_size=0.3, shuffle=True, random_state=0, params=None, use_grid_search=False, cv=5):
    """
    Generic function to split, scale, and train a machine learning model based on the selected type.
    Supports hyperparameter tuning using GridSearchCV.

    Supported Models:
     Supervised Learning

     Regression Models:
    - linear_regression
    - ridge_regression
    - lasso_regression
    - elastic_net
    - decision_tree_regressor
    - random_forest_regressor
    - gradient_boosting_regressor
    - xgboost_regressor
    - lightgbm_regressor
    - svr
    - knn_regressor
    - mlp_regressor

     Classification Models:
    - logistic_regression
    - knn_classifier
    - decision_tree_classifier
    - random_forest_classifier
    - gradient_boosting_classifier
    - xgboost_classifier
    - lightgbm_classifier
    - svm_classifier
    - gaussian_nb
    - multinomial_nb
    - mlp_classifier

    Parameters:
    - x (array-like): Feature data
    - y (array-like): Target data
    - model_type (str): Model key from the above list
    - test_size (float): Proportion for test split
    - shuffle (bool): Whether to shuffle before splitting
    - random_state (int): Seed for reproducibility
    - params (dict, optional): Custom parameters for the model, overrides defaults
    - use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
    - cv (int): Number of cross-validation folds for GridSearchCV

    Returns:
    - scaler (StandardScaler): Scaler used to transform the features
    - model: Trained model (or GridSearchCV object if use_grid_search=True)
    - x_train, x_test, y_train, y_test: Split datasets
    """
    # Get default parameters for the model
    default_params = get_default_params(model_type)
    
    # Override defaults with custom params if provided
    if params is not None:
        default_params.update(params)
    
    # Create the base model with parameters
    model_dict = {
        # Regression models
        "linear_regression": LinearRegression(**default_params),
        "ridge_regression": Ridge(**default_params),
        "lasso_regression": Lasso(**default_params),
        "elastic_net": ElasticNet(**default_params),
        "decision_tree_regressor": DecisionTreeRegressor(**default_params),
        "random_forest_regressor": RandomForestRegressor(**default_params),
        "gradient_boosting_regressor": GradientBoostingRegressor(**default_params),
        "xgboost_regressor": XGBRegressor(**default_params),
        "lightgbm_regressor": LGBMRegressor(**default_params),
        "svr": SVR(**default_params),
        "knn_regressor": KNeighborsRegressor(**default_params),
        "mlp_regressor": MLPRegressor(**default_params),

        # Classification models
        "logistic_regression": LogisticRegression(**default_params),
        "knn_classifier": KNeighborsClassifier(**default_params),
        "decision_tree_classifier": DecisionTreeClassifier(**default_params),
        "random_forest_classifier": RandomForestClassifier(**default_params),
        "gradient_boosting_classifier": GradientBoostingClassifier(**default_params),
        "xgboost_classifier": XGBClassifier(**default_params),
        "lightgbm_classifier": LGBMClassifier(**default_params),
        "svm_classifier": SVC(**default_params),
        "gaussian_nb": GaussianNB(**default_params),
        "multinomial_nb": MultinomialNB(**default_params),
        "mlp_classifier": MLPClassifier(**default_params),
    }

    if model_type not in model_dict:
        raise ValueError(f"Model '{model_type}' is not recognized. Please choose from:\n{list(model_dict.keys())}")

    base_model = model_dict[model_type]

    # Split and scale
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle, random_state=random_state)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Use GridSearchCV if requested
    if use_grid_search:
        param_grid = get_param_grid(model_type)
        if param_grid:
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error' if 'regressor' in model_type else 'accuracy',
                n_jobs=-1
            )
            grid_search.fit(x_train_scaled, y_train)
            model = grid_search
            print(f"Best parameters found: {grid_search.best_params_}")
        else:
            print(f"No parameter grid defined for {model_type}. Using default model.")
            base_model.fit(x_train_scaled, y_train)
            model = base_model
    else:
        # Train with provided parameters
        base_model.fit(x_train_scaled, y_train)
        model = base_model

    return scaler, model, x_train, x_test, y_train, y_test

# Update each individual training function to use default parameters
def prepare_and_train_linear_regression_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Linear Regression model after splitting and scaling the data."""
    custom_params = params or get_default_params("linear_regression")
    return prepare_and_train_model(x, y, "linear_regression", test_size, shuffle, random_state, custom_params)

def prepare_and_train_ridge_regression_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Ridge Regression model after splitting and scaling the data."""
    custom_params = params or get_default_params("ridge_regression")
    return prepare_and_train_model(x, y, "ridge_regression", test_size, shuffle, random_state, custom_params)

def prepare_and_train_lasso_regression_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Lasso Regression model after splitting and scaling the data."""
    custom_params = params or get_default_params("lasso_regression")
    return prepare_and_train_model(x, y, "lasso_regression", test_size, shuffle, random_state, custom_params)

def prepare_and_train_elastic_net_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Elastic Net model after splitting and scaling the data."""
    custom_params = params or get_default_params("elastic_net")
    return prepare_and_train_model(x, y, "elastic_net", test_size, shuffle, random_state, custom_params)

def prepare_and_train_decision_tree_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Decision Tree Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("decision_tree_regressor")
    return prepare_and_train_model(x, y, "decision_tree_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_random_forest_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Random Forest Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("random_forest_regressor")
    return prepare_and_train_model(x, y, "random_forest_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_gradient_boosting_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Gradient Boosting Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("gradient_boosting_regressor")
    return prepare_and_train_model(x, y, "gradient_boosting_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_xgboost_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a XGBoost Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("xgboost_regressor")
    return prepare_and_train_model(x, y, "xgboost_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_lightgbm_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a LightGBM Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("lightgbm_regressor")
    return prepare_and_train_model(x, y, "lightgbm_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_svr_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Support Vector Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("svr")
    return prepare_and_train_model(x, y, "svr", test_size, shuffle, random_state, custom_params)

def prepare_and_train_knn_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a K-Nearest Neighbors Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("knn_regressor")
    return prepare_and_train_model(x, y, "knn_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_mlp_regressor_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a MLP Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("mlp_regressor")
    return prepare_and_train_model(x, y, "mlp_regressor", test_size, shuffle, random_state, custom_params)

def prepare_and_train_logistic_regression_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Logistic Regression model after splitting and scaling the data."""
    custom_params = params or get_default_params("logistic_regression")
    return prepare_and_train_model(x, y, "logistic_regression", test_size, shuffle, random_state, custom_params)

def prepare_and_train_knn_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a K-Nearest Neighbors Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("knn_classifier")
    return prepare_and_train_model(x, y, "knn_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_decision_tree_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Decision Tree Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("decision_tree_classifier")
    return prepare_and_train_model(x, y, "decision_tree_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_random_forest_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Random Forest Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("random_forest_classifier")
    return prepare_and_train_model(x, y, "random_forest_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_gradient_boosting_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Gradient Boosting Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("gradient_boosting_classifier")
    return prepare_and_train_model(x, y, "gradient_boosting_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_xgboost_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a XGBoost Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("xgboost_classifier")
    return prepare_and_train_model(x, y, "xgboost_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_lightgbm_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a LightGBM Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("lightgbm_classifier")
    return prepare_and_train_model(x, y, "lightgbm_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_svm_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Support Vector Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("svm_classifier")
    return prepare_and_train_model(x, y, "svm_classifier", test_size, shuffle, random_state, custom_params)

def prepare_and_train_gaussian_nb_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Gaussian Naive Bayes model after splitting and scaling the data."""
    custom_params = params or get_default_params("gaussian_nb")
    return prepare_and_train_model(x, y, "gaussian_nb", test_size, shuffle, random_state, custom_params)

def prepare_and_train_multinomial_nb_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Multinomial Naive Bayes model after splitting and scaling the data."""
    custom_params = params or get_default_params("multinomial_nb")
    return prepare_and_train_model(x, y, "multinomial_nb", test_size, shuffle, random_state, custom_params)

def prepare_and_train_mlp_classifier_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a MLP Classifier model after splitting and scaling the data."""
    custom_params = params or get_default_params("mlp_classifier")
    return prepare_and_train_model(x, y, "mlp_classifier", test_size, shuffle, random_state, custom_params)

# Update the random forest function to use the standard naming pattern
def prepare_and_train_random_forest_model(x, y, test_size=0.3, shuffle=True, random_state=0, params=None):
    """Trains a Random Forest Regressor model after splitting and scaling the data."""
    custom_params = params or get_default_params("random_forest_regressor")
    return prepare_and_train_model(x, y, "random_forest_regressor", test_size, shuffle, random_state, custom_params)

# Example usage showing how to use the new parameters and GridSearchCV
def example_with_hyperparameter_tuning():
    """Example of how to use the functions with hyperparameter tuning."""
    # Sample data (replace with your actual data)
    import numpy as np
    from sklearn.datasets import make_regression
    
    # Generate sample regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    
    # Example 1: Train with default parameters
    scaler, model, x_train, x_test, y_train, y_test = prepare_and_train_random_forest_regressor_model(X, y)
    
    # Example 2: Train with custom parameters
    custom_params = {
        'n_estimators': 200,
        'max_depth': 5
    }
    scaler, model, x_train, x_test, y_train, y_test = prepare_and_train_random_forest_regressor_model(
        X, y, params=custom_params
    )
    
    # Example 3: Use the generic function with grid search
    scaler, model, x_train, x_test, y_train, y_test = prepare_and_train_model(X, y, "random_forest_regressor", use_grid_search=True)