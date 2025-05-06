import numpy as np
import joblib
import streamlit as st
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

def make_prediction(user_input, ModelPath, ScalerPath):
    """
    Makes a prediction using a pre-trained machine learning model and a saved scaler.
    
    This function:
    1. Loads the model and scaler from the provided paths.
    2. Processes the user input (converts to a 2D array and scales it).
    3. Returns the prediction result from the model.
    
    Parameters:
    - user_input (dict): A dictionary containing feature names and their respective values for prediction.
    - model_path (str): The file path to the saved trained model (in .pkl format).
    - scaler_path (str): The file path to the saved scaler (in .pkl format) used to scale the input data.
    
    Returns:
    - result: The prediction made by the model based on the input data.
    
    Exceptions:
    - FileNotFoundError: If the model or scaler file is not found at the provided path.
    - Any other errors during input processing, scaling, or prediction will be logged and re-raised.
    """
    try:
        # Load the trained model
        LinearRegMod = joblib.load(ModelPath)
    except FileNotFoundError:
        logging.error(f"Model file not found: {ModelPath}")
        raise  # Rethrow the error after logging it
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model from {ModelPath}: {e}")
        raise  # Rethrow the error after logging it

    try:
        # Load the saved scaler
        MyScaler = joblib.load(ScalerPath)
    except FileNotFoundError:
        logging.error(f"Scaler file not found: {ScalerPath}")
        raise  # Rethrow the error after logging it
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the scaler from {ScalerPath}: {e}")
        raise  # Rethrow the error after logging it

    try:
        # Convert dict to 2D array if needed
        input_array = np.array([list(user_input.values())])
        input_array = MyScaler.transform(input_array)  # Scale the input data
    except Exception as e:
        logging.error(f"Error occurred while processing user input: {e}")
        raise  # Rethrow the error after logging it

    try:
        # Perform the prediction
        result = LinearRegMod.predict(input_array)
        return result
    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")
        raise  # Rethrow the error after logging it

def load_data(file_path: str, sheet_name: str = 0):
    """
    Loads a CSV or Excel file into a pandas DataFrame and processes the data.

    This function:
    1. Detects the file type by extension and reads it accordingly.
    2. For Excel files, reads the first sheet by default or a specified sheet name/index.
    3. Converts all column names to lowercase for consistency.
    4. Strips any leading/trailing spaces from column names and replaces multiple spaces with a single space.
    5. Prints the initial dimensions (rows and columns) of the dataset.
    6. Prints the updated column names after loading the data.
    7. Returns the DataFrame containing the loaded data.

    Parameters:
    - file_path (str): The path to the data file to be loaded.
    - sheet_name (str|int, optional): The Excel sheet name or index to read. Defaults to 0 (first sheet).

    Returns:
    - df (pandas.DataFrame): The loaded dataset as a DataFrame.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print("=============================================================================")
    print(f"Original column names: {df.columns.tolist()}")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

    print("=============================================================================")
    print(f"Updated column names (strip, lowercase, and standardize spaces): {df.columns.tolist()}")
    print("=============================================================================")
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("=============================================================================")

    return df

def full_dataframe_report(
    df: pd.DataFrame,
    verbose: bool = True,
    return_dict: bool = True,
    return_text: bool = True,
    show_plots: bool = False,
    top_n: int = 3
):
    """
    Provides a full summary of a DataFrame including structure, missing data, 
    duplicates, unique values, outliers, and memory usage.
    """

    # Set display format for floats (human-readable, no scientific notation)
    pd.set_option('display.float_format', '{:,.2f}'.format)

    report = {}
    text = []

    def section(title):
        line = "=" * 80
        text.extend([line, f"{title}", line])

    # Shape
    shape = df.shape
    report['shape'] = shape
    section("Shape of DataFrame")
    text.append(f"{shape[0]} rows × {shape[1]} columns")

    # Validate input
    is_empty = df.empty
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    mixed_type_cols = [col for col in df.columns if df[col].apply(type).nunique() > 1]
    report['validation'] = {
        'is_empty': is_empty,
        'duplicated_columns': duplicated_cols,
        'constant_columns': constant_cols,
        'mixed_type_columns': mixed_type_cols
    }
    section("Validation Checks")
    text.append(f"Empty: {is_empty}")
    text.append(f"Duplicated Columns: {duplicated_cols or 'None'}")
    text.append(f"Constant Columns: {constant_cols or 'None'}")
    text.append(f"Mixed-Type Columns: {mixed_type_cols or 'None'}")

    # Missing data
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_pct
    })
    report['missing_data'] = missing_df.to_dict()
    section("Missing Data")
    text.extend(missing_df[missing_count > 0].to_string().splitlines() or ["No missing values."])

    # Duplicates
    dup_rows = df.duplicated().sum()
    dup_pct = (dup_rows / len(df)) * 100
    report['duplicates'] = {'count': dup_rows, 'percentage': dup_pct}
    section("Duplicate Rows")
    text.append(f"{dup_rows} duplicate rows ({dup_pct:.2f}%)")

    # Data types
    dtype_info = pd.DataFrame(df.dtypes, columns=["Data Type"])
    report['data_types'] = dtype_info.to_dict()
    section("Data Types")
    text.extend(dtype_info.to_string().splitlines())

    # Unique values
    unique_vals = df.nunique()
    unique_pct = (unique_vals / len(df)) * 100
    unique_df = pd.DataFrame({
        'Unique Count': unique_vals,
        'Unique Percentage': unique_pct
    }).sort_values(by='Unique Count')
    report['unique_values'] = unique_df.to_dict()
    section("Unique Values")
    text.extend(unique_df.to_string().splitlines())

    # Top frequent values for categoricals
    top_freq = {}
    section("Top Frequent Values (Categorical Columns)")
    for col in df.select_dtypes(include=['object', 'category']):
        top_values = df[col].value_counts().head(top_n)
        top_freq[col] = top_values.to_dict()
        line = f"{col}: " + ", ".join([f"{k} ({v})" for k, v in top_values.items()])
        text.append(line)
    if not top_freq:
        text.append("No categorical columns.")
    report['top_frequent_values'] = top_freq

    # Outliers using IQR
    outliers = {}
    section("Outliers (IQR method)")
    for col in df.select_dtypes(include=np.number):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = mask.sum()
        outlier_pct = (outlier_count / len(df)) * 100
        outliers[col] = {'Outlier Count': outlier_count, 'Outlier Percentage': outlier_pct}
        text.append(f"{col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
    report['outliers'] = outliers

    # Skewness
    skewness = df.select_dtypes(include=np.number).skew()
    skewed_cols = skewness[skewness.abs() > 1].sort_values(ascending=False)
    report['highly_skewed_columns'] = skewed_cols.to_dict()
    section("Highly Skewed Numeric Columns (|skew| > 1)")
    if not skewed_cols.empty:
        for col, val in skewed_cols.items():
            text.append(f"{col}: {val:.2f}")
            if show_plots:
                plt.figure(figsize=(6, 3))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Skewed: {col}")
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()
    else:
        text.append("No highly skewed numeric columns.")

    # Memory usage
    mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    report['memory_usage_MB'] = round(mem_usage, 2)
    section("Memory Usage")
    text.append(f"Total memory usage: {mem_usage:.2f} MB")

    # Descriptive statistics
    desc_stats = df.describe().T
    report['describe'] = desc_stats.to_dict()
    section("Descriptive Statistics")
    text.extend(desc_stats.to_string().splitlines())

    # Reset display format after function if needed
    # pd.reset_option('display.float_format')

    if verbose:
        print("\n".join(text))

    if return_text:
        return text
    elif return_dict:
        return report
    else:
        return None

def check_data_for_preprocessing(
    df: pd.DataFrame, 
    verbose: bool = True, 
    return_summary: bool = True, 
    return_text_report: bool = True, 
    show_plots: bool = True,
    top_n_unique: int = 3
):
    """
    Analyzes a dataset and provides preprocessing recommendations with optional summaries, text reports, and plots.

    Parameters:
    - df (pd.DataFrame): Dataset to analyze.
    - verbose (bool): Print the analysis summary if True.
    - return_summary (bool): Return insights as a dictionary if True.
    - return_text_report (bool): Return the report as a list of strings if True.
    - show_plots (bool): Display histograms for highly skewed numeric columns if True.
    - top_n_unique (int): Number of top frequent values to show for categorical columns.

    Returns:
    - dict or list or None: Depending on the flags.
    """

    insights = {}
    report_lines = []

    # 0. Basic Shape
    shape = df.shape
    insights['shape'] = shape
    report_lines.append("="*90)
    report_lines.append(f" Dataset Shape: {shape[0]} rows × {shape[1]} columns")
    report_lines.append("="*90)

    # 1. Missing Values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_cols = missing_values[missing_values > 0].index.tolist()
    insights['missing_values'] = missing_values.to_dict()
    insights['missing_percentage'] = missing_pct.to_dict()
    insights['missing_columns'] = missing_cols

    if missing_cols:
        report_lines.append(" Handle Missing Values:")
        for col in missing_cols:
            report_lines.append(f" - {col}: {missing_values[col]} missing ({missing_pct[col]:.2f}%)")
    else:
        report_lines.append(" No missing values found.")
    report_lines.append("="*90)

    # 2. Duplicate Rows
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / len(df)) * 100
    insights['duplicate_rows'] = {'count': duplicate_rows, 'percentage': duplicate_pct}
    report_lines.append(f" Duplicate Rows: {duplicate_rows} ({duplicate_pct:.2f}%)")
    report_lines.append("="*90)

    # 3. Unique Values & Constant Columns
    unique_values = df.nunique()
    constant_columns = unique_values[unique_values == 1].index.tolist()
    high_cardinality_columns = unique_values[unique_values > 50].index.tolist()
    insights['unique_values_per_column'] = unique_values.to_dict()
    insights['constant_columns'] = constant_columns
    insights['high_cardinality_columns'] = high_cardinality_columns

    if constant_columns:
        report_lines.append(" Drop Constant Columns (only one unique value):")
        report_lines.extend([f" - {col}" for col in constant_columns])
    else:
        report_lines.append(" No constant columns found.")
    report_lines.append("="*90)

    if high_cardinality_columns:
        report_lines.append(" High Cardinality Columns (>50 unique values):")
        report_lines.extend([f" - {col}: {unique_values[col]} unique values" for col in high_cardinality_columns])
    else:
        report_lines.append(" No high-cardinality columns found.")
    report_lines.append("="*90)

    # 4. Object Columns and Mixed Types
    object_cols = df.select_dtypes(include='object').columns.tolist()
    mixed_type_columns = [col for col in df.columns if df[col].apply(type).nunique() > 1]
    insights['object_columns'] = object_cols
    insights['mixed_type_columns'] = mixed_type_columns

    if object_cols:
        report_lines.append(" Categorical (object) Columns for Encoding:")
        for col in object_cols:
            top_values = df[col].value_counts().head(top_n_unique)
            top_values_text = ', '.join([f"{k} ({v})" for k, v in top_values.items()])
            report_lines.append(f" - {col}: {top_values_text}")
    else:
        report_lines.append(" No categorical (object) columns found.")
    report_lines.append("="*90)

    if mixed_type_columns:
        report_lines.append(" Columns with Mixed Data Types:")
        report_lines.extend([f" - {col}" for col in mixed_type_columns])
    else:
        report_lines.append(" No mixed-type columns detected.")
    report_lines.append("="*90)

    # 5. Numeric Columns & Skewness
    numeric_cols = df.select_dtypes(include=np.number).columns
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    highly_skewed = skewness[skewness.abs() > 1].index.tolist()
    insights['numeric_summary'] = df[numeric_cols].describe().T.to_dict()
    insights['highly_skewed_columns'] = {col: skewness[col] for col in highly_skewed}

    if highly_skewed:
        report_lines.append(" Highly Skewed Numeric Columns (|skew| > 1):")
        for col in highly_skewed:
            report_lines.append(f" - {col}: Skewness = {skewness[col]:.2f}")
    else:
        report_lines.append(" No highly skewed numeric columns found.")
    report_lines.append("="*90)

    # 6. Memory Usage
    memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
    insights['memory_usage_MB'] = memory_usage
    report_lines.append(f" Memory Usage: {memory_usage:.2f} MB")
    report_lines.append("="*90)

    # 7. Suggested Next Steps
    report_lines.append(" Suggested Next Steps for Preprocessing:")
    if missing_cols: report_lines.append(" - Handle missing values (impute/drop).")
    if constant_columns: report_lines.append(" - Drop constant columns.")
    if object_cols: report_lines.append(" - Encode categorical variables.")
    if high_cardinality_columns: report_lines.append(" - Consider reducing cardinality.")
    if highly_skewed: report_lines.append(" - Apply transformations (log, box-cox, etc.) to skewed features.")
    if mixed_type_columns: report_lines.append(" - Standardize mixed-type columns.")
    report_lines.append("="*90)

    # 8. Show Plots
    if show_plots and highly_skewed:
        for col in highly_skewed:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of Skewed Feature: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    # 9. Output Control
    if verbose:
        for line in report_lines:
            print(line)

    if return_text_report:
        return report_lines

    if return_summary:
        return insights

class DataFrameStatistics:
    """
    A class to perform various statistical operations and validation on a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def validate_input(self) -> dict:
        validations = {}
        validations['is_empty'] = self.df.empty
        validations['duplicated_columns'] = self.df.columns[self.df.columns.duplicated()].tolist()
        validations['constant_columns'] = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        validations['mixed_type_columns'] = [
            col for col in self.df.columns 
            if self.df[col].apply(type).nunique() > 1
        ]
        return validations

    def missing_data_info(self) -> pd.DataFrame:
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percentage
        })

    def duplicate_info(self) -> pd.DataFrame:
        duplicate_rows = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(self.df)) * 100
        return pd.DataFrame({
            'Duplicate Count': [duplicate_rows],
            'Duplicate Percentage': [duplicate_percentage]
        })

    def data_info(self) -> None:
        print(self.df.info())

    def shape(self) -> tuple:
        return self.df.shape

    def unique_values_info(self) -> pd.DataFrame:
        unique_values = self.df.nunique()
        unique_percentage = (unique_values / len(self.df)) * 100
        return pd.DataFrame({
            'Unique Count': unique_values,
            'Unique Percentage': unique_percentage
        }).sort_values(by='Unique Count')

    def describe(self) -> pd.DataFrame:
        return self.df.describe()

    def data_types_info(self) -> pd.DataFrame:
        return pd.DataFrame(self.df.dtypes, columns=['Data Type'])

    def top_frequent_values(self, top_n=3) -> dict:
        frequent_values = {}
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            frequent_values[col] = self.df[col].value_counts().head(top_n).to_dict()
        return frequent_values

    def outlier_info(self) -> dict:
        outlier_summary = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            outlier_summary[col] = {
                'Outlier Count': outliers.shape[0],
                'Outlier Percentage': (outliers.shape[0] / len(self.df)) * 100
            }
        return outlier_summary

    def memory_usage_info(self) -> str:
        mem_usage = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        return f"Total memory usage: {mem_usage:.2f} MB"

    def statistics(self) -> None:
        print("="*75)
        print("Validation Results:")
        print(self.validate_input())
        print("="*75)
        print("\nMissing Data Info (count and percentage):")
        print(self.missing_data_info())
        print("="*75)
        print("\nDuplicate Row Info (count and percentage):")
        print(self.duplicate_info())
        print("="*75)
        print("\nData Types Info:")
        print(self.data_types_info())
        print("="*75)
        print("\nShape of DataFrame:")
        print(self.shape())
        print("="*75)
        print("\nUnique Values Info (count and percentage):")
        print(self.unique_values_info())
        print("="*75)
        print("\nTop Frequent Values (for categorical columns):")
        print(self.top_frequent_values())
        print("="*75)
        print("\nOutlier Info (for numeric columns):")
        print(self.outlier_info())
        print("="*75)
        print("\nMemory Usage:")
        print(self.memory_usage_info())
        print("="*75)
        print("\nDescriptive Statistics (for numerical columns):")
        print(self.describe())
        print("="*75)

    def generate_report(self) -> dict:
        """
        Generate all statistics in a structured dictionary format for easy export.
        """
        return {
            "Validation": self.validate_input(),
            "MissingData": self.missing_data_info().to_dict(),
            "DuplicateInfo": self.duplicate_info().to_dict(),
            "DataTypes": self.data_types_info().to_dict(),
            "Shape": self.shape(),
            "UniqueValues": self.unique_values_info().to_dict(),
            "TopFrequentValues": self.top_frequent_values(),
            "Outliers": self.outlier_info(),
            "MemoryUsage": self.memory_usage_info(),
            "Describe": self.describe().to_dict()
        }

def standardize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the column headers of the DataFrame by:
    - Converting all characters to lowercase
    - Removing leading and trailing spaces
    
    Args:
        df (pd.DataFrame): The input DataFrame whose column headers need to be normalized.
    
    Returns:
        pd.DataFrame: The DataFrame with normalized column headers.
    """
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df.columns = df.columns.str.lower()  # Convert to lowercase
    return df

def identify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies outliers in all numerical columns of the DataFrame using the IQR method.
    Returns a DataFrame with the number of outliers, the outlier range, and the min/max values for each column.
    
    Args:
        df (pd.DataFrame): The DataFrame for which outliers need to be identified.
    
    Returns:
        pd.DataFrame: A DataFrame containing the number of outliers, the outlier range, and the min/max values for each numerical column.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns  # Select numeric columns
    
    if len(numerical_cols) == 0:
        print("No numeric columns found in the DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if no numeric columns exist
    
    outlier_info = []

    for col in numerical_cols:
        # Calculate the IQR
        Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        
        # Calculate lower and upper bounds
        minw = Q1 - 1.5 * IQR  # Lower bound
        maxw = Q3 + 1.5 * IQR  # Upper bound

        # Identify outliers for each column
        outliers = df[(df[col] < minw) | (df[col] > maxw)]

        # Get the min and max values for the column
        col_min = df[col].min()
        col_max = df[col].max()

        # Store the outlier count, range, and min/max values
        outlier_info.append({
            'Column': col,
            'Outlier Count': len(outliers),
            'Column Min': col_min,
            'Outlier Range (min)': minw,
            'Outlier Range (max)': maxw,
            'Column Max': col_max
        })
    
    # Create a DataFrame with outlier information
    outlier_df = pd.DataFrame(outlier_info)
    
    # Reorder columns as requested
    outlier_df = outlier_df[['Column', 'Outlier Count', 'Column Min', 'Outlier Range (min)', 'Outlier Range (max)', 'Column Max']]
    
    return outlier_df

def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Drops specified columns from the DataFrame after normalizing column names
    (trimming and converting to lowercase). Prints status for each column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (list): A list of column names to drop from the DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
    """
    # Normalize DataFrame column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Normalize columns to drop
    normalized_columns_to_drop = [col.strip().lower() for col in columns_to_drop]
    
    # Check and print status
    for col in normalized_columns_to_drop:
        if col in df.columns:
            print(f"Column '{col}' found and will be dropped.")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    
    # Drop existing columns
    df_dropped = df.drop(columns=[col for col in normalized_columns_to_drop if col in df.columns])
    
    return df_dropped

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate rows from the DataFrame and prints the number of rows before and after, 
    as well as the number of duplicated rows.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with duplicate rows dropped.
    """
    # Print the number of rows before dropping duplicates
    rows_before = len(df)
    print(f"Number of rows before dropping duplicates: {rows_before}")
    
    # Find duplicated rows
    duplicated_rows = df.duplicated().sum()
    
    # Drop duplicate rows
    df_dropped = df.drop_duplicates()
    
    # Print the number of rows after dropping duplicates
    rows_after = len(df_dropped)
    print(f"Number of rows after dropping duplicates: {rows_after}")
    
    # Print the number of duplicated rows
    print(f"Number of duplicated rows: {duplicated_rows}")
    
    return df_dropped

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value=None) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame based on the specified strategy.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): The strategy to handle missing values. Options are:
                         'drop' to drop rows with missing values,
                         'fill' to fill missing values with a constant value or method,
                         'forward' to forward fill missing values,
                         'backward' to backward fill missing values.
        fill_value (optional): The value to use for filling missing values if the strategy is 'fill'.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    # Print the number of missing values before handling
    missing_before = df.isnull().sum().sum()
    print(f"Number of missing values before handling: {missing_before}")
    
    # Handle missing values based on the specified strategy
    if strategy == 'drop':
        # Drop rows with missing values
        df_handled = df.dropna()
        print("Rows with missing values have been dropped.")
    elif strategy == 'fill':
        if fill_value is not None:
            # Fill missing values with a specified constant value
            df_handled = df.fillna(fill_value)
            print(f"Missing values have been filled with {fill_value}.")
        else:
            print("Error: 'fill' strategy requires a 'fill_value'.")
            return df
    elif strategy == 'forward':
        # Forward fill missing values
        df_handled = df.ffill()
        print("Missing values have been forward-filled.")
    elif strategy == 'backward':
        # Backward fill missing values
        df_handled = df.bfill()
        print("Missing values have been backward-filled.")
    else:
        print(f"Error: Unknown strategy '{strategy}'.")
        return df
    
    # Print the number of missing values after handling
    missing_after = df_handled.isnull().sum().sum()
    print(f"Number of missing values after handling: {missing_after}")
    
    return df_handled

def encode_column(data, column_name, encoding_type="onehot"):
    """
    Encodes the specified column in the dataframe based on the encoding type,
    and creates a new column(s) instead of replacing the original one.

    Parameters:
    - data: pandas DataFrame
    - column_name: str, the name of the column to be encoded
    - encoding_type: str, encoding method - either 'onehot' or 'label'. Default is 'onehot'.

    Returns:
    - DataFrame with new encoded column(s)
    """
    
    # Ensure the column exists in the dataframe
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    
    data = data.copy()  # To avoid modifying the original dataframe

    if encoding_type == "label":
        label_encoder = LabelEncoder()
        new_col = f"{column_name}label"
        data[new_col] = label_encoder.fit_transform(data[column_name])

    elif encoding_type == "onehot":
        onehot_df = pd.get_dummies(data[column_name], prefix=column_name, prefix_sep='', drop_first=False)
        onehot_df.columns = onehot_df.columns.str.lower()  # Make new column names lowercase
        data = pd.concat([data, onehot_df], axis=1)

    else:
        raise ValueError("Invalid encoding_type. Choose either 'label' or 'onehot'.")
    
    return data

def encode_by_ranges(df: pd.DataFrame, column: str, new_column: str, bins: list, labels: list) -> pd.DataFrame:
    """
    Assigns encode labels to a column based on specified bins using a for loop.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to scale.
        new_column (str): The name of the new column to create.
        bins (list): A list of numeric boundaries for the bins.
        labels (list): A list of labels for each bin.
    
    Returns:
        pd.DataFrame: The modified DataFrame with the new scaled column.
    """
    # Initialize the new column with NaN or a default value
    df[new_column] = None  # Or you could initialize it with NaN
    
    # Iterate through each row in the DataFrame
    for index in df.index:
        value = df.loc[index, column]  # Get the value from the column
        
        # Assign the appropriate label based on the bins
        for i in range(1, len(bins)):
            if bins[i-1] <= value < bins[i]:
                df.loc[index, new_column] = labels[i-1]
                break  # Stop once the appropriate bin is found
    
    return df

def save_to_csv(data, filename):
    """
    Function to save data to a CSV file using pandas DataFrame.
    
    Args:
        data (dict or list of dicts): Data to be saved in the CSV file.
        filename (str): The name of the CSV file where data will be saved.
        
    Returns:
        str: Full path of the saved CSV file.
    """
    # Convert the data into a DataFrame (if it is not already a DataFrame)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Get the absolute path of the filename
    full_path = os.path.abspath(filename)
    
    # Save the DataFrame to a CSV file
    data.to_csv(full_path, index=False)
    
    # Print confirmation
    print(f"Data has been saved to {full_path}")
    
    # Return the full path of the saved file
    return full_path

def write_to_text_file(data, filename='output.txt'):
    """
    Writes the given data to a text file.

    Parameters:
    - data (str or list): Text content to write. If a list, each item will be a new line.
    - filename (str): Name of the file to write to (default is 'output.txt').

    Returns:
    - None
    """
    with open(filename, 'w', encoding='utf-8') as file:
        if isinstance(data, list):
            file.write('\n'.join(str(line) for line in data))
        else:
            file.write(str(data))
    print(f" Data written to {filename}")

def feature_engineering(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Performs advanced feature engineering on the dataset.
    
    This function:
    1. Creates interaction features between numeric columns
    2. Generates polynomial features
    3. Creates time-based features for datetime columns
    4. Handles categorical feature interactions
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - target_column (str, optional): Target column name for feature importance
    
    Returns:
    - pd.DataFrame: DataFrame with engineered features
    """
    df_engineered = df.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create interaction features
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            df_engineered[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df_engineered[f'{col1}_div_{col2}'] = df[col1] / df[col2]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    poly_cols = [f'poly_{i}' for i in range(poly_features.shape[1])]
    df_poly = pd.DataFrame(poly_features, columns=poly_cols)
    df_engineered = pd.concat([df_engineered, df_poly], axis=1)
    
    # Handle datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df_engineered[f'{col}_year'] = df[col].dt.year
        df_engineered[f'{col}_month'] = df[col].dt.month
        df_engineered[f'{col}_day'] = df[col].dt.day
        df_engineered[f'{col}_dayofweek'] = df[col].dt.dayofweek
    
    return df_engineered

def validate_data(df: pd.DataFrame, schema: dict) -> dict:
    """
    Validates data against a predefined schema.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - schema (dict): Dictionary defining expected data types and constraints
    
    Returns:
    - dict: Dictionary containing validation results and errors
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check column existence
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing columns: {missing_cols}")
    
    # Check data types
    for col, expected_type in schema.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type != actual_type:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Column {col}: Expected type {expected_type}, got {actual_type}"
                )
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation_results['warnings'].append(
            f"Columns with null values: {null_counts[null_counts > 0].to_dict()}"
        )
    
    return validation_results

def detect_anomalies(df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
    """
    Detects anomalies in the dataset using various methods.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - method (str): Method to use for anomaly detection ('zscore', 'iqr', 'isolation_forest')
    - threshold (float): Threshold for anomaly detection
    
    Returns:
    - pd.DataFrame: DataFrame with anomaly flags
    """
    from sklearn.ensemble import IsolationForest
    import numpy as np
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    anomalies = pd.DataFrame(index=df.index)
    
    if method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            anomalies[f'{col}_anomaly'] = z_scores > threshold
            
    elif method == 'iqr':
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies[f'{col}_anomaly'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
    elif method == 'isolation_forest':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        for col in numeric_cols:
            anomalies[f'{col}_anomaly'] = iso_forest.fit_predict(df[[col]]) == -1
            
    return anomalies

def create_time_series_features(df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
    """
    Creates time series features from a datetime column.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - date_column (str): Name of the datetime column
    - target_column (str): Name of the target column
    
    Returns:
    - pd.DataFrame: DataFrame with time series features
    """
    df_ts = df.copy()
    
    # Basic time features
    df_ts[f'{date_column}_year'] = df[date_column].dt.year
    df_ts[f'{date_column}_month'] = df[date_column].dt.month
    df_ts[f'{date_column}_day'] = df[date_column].dt.day
    df_ts[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df_ts[f'{date_column}_quarter'] = df[date_column].dt.quarter
    
    # Lag features
    for lag in [1, 7, 30]:
        df_ts[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    # Rolling statistics
    for window in [7, 30]:
        df_ts[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
        df_ts[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
    
    # Expanding statistics
    df_ts[f'{target_column}_expanding_mean'] = df[target_column].expanding().mean()
    df_ts[f'{target_column}_expanding_std'] = df[target_column].expanding().std()
    
    return df_ts