import numpy as np
import joblib
import streamlit as st
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import os

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

def get_user_input():
    """
    Collects input data from the user in both English and Arabic.
    Returns the user input as a dictionary.
    """
    # Create inputs for all required features (English and Arabic)
    bedrooms = st.number_input("Number of Bedrooms: / عدد الغرف:", min_value=1, max_value=50)
    bathrooms = st.number_input("Number of Bathrooms: / عدد الحمامات:", min_value=1, max_value=10)
    sqft_living = st.number_input("Square Footage of Living Area (sqft): / مساحة المنطقة السكنية (قدم مربع):", min_value=200, max_value=10000)
    floors = st.number_input("Number of Floors: / عدد الطوابق:", min_value=1, max_value=5)
    waterfront = st.selectbox("Waterfront (1 = Yes, 0 = No): / بالقرب من الماء (1 = نعم، 0 = لا):", [0, 1])
    view = st.selectbox("View Quality (0 to 4): / جودة الإطلالة (من 0 إلى 4):", [0, 1, 2, 3, 4])
    condition = st.selectbox("Condition (1 to 5): / الحالة (من 1 إلى 5):", [1, 2, 3, 4, 5])
    grade = st.selectbox("Grade (1 to 13): / الدرجة (من 1 إلى 13):", [i for i in range(1, 14)])
    yr_built = st.number_input("Year Built: / سنة البناء:", min_value=1900, max_value=2025)
    Renovated = st.selectbox("Renovated (1 = Yes, 0 = No): / تم تجديده (1 = نعم، 0 = لا):", [0, 1])
    sqft_lot = st.number_input("Lot Size (sqft): / حجم الأرض (قدم مربع):", min_value=500, max_value=100000)
    sqft_above = st.number_input("Square Footage of Area Above Ground (sqft): / مساحة المنطقة فوق الأرض (قدم مربع):", min_value=100, max_value=10000)
    sqft_basement = st.number_input("Square Footage of Basement (sqft): / مساحة الطابق السفلي (قدم مربع):", min_value=0, max_value=5000)

    # Create the feature array for prediction
    user_input = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'yr_built': yr_built,
        'Renovated':Renovated,
        'sqft_lot': sqft_lot,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement
    }

    return user_input

def load_data(file_path: str):
    """
    Loads a CSV file into a pandas DataFrame and processes the data.

    This function:
    1. Attempts to read the CSV file from the provided path.
    2. Converts all column names to lowercase for consistency.
    3. Strips any leading/trailing spaces from column names and replaces multiple spaces with a single space.
    4. Prints the initial dimensions (rows and columns) of the dataset.
    5. Prints the updated column names after loading the data.
    6. Prints insights about the dataset such as shape, column types, missing values, and unique values.
    7. Returns the DataFrame containing the loaded data.

    Parameters:
    - file_path (str): The path to the CSV file to be loaded.

    Returns:
    - df (pandas.DataFrame): The loaded dataset as a DataFrame.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    print("=============================================================================")
    # Print updated column names
    print(f"column names: {df.columns.tolist()}")

    # Clean column names: strip spaces, convert to lowercase, and replace multiple spaces with a single one
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    
    # Store initial dimensions
    initial_number_of_rows = df.shape[0]
    initial_number_of_columns = df.shape[1]
    print("=============================================================================")
    # Print updated column names
    print(f"Updated column names(strip, lowercase, and standardize spaces): {df.columns.tolist()}")
    print("=============================================================================")
    
    # Confirm successful data loading
    print(f"Dataset loaded successfully with {initial_number_of_rows} rows and {initial_number_of_columns} columns.")
    print("=============================================================================")
    
    return df

def check_data_for_preprocessing(df: pd.DataFrame, verbose: bool = True, return_summary: bool = False, return_text_report: bool = False, show_plots: bool = False):
    """
    Analyzes the dataset and provides clear, column-specific preprocessing recommendations.

    Parameters:
    - df (pd.DataFrame): Dataset to analyze.
    - verbose (bool): If True, prints the analysis summary.
    - return_summary (bool): If True, returns the insights as a dictionary.
    - return_text_report (bool): If True, returns the summary as a list of strings for saving to a text file.
    - show_plots (bool): If True, shows histograms for skewed numeric columns.

    Returns:
    - dict, list or None: Depending on the flags, returns insights or text report.
    """
    
    insights = {}
    report_lines = []

    # Basic shape
    shape = df.shape
    insights['shape'] = shape
    report_lines.append("="*75)
    report_lines.append(f"📊 Dataset Shape: {shape}")
    
    # 1. Missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_cols = missing_values[missing_values > 0].index.tolist()
    insights['missing_values'] = pd.DataFrame({'Missing Count': missing_values, 'Missing %': missing_pct})
    insights['missing_cols'] = missing_cols

    # 2. Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / len(df)) * 100
    insights['duplicate_rows'] = {'Count': duplicate_rows, 'Percentage': duplicate_pct}
    report_lines.append(f"📌 Duplicate Rows: {duplicate_rows} ({duplicate_pct:.2f}%)")
    report_lines.append("="*75)

    # 3. Unique values and constants
    unique_values = df.nunique()
    constant_columns = unique_values[unique_values == 1].index.tolist()
    high_cardinality_columns = unique_values[unique_values > 50].index.tolist()
    insights['unique_values'] = unique_values
    insights['constant_columns'] = constant_columns
    insights['high_cardinality_columns'] = high_cardinality_columns

    # 4. Object and mixed types
    object_cols = df.select_dtypes(include='object').columns.tolist()
    mixed_types = [col for col in df.columns if df[col].apply(type).nunique() > 1]
    insights['object_columns'] = object_cols
    insights['mixed_type_columns'] = mixed_types

    # 5. Numeric and skewed
    numeric_cols = df.select_dtypes(include=np.number).columns
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    highly_skewed = skewness[skewness > 1].index.tolist()
    insights['numeric_summary'] = df[numeric_cols].describe().T
    insights['skewed_columns'] = highly_skewed

    # Report: Missing
    if missing_cols:
        report_lines.append("🧹 Handle Missing Values:")
        for col in missing_cols:
            report_lines.append(f" - {col}: {missing_values[col]} missing ({missing_pct[col]:.2f}%)")
    else:
        report_lines.append("✅ No missing values.")
    report_lines.append("="*75)

    # Report: Constants
    if constant_columns:
        report_lines.append("🗑️ Drop Constant Columns:")
        for col in constant_columns:
            report_lines.append(f" - {col}")
    else:
        report_lines.append("✅ No constant columns.")
    report_lines.append("="*75)

    # Report: Categorical
    if object_cols:
        report_lines.append("🧾 Encode Categorical Columns:")
        for col in object_cols:
            report_lines.append(f" - {col}")
    else:
        report_lines.append("✅ No object-type columns.")
    report_lines.append("="*75)

    # Report: Mixed types
    if mixed_types:
        report_lines.append("⚠️ Mixed Type Columns:")
        for col in mixed_types:
            report_lines.append(f" - {col}")
    else:
        report_lines.append("✅ No mixed-type columns.")
    report_lines.append("="*75)

    # Report: High Cardinality
    if high_cardinality_columns:
        report_lines.append("📊 High Cardinality Columns:")
        for col in high_cardinality_columns:
            report_lines.append(f" - {col}: {unique_values[col]} unique values")
    else:
        report_lines.append("✅ No high-cardinality columns.")
    report_lines.append("="*75)

    # Report: Skewed
    if highly_skewed:
        report_lines.append("📈 Skewed Numeric Columns:")
        for col in highly_skewed:
            report_lines.append(f" - {col}: Skewness = {skewness[col]:.2f}")
    else:
        report_lines.append("✅ No highly skewed numeric columns.")
    report_lines.append("="*75)

    # Report: Summary Actions
    report_lines.append("✅ Suggested Next Steps:")
    if missing_cols: report_lines.append(" - Handle missing data.")
    if constant_columns: report_lines.append(" - Drop constant columns.")
    if object_cols: report_lines.append(" - Encode categorical features.")
    if high_cardinality_columns: report_lines.append(" - Consider binning/embedding for high cardinality.")
    if highly_skewed: report_lines.append(" - Apply transformations to skewed features.")
    if mixed_types: report_lines.append(" - Resolve inconsistent data types.")
    report_lines.append("="*75)

    if verbose:
        for line in report_lines:
            print(line)

    if show_plots and highly_skewed:
        import matplotlib.pyplot as plt
        import seaborn as sns
        for col in highly_skewed:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of Skewed Feature: {col}")
            plt.tight_layout()
            plt.show()

    if return_text_report:
        return report_lines

    if return_summary:
        return insights

    if show_plots and highly_skewed:
        import matplotlib.pyplot as plt
        import seaborn as sns
        for col in highly_skewed:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of Skewed Feature: {col}")
            plt.show()

    if return_summary:
        return insights

class DataFrameStatistics:
    """
    A class to perform various statistical operations on a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

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

    def statistics(self) -> None:
        print("="*75)
        print("Missing Data Info (count and percentage):")
        print(self.missing_data_info())
        print("="*75)
        print("\nDuplicate Row Info (count and percentage):")
        print(self.duplicate_info())
        print("="*75)
        print("\nData Info:")
        self.data_info()
        print("="*75)
        print("\nShape of DataFrame:")
        print(self.shape())
        print("="*75)
        print("\nUnique Values Info (count and percentage):")
        print(self.unique_values_info())
        print("="*75)
        print("\nDescriptive Statistics (for numerical columns):")
        print(self.describe())
        print("="*75)

    def generate_report_lines(self) -> list:
        lines = []
        lines.append("="*75)
        lines.append("📌 Missing Data Info (count and percentage):")
        lines.append(str(self.missing_data_info()))
        lines.append("="*75)

        lines.append("\n📌 Duplicate Row Info (count and percentage):")
        lines.append(str(self.duplicate_info()))
        lines.append("="*75)

        lines.append("\n📌 Shape of DataFrame:")
        lines.append(str(self.shape()))
        lines.append("="*75)

        lines.append("\n📌 Unique Values Info (count and percentage):")
        lines.append(str(self.unique_values_info()))
        lines.append("="*75)

        lines.append("\n📌 Descriptive Statistics (for numerical columns):")
        lines.append(str(self.describe()))
        lines.append("="*75)

        return lines

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
    Encodes the specified column in the dataframe based on the encoding type.
    
    Parameters:
    - data: pandas DataFrame
    - column_name: str, the name of the column to be encoded
    - encoding_type: str, encoding method - either 'onehot' or 'label'. Default is 'onehot'.
    
    Returns:
    - Encoded dataframe or Series
    """
    
    # Ensure the column exists in the dataframe
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    
    if encoding_type == "label":
        # Label Encoding: Convert categories to integers
        label_encoder = LabelEncoder()
        data[column_name] = label_encoder.fit_transform(data[column_name])
        
    elif encoding_type == "onehot":
        # One-Hot Encoding: Create binary columns for each category
        data = pd.get_dummies(data, columns=[column_name], drop_first=False)
        
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