import numpy as np
import joblib
import streamlit as st
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

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

def check_data_for_preprocessing(df: pd.DataFrame):
    """
    Analyzes the given dataset and provides insights for preprocessing.
    
    This function checks:
    1. Missing values in each column.
    2. Duplicates in the dataset.
    3. Data types of each column and suggests conversion if needed.
    4. Number of unique values in each column for potential categorical features.
    5. Potentially problematic columns for preprocessing (e.g., constant columns, highly skewed numeric columns).
    6. Basic statistics for numerical columns to identify outliers or scaling issues.

    Parameters:
    - df (pandas.DataFrame): The dataset to analyze.

    Returns:
    - None (prints insights about the dataset and preprocessing recommendations).
    """
    
    # 1. Check missing values in each column
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # 2. Check duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / len(df)) * 100
    
    # 3. Check the number of unique values in each column to identify categorical columns
    unique_values = df.nunique()
    unique_percentage = (unique_values / len(df)) * 100
    
    # 4. Identify constant columns (columns with only one unique value)
    constant_columns = unique_values[unique_values == 1].index
    
    # 5. Identify numeric columns for potential outliers or scaling
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # 6. Identify columns with high cardinality (many unique values) that may need encoding
    high_cardinality_columns = unique_values[unique_values > 50].index
    
    # Output Analysis
    print("=============================================================================")
    print(f"Duplicate rows: {duplicate_rows} ({duplicate_percentage:.2f}%)")
    print("=============================================================================")
    
    # 1. Missing data
    print(f"Missing data in each column (count and percentage):")
    for col in missing_values.index:
        print(f"  - {col}: {missing_values[col]} missing values ({missing_percentage[col]:.2f}%)")
    print("=============================================================================")

    # 2. Unique values per column (count and percentage)
    print(f"Unique values in each column (count and percentage):")
    for col in unique_values.index:
        print(f"  - {col}: {unique_values[col]} unique values ({unique_percentage[col]:.2f}%)")
    print("=============================================================================")
    
    # 3. High cardinality columns (might need encoding or binning)
    print(f"High cardinality columns (might need encoding or binning):\n{high_cardinality_columns}")
    print("=============================================================================")
    
    # 4. Suggest potential preprocessing actions
    print("Suggested preprocessing actions:")
    print("=============================================================================")
    # Handling missing values
    if not missing_values[missing_values > 0].empty:
        print("- Consider filling missing values using imputation or dropping rows/columns with too many missing values.")
    
    # Handling data type conversions
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"- The column '{col}' is of type object, consider encoding it (e.g., one-hot encoding or label encoding).")
    
    # Scaling of numeric columns (if needed)
    skewed_columns = df[numeric_columns].skew().sort_values(ascending=False)
    print(f"Skewed numeric columns (might need scaling or transformation):\n{skewed_columns[skewed_columns > 1]}\n")
    
    print("- Consider scaling numerical features (e.g., Min-Max Scaling, Standardization).")
    print("- Review constant columns; these might not add value to the analysis or model.")
    print("- Review high cardinality columns for potential encoding or binning.")
    print("=============================================================================")

class DataFrameStatistics:
    """
    A class to perform various statistical operations on a Pandas DataFrame.
    
    Attributes:
        df (pd.DataFrame): The DataFrame on which statistical operations will be performed.
    
    Methods:
        missing_data_info() -> pd.DataFrame:
            Returns a DataFrame containing the count and percentage of missing data for each column.
        
        duplicate_info() -> pd.DataFrame:
            Returns the count and percentage of duplicate rows in the DataFrame.
        
        data_info() -> None:
            Prints the DataFrame information, including data types and non-null counts for columns.
        
        shape() -> tuple:
            Returns the shape (rows, columns) of the DataFrame.
        
        unique_values_info() -> pd.DataFrame:
            Returns the count and percentage of unique values for each column.
        
        describe() -> pd.DataFrame:
            Returns the descriptive statistics for the numerical columns in the DataFrame.
        
        statistics() -> None:
            Prints a summary of missing data, duplicate count, data info, shape, unique values, 
            and descriptive statistics.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataFrameStatistics class with the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df

    def missing_data_info(self) -> pd.DataFrame:
        """
        Calculates the count and percentage of missing values for each column in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the count and percentage of missing values for each column.
        """
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percentage
        })

    def duplicate_info(self) -> pd.DataFrame:
        """
        Calculates the count and percentage of duplicate rows in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the duplicate row count and percentage.
        """
        duplicate_rows = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(self.df)) * 100
        return pd.DataFrame({
            'Duplicate Count': [duplicate_rows],
            'Duplicate Percentage': [duplicate_percentage]
        })

    def data_info(self) -> None:
        """
        Prints the summary of the DataFrame, including the data types and the count of non-null values 
        for each column.
        """
        print(self.df.info())

    def shape(self) -> tuple:
        """
        Returns the shape of the DataFrame (number of rows and columns).

        Returns:
            tuple: A tuple representing the shape of the DataFrame (rows, columns).
        """
        return self.df.shape

    def unique_values_info(self) -> pd.DataFrame:
        """
        Calculates the count and percentage of unique values for each column in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the count and percentage of unique values for each column.
        """
        unique_values = self.df.nunique()
        unique_percentage = (unique_values / len(self.df)) * 100
        return pd.DataFrame({
            'Unique Count': unique_values,
            'Unique Percentage': unique_percentage
        }).sort_values(by='Unique Count')

    def describe(self) -> pd.DataFrame:
        """
        Returns the descriptive statistics for the numerical columns in the DataFrame.
        
        Descriptive statistics include count, mean, standard deviation, min, max, 25th, 50th, and 75th percentiles.

        Returns:
            pd.DataFrame: A DataFrame containing descriptive statistics for numerical columns.
        """
        return self.df.describe()

    def statistics(self) -> None:
        """
        Prints a comprehensive summary of the DataFrame's statistics, including missing data percentage,
        duplicate row count, data info, shape, unique values, and descriptive statistics.
        """
        print("=============================================================================")
        print("Missing Data Info (count and percentage):")
        print(self.missing_data_info())
        print("=============================================================================")
        print("\nDuplicate Row Info (count and percentage):")
        print(self.duplicate_info())
        print("=============================================================================")
        print("\nData Info:")
        self.data_info()  # This method prints the info directly
        print("=============================================================================")
        print("\nShape of DataFrame:")
        print(self.shape())
        print("=============================================================================")
        print("\nUnique Values Info (count and percentage):")
        print(self.unique_values_info())
        print("=============================================================================")
        print("\nDescriptive Statistics (for numerical columns):")
        print(self.describe())
        print("=============================================================================")

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
    """
    # Convert the data into a DataFrame (if it is not already a DataFrame)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    data.to_csv(filename, index=False)
    print(f"Data has been saved to {filename}")