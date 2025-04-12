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
    3. Logs the initial dimensions (rows and columns) of the dataset.
    4. Logs the updated column names after loading the data.
    5. Returns the DataFrame containing the loaded data.

    Parameters:
    - file_path (str): The path to the CSV file to be loaded.

    Returns:
    - df (pandas.DataFrame): The loaded dataset as a DataFrame.

    Exceptions:
    - FileNotFoundError: If the file cannot be found at the provided path.
    - pd.errors.ParserError: If the CSV file cannot be parsed due to format issues.
    - Any other errors during loading or processing the data will be logged.
    """
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Store initial dimensions
        initial_number_of_rows = df.shape[0]
        initial_number_of_columns = df.shape[1]

        # Display updated column names
        logging.info(f"Updated column names: {df.columns}")
        
        # Confirm successful data loading
        logging.info(f"Dataset loaded successfully with {initial_number_of_rows} rows and {initial_number_of_columns} columns.")
        
        return df
    
    # Handle the case where the file is not found
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found. Please upload the file or provide the correct path.")
    
    # Handle errors related to CSV parsing issues (e.g., incorrect formatting)
    except pd.errors.ParserError:
        logging.error(f"Error: Unable to parse the CSV file '{file_path}'. Please check the file format.")
    
    # Catch any other unexpected errors and display the error message
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

class DataFrameStatistics:
    """
    A class for performing various statistical operations on a Pandas DataFrame.
    
    Attributes:
        df (pd.DataFrame): The input DataFrame for analysis.
    
    Methods:
        missing_data_percentage() -> pd.Series:
            Returns the percentage of missing data for each column in the DataFrame.
        
        duplicate_count() -> int:
            Returns the count of duplicate rows in the DataFrame.
        
        data_info() -> None:
            Prints the DataFrame info, such as the data types and non-null counts of columns.
        
        shape() -> tuple:
            Returns the shape (rows, columns) of the DataFrame.
        
        unique_values() -> pd.Series:
            Returns the count of unique values for each column, sorted in ascending order.
        
        statistics() -> None:
            Prints a summary of missing data percentage, duplicate count, data info, shape, and unique values.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def missing_data_percentage(self):
        """Returns the percentage of missing data for each column"""
        return self.df.isnull().sum() / len(self.df) * 100

    def duplicate_count(self):
        """Returns the count of duplicate rows"""
        return self.df.duplicated().sum()

    def data_info(self):
        """Returns the DataFrame info"""
        return self.df.info()

    def shape(self):
        """Returns the shape of the DataFrame"""
        return self.df.shape

    def unique_values(self):
        """Returns the unique values count in sorted order"""
        return self.df.nunique().sort_values()

    def statistics(self):
        """Returns all statistics as separate values"""
        print("Missing Data Percentage:")
        print(self.missing_data_percentage())
        print("\nDuplicate Count:")
        print(self.duplicate_count())
        print("\nData Info:")
        self.data_info()  # This method prints the info directly, no return value
        print("\nShape:")
        print(self.shape())
        print("\nUnique Values Count:")
        print(self.unique_values())

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

def identify_outliers(df: pd.DataFrame):
    """
    Identifies outliers in all numerical columns of the DataFrame using the IQR method.
    Also prints the number of outliers in each column.
    
    Args:
        df (pd.DataFrame): The DataFrame for which outliers need to be identified.
    
    Returns:
        None: The function prints out the number of outliers in each column.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns  # Select numeric columns
    
    if len(numerical_cols) == 0:
        print("No numeric columns found in the DataFrame.")
        return

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

        # Print the number of outliers in the column
        print(f"Outliers in '{col}': {len(outliers)} rows")

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

