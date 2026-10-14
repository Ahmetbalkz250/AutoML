import pandas as pd
import os

def load_and_prepare_data(file_path, target_column):

    """
    Reads the CSV or Excel file chosen by the user, splits the target column, and returns X and y ready for modeling.
    """
    # 1. Fetches the file extension (.csv or .xlsx) 
    _, file_extension = os.path.splitext(file_path)
    
    try:
        # 2. Read the file with pandas according to its formats
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            # If it is neither CSV nor Excel, send 'False' and an error message
            return False, "Error: Only files with .csv or .xlsx extensions are supported!"
        
        # 3. Does the target column entered by the user in the interface actually exist in the table?
        if target_column not in df.columns:
            return False, f"Error: 'A column named {target_column}' was not found!"
        
        # 4. MVP RULES
        # If NaN cells is exist in table, we are deleteing the those rows from the table for the model to not crash.
        df = df.dropna()
        
        # We only collect numerical data. (We are excluding text/strings for now so the system doesn't crash.)
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
        
        if target_column not in df_numeric.columns:
            return False, "Error: The target column must be numeric!"
            
        # 5. Splitting Process: X (Variables) and y (Target)
        y = df_numeric[target_column]
        X = df_numeric.drop(columns=[target_column])
        
        # If the operation is successful, return True and the (X, y) tuple.
        return True, (X, y)
        
    except Exception as e:
        # To prevent the program from crashing in the event of an unexpected error (e.g., a corrupted file)
        return False, f"An error occurred while reading the file: {str(e)}"


def get_column_names(file_path):
    """
    Reads only the headers (column names) of the file.
    Since it doesn't load the entire data, it runs instantly.
    """
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == '.csv':
            # Reads only the 0th row (headers), works at light speed even on massive files
            df = pd.read_csv(file_path, nrows=0) 
        elif file_extension.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, nrows=0)
        else:
            return False, []
            
        return True, list(df.columns)
    except Exception:
        return False, []