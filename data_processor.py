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
        target_column = str(target_column).strip()
        df.columns = df.columns.astype(str).str.strip()
        # 3. Does the target column entered by the user in the interface actually exist in the table?
        if target_column not in df.columns:
            return False, f"Error: 'A column named {target_column}' was not found!"
        
        df = df.dropna()

        y = df[target_column]
        X_numeric = df.drop(columns= [target_column])

        X = X_numeric.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

        if X.empty:
            return False, "Error: Feature columns must be numeric!"
            
        return True, (X, y)
        
    except Exception as e:
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