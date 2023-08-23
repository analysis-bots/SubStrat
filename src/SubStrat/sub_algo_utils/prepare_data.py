import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype



def prepare_dataset(data: pd.DataFrame, target_col: str, hard_mode=False) -> pd.DataFrame:
    """
    Prepare the input DataFrame by performing the following steps:
    
    1. Drop rows containing NaN values if any column has data type 'object'.
    2. For columns of type 'object':
       a. If the column has only two unique values (binary), it's label-encoded.
       b. If the column has more than two unique values, it's one-hot encoded. The first 
          dummy column is dropped to avoid multicollinearity.
    3. Convert boolean columns to integer type (True becomes 1, False becomes 0).

    Note: This function returns a modified version of the DataFrame. The original DataFrame 
    remains unchanged.

    Parameters:
    - data (pd.DataFrame): Input DataFrame to be processed.
    
    Returns:
    - pd.DataFrame: The processed DataFrame.
    
    Example:
    >>> df = pd.DataFrame({
    ...     'gender': ['male', 'female', 'male', 'female'],
    ...     'likes_chocolate': [True, False, True, True],
    ...     'city': ['NY', 'LA', 'NY', 'SF']
    ... })
    >>> df_processed = prepare_dataset(df)
    >>> print(df_processed)
       gender  likes_chocolate  city_LA  city_NY  city_SF
    0       1                1        0        1        0
    1       0                0        1        0        0
    2       1                1        0        1        0
    3       0                1        0        0        1

    """
    if (data.dtypes == 'object').any():
        pd.set_option('mode.chained_assignment',None)
        if hard_mode:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            data = data.dropna()
        else:
            data = data.dropna()
        target_data = data[target_col].copy()
        data = data.drop(columns=target_col)

        for col in data.columns:
            if data[col].dtype == 'object':
                if len(data[col].unique()) <= 2:  # Label encode binary columns
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                else:  # One-hot encode non-binary columns
                    data = pd.get_dummies(data, columns=[col], drop_first=True)
            
        for col in data.columns:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype(int)
        if target_data.dtype == 'object':
            le = LabelEncoder()
            target_data = le.fit_transform(target_data)
        data[target_col] = target_data
        pd.set_option('mode.chained_assignment',"warn")
    return data

# def soft_prepare_dataset(data: pd.DataFrame, target_col: str, hard_mode=False):
#     # remove what we do not need
#     pd.set_option('mode.chained_assignment',None)
#     if hard_mode:
#             numeric_cols = data.select_dtypes(include=[np.number]).columns
#             data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
#     data.dropna(inplace=True)      
#     target_data = data[target_col].copy()
#     data = data.drop(columns=target_col)
#     data.drop([col for col in list(data) if not is_numeric_dtype(data[col])], axis=1, inplace=True)
#     # remove _rows with nan
#     if target_data.dtype == 'object':
#             le = LabelEncoder()
#             target_data = le.fit_transform(target_data)
#     data[target_col] = target_data
#     pd.set_option('mode.chained_assignment',"warn")
#     return data

def soft_prepare_dataset(data: pd.DataFrame, target_col, hard_mode=False):
    if hard_mode:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    data = data.dropna(axis = 0, how = 'any')
    cat_columns = data.select_dtypes(include = ['object'])
    if len(cat_columns.columns) > 0:
        le = LabelEncoder()
        cat_encoded = cat_columns.apply(le.fit_transform)
        cat_labels = cat_encoded.columns
        data = data.drop(columns = cat_labels, axis = 1)   
        data = pd.concat([data, cat_encoded], axis = 1)  
    return data   