from typing import Tuple
import math
import pandas as pd

def get_sub_dataset_size(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Determines the size of the sub-dataset according to the number of columns present in the DataFrame. 
    Different rules are applied for datasets with less or equal to 5 columns and more than 5 columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame for which sub-dataset size needs to be determined.
        
    Returns:
        subset_rows (int): The number of rows for the sub-dataset.
        subset_cols (int): The number of columns for the sub-dataset.
    """
    _, num_cols = df.shape
    if num_cols<=5:
        return less_5_columns_subdataset_size(df)
    return get_reuglar_subdataset_size(df)


def get_reuglar_subdataset_size(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Returns the size of the sub-dataset when the number of columns in the DataFrame is more than 5.
    The size is calculated as the square root of the number of rows (for rows) and 
    a quarter of the number of columns (for columns).

    Parameters:
        df (pd.DataFrame): The input DataFrame for which sub-dataset size needs to be determined.
        
    Returns:
        subset_rows (int): The number of rows for the sub-dataset.
        subset_cols (int): The number of columns for the sub-dataset.
    """
    num_rows, num_cols = df.shape
    subset_rows = int(math.sqrt(num_rows))
    subset_cols =  int(0.75 * num_cols)
    
    return subset_rows, subset_cols

def less_5_columns_subdataset_size(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Returns the size of the sub-dataset when the number of columns in the DataFrame is more than 5.
    The size is calculated as the square root of the number of rows (for rows) and 
    a quarter of the number of columns (for columns).

    Parameters:
        df (pd.DataFrame): The input DataFrame for which sub-dataset size needs to be determined.
        
    Returns:
        subset_rows (int): The number of rows for the sub-dataset.
        subset_cols (int): The number of columns for the sub-dataset.
    """
    num_rows, num_cols = df.shape
    subset_rows = int(math.sqrt(num_rows))
    subset_cols = round(0.75 * num_cols)
    return subset_rows, subset_cols