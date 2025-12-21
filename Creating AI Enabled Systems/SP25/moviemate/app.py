# app.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(file_path, separation_type=None):
    """
    Load the dataset using pandas and inspect its structure.
    
    :param file_path: Path to the data file.
    :param separation_type: 'tab' or '\\t' for tab-separated, 'bar' or '|' for pipe-separated.
    :return: pandas.DataFrame of the dataset.
    """
    # Determine separator
    if separation_type in ['tab', '\t']:
        sep = '\t'
    elif separation_type in ['bar', '|']:
        sep = '|'
    else:
        sep = None  # let pandas infer or default to comma

    # Load
    df = pd.read_csv(file_path, sep=sep, header=None)
    
    # Quick inspection
    print("First five rows:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())
    return df


def partition_data(ratings_df, split: float = 0.8, partition_type='stratified', stratify_by='user'):
    """
    Split the data into training and testing sets using:
    - stratified sampling (by user),
    - temporal sampling (by timestamp).

    :param ratings_df: DataFrame containing at least ['user', 'item', 'rating', 'timestamp'].
    :param split: Fraction of data to use for training (e.g., 0.8 → 80% train, 20% test).
    :param partition_type: 'stratified' or 'temporal'.
    :param stratify_by: Column to stratify on for stratified sampling (default 'user').
    :return: (train_df, test_df)
    """
    if partition_type == 'stratified':
        train_df, test_df = train_test_split(
            ratings_df,
            test_size=1 - split,
            random_state=42,
            stratify=ratings_df[stratify_by]
        )
    elif partition_type == 'temporal':
        df_sorted = ratings_df.sort_values('timestamp')
        cutoff = int(len(df_sorted) * split)
        train_df = df_sorted.iloc[:cutoff]
        test_df = df_sorted.iloc[cutoff:]
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")
    
    return train_df, test_df
