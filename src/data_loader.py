import pandas as pd

def load_full_data(filepath):
    """
    Load the full dataset without filtering.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        df (DataFrame): Full dataset.
    """
    df = pd.read_csv(filepath)
    return df

def load_filtered_data(filepath, prefix="HC"):
    """
    Load the dataset filtered by stem prefix (e.g., 'HC', 'LC').

    Args:
        filepath (str): Path to the CSV file.
        prefix (str): Prefix to filter stems on (default 'HC').

    Returns:
        df (DataFrame): Filtered dataset.

    Note:
        HC stands for authentic cadence and LC stands for non-authentic cadence.
    """
    df = pd.read_csv(filepath)
    df = df[df['stem'].str.startswith(prefix)]
    return df