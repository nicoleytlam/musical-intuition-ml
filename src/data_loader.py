import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

class MelodyOneHotDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def get_dataloader(X, y=None, batch_size=32, shuffle=True):
    dataset = MelodyOneHotDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)