import numpy as np

def split_data(X, y, train_size=0.8, val_size=0.1):
    """
    Shuffle and split X and y into train/val/test sets.
    
    Args:
        X (ndarray): Features
        y (ndarray): Targets
        train_size (float): Proportion for training set
        val_size (float): Proportion for validation set

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    shuffled_indices = np.random.permutation(X.shape[0])

    train_split_idx = int(train_size * X.shape[0])
    val_split_idx = int((train_size + val_size) * X.shape[0])

    train_indices = shuffled_indices[0:train_split_idx]
    val_indices = shuffled_indices[train_split_idx:val_split_idx]
    test_indices = shuffled_indices[val_split_idx:]

    x_train, y_train = X[train_indices], y[train_indices]
    x_val, y_val = X[val_indices], y[val_indices]
    x_test, y_test = X[test_indices], y[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test