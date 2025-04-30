from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
import pandas as pd
import numpy as np

def get_onehot_midi_sequences(csv_path, feature_steps=range(1, 7), target_col=None):
    """
    Converts CSV MIDI step data into one-hot vectors using a fixed range
    from min to max MIDI pitch in the dataset.

    Returns:
    - X: shape (num_samples, sequence_length, num_pitches)
    - y: shape (num_samples,) or None
    - midi_range: (min_midi, max_midi)
    """
    df = pd.read_csv(csv_path)
    feature_cols = [f'step{i}' for i in feature_steps]

    X_raw = df[feature_cols].values

    # Determine full MIDI range
    min_midi = X_raw.min()
    max_midi = X_raw.max()
    num_pitches = max_midi - min_midi + 1

    # Convert each MIDI pitch to one-hot index
    num_samples = X_raw.shape[0]
    sequence_length = len(feature_cols)
    X = np.zeros((num_samples, sequence_length, num_pitches), dtype=np.float32)

    for i, row in enumerate(X_raw):
        for j, midi_note in enumerate(row):
            if pd.notna(midi_note):  # Skip NaNs
                index = int(midi_note) - min_midi
                X[i, j, index] = 1.0

    # Optional target
    if target_col:
        y = df[target_col].astype(int).values - min_midi  # match output to same index scale
    else:
        y = None

    return X, y, (min_midi, max_midi)


def get_features_and_target(df, feature_steps=range(1, 7), extra_feature_cols=None, target_col='step7'):
    """
    Extracts features and target from a DataFrame of melodic sequences.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing step columns and optional contextual features
        feature_steps (iterable): Sequence of integers indicating which steps to include (e.g., range(1, 7))
        extra_feature_cols (list of str): Optional list of additional column names to include
        target_col (str): Name of the target column
    
    Returns:
        X (np.ndarray): Features array
        y (np.ndarray): Target array
    """
    feature_cols = [f'step{i}' for i in feature_steps]
    
    if extra_feature_cols:
        feature_cols += extra_feature_cols

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y

def apply_polynomial_expansion(X, degree=2):
    """
    Flattens and applies polynomial expansion.
    Args:
        X (np.array): shape (n_samples, sequence_len, num_pitches)
        degree (int): polynomial degree
    Returns:
        X_poly: expanded features
    """
    X_flat = X.reshape(X.shape[0], -1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_flat)
    return X_poly

def apply_rbf_kernel_approximation(X, gamma=0.1, n_components=500):
    """
    Applies random Fourier feature approximation to simulate RBF kernel.
    """
    X_flat = X.reshape(X.shape[0], -1)
    rbf = RBFSampler(gamma=gamma, n_components=n_components)
    X_rbf = rbf.fit_transform(X_flat)
    return X_rbf