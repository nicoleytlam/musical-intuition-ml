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
