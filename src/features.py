def get_features_and_target(df, feature_steps=range(1, 7), target_col='step7'):
    """
    Extract features (steps 1-6) and target (step 7) from the df.

    Args:
        df (DataFrame): DataFrame after loading (full or filtered).

    Returns:
        X (ndarray): Features matrix.
        y (ndarray): Target vector.
    """
    step_cols = [f'step{i}' for i in feature_steps]
    X = df[step_cols].values
    y = df[target_col].values
    return X, y