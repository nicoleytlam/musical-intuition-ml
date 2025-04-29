from sklearn import metrics as skmetrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Evaluate model on train/val/test sets and print results.
    """
    print('***** Evaluation Results *****')

    # Train
    predictions_train = model.predict(x_train)
    score_mse_train = skmetrics.mean_squared_error(y_train, predictions_train)
    score_r2_train = skmetrics.r2_score(y_train, predictions_train)
    print('Training set mean squared error: {:.4f}'.format(score_mse_train))
    print('Training set r-squared score: {:.4f}'.format(score_r2_train))

    # Validation
    predictions_val = model.predict(x_val)
    score_mse_val = skmetrics.mean_squared_error(y_val, predictions_val)
    score_r2_val = skmetrics.r2_score(y_val, predictions_val)
    print('Validation set mean squared error: {:.4f}'.format(score_mse_val))
    print('Validation set r-squared score: {:.4f}'.format(score_r2_val))

    # Test
    predictions_test = model.predict(x_test)
    score_mse_test = skmetrics.mean_squared_error(y_test, predictions_test)
    score_r2_test = skmetrics.r2_score(y_test, predictions_test)
    print('Testing set mean squared error: {:.4f}'.format(score_mse_test))
    print('Testing set r-squared score: {:.4f}'.format(score_r2_test))

def print_coefficients(model):
    """
    Print coefficients of trained linear model.
    """
    print("\nLearned coefficients (weights for each step):")
    for i, coef in enumerate(model.coef_):
        print(f"Step {i+1}: {coef:.4f}")

def cross_validate_model(X, y, k=5, model_type="linear", alpha=1.0):
    """
    Perform K-Fold Cross-Validation and print results.
    
    Args:
        X (ndarray): Feature matrix
        y (ndarray): Target vector
        k (int): Number of folds
        model_type (str): 'linear' or 'ridge'
        alpha (float): Regularization strength for Ridge
    """
    print(f'\n***** {k}-Fold Cross-Validation ({model_type.title()}) Results *****')

    kf = KFold(n_splits=k, shuffle=True, random_state=1)

    mse_scores = []
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train_cv, x_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        if model_type == "linear":
            model_cv = LinearRegression(fit_intercept=False)
        elif model_type == "ridge":
            model_cv = Ridge(alpha=alpha, fit_intercept=False)
        elif model_type == "lasso":
            model_cv = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_cv.fit(x_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(x_test_cv)

        mse = skmetrics.mean_squared_error(y_test_cv, y_pred_cv)
        r2 = skmetrics.r2_score(y_test_cv, y_pred_cv)

        print(f"Fold {fold + 1}: MSE = {mse:.4f}, R² = {r2:.4f}")

        mse_scores.append(mse)
        r2_scores.append(r2)

    print("\n***** Average over {} folds *****".format(k))
    print("Average MSE: {:.4f}".format(np.mean(mse_scores)))
    print("Average R²: {:.4f}".format(np.mean(r2_scores)))

def bootstrap_prediction_intervals(
    model_type, X, y_true, n_bootstraps=1000, alpha=0.05, model_kwargs=None, plot=True
):
    """
    Bootstrap 95% prediction intervals and evaluate whether the true target falls within each.

    Args:
        model_type (str): 'linear', 'ridge', or 'lasso'
        X (ndarray): Input features
        y_true (ndarray): True targets
        n_bootstraps (int): Number of bootstrap iterations
        alpha (float): Significance level for CI (e.g., 0.05 for 95%)
        model_kwargs (dict): Additional args to pass to model (e.g., alpha for Ridge)
        plot (bool): Whether to plot predictions with intervals

    Returns:
        coverage (float): Proportion of points where true target is within CI
    """
    model_kwargs = model_kwargs or {}
    
    model_map = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from 'linear', 'ridge', 'lasso'.")

    all_preds = np.zeros((n_bootstraps, len(y_true)))

    for i in range(n_bootstraps):
        X_boot, y_boot = resample(X, y_true, replace=True)
        model = model_map[model_type](fit_intercept=False, **model_kwargs)
        model.fit(X_boot, y_boot)
        all_preds[i] = model.predict(X)

    lower_bounds = np.percentile(all_preds, 100 * (alpha / 2), axis=0)
    upper_bounds = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)
    mean_preds = np.mean(all_preds, axis=0)

    inside_ci = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    coverage = np.mean(inside_ci)

    # Print evaluation
    print("=== Bootstrap CI Evaluation ===")
    for i, (pred, low, high, actual, within) in enumerate(
        zip(mean_preds, lower_bounds, upper_bounds, y_true, inside_ci)
    ):
        status = "Correct" if within else "Missed"
        print(
            f"Point {i+1}: Pred = {pred:.2f}, CI = [{low:.2f}, {high:.2f}], True = {actual:.2f} → {status}"
        )

    # Plot
    if plot:
        plt.figure(figsize=(12, 5))
        x_axis = np.arange(len(y_true))
        plt.plot(x_axis, y_true, "o", label="True", color="black")
        plt.plot(x_axis, mean_preds, "r-", label="Prediction")
        plt.fill_between(x_axis, lower_bounds, upper_bounds, color="gray", alpha=0.3, label="95% CI")
        plt.title(f"{model_type.title()} Model: Predictions with 95% Confidence Intervals")
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"\nOverall coverage: {coverage:.2%} within 95% CI")
    return coverage

def kfold_bootstrap_ci_analysis(
    model_type, X, y, k=5, n_bootstraps=1000, alpha=0.05, model_kwargs=None, plot=True
):
    """
    Perform bootstrap CI estimation in K-fold CV so every data point is evaluated.

    Args:
        model_type (str): 'linear', 'ridge', or 'lasso'
        X (ndarray): Features
        y (ndarray): Targets
        k (int): Number of CV folds
        n_bootstraps (int): Number of bootstrap iterations per fold
        alpha (float): Significance level
        model_kwargs (dict): Extra args for model (e.g. Ridge alpha)
        plot (bool): Whether to plot final results

    Returns:
        DataFrame with true values, predictions, CI bounds, and coverage info
    """
    import pandas as pd
    from sklearn.model_selection import KFold

    model_kwargs = model_kwargs or {}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Run bootstrap CI on this fold's test set
        all_preds = np.zeros((n_bootstraps, len(y_test)))

        for b in range(n_bootstraps):
            X_boot, y_boot = resample(X_train, y_train, replace=True)
            model = {
                "linear": LinearRegression,
                "ridge": Ridge,
                "lasso": Lasso
            }[model_type](fit_intercept=False, **model_kwargs)

            model.fit(X_boot, y_boot)
            all_preds[b] = model.predict(X_test)

        lower_bounds = np.percentile(all_preds, 100 * (alpha / 2), axis=0)
        upper_bounds = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)
        mean_preds = np.mean(all_preds, axis=0)
        in_ci = (y_test >= lower_bounds) & (y_test <= upper_bounds)

        # Store results
        fold_df = pd.DataFrame({
            "fold": fold + 1,
            "true_y": y_test,
            "pred_mean": mean_preds,
            "ci_lower": lower_bounds,
            "ci_upper": upper_bounds,
            "in_ci": in_ci
        })
        all_results.append(fold_df)

    results_df = pd.concat(all_results, ignore_index=True)

    # Print summary
    coverage = results_df["in_ci"].mean()
    print(f"\nOverall coverage across all folds: {coverage:.2%} within {100*(1-alpha):.0f}% CI")

    # Plot all
    if plot:
        plt.figure(figsize=(12, 5))
        x_axis = np.arange(len(results_df))
        plt.plot(x_axis, results_df["true_y"], "o", label="True", color="black")
        plt.plot(x_axis, results_df["pred_mean"], "r-", label="Prediction")
        plt.fill_between(x_axis, results_df["ci_lower"], results_df["ci_upper"], color="gray", alpha=0.3, label="95% CI")
        plt.title(f"K-Fold Bootstrap CI: {model_type.title()} Model")
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results_df
