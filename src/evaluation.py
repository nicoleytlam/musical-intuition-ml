from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "r2": r2}

def confidence_interval(preds, std=1.96):
    return preds - std, preds + std
