import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings

warnings.filterwarnings("ignore")
np.random.seed(1)

def train_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_ridge(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def train_lasso(X, y, alpha=1.0):
    """
    Train a Lasso regression model.
    """
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(X, y)
    return model