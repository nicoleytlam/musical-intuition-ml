from sklearn.linear_model import LinearRegression, Ridge

def train_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_ridge(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model