import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch

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

class OneHotMLP(nn.Module):
    def __init__(self, sequence_len, num_pitches, hidden_dim=128, output_dim=88):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(sequence_len * num_pitches, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def simple_train_fn(model, X, y, criterion, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

class PolyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=88):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DeepMLP(nn.Module):
    def __init__(self, sequence_len, num_pitches, hidden_dims=[256, 128, 64], output_dim=88):
        super().__init__()
        self.flatten = nn.Flatten()
        dims = [sequence_len * num_pitches] + hidden_dims
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)
        ])
        self.out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

