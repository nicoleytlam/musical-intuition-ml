from src.data_loader import load_data
from src.features import add_features
from src.models import train_linear, train_ridge
from src.evaluation import evaluate
from sklearn.model_selection import train_test_split

# Load and prepare data
df = load_data()
df = add_features(df)

X = df.drop(columns='target_note')  # Adjust as necessary
y = df['target_note']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate linear model
model = train_linear(X_train, y_train)
results = evaluate(model, X_test, y_test)

print("Linear Regression Evaluation:", results)
