import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data():
    """Load Iris dataset."""
    data = load_iris()
    return data.data, data.target


def train_model(X_train, y_train):
    """Train logistic regression model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    return acc


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, "iris_log_reg.joblib")


if __name__ == "__main__":
    main()
