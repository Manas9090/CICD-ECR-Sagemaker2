# train.py
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    print("✅ Model trained successfully!")

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "model.pkl")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved at {model_path}")

if __name__ == "__main__":
    print("🚀 Starting training process...")
    train_and_save_model()
    print("🎉 Training complete! Model ready for Docker build.")
