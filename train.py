import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    """
    Train a simple RandomForest model on the Iris dataset
    and save it to the 'model/' directory.
    """
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    print("✅ Model trained successfully!")

    # Ensure model directory exists
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"📁 Created directory: {model_dir}")

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved at {model_path}")

if __name__ == "__main__":
    print("🚀 Starting training process...")
    train_and_save_model()
    print("🎉 Training complete! Model ready for Docker build.")

