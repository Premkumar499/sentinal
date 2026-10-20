import xgboost as xgb
import numpy as np
import yaml
from pathlib import Path

class XGBoostBaseline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the XGBoost model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels using a specified threshold."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(np.int32)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class (change)."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, path: str):
        """Save the XGBoost model to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p))

    def load_model(self, path: str):
        """Load the XGBoost model from a JSON file."""
        self.model.load_model(path)
        self.is_trained = True
