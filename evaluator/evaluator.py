# models/evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import torch
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, y_true):
        """
        Initialize evaluator with ground truth values (original).
        
        Args:
            y_true (array-like): True target values.
        """
        self.y_true = np.array(y_true).reshape(-1, 1)

    def _to_numpy(self, preds):
        """Convert predictions to numpy regardless of framework."""
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        elif isinstance(preds, tf.Tensor):
            preds = preds.numpy()
        else:
            preds = np.array(preds)
        return preds.reshape(-1, 1)

    def evaluate(self, model, X, model_name="model", scaler=None):
        """
        Evaluate a model and return a DataFrame with predictions and metrics.

        Args:
            model: sklearn, PyTorch, or TensorFlow model.
            X: Input features (numpy, pandas, tensor).
            model_name (str): Name of the model.
            scaler: Optional scaler with inverse_transform (e.g., StandardScaler, MinMaxScaler).

        Returns:
            pd.DataFrame: y_true, y_pred, and metrics.
        """
        # --- Get predictions ---
        if hasattr(model, "predict"):  # sklearn or tf.keras.Model
            preds = model.predict(X)
        elif isinstance(model, torch.nn.Module):  # PyTorch
            model.eval()
            with torch.no_grad():
                preds = model(X)
        else:
            raise TypeError("Unsupported model type")

        y_pred = self._to_numpy(preds)

        # --- Inverse transform if scaler is provided ---
        if scaler is not None:
            y_true_scaled = scaler.fit_transform(self.y_true.values.reshape(-1, 1))
            y_pred = scaler.inverse_transform(y_pred).ravel()
        else:
            y_pred = y_pred.ravel()

        # --- Metrics ---
        r2 = r2_score(self.y_true, y_pred)
        mse = mean_squared_error(self.y_true, y_pred) if scaler is not None else mean_squared_error(y_true_scaled, y_pred)
        mape = mean_absolute_percentage_error(self.y_true, y_pred)

        # --- Build DataFrame ---
        results = pd.DataFrame({
            "y_true": self.y_true,
            "y_pred": y_pred
        })
        results["model"] = model_name
        results["R2"] = r2
        results["MSE"] = mse
        results["MAPE"] = mape

        return results
