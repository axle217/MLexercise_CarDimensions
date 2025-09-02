import pandas as pd

def compare_predictions(y_true, y_pred):
    """
    Create a DataFrame comparing actual vs. predicted values,
    including error and percentage error.

    Parameters
    ----------
    y_true : array-like
        Ground truth (actual values).
    y_pred : array-like
        Model predictions (can be 1D or 2D from TF/PyTorch/sklearn).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: Actual, Predicted, Error, Error %.
    """
    # Ensure predictions are 1D
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred

    # Build comparison DataFrame
    comparison = pd.DataFrame({
        "Actual": pd.Series(y_true).reset_index(drop=True),
        "Predicted": pd.Series(y_pred_flat).reset_index(drop=True)
    })

    # Calculate error and percentage error
    comparison["Error"] = comparison["Predicted"] - comparison["Actual"]
    comparison["Error %"] = comparison["Error"].abs() / comparison["Actual"] * 100

    return comparison
