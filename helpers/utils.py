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
    y_true_flat = y_true.flatten() if hasattr(y_true, "flatten") else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred

    # Build comparison DataFrame
    comparison = pd.DataFrame({
        "Actual": pd.Series(y_true_flat).reset_index(drop=True),
        "Predicted": pd.Series(y_pred_flat).reset_index(drop=True)
    })

    # Calculate error and percentage error
    comparison["Error"] = comparison["Predicted"] - comparison["Actual"]
    comparison["Error %"] = comparison["Error"].abs() / comparison["Actual"] * 100

    return comparison




import matplotlib.pyplot as plt

def plot_scatter_comparison(
    dfs, labels, 
    x_col="", y_col="", 
    xlabel="", ylabel="", 
    title="Scatter Plot",
    xrange=None, yrange=None, drawline=False
):
    """
    Plots scatter plots for multiple DataFrames on the same figure.
    
    Parameters:
        dfs (list): List of pandas DataFrames to plot.
        labels (list): List of labels for each DataFrame.
        x_col (str): Column name for x-axis values. Default is "Actual".
        y_col (str): Column name for y-axis values. Default is "Predicted".
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        xrange (tuple): (min, max) for x-axis. Default None (auto).
        yrange (tuple): (min, max) for y-axis. Default None (auto).
        drawline (bool): Draw y=x.
    """
    plt.figure(figsize=(8, 6))
    
    markers = ["o", "v", "s", "x", "D", "^", "<", ">"]  
    colors = ["red", "black", "blue", "green", "purple", "orange"]
    
    for i, df in enumerate(dfs):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        plt.scatter(df[x_col], df[y_col], 
                    label=labels[i], 
                    marker=marker, 
                    facecolors="none", 
                    edgecolors=color, 
                    s=80)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if xrange is not None:
        plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)
    
    if drawline: 
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        line_min = max(xmin, ymin)
        line_max = min(xmax, ymax)
        plt.plot([line_min, line_max], [line_min, line_max],
             'g--', label="y = x", zorder=4) # r=red, g=green -- dashed line
    
    plt.legend(facecolor='white', edgecolor='black', framealpha=1.0)
    plt.grid(True, zorder=0)
    plt.show()
