import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_actual_vs_fitted(y_pred: pd.Series, y_test: pd.Series) -> None:
    """Plots the actual vs fitted values with the `y=x` line as a reference for perfect predictions.

    Args:
        y_pred (pd.Series): A Pandas Series of the predicted values.
        y_test (pd.Series): A Pandas Series of the actual values.
    """
    sns.scatterplot(x=y_pred, y=y_test)
    plt.title(f"Predicted vs Actual Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    x = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
    plt.plot(x, x, c="red")
    plt.show()