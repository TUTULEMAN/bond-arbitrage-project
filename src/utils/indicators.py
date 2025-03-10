"""
Utility functions for calculating technical indicators and risk metrics used in the strategy.
Includes functions for:
  - Simple and exponential moving averages
  - Rolling standard deviation
  - Standard and Bayesian z-score calculations
  - Yield spread calculation (if needed)
"""

import numpy as np
import pandas as pd

def moving_average(series, window):
    """
    Calculate the simple moving average of a series.

    Parameters:
        series (pd.Series or np.array): Input data.
        window (int): Window size for the moving average.

    Returns:
        pd.Series: Simple moving average.
    """
    return pd.Series(series).rolling(window=window).mean()

def exponential_moving_average(series, window):
    """
    Calculate the exponential moving average (EMA) of a series.

    Parameters:
        series (pd.Series or np.array): Input data.
        window (int): Window size for the EMA.

    Returns:
        pd.Series: Exponential moving average.
    """
    return pd.Series(series).ewm(span=window, adjust=False).mean()

def rolling_std(series, window):
    """
    Calculate the rolling standard deviation of a series.

    Parameters:
        series (pd.Series or np.array): Input data.
        window (int): Window size for the rolling standard deviation.

    Returns:
        pd.Series: Rolling standard deviation.
    """
    return pd.Series(series).rolling(window=window).std()

def compute_z_score(value, mean, std):
    """
    Compute the standard z-score for a given value.

    Parameters:
        value (float): Current value.
        mean (float): Mean value.
        std (float): Standard deviation.

    Returns:
        float: The z-score. Returns 0 if std is 0.
    """
    if std == 0:
        return 0.0
    return (value - mean) / std

def compute_bayesian_z_score(window_data, current_value, prior_mean=0.0, prior_variance=1.0, observation_variance=0.01):
    """
    Compute a Bayesian z-score for the current value using a rolling window of data.
    This updates the prior with observed data and calculates the z-score based on the posterior.

    Parameters:
        window_data (np.array or pd.Series): Recent observations.
        current_value (float): Current observed value.
        prior_mean (float): Prior mean for Bayesian update.
        prior_variance (float): Prior variance (tau^2).
        observation_variance (float): Known variance of observations (sigma^2).

    Returns:
        tuple: (posterior_mean, posterior_std, z_score)
    """
    n = len(window_data)
    if n == 0:
        raise ValueError("window_data must contain at least one element")

    sample_mean = np.mean(window_data)

    # Compute posterior variance and mean using Normal-Normal conjugacy
    posterior_variance = 1.0 / (n / observation_variance + 1.0 / prior_variance)
    posterior_mean = posterior_variance * (n * sample_mean / observation_variance + prior_mean / prior_variance)
    posterior_std = np.sqrt(posterior_variance)
    z_score = compute_z_score(current_value, posterior_mean, posterior_std)
    
    return posterior_mean, posterior_std, z_score

def compute_yield_spread(yield1, yield2):
    """
    Compute the yield spread between two yields.
    
    Parameters:
        yield1 (float): Yield of the first bond.
        yield2 (float): Yield of the second bond.
    
    Returns:
        float: The yield spread.
    """
    return yield1 - yield2

if __name__ == "__main__":
    # Quick demo of the indicators functions
    import matplotlib.pyplot as plt

    # Generate sample data: a random walk for demonstration
    np.random.seed(42)
    data = np.cumsum(np.random.normal(0, 1, 100))

    # Compute indicators
    ma = moving_average(data, window=10)
    ema = exponential_moving_average(data, window=10)
    std_dev = rolling_std(data, window=10)

    # Compute Bayesian z-score for the last value using the previous 10 observations
    try:
        posterior_mean, posterior_std, bayes_z = compute_bayesian_z_score(data[-10:], data[-1])
        print("Bayesian update:")
        print("Posterior Mean:", posterior_mean)
        print("Posterior Std:", posterior_std)
        print("Z-Score:", bayes_z)
    except ValueError as e:
        print(e)
    
    # Plot the indicators
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Data", lw=2)
    plt.plot(ma, label="Moving Average (10)", lw=2)
    plt.plot(ema, label="Exponential MA (10)", lw=2)
    plt.fill_between(range(len(std_dev)), ma - std_dev, ma + std_dev, color='gray', alpha=0.2, label="Rolling Std Dev")
    plt.legend()
    plt.title("Indicators Demo")
    plt.show()
