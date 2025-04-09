"""
Evaluation metrics for benchmarking model performance.
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf


def mean_square_error(real_data, generated_data):
    """
    Calculate Mean Square Error between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data with shape (n_samples, ...)
    generated_data : ndarray
        Generated data with shape (n_samples, ...)
        
    Returns:
    --------
    float
        Mean Square Error
    """
    # Ensure data has the same shape
    if real_data.shape != generated_data.shape:
        raise ValueError(f"Data shapes do not match: {real_data.shape} vs {generated_data.shape}")
    
    return np.mean((real_data - generated_data) ** 2)


def mean_absolute_error(real_data, generated_data):
    """
    Calculate Mean Absolute Error between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data with shape (n_samples, ...)
    generated_data : ndarray
        Generated data with shape (n_samples, ...)
        
    Returns:
    --------
    float
        Mean Absolute Error
    """
    # Ensure data has the same shape
    if real_data.shape != generated_data.shape:
        raise ValueError(f"Data shapes do not match: {real_data.shape} vs {generated_data.shape}")
    
    return np.mean(np.abs(real_data - generated_data))


def wasserstein_metric(real_data, generated_data, axis=0):
    """
    Calculate Wasserstein distance (Earth Mover's Distance) between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data
    generated_data : ndarray
        Generated data
    axis : int
        Axis along which to compute the distance
        
    Returns:
    --------
    float or ndarray
        Wasserstein distance(s)
    """
    if axis == 0:
        # Compute distance across the entire dataset
        return wasserstein_distance(real_data.flatten(), generated_data.flatten())
    else:
        # Compute distance for each feature
        distances = []
        for i in range(real_data.shape[1]):
            distances.append(wasserstein_distance(real_data[:, i], generated_data[:, i]))
        return np.array(distances)


def kl_divergence(real_data, generated_data, bins=100, epsilon=1e-10):
    """
    Calculate Kullback-Leibler divergence between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data
    generated_data : ndarray
        Generated data
    bins : int
        Number of bins for histogram
    epsilon : float
        Small constant to avoid division by zero
        
    Returns:
    --------
    float
        KL divergence
    """
    # Flatten data if multidimensional
    real_flat = real_data.flatten()
    gen_flat = generated_data.flatten()
    
    # Find common range for both datasets
    min_val = min(real_flat.min(), gen_flat.min())
    max_val = max(real_flat.max(), gen_flat.max())
    
    # Create histograms
    real_hist, bin_edges = np.histogram(real_flat, bins=bins, range=(min_val, max_val), density=True)
    gen_hist, _ = np.histogram(gen_flat, bins=bins, range=(min_val, max_val), density=True)
    
    # Add epsilon to avoid log(0)
    real_hist += epsilon
    gen_hist += epsilon
    
    # Normalize histograms
    real_hist /= real_hist.sum()
    gen_hist /= gen_hist.sum()
    
    # Calculate KL divergence
    kl_div = np.sum(real_hist * np.log(real_hist / gen_hist))
    
    return kl_div


def js_divergence(real_data, generated_data, bins=100, epsilon=1e-10):
    """
    Calculate Jensen-Shannon divergence between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data
    generated_data : ndarray
        Generated data
    bins : int
        Number of bins for histogram
    epsilon : float
        Small constant to avoid division by zero
        
    Returns:
    --------
    float
        JS divergence
    """
    # Flatten data if multidimensional
    real_flat = real_data.flatten()
    gen_flat = generated_data.flatten()
    
    # Find common range for both datasets
    min_val = min(real_flat.min(), gen_flat.min())
    max_val = max(real_flat.max(), gen_flat.max())
    
    # Create histograms
    real_hist, bin_edges = np.histogram(real_flat, bins=bins, range=(min_val, max_val), density=True)
    gen_hist, _ = np.histogram(gen_flat, bins=bins, range=(min_val, max_val), density=True)
    
    # Add epsilon to avoid log(0)
    real_hist += epsilon
    gen_hist += epsilon
    
    # Normalize histograms
    real_hist /= real_hist.sum()
    gen_hist /= gen_hist.sum()
    
    # Calculate mixture distribution
    m_hist = 0.5 * (real_hist + gen_hist)
    
    # Calculate KL divergences
    kl_real_m = np.sum(real_hist * np.log(real_hist / m_hist))
    kl_gen_m = np.sum(gen_hist * np.log(gen_hist / m_hist))
    
    # Jensen-Shannon divergence
    js_div = 0.5 * (kl_real_m + kl_gen_m)
    
    return js_div


def autocorrelation_comparison(real_data, generated_data, max_lag=20):
    """
    Compare autocorrelation functions between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data time series with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data time series with shape (n_samples, n_features)
    max_lag : int
        Maximum lag for autocorrelation
        
    Returns:
    --------
    dict
        Dictionary containing autocorrelation error metrics
    """
    results = {}
    
    # Compute autocorrelation for each feature
    real_acf = np.zeros((real_data.shape[1], max_lag+1))
    gen_acf = np.zeros((generated_data.shape[1], max_lag+1))
    
    for i in range(real_data.shape[1]):
        real_acf[i] = acf(real_data[:, i], nlags=max_lag, fft=True)
        gen_acf[i] = acf(generated_data[:, i], nlags=max_lag, fft=True)
    
    # Calculate errors
    acf_mse = mean_square_error(real_acf, gen_acf)
    acf_mae = mean_absolute_error(real_acf, gen_acf)
    
    results['acf_mse'] = acf_mse
    results['acf_mae'] = acf_mae
    results['real_acf'] = real_acf
    results['gen_acf'] = gen_acf
    
    return results


def cross_correlation_comparison(real_data, generated_data):
    """
    Compare cross-correlation matrices between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data with shape (n_samples, n_features)
        
    Returns:
    --------
    dict
        Dictionary containing cross-correlation error metrics
    """
    results = {}
    
    # Compute cross-correlation matrices
    real_corr = np.corrcoef(real_data, rowvar=False)
    gen_corr = np.corrcoef(generated_data, rowvar=False)
    
    # Calculate errors
    corr_mse = mean_square_error(real_corr, gen_corr)
    corr_mae = mean_absolute_error(real_corr, gen_corr)
    
    results['corr_mse'] = corr_mse
    results['corr_mae'] = corr_mae
    results['real_corr'] = real_corr
    results['gen_corr'] = gen_corr
    
    return results


def volatility_clustering(data, window_size=10):
    """
    Calculate volatility clustering metric.
    
    Parameters:
    -----------
    data : ndarray
        Time series data with shape (n_samples, n_features)
    window_size : int
        Window size for rolling volatility
        
    Returns:
    --------
    ndarray
        Autocorrelation of squared returns
    """
    # Calculate returns
    returns = np.diff(data, axis=0) / data[:-1]
    
    # Calculate squared returns (proxy for volatility)
    squared_returns = returns ** 2
    
    # Calculate autocorrelation of squared returns
    vol_acf = np.zeros((returns.shape[1], window_size+1))
    
    for i in range(returns.shape[1]):
        vol_acf[i] = acf(squared_returns[:, i], nlags=window_size, fft=True)
    
    return vol_acf


def volatility_clustering_comparison(real_data, generated_data, window_size=10):
    """
    Compare volatility clustering between real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data time series with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data time series with shape (n_samples, n_features)
    window_size : int
        Window size for rolling volatility
        
    Returns:
    --------
    dict
        Dictionary containing volatility clustering error metrics
    """
    results = {}
    
    # Calculate volatility clustering metrics
    real_vol_acf = volatility_clustering(real_data, window_size)
    gen_vol_acf = volatility_clustering(generated_data, window_size)
    
    # Calculate errors
    vol_mse = mean_square_error(real_vol_acf, gen_vol_acf)
    vol_mae = mean_absolute_error(real_vol_acf, gen_vol_acf)
    
    results['vol_mse'] = vol_mse
    results['vol_mae'] = vol_mae
    results['real_vol_acf'] = real_vol_acf
    results['gen_vol_acf'] = gen_vol_acf
    
    return results


def check_no_arbitrage(dlvs):
    """
    Check if DLVs satisfy no-arbitrage conditions.
    
    Parameters:
    -----------
    dlvs : ndarray
        Discrete local volatility surface with shape (n_samples, n_strikes, n_maturities)
        
    Returns:
    --------
    dict
        Dictionary containing arbitrage metrics
    """
    results = {}
    
    # For DLVs, the no-arbitrage condition is non-negativity
    violation_mask = dlvs <= 0
    
    # Calculate overall violation rate
    total_elements = np.prod(dlvs.shape)
    num_violations = np.sum(violation_mask)
    violation_rate = num_violations / total_elements
    
    results['violation_rate'] = violation_rate
    results['num_violations'] = num_violations
    results['total_elements'] = total_elements
    
    return results


def evaluate_model(real_data, generated_data, is_dlv_surface=False, n_strikes=None, n_maturities=None):
    """
    Comprehensive evaluation of a model comparing real and generated data.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data
    generated_data : ndarray
        Generated data
    is_dlv_surface : bool
        Whether the data represents a DLV surface
    n_strikes : int, optional
        Number of strikes if is_dlv_surface is True
    n_maturities : int, optional
        Number of maturities if is_dlv_surface is True
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    results = {}
    
    # Reshape data if it's a DLV surface
    if is_dlv_surface:
        if n_strikes is None or n_maturities is None:
            raise ValueError("n_strikes and n_maturities must be provided if is_dlv_surface is True")
        
        real_data_reshaped = real_data.reshape(real_data.shape[0], -1)
        generated_data_reshaped = generated_data.reshape(generated_data.shape[0], -1)
    else:
        real_data_reshaped = real_data
        generated_data_reshaped = generated_data
    
    # Basic error metrics
    results['mse'] = mean_square_error(real_data_reshaped, generated_data_reshaped)
    results['mae'] = mean_absolute_error(real_data_reshaped, generated_data_reshaped)
    
    # Distribution metrics
    results['wasserstein'] = wasserstein_metric(real_data_reshaped, generated_data_reshaped)
    results['kl_divergence'] = kl_divergence(real_data_reshaped, generated_data_reshaped)
    results['js_divergence'] = js_divergence(real_data_reshaped, generated_data_reshaped)
    
    # Time series metrics
    acf_results = autocorrelation_comparison(real_data_reshaped, generated_data_reshaped)
    corr_results = cross_correlation_comparison(real_data_reshaped, generated_data_reshaped)
    vol_results = volatility_clustering_comparison(real_data_reshaped, generated_data_reshaped)
    
    results.update(acf_results)
    results.update(corr_results)
    results.update(vol_results)
    
    # No-arbitrage check for DLVs
    if is_dlv_surface:
        arbitrage_results = check_no_arbitrage(generated_data)
        results.update(arbitrage_results)
    
    return results


def summarize_evaluation(results):
    """
    Summarize evaluation results in a table.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing evaluation metrics
        
    Returns:
    --------
    pandas.DataFrame
        Summary table of evaluation metrics
    """
    summary = {}
    
    # Extract key metrics
    summary['MSE'] = results['mse']
    summary['MAE'] = results['mae']
    summary['Wasserstein'] = results['wasserstein']
    summary['KL Divergence'] = results['kl_divergence']
    summary['JS Divergence'] = results['js_divergence']
    summary['ACF MSE'] = results['acf_mse']
    summary['Cross-Correlation MSE'] = results['corr_mse']
    summary['Volatility ACF MSE'] = results['vol_mse']
    
    if 'violation_rate' in results:
        summary['Arbitrage Violation Rate'] = results['violation_rate']
    
    return pd.DataFrame(summary, index=['Value']).T 