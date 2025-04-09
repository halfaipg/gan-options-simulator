"""
Option data transformations module.

This module provides functions to convert between different representations
of option prices: raw prices, implied volatilities, and discrete local volatilities (DLVs).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd


def bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.
    
    Parameters:
    -----------
    S : float
        Underlying asset price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Implied volatility (annualized)
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price


def implied_volatility(price, S, K, T, r, option_type='call', tol=1e-8, max_iterations=1000):
    """
    Calculate implied volatility using Brent's method.
    
    Parameters:
    -----------
    price : float
        Market price of the option
    S : float
        Underlying asset price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    option_type : str
        'call' or 'put'
    tol : float
        Tolerance for solving the equation
    max_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    float
        Implied volatility
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price
    
    # Handle edge cases
    if price <= 0:
        return 0.0
    
    # Set bounds for Brent's method
    sigma_low = 0.001
    sigma_high = 5.0
    
    try:
        # Use Brent's method to find the implied volatility
        sigma = brentq(objective, sigma_low, sigma_high, 
                       xtol=tol, maxiter=max_iterations)
        return sigma
    except ValueError:
        # If Brent's method fails, use a simple approximation
        # This is a crude approximation and should be improved
        return 0.2  # Default to 20% volatility


def implied_vol_surface(option_prices, S, K_grid, T_grid, r, option_types):
    """
    Calculate implied volatility surface from option prices.
    
    Parameters:
    -----------
    option_prices : ndarray
        Array of option prices with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
    option_types : ndarray
        Array of option types ('call' or 'put') with shape (n_strikes, n_maturities)
        
    Returns:
    --------
    ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    """
    n_strikes = len(K_grid)
    n_maturities = len(T_grid)
    
    implied_vols = np.zeros((n_strikes, n_maturities))
    
    for i in range(n_strikes):
        for j in range(n_maturities):
            K = K_grid[i]
            T = T_grid[j]
            price = option_prices[i, j]
            option_type = option_types[i, j]
            
            implied_vols[i, j] = implied_volatility(price, S, K, T, r, option_type)
    
    return implied_vols


def local_volatility_dupire(implied_vols, S, K_grid, T_grid, r):
    """
    Calculate local volatility using Dupire's formula.
    
    Parameters:
    -----------
    implied_vols : ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    ndarray
        Local volatility surface with shape (n_strikes, n_maturities)
    """
    n_strikes = len(K_grid)
    n_maturities = len(T_grid)
    
    # Initialize local volatility surface
    local_vols = np.zeros((n_strikes, n_maturities))
    
    # Compute finite differences for each point in the grid
    for i in range(1, n_strikes - 1):
        for j in range(1, n_maturities - 1):
            K = K_grid[i]
            T = T_grid[j]
            sigma = implied_vols[i, j]
            
            # Calculate derivatives using finite differences
            d_sigma_dK = (implied_vols[i+1, j] - implied_vols[i-1, j]) / (K_grid[i+1] - K_grid[i-1])
            d2_sigma_dK2 = (implied_vols[i+1, j] - 2*implied_vols[i, j] + implied_vols[i-1, j]) / ((K_grid[i+1] - K_grid[i]) ** 2)
            d_sigma_dT = (implied_vols[i, j+1] - implied_vols[i, j-1]) / (T_grid[j+1] - T_grid[j-1])
            
            # Calculate d1 and d2 from Black-Scholes
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            
            # Apply Dupire's formula
            numerator = sigma**2
            denominator = (1 + K * d1 * d_sigma_dK)**2 + K**2 * sigma * d2_sigma_dK2 + 2 * K * sigma * d_sigma_dT
            
            # Handle numerical instabilities
            if denominator <= 0:
                local_vols[i, j] = sigma  # Fallback to implied vol
            else:
                local_vols[i, j] = sigma * np.sqrt(numerator / denominator)
    
    # Fill in boundary values (simple approximation)
    local_vols[0, :] = local_vols[1, :]
    local_vols[-1, :] = local_vols[-2, :]
    local_vols[:, 0] = local_vols[:, 1]
    local_vols[:, -1] = local_vols[:, -2]
    
    return local_vols


def discrete_local_volatility(implied_vols, S, K_grid, T_grid, r):
    """
    Convert implied volatility surface to discrete local volatilities (DLVs).
    
    This is a simplified implementation that needs to be enhanced based on
    the specific methodology described in Buehler (2017) and Wissel (2007).
    
    Parameters:
    -----------
    implied_vols : ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    ndarray
        Discrete local volatility surface with shape (n_strikes, n_maturities)
    """
    # For now, we'll use Dupire's formula as an approximation
    # In a real implementation, this should be replaced with the actual DLV calculation
    # as described in the referenced papers
    return local_volatility_dupire(implied_vols, S, K_grid, T_grid, r)


def dlv_to_implied_vol(dlvs, S, K_grid, T_grid, r):
    """
    Convert DLVs back to implied volatilities.
    
    This is a placeholder function that needs to be implemented based on
    the methodology described in the referenced papers.
    
    Parameters:
    -----------
    dlvs : ndarray
        Discrete local volatility surface with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    """
    # This requires implementing the inverse transformation
    # For now, we'll return a simple approximation
    return dlvs  # This is not correct and needs to be implemented properly


def check_no_arbitrage(option_prices, S, K_grid, T_grid, r):
    """
    Check if option prices satisfy no-arbitrage conditions.
    
    Parameters:
    -----------
    option_prices : ndarray
        Array of option prices with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    bool
        True if no-arbitrage conditions are satisfied, False otherwise
    """
    n_strikes = len(K_grid)
    n_maturities = len(T_grid)
    
    # Check monotonicity with respect to strike (for calls)
    for j in range(n_maturities):
        for i in range(n_strikes - 1):
            # Call prices should decrease with strike
            if option_prices[i, j] < option_prices[i+1, j]:
                return False
    
    # Check monotonicity with respect to maturity
    for i in range(n_strikes):
        for j in range(n_maturities - 1):
            # Option prices should increase with maturity
            if option_prices[i, j] > option_prices[i, j+1]:
                return False
    
    # Add more no-arbitrage checks as needed
    
    return True


def check_dlv_no_arbitrage(dlvs):
    """
    Check if DLVs satisfy no-arbitrage conditions.
    
    Parameters:
    -----------
    dlvs : ndarray
        Discrete local volatility surface with shape (n_strikes, n_maturities)
        
    Returns:
    --------
    bool
        True if no-arbitrage conditions are satisfied, False otherwise
    """
    # For DLVs, the no-arbitrage condition is simply non-negativity
    return np.all(dlvs > 0) 