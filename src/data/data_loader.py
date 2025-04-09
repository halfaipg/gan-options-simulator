"""
Data loading and preprocessing module for option market data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ..utils.helpers import calculate_relative_strike
from src.utils.helpers import calculate_relative_strike

class OptionDataLoader:
    """
    Class for loading and preprocessing option market data.
    """
    
    def __init__(self, symbol, start_date=None, end_date=None):
        """
        Initialize the OptionDataLoader.
        
        Parameters:
        -----------
        symbol : str
            Symbol of the underlying asset (e.g., '^STOXX50E' for EURO STOXX 50)
        start_date : str or datetime, optional
            Start date for historical data (format: 'YYYY-MM-DD')
        end_date : str or datetime, optional
            End date for historical data (format: 'YYYY-MM-DD')
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Load underlying asset data
        self.underlying_data = None
        
        # Option data containers
        self.option_chains = {}
        self.option_prices = None
        self.implied_vols = None
        self.dlvs = None
        
        # Initialize grid parameters
        self.K_grid = None
        self.T_grid = None
        self.relative_K_grid = None
    
    def load_underlying_data(self):
        """
        Load historical data for the underlying asset.
        
        Returns:
        --------
        pandas.DataFrame
            Historical data for the underlying asset
        """
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            self.underlying_data = data
            print(f"Loaded {len(data)} days of data for {self.symbol}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def fetch_option_chains(self, date=None):
        """
        Fetch option chain data for a specific date or the latest available date.
        
        Parameters:
        -----------
        date : str or datetime, optional
            Date for which to fetch option data (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        dict
            Dictionary of option chains indexed by expiration date
        """
        # Use yfinance to get option data
        ticker = yf.Ticker(self.symbol)
        
        # Get expiration dates
        try:
            expirations = ticker.options
            
            if not expirations:
                print(f"No options data available for {self.symbol}")
                return None
            
            # Fetch option chain for each expiration
            chains = {}
            for exp in expirations:
                opt = ticker.option_chain(exp)
                chains[exp] = {
                    'calls': opt.calls,
                    'puts': opt.puts
                }
            
            self.option_chains = chains
            return chains
        
        except Exception as e:
            print(f"Error fetching option chains: {e}")
            return None
    
    def process_option_data(self, risk_free_rate=0.02, price_column='lastPrice'):
        """
        Process option chains to create structured option price data.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate to use for calculations
        price_column : str
            Column name in the option chain data that contains prices
            
        Returns:
        --------
        tuple
            (option_prices, implied_vols, strike_grid, maturity_grid)
        """
        if not self.option_chains:
            print("No option chains loaded. Call fetch_option_chains() first.")
            return None
        
        if self.underlying_data is None:
            self.load_underlying_data()
        
        # Get current price of underlying
        current_price = self.underlying_data.iloc[-1]['Close']
        
        # Extract unique strikes and maturities
        all_strikes = set()
        all_maturities = set()
        
        for exp_date, chain in self.option_chains.items():
            # Convert expiration date to datetime
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            
            # Calculate time to maturity in years
            days_to_expiry = (exp_datetime - datetime.now()).days
            T = max(1, days_to_expiry) / 365.0  # Ensure at least 1 day
            
            all_maturities.add(T)
            
            # Extract strikes from calls
            for strike in chain['calls']['strike']:
                all_strikes.add(float(strike))
        
        # Create sorted grids
        K_grid = sorted(list(all_strikes))
        T_grid = sorted(list(all_maturities))
        
        # Initialize option price and option type matrices
        n_strikes = len(K_grid)
        n_maturities = len(T_grid)
        
        option_prices = np.zeros((n_strikes, n_maturities))
        option_types = np.full((n_strikes, n_maturities), 'call', dtype=object)
        
        # Fill the price matrix
        for i, K in enumerate(K_grid):
            for j, T in enumerate(T_grid):
                # Find the corresponding expiration date
                for exp_date, chain in self.option_chains.items():
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    days_to_expiry = (exp_datetime - datetime.now()).days
                    maturity = max(1, days_to_expiry) / 365.0
                    
                    if abs(maturity - T) < 1e-6:  # Close enough to be the same maturity
                        # Find the closest strike in calls
                        calls_df = chain['calls']
                        puts_df = chain['puts']
                        
                        # Determine whether to use call or put based on moneyness
                        # (typically use calls for K >= S and puts for K < S)
                        if K >= current_price:
                            # Use call option
                            option_df = calls_df
                            option_types[i, j] = 'call'
                        else:
                            # Use put option
                            option_df = puts_df
                            option_types[i, j] = 'put'
                        
                        # Find the closest strike
                        strike_idx = (option_df['strike'] - K).abs().idxmin()
                        price = option_df.loc[strike_idx, price_column]
                        
                        option_prices[i, j] = price
                        break
        
        # Store the results
        self.option_prices = option_prices
        self.K_grid = np.array(K_grid)
        self.T_grid = np.array(T_grid)
        
        # Calculate relative strikes (K/S)
        self.relative_K_grid = np.array([K / current_price for K in K_grid])
        
        return option_prices, self.K_grid, self.T_grid, option_types
    
    def create_synthetic_data(self, n_strikes=8, n_maturities=4, base_volatility=0.2):
        """
        Create synthetic option data when real market data is not available.
        
        Parameters:
        -----------
        n_strikes : int
            Number of strikes to generate
        n_maturities : int
            Number of maturities to generate
        base_volatility : float
            Base volatility level for generating prices
            
        Returns:
        --------
        tuple
            (option_prices, strike_grid, maturity_grid, option_types)
        """
        # Create a synthetic underlying price
        S = 100.0
        
        # Create strike grid centered around S
        K_min = S * 0.8
        K_max = S * 1.2
        K_grid = np.linspace(K_min, K_max, n_strikes)
        
        # Create maturity grid (in years)
        T_min = 20 / 365.0  # 20 days
        T_max = 120 / 365.0  # 120 days
        T_grid = np.linspace(T_min, T_max, n_maturities)
        
        # Risk-free rate
        r = 0.02
        
        # Generate a realistic volatility surface with smile
        implied_vols = np.zeros((n_strikes, n_maturities))
        for j, T in enumerate(T_grid):
            for i, K in enumerate(K_grid):
                # Create volatility smile (higher vols for OTM options)
                moneyness = K / S
                vol_adj = 0.1 * (moneyness - 1.0) ** 2
                term_adj = 0.05 * np.sqrt(T)
                implied_vols[i, j] = base_volatility + vol_adj + term_adj
        
        # Generate option prices from implied vols
        option_prices = np.zeros((n_strikes, n_maturities))
        option_types = np.full((n_strikes, n_maturities), 'call', dtype=object)
        
        for j, T in enumerate(T_grid):
            for i, K in enumerate(K_grid):
                sigma = implied_vols[i, j]
                
                # Determine whether to use call or put based on moneyness
                if K >= S:
                    option_type = 'call'
                else:
                    option_type = 'put'
                
                option_types[i, j] = option_type
                
                from ..data.transformations import bs_price
                price = bs_price(S, K, T, r, sigma, option_type)
                option_prices[i, j] = price
        
        # Store the results
        self.option_prices = option_prices
        self.K_grid = K_grid
        self.T_grid = T_grid
        self.relative_K_grid = np.array([K / S for K in K_grid])
        
        return option_prices, K_grid, T_grid, option_types
    
    def calculate_implied_volatilities(self, risk_free_rate=0.02):
        """
        Calculate implied volatilities from option prices.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate to use for calculations
            
        Returns:
        --------
        ndarray
            Implied volatility surface
        """
        if self.option_prices is None:
            print("No option prices available. Process option data first.")
            return None
        
        # Get current price of underlying
        if self.underlying_data is not None:
            current_price = self.underlying_data.iloc[-1]['Close']
        else:
            current_price = 100.0  # Default for synthetic data
        
        # Create option types matrix (assuming calls for K >= S, puts for K < S)
        option_types = np.full_like(self.option_prices, 'call', dtype=object)
        for i, K in enumerate(self.K_grid):
            for j in range(len(self.T_grid)):
                if K < current_price:
                    option_types[i, j] = 'put'
        
        from ..data.transformations import implied_vol_surface
        implied_vols = implied_vol_surface(
            self.option_prices, current_price, self.K_grid, self.T_grid, risk_free_rate, option_types
        )
        
        self.implied_vols = implied_vols
        return implied_vols
    
    def calculate_dlvs(self, risk_free_rate=0.02):
        """
        Calculate discrete local volatilities (DLVs) from implied volatilities.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate to use for calculations
            
        Returns:
        --------
        ndarray
            Discrete local volatility surface
        """
        if self.implied_vols is None:
            self.calculate_implied_volatilities(risk_free_rate)
        
        # Get current price of underlying
        if self.underlying_data is not None:
            current_price = self.underlying_data.iloc[-1]['Close']
        else:
            current_price = 100.0  # Default for synthetic data
        
        from ..data.transformations import discrete_local_volatility
        dlvs = discrete_local_volatility(
            self.implied_vols, current_price, self.K_grid, self.T_grid, risk_free_rate
        )
        
        self.dlvs = dlvs
        return dlvs
    
    def save_data(self, filename):
        """
        Save processed data to a file.
        
        Parameters:
        -----------
        filename : str
            Filename to save data to
        """
        data = {
            'option_prices': self.option_prices,
            'implied_vols': self.implied_vols,
            'dlvs': self.dlvs,
            'K_grid': self.K_grid,
            'T_grid': self.T_grid,
            'relative_K_grid': self.relative_K_grid
        }
        
        np.savez(filename, **data)
        print(f"Data saved to {filename}")
    
    def load_saved_data(self, filename):
        """
        Load processed data from a file.
        
        Parameters:
        -----------
        filename : str
            Filename to load data from
            
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            data = np.load(filename)
            
            self.option_prices = data['option_prices']
            self.implied_vols = data['implied_vols']
            self.dlvs = data['dlvs']
            self.K_grid = data['K_grid']
            self.T_grid = data['T_grid']
            self.relative_K_grid = data['relative_K_grid']
            
            print(f"Data loaded from {filename}")
            return True
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False 