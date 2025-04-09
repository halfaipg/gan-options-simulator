"""
Dimension reduction module for DLV data.

This module provides functions for compressing and decompressing DLV data
using Principal Component Analysis (PCA).
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DLVDimensionReducer:
    """
    Class for reducing dimensionality of DLV data and reconstructing it.
    """
    
    def __init__(self, n_components=5):
        """
        Initialize the DLVDimensionReducer.
        
        Parameters:
        -----------
        n_components : int
            Number of principal components to use
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        self.explained_variance_ratio = None
        self.cumulative_explained_variance = None
    
    def fit(self, log_dlvs):
        """
        Fit the PCA model to the log-DLV data.
        
        Parameters:
        -----------
        log_dlvs : ndarray
            Array of log-DLV data with shape (n_samples, n_features)
            where n_features = n_strikes * n_maturities
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        # Reshape if necessary
        if log_dlvs.ndim > 2:
            n_samples = log_dlvs.shape[0]
            log_dlvs_reshaped = log_dlvs.reshape(n_samples, -1)
        else:
            log_dlvs_reshaped = log_dlvs
        
        # Fit PCA model
        self.pca.fit(log_dlvs_reshaped)
        
        # Store explained variance information
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
        
        self.is_fitted = True
        return self
    
    def transform(self, log_dlvs):
        """
        Transform log-DLV data to PCA components.
        
        Parameters:
        -----------
        log_dlvs : ndarray
            Array of log-DLV data with shape (n_samples, n_features)
            or (n_samples, n_strikes, n_maturities)
            
        Returns:
        --------
        ndarray
            Transformed data with shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("The PCA model must be fitted before transformation")
        
        # Reshape if necessary
        original_shape = log_dlvs.shape
        if log_dlvs.ndim > 2:
            n_samples = log_dlvs.shape[0]
            log_dlvs_reshaped = log_dlvs.reshape(n_samples, -1)
        else:
            log_dlvs_reshaped = log_dlvs
        
        # Transform data
        transformed_data = self.pca.transform(log_dlvs_reshaped)
        return transformed_data
    
    def inverse_transform(self, pca_components, original_shape=None):
        """
        Transform PCA components back to log-DLV data.
        
        Parameters:
        -----------
        pca_components : ndarray
            Array of PCA components with shape (n_samples, n_components)
        original_shape : tuple, optional
            Shape to reshape the reconstructed data to, e.g., (n_samples, n_strikes, n_maturities)
            
        Returns:
        --------
        ndarray
            Reconstructed log-DLV data with shape determined by original_shape or
            (n_samples, n_features) if original_shape is None
        """
        if not self.is_fitted:
            raise ValueError("The PCA model must be fitted before inverse transformation")
        
        # Inverse transform
        reconstructed_data = self.pca.inverse_transform(pca_components)
        
        # Reshape if necessary
        if original_shape is not None and len(original_shape) > 2:
            reconstructed_data = reconstructed_data.reshape(original_shape)
        
        return reconstructed_data
    
    def fit_transform(self, log_dlvs):
        """
        Fit the PCA model and transform the data in one step.
        
        Parameters:
        -----------
        log_dlvs : ndarray
            Array of log-DLV data with shape (n_samples, n_features)
            or (n_samples, n_strikes, n_maturities)
            
        Returns:
        --------
        ndarray
            Transformed data with shape (n_samples, n_components)
        """
        self.fit(log_dlvs)
        return self.transform(log_dlvs)
    
    def reconstruct(self, log_dlvs, original_shape=None):
        """
        Compress and reconstruct log-DLV data to assess reconstruction quality.
        
        Parameters:
        -----------
        log_dlvs : ndarray
            Array of log-DLV data with shape (n_samples, n_features)
            or (n_samples, n_strikes, n_maturities)
        original_shape : tuple, optional
            Shape to reshape the reconstructed data to
            
        Returns:
        --------
        ndarray
            Reconstructed log-DLV data with the same shape as the input
        """
        if not self.is_fitted:
            self.fit(log_dlvs)
        
        # Save original shape
        if original_shape is None:
            original_shape = log_dlvs.shape
        
        # Transform and inverse transform
        pca_components = self.transform(log_dlvs)
        reconstructed_data = self.inverse_transform(pca_components, original_shape)
        
        return reconstructed_data
    
    def plot_explained_variance(self, figsize=(10, 6)):
        """
        Plot the explained variance ratio and cumulative explained variance.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        tuple
            (fig, ax1, ax2) containing the figure and axis objects
        """
        if not self.is_fitted:
            raise ValueError("The PCA model must be fitted before plotting")
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot explained variance
        components = range(1, len(self.explained_variance_ratio) + 1)
        ax1.bar(components, self.explained_variance_ratio, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Principal Components')
        
        # Plot cumulative explained variance
        ax2 = ax1.twinx()
        ax2.plot(components, self.cumulative_explained_variance, 'r-', marker='o')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.grid(False)
        
        # Add threshold line
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7)
        ax2.text(len(components) * 0.7, 0.96, '95% Threshold', color='r')
        
        fig.tight_layout()
        return fig, ax1, ax2
    
    def plot_component_weights(self, component_idx=0, n_strikes=8, n_maturities=4, figsize=(10, 6)):
        """
        Plot the weights of a specific principal component as a surface.
        
        Parameters:
        -----------
        component_idx : int
            Index of the principal component to plot (0 for first component)
        n_strikes : int
            Number of strikes in the original DLV grid
        n_maturities : int
            Number of maturities in the original DLV grid
        figsize : tuple
            Figure size
            
        Returns:
        --------
        tuple
            (fig, ax) containing the figure and axis objects
        """
        if not self.is_fitted:
            raise ValueError("The PCA model must be fitted before plotting")
        
        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} out of range (0-{self.n_components-1})")
        
        # Get component weights
        component_weights = self.pca.components_[component_idx].reshape(n_strikes, n_maturities)
        
        # Create strike and maturity grids for plotting
        strike_grid = np.linspace(0.8, 1.2, n_strikes)  # Relative strikes
        maturity_grid = np.linspace(20/365, 120/365, n_maturities)  # In years
        
        # Plot as a surface
        from ..utils.helpers import plot_surface
        fig, ax = plot_surface(
            strike_grid, maturity_grid, component_weights,
            f'Weights of Principal Component {component_idx+1}',
            'Relative Strike (K/S)', 'Maturity (years)', 'Weight',
            cmap='coolwarm'
        )
        
        return fig, ax
    
    def save(self, filename):
        """
        Save the PCA model to a file.
        
        Parameters:
        -----------
        filename : str
            Filename to save the model to
        """
        if not self.is_fitted:
            raise ValueError("The PCA model must be fitted before saving")
        
        model_data = {
            'n_components': self.n_components,
            'pca_components': self.pca.components_,
            'pca_mean': self.pca.mean_,
            'explained_variance': self.pca.explained_variance_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'singular_values': self.pca.singular_values_
        }
        
        np.savez(filename, **model_data)
        print(f"PCA model saved to {filename}")
    
    def load(self, filename):
        """
        Load a saved PCA model from a file.
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        try:
            data = np.load(filename)
            
            # Ensure the loaded model has the same number of components
            loaded_n_components = data['n_components']
            if loaded_n_components != self.n_components:
                print(f"Warning: Loaded model has {loaded_n_components} components, "
                      f"but {self.n_components} was requested. Using loaded value.")
                self.n_components = loaded_n_components
                self.pca = PCA(n_components=self.n_components)
            
            # Set PCA attributes
            self.pca.components_ = data['pca_components']
            self.pca.mean_ = data['pca_mean']
            self.pca.explained_variance_ = data['explained_variance']
            self.pca.explained_variance_ratio_ = data['explained_variance_ratio']
            self.pca.singular_values_ = data['singular_values']
            
            # Set other attributes
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
            
            self.is_fitted = True
            print(f"PCA model loaded from {filename}")
            
            return self
        
        except Exception as e:
            print(f"Error loading PCA model: {e}")
            return self 