"""
Main module that ties all components together for option market simulation.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

from src.data.data_loader import OptionDataLoader
from src.data.dimension_reduction import DLVDimensionReducer
from src.models.lstm_gan import LSTMGAN
from src.models.tcn_gan import TCNGAN
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import create_evaluation_dashboard, summarize_evaluation_metrics
from src.utils.helpers import create_sequences, log_transform, exp_transform, ensure_dir


def create_synthetic_dataset(n_samples=1000, n_strikes=8, n_maturities=4, base_volatility=0.2):
    """
    Create a synthetic dataset for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of time samples to generate
    n_strikes : int
        Number of strikes in the option grid
    n_maturities : int
        Number of maturities in the option grid
    base_volatility : float
        Base volatility level for options
        
    Returns:
    --------
    tuple
        (log_dlv_series, data_loader) containing the synthetic data and data loader
    """
    print(f"Creating synthetic dataset with {n_samples} samples...")
    
    # Create data loader for synthetic data
    data_loader = OptionDataLoader(symbol='^STOXX50E')  # EURO STOXX 50 index
    
    # Create synthetic option data
    option_prices, strikes, maturities, option_types = data_loader.create_synthetic_data(
        n_strikes=n_strikes, 
        n_maturities=n_maturities, 
        base_volatility=base_volatility
    )
    
    # Calculate implied volatilities and DLVs
    implied_vols = data_loader.calculate_implied_volatilities(risk_free_rate=0.02)
    dlvs = data_loader.calculate_dlvs(risk_free_rate=0.02)
    
    # Convert to log-DLVs
    log_dlvs = log_transform(dlvs)
    
    # Create a time series with some typical dynamics
    log_dlv_series = np.zeros((n_samples, n_strikes, n_maturities))
    
    for i in range(n_samples):
        if i == 0:
            log_dlv_series[i] = log_dlvs.copy()
        else:
            # Add some autocorrelation and noise
            log_dlv_series[i] = 0.98 * log_dlv_series[i-1] + 0.02 * log_dlvs + 0.01 * np.random.randn(n_strikes, n_maturities)
    
    print(f"Created log-DLV series with shape {log_dlv_series.shape}")
    return log_dlv_series, data_loader


def apply_pca(log_dlv_series, n_components=5):
    """
    Apply PCA dimension reduction to log-DLV series.
    
    Parameters:
    -----------
    log_dlv_series : ndarray
        Log-DLV time series with shape (n_samples, n_strikes, n_maturities)
    n_components : int
        Number of principal components to use
        
    Returns:
    --------
    tuple
        (pca_components, pca_reducer) containing the transformed data and the PCA model
    """
    print(f"Applying PCA with {n_components} components...")
    
    # Reshape for PCA
    n_samples = log_dlv_series.shape[0]
    log_dlv_series_reshaped = log_dlv_series.reshape(n_samples, -1)
    
    # Create and fit PCA model
    pca_reducer = DLVDimensionReducer(n_components=n_components)
    pca_components = pca_reducer.fit_transform(log_dlv_series_reshaped)
    
    print(f"PCA explained variance: {pca_reducer.explained_variance_ratio}")
    print(f"Cumulative explained variance: {pca_reducer.cumulative_explained_variance}")
    
    return pca_components, pca_reducer


def prepare_training_data(pca_components, sequence_length=10, batch_size=32, train_ratio=0.8):
    """
    Prepare training and validation datasets.
    
    Parameters:
    -----------
    pca_components : ndarray
        PCA components with shape (n_samples, n_components)
    sequence_length : int
        Length of input sequences
    batch_size : int
        Batch size for training
    train_ratio : float
        Ratio of data to use for training (vs. validation)
        
    Returns:
    --------
    tuple
        (train_dataset, val_dataset, X_train, y_train, X_val, y_val) containing
        TensorFlow datasets and numpy arrays for training and validation
    """
    print(f"Preparing training data with sequence length {sequence_length}...")
    
    # Create sequences
    X, y = create_sequences(pca_components, sequence_length=sequence_length, target_steps=1)
    
    # Create training and validation sets
    train_size = int(train_ratio * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    # Create TensorFlow datasets
    buffer_size = min(1000, len(X_train))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    print(f"Created training dataset with {len(X_train)} sequences")
    print(f"Created validation dataset with {len(X_val)} sequences")
    
    return train_dataset, val_dataset, X_train, y_train, X_val, y_val


def train_lstm_gan(train_dataset, val_dataset, state_dim, noise_dim, output_dim, 
                  log_dir='logs', epochs=100):
    """
    Train an LSTM-GAN model.
    
    Parameters:
    -----------
    train_dataset : tf.data.Dataset
        Training dataset
    val_dataset : tf.data.Dataset
        Validation dataset
    state_dim : tuple
        Dimension of state input (batch_size, seq_length, features)
    noise_dim : int
        Dimension of noise input
    output_dim : int
        Dimension of output (number of PCA components)
    log_dir : str
        Directory for logging
    epochs : int
        Number of epochs to train
        
    Returns:
    --------
    LSTMGAN
        Trained LSTM-GAN model
    """
    print(f"Training LSTM-GAN model for {epochs} epochs...")
    
    # Create LSTM-GAN model
    lstm_gan = LSTMGAN(
        state_dim=state_dim,
        noise_dim=noise_dim,
        output_dim=output_dim,
        generator_units=[64, 128],
        discriminator_units=[128, 64],
        use_pca=True,
        n_pca_components=output_dim,
        log_dir=log_dir
    )
    
    # Compile model
    lstm_gan.compile(
        generator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )
    
    # Train model
    history = lstm_gan.train(train_dataset, validation_dataset=val_dataset, epochs=epochs, verbose=1)
    
    return lstm_gan


def train_tcn_gan(train_dataset, val_dataset, state_dim, noise_dim, output_dim, 
                 log_dir='logs', epochs=100):
    """
    Train a TCN-GAN model.
    
    Parameters:
    -----------
    train_dataset : tf.data.Dataset
        Training dataset
    val_dataset : tf.data.Dataset
        Validation dataset
    state_dim : tuple
        Dimension of state input (batch_size, seq_length, features)
    noise_dim : int
        Dimension of noise input
    output_dim : int
        Dimension of output (number of PCA components)
    log_dir : str
        Directory for logging
    epochs : int
        Number of epochs to train
        
    Returns:
    --------
    TCNGAN
        Trained TCN-GAN model
    """
    print(f"Training TCN-GAN model for {epochs} epochs...")
    
    # Create TCN-GAN model
    tcn_gan = TCNGAN(
        state_dim=state_dim,
        noise_dim=noise_dim,
        output_dim=output_dim,
        generator_filters=[64, 128, 64],
        discriminator_filters=[64, 128, 64],
        kernel_size=3,
        use_pca=True,
        n_pca_components=output_dim,
        log_dir=log_dir
    )
    
    # Compile model
    tcn_gan.compile(
        generator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )
    
    # Train model
    history = tcn_gan.train(train_dataset, validation_dataset=val_dataset, epochs=epochs, verbose=1)
    
    return tcn_gan


def generate_and_evaluate(models, initial_state, pca_reducer, sequence_length, 
                         real_log_dlvs, n_strikes, n_maturities, plots_dir='plots'):
    """
    Generate synthetic data and evaluate models.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    initial_state : ndarray
        Initial state for generation
    pca_reducer : DLVDimensionReducer
        PCA dimension reducer
    sequence_length : int
        Length of sequence to generate
    real_log_dlvs : ndarray
        Real log-DLV data for comparison
    n_strikes : int
        Number of strikes in the option grid
    n_maturities : int
        Number of maturities in the option grid
    plots_dir : str
        Directory for saving plots
        
    Returns:
    --------
    tuple
        (generated_sequences, evaluation_results, figures) containing the generated data,
        evaluation results, and visualization figures
    """
    print("Generating synthetic data and evaluating models...")
    
    # Ensure plots directory exists
    ensure_dir(plots_dir)
    
    # Initialize results dictionaries
    generated_sequences = {}
    evaluation_results = {}
    all_figures = {}
    
    # Get a subset of real data for comparison
    real_log_dlvs_subset = real_log_dlvs[:sequence_length]
    real_log_dlvs_flat = real_log_dlvs_subset.reshape(sequence_length, -1)
    
    # Generate and evaluate for each model
    for model_name, model in models.items():
        print(f"Generating and evaluating {model_name}...")
        
        # Generate PCA components
        generated_sequence = model.generate_sequences(
            initial_state, 
            sequence_length=sequence_length, 
            use_generated_state=True
        )
        
        # Transform PCA components back to log-DLVs
        generated_log_dlvs = pca_reducer.inverse_transform(
            generated_sequence, 
            original_shape=(sequence_length, n_strikes, n_maturities)
        )
        
        # Reshape for evaluation
        generated_log_dlvs_flat = generated_log_dlvs.reshape(sequence_length, -1)
        
        # --- DEBUG PRINTS ---
        print(f"\n--- Debugging inputs for evaluate_model ({model_name}) ---")
        print(f"Real data shape: {real_log_dlvs_flat.shape}")
        print(f"Generated data shape: {generated_log_dlvs_flat.shape}")
        print(f"Real data contains NaN: {np.isnan(real_log_dlvs_flat).any()}")
        print(f"Generated data contains NaN: {np.isnan(generated_log_dlvs_flat).any()}")
        print(f"Real data column variances: {np.var(real_log_dlvs_flat, axis=0)}")
        print(f"Generated data column variances: {np.var(generated_log_dlvs_flat, axis=0)}")
        print(f"---------------------------------------------------")
        # --- END DEBUG PRINTS ---

        # Evaluate the model
        results = evaluate_model(
            real_log_dlvs_flat, 
            generated_log_dlvs_flat,
            is_dlv_surface=True,
            n_strikes=n_strikes,
            n_maturities=n_maturities
        )
        
        # Create evaluation dashboard
        figures = create_evaluation_dashboard(
            results,
            real_log_dlvs_flat,
            generated_log_dlvs_flat,
            save_path=os.path.join(plots_dir, model_name.lower())
        )
        
        # Store results
        generated_sequences[model_name] = generated_log_dlvs
        evaluation_results[model_name] = results
        all_figures[model_name] = figures
    
    # Compare models
    comparison_df = summarize_evaluation_metrics(evaluation_results)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(plots_dir, 'model_comparison.csv'))
    
    return generated_sequences, evaluation_results, all_figures


def main(args):
    """
    Main function to run the complete option market simulation pipeline.
    
    Parameters:
    -----------
    args : Namespace
        Command-line arguments
    """
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, timestamp)
    ensure_dir(log_dir)
    
    # Create plots directory
    plots_dir = os.path.join(args.plots_dir, timestamp)
    ensure_dir(plots_dir)
    
    # Step 1: Create synthetic dataset
    log_dlv_series, data_loader = create_synthetic_dataset(
        n_samples=args.n_samples,
        n_strikes=args.n_strikes,
        n_maturities=args.n_maturities,
        base_volatility=args.base_volatility
    )
    
    # Step 2: Apply PCA
    pca_components, pca_reducer = apply_pca(log_dlv_series, n_components=args.n_components)
    
    # Step 3: Prepare training data
    train_dataset, val_dataset, X_train, y_train, X_val, y_val = prepare_training_data(
        pca_components, 
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio
    )
    
    # Define state, noise, and output dimensions
    state_dim = (None, args.sequence_length, args.n_components)
    noise_dim = args.noise_dim
    output_dim = args.n_components
    
    # Dictionary to store models
    models = {}
    
    # Step 4: Train models
    if args.train_lstm:
        lstm_gan = train_lstm_gan(
            train_dataset, 
            val_dataset, 
            state_dim, 
            noise_dim, 
            output_dim,
            log_dir=os.path.join(log_dir, 'lstm_gan'),
            epochs=args.epochs
        )
        models['LSTM-GAN'] = lstm_gan
    
    if args.train_tcn:
        tcn_gan = train_tcn_gan(
            train_dataset, 
            val_dataset, 
            state_dim, 
            noise_dim, 
            output_dim,
            log_dir=os.path.join(log_dir, 'tcn_gan'),
            epochs=args.epochs
        )
        models['TCN-GAN'] = tcn_gan
    
    # Step 5: Generate and evaluate
    if models:
        initial_state = X_val[0:1]  # Take first validation sequence as initial state
        generated_sequences, evaluation_results, figures = generate_and_evaluate(
            models,
            initial_state,
            pca_reducer,
            args.generate_length,
            log_dlv_series,
            args.n_strikes,
            args.n_maturities,
            plots_dir=plots_dir
        )
    
    print(f"Results saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Option Market Simulator")
    
    # Data generation parameters
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of time samples to generate")
    parser.add_argument("--n_strikes", type=int, default=8, help="Number of strikes in option grid")
    parser.add_argument("--n_maturities", type=int, default=4, help="Number of maturities in option grid")
    parser.add_argument("--base_volatility", type=float, default=0.2, help="Base volatility for options")
    
    # PCA parameters
    parser.add_argument("--n_components", type=int, default=5, help="Number of PCA components")
    
    # Training parameters
    parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training")
    parser.add_argument("--noise_dim", type=int, default=32, help="Dimension of noise input")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    # Model selection
    parser.add_argument("--train_lstm", action="store_true", help="Train LSTM-GAN model")
    parser.add_argument("--train_tcn", action="store_true", help="Train TCN-GAN model")
    
    # Generation parameters
    parser.add_argument("--generate_length", type=int, default=100, help="Length of sequence to generate")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory for plots")
    
    args = parser.parse_args()
    
    # Default to training both if none specified
    if not args.train_lstm and not args.train_tcn:
        args.train_lstm = True
        args.train_tcn = True
    
    main(args) 