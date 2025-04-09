#!/usr/bin/env python3
import os
import sys

# Add the current directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now try to run with the system Python
if __name__ == "__main__":
    import argparse
    from src.main import main
    
    parser = argparse.ArgumentParser(description='Run GAN-based option market simulator')
    parser.add_argument('--train_lstm', action='store_true', help='Train LSTM GAN model')
    parser.add_argument('--train_tcn', action='store_true', help='Train TCN GAN model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for saving logs')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory for saving plots')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument("--n_strikes", type=int, default=8, help="Number of strikes in option grid")
    parser.add_argument("--n_maturities", type=int, default=4, help="Number of maturities in option grid")
    parser.add_argument("--base_volatility", type=float, default=0.2, help="Base volatility for options")
    parser.add_argument("--n_components", type=int, default=5, help="Number of PCA components")
    parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training")
    parser.add_argument("--noise_dim", type=int, default=32, help="Dimension of noise input for GAN")
    parser.add_argument("--generate_length", type=int, default=100, help="Length of sequence to generate for evaluation")
    
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(args) 