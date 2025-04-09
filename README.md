# Realistic Equity Option Market Simulation using GANs (Replication of arXiv:1911.01700)

This project provides a framework for generating realistic synthetic equity option market data using Generative Adversarial Networks (GANs). It is an implementation based on the methodologies presented in the paper "Deep Hedging: Learning to Simulate Equity Option Markets" by Wiese et al. (2019) [[arXiv:1911.01700]](https://arxiv.org/abs/1911.01700).

Financial modeling and the backtesting of trading strategies often require large datasets that capture complex market dynamics. Real-world data can be limited or costly. This framework aims to address this by providing tools to simulate option market behavior, specifically focusing on the generation of realistic Discrete Local Volatility (DLV) surfaces over time.

## Features

*   **GAN-based Simulation:** Leverages Generative Adversarial Networks to learn the complex dynamics of multivariate financial time series (option surfaces).
*   **Paper Replication:** Implements key components described in Wiese et al. (2019):
    *   **LSTM-GAN:** A recurrent architecture for sequence generation.
    *   **TCN-GAN:** A temporal convolutional network architecture for sequence generation.
    *   **PCA Compression:** Uses Principal Component Analysis (PCA) to reduce the dimensionality of high-dimensional option surface data (DLVs) before feeding it to the GANs.
*   **Modular Design:** Code is organized into distinct modules for:
    *   Data Loading & Preprocessing (`src/data`)
    *   Generative Models (GANs) (`src/models`)
    *   Evaluation Metrics & Visualization (`src/evaluation`)
    *   Utility Functions (`src/utils`)
*   **Synthetic Data Generation:** Includes functionality to create an initial synthetic dataset for demonstration and testing purposes.
*   **Evaluation Framework:** Provides tools to quantitatively and qualitatively assess the realism of the generated data compared to the source data (using metrics like ACF, distribution comparison, etc.).
*   **Extensibility:** Designed with the potential to incorporate new generative models, different data sources (e.g., real market data), other asset classes (e.g., futures), or custom evaluation metrics.

## Installation

A Conda environment is recommended for managing dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate Conda environment:**
    ```bash
    conda create -n gansynth python=3.11 # Or your preferred Python 3.x version
    conda activate gansynth
    ```

3.  **Install dependencies:**
    *   Install core dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   **TensorFlow Note:** Ensure you have a compatible version of TensorFlow installed for your system (CPU/GPU). The `requirements.txt` specifies `tensorflow>=2.4.0`. Installation might vary based on your OS and hardware (especially on M1/M2 Macs). You might have installed this separately in your conda environment already.
    *   **Other Dependencies:** The `requirements.txt` should cover other necessary packages like `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`. If you encounter `ModuleNotFoundError` during runtime, install the missing package using pip or conda.

## Usage

The main script to run the simulation pipeline is `run.py`.

1.  **Activate your environment:**
    ```bash
    conda activate gansynth
    ```

2.  **Run the simulation:**
    ```bash
    python run.py [options]
    ```

    **Example:** Train both LSTM-GAN and TCN-GAN for 50 epochs using the generated synthetic data:
    ```bash
    python run.py --train_lstm --train_tcn --epochs 50
    ```

    **Key Command-Line Arguments:**
    *   `--train_lstm`: Train the LSTM-GAN model.
    *   `--train_tcn`: Train the TCN-GAN model.
    *   `--epochs INT`: Number of training epochs (default: 10).
    *   `--n_samples INT`: Number of time samples for the synthetic dataset (default: 1000).
    *   `--n_strikes INT`: Number of option strikes (default: 8).
    *   `--n_maturities INT`: Number of option maturities (default: 4).
    *   `--n_components INT`: Number of PCA components for dimension reduction (default: 5).
    *   `--sequence_length INT`: Input sequence length for models (default: 10).
    *   `--noise_dim INT`: Dimension of the noise vector for GAN generators (default: 32).
    *   `--generate_length INT`: Length of the sequence to generate during evaluation (default: 100).
    *   `--log_dir PATH`: Directory to save logs and model weights (default: 'logs').
    *   `--plots_dir PATH`: Directory to save evaluation plots and metrics (default: 'plots').
    *   `--seed INT`: Random seed for reproducibility (default: 42).

    Use `python run.py --help` to see all available options.

3.  **Outputs:**
    *   **Logs & Models:** Training logs and saved model weights (`.weights.h5` files) are stored in timestamped subdirectories within the specified `--log_dir` (e.g., `logs/20231027-120000/LSTM-GAN/`).
    *   **Evaluation Results:** Comparison metrics (`model_comparison.csv`) and visualization plots (`.png` files comparing statistical properties) are saved in a timestamped subdirectory within the specified `--plots_dir` (e.g., `plots/20231027-120000/`).

4.  **Demo Notebook:**
    *   Explore the framework interactively using the Jupyter notebook: `notebooks/demo.ipynb`.

## Project Structure

```
.
├── logs/                  # Stores training logs and model weights
├── notebooks/
│   └── demo.ipynb         # Interactive demo notebook
├── plots/                 # Stores evaluation results (metrics, plots)
├── src/                   # Source code
│   ├── data/              # Data loading, transformation, PCA
│   ├── evaluation/        # Evaluation metrics and visualization
│   ├── models/            # GAN model implementations (Base, LSTM, TCN)
│   ├── utils/             # Helper functions
│   ├── __init__.py
│   └── main.py            # Main simulation pipeline logic
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── run.py                 # Script to execute the main pipeline
```

## Evaluation

The framework evaluates the quality of the generated time series by comparing their statistical properties to the original data. This includes:

*   Visual comparison of time series paths.
*   Distribution analysis (histograms, Q-Q plots).
*   Autocorrelation Function (ACF) comparison.
*   Other relevant financial time series metrics (defined in `src/evaluation/metrics.py`).

Results are saved in the `--plots_dir` for each run, allowing comparison between different models (LSTM vs. TCN) and configurations.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or suggestions.

## License

[Specify License Here - e.g., MIT License]

## Citation

If you use this code or build upon the concepts in your research, please cite the original paper:

```bibtex
@misc{wiese2019deep,
      title={Deep Hedging: Learning to Simulate Equity Option Markets},
      author={Magnus Wiese and Lianjun Bai and Ben Wood and Hans Buehler},
      year={2019},
      eprint={1911.01700},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```

Link: [https://arxiv.org/abs/1911.01700](https://arxiv.org/abs/1911.01700)

## Acknowledgements

This project is based on the work of Magnus Wiese, Lianjun Bai, Ben Wood, and Hans Buehler. 