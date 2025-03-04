# Crop Classification using Sentinel-2 Imagery

## Project Overview

This project implements a deep learning model for crop classification using Sentinel-2 satellite imagery. The goal is to develop a machine learning solution that can identify different crop types from multi-temporal satellite data.

## Features

- Multi-temporal image processing using Sentinel-2 satellite data
- GRU-based neural network for temporal feature extraction
- ResNet-50 feature extractor pre-trained on Sentinel-2 imagery
- Custom loss function with class balancing
- Comprehensive metrics tracking (Accuracy, F1-Score)
- Experiment tracking with Weights & Biases (wandb)

## Project Structure

```
.
├── config/
│   └── config.yaml           # Configuration settings
├── notebooks/
│   └── data_exploration.ipynb# Jupyter notebook for data exploration
├── src/
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── losses.py             # Custom loss functions
│   ├── metrics.py            # Custom evaluation metrics
│   ├── model.py              # Neural network model
│   ├── trainer.py            # Training loop and utilities
│   └── utils.py              # Utility functions
├── tests/
│   ├── test_data_loader.py   # Tests for data loading
│   ├── test_losses.py        # Tests for loss functions
│   └── test_model.py         # Tests for model architecture
└── main.py                   # Main training script
```

## Requirements

- Python 3.10+
- PyTorch
- TorchGeo
- TorchMetrics
- Weights & Biases (wandb)
- NumPy
- Pandas
- PyYAML
- tqdm

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/crop-classification.git
cd crop-classification
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

Modify the `config/config.yaml` file to adjust:
- Data paths
- Model hyperparameters
- Training settings
- Device configuration

## Training

Run the main training script:
```bash
python main.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Experiment Tracking

This project uses Weights & Biases (wandb) for experiment tracking. Make sure to:
1. Create a wandb account
2. Log in via CLI: `wandb login`

## Model Architecture

The model combines:
- ResNet-50 (pre-trained on Sentinel-2 imagery)
- GRU layer for temporal feature extraction
- Fully connected classification head

## Key Components

- **Data Loader**: Handles Sentinel-2 image chips with temporal and spectral information
- **Loss Function**: Binary Cross-Entropy with class balancing
- **Metrics**: Accuracy, F1-Score (macro, micro, weighted, and per-class)


