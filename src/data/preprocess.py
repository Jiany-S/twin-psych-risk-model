"""
Data preprocessing module for time series windowing and train/val/test splits.

This module handles:
- Time series windowing for sequential data
- Train/validation/test data splitting
- Data loading and basic transformations
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Handles time series preprocessing, windowing, and data splits."""
    
    def __init__(
        self,
        window_size: int = 30,
        forecast_horizon: int = 7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Initialize the preprocessor.
        
        Args:
            window_size: Number of time steps in each window (lookback period)
            forecast_horizon: Number of time steps to forecast ahead
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, validation, and test ratios must sum to 1.0"
    
    def create_windows(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding windows from time series data.
        
        Args:
            data: Input time series data of shape (n_samples, n_features)
            timestamps: Optional timestamps for each sample
            
        Returns:
            Tuple of (windows, targets, window_timestamps)
            - windows: shape (n_windows, window_size, n_features)
            - targets: shape (n_windows, forecast_horizon, n_features)
            - window_timestamps: shape (n_windows, window_size) if provided
        """
        n_samples = len(data)
        total_length = self.window_size + self.forecast_horizon
        
        if n_samples < total_length:
            logger.warning(
                f"Data length {n_samples} is less than required {total_length}"
            )
            return np.array([]), np.array([]), None
        
        n_windows = n_samples - total_length + 1
        windows = []
        targets = []
        window_times = [] if timestamps is not None else None
        
        for i in range(n_windows):
            window = data[i:i + self.window_size]
            target = data[i + self.window_size:i + total_length]
            windows.append(window)
            targets.append(target)
            
            if timestamps is not None:
                window_times.append(timestamps[i:i + self.window_size])
        
        windows = np.array(windows)
        targets = np.array(targets)
        window_times = np.array(window_times) if window_times else None
        
        logger.info(f"Created {n_windows} windows from {n_samples} samples")
        return windows, targets, window_times
    
    def train_val_test_split(
        self,
        data: np.ndarray,
        shuffle: bool = False,
        random_state: Optional[int] = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data to split
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(data)
        
        if shuffle:
            np.random.seed(random_state)
            indices = np.random.permutation(n_samples)
            data = data[indices]
        
        train_end = int(n_samples * self.train_ratio)
        val_end = train_end + int(n_samples * self.val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        logger.info(
            f"Split data: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )
        
        return train_data, val_data, test_data
    
    def load_and_preprocess(
        self,
        input_path: Path,
        output_dir: Path
    ) -> None:
        """
        Load raw data, create windows, split, and save processed data.
        
        Args:
            input_path: Path to raw data file (CSV)
            output_dir: Directory to save processed data
        """
        logger.info(f"Loading data from {input_path}")
        
        # Load data (stub - assumes CSV format)
        # In practice, this would handle various formats
        df = pd.read_csv(input_path)
        data = df.values
        
        # Create windows
        windows, targets, _ = self.create_windows(data)
        
        if len(windows) == 0:
            logger.error("No windows created. Check data length and window size.")
            return
        
        # Split into train/val/test
        combined = np.concatenate([windows, targets], axis=1)
        train, val, test = self.train_val_test_split(combined)
        
        # Save processed data
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "train.npy", train)
        np.save(output_dir / "val.npy", val)
        np.save(output_dir / "test.npy", test)
        
        logger.info(f"Processed data saved to {output_dir}")


def main():
    """Main entry point for preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocess time series data for risk forecasting"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw input data (CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Number of time steps in each window"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=7,
        help="Number of time steps to forecast"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for testing"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Process data
    preprocessor.load_and_preprocess(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir)
    )
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
