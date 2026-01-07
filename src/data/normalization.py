"""
Data normalization module implementing z-score normalization.

This module provides utilities for:
- Z-score (standard) normalization
- Fitting normalizers on training data
- Applying normalization to new data
- Inverse transformation for predictions
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZScoreNormalizer:
    """Z-score (standard) normalization for time series data."""
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the normalizer.
        
        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted_ = False
    
    def fit(self, data: np.ndarray) -> "ZScoreNormalizer":
        """
        Compute mean and standard deviation from training data.
        
        Args:
            data: Training data of shape (n_samples, ..., n_features)
            
        Returns:
            Self for method chaining
        """
        # Compute statistics along the sample axis (axis=0)
        # This preserves feature dimensions
        self.mean_ = np.mean(data, axis=0, keepdims=True)
        self.std_ = np.std(data, axis=0, keepdims=True)
        
        # Ensure std is not zero
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        
        self.fitted_ = True
        logger.info(f"Fitted normalizer on data with shape {data.shape}")
        logger.debug(f"Mean shape: {self.mean_.shape}, Std shape: {self.std_.shape}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to data.
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data with same shape as input
        """
        if not self.fitted_:
            raise ValueError("Normalizer must be fitted before transform")
        
        normalized = (data - self.mean_) / self.std_
        return normalized
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform data in one step.
        
        Args:
            data: Training data to fit and transform
            
        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        
        Args:
            normalized_data: Z-score normalized data
            
        Returns:
            Data in original scale
        """
        if not self.fitted_:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        original = (normalized_data * self.std_) + self.mean_
        return original
    
    def save(self, path: Path) -> None:
        """
        Save normalizer parameters to disk.
        
        Args:
            path: Path to save the normalizer (pickle format)
        """
        if not self.fitted_:
            logger.warning("Saving an unfitted normalizer")
        
        state = {
            'mean': self.mean_,
            'std': self.std_,
            'epsilon': self.epsilon,
            'fitted': self.fitted_
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Normalizer saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ZScoreNormalizer":
        """
        Load normalizer parameters from disk.
        
        Args:
            path: Path to saved normalizer
            
        Returns:
            Loaded normalizer instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        normalizer = cls(epsilon=state['epsilon'])
        normalizer.mean_ = state['mean']
        normalizer.std_ = state['std']
        normalizer.fitted_ = state['fitted']
        
        logger.info(f"Normalizer loaded from {path}")
        return normalizer
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get normalizer parameters.
        
        Returns:
            Dictionary with mean and std statistics
        """
        return {
            'mean': self.mean_,
            'std': self.std_,
            'fitted': self.fitted_
        }


def normalize_dataset(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    normalizer_path: Path
) -> None:
    """
    Normalize train/val/test datasets and save results.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        output_dir: Directory to save normalized data
        normalizer_path: Path to save the fitted normalizer
    """
    logger.info("Loading datasets...")
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)
    
    logger.info(
        f"Loaded: train={train_data.shape}, "
        f"val={val_data.shape}, test={test_data.shape}"
    )
    
    # Fit normalizer on training data only
    normalizer = ZScoreNormalizer()
    train_normalized = normalizer.fit_transform(train_data)
    
    # Transform validation and test data
    val_normalized = normalizer.transform(val_data)
    test_normalized = normalizer.transform(test_data)
    
    # Save normalized data
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_normalized.npy", train_normalized)
    np.save(output_dir / "val_normalized.npy", val_normalized)
    np.save(output_dir / "test_normalized.npy", test_normalized)
    
    logger.info(f"Normalized data saved to {output_dir}")
    
    # Save normalizer
    normalizer.save(normalizer_path)


def main():
    """Main entry point for normalization CLI."""
    parser = argparse.ArgumentParser(
        description="Normalize time series data using z-score normalization"
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training data (.npy file)"
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to validation data (.npy file)"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test data (.npy file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save normalized data"
    )
    parser.add_argument(
        "--normalizer-path",
        type=str,
        default="data/processed/normalizer.pkl",
        help="Path to save the fitted normalizer"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small constant to avoid division by zero"
    )
    
    args = parser.parse_args()
    
    # Normalize datasets
    normalize_dataset(
        train_path=Path(args.train),
        val_path=Path(args.val),
        test_path=Path(args.test),
        output_dir=Path(args.output_dir),
        normalizer_path=Path(args.normalizer_path)
    )
    
    logger.info("Normalization complete!")


if __name__ == "__main__":
    main()
