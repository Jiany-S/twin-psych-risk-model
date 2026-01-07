"""
Worker profile management module for computing EMA (Ecological Momentary Assessment) 
statistics per worker.

This module handles:
- Computing mean (μ) and standard deviation (σ) per worker
- Worker-specific baseline statistics
- Profile persistence and loading
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkerProfile:
    """Manages individual worker profile with EMA statistics."""
    
    def __init__(self, worker_id: str):
        """
        Initialize a worker profile.
        
        Args:
            worker_id: Unique identifier for the worker
        """
        self.worker_id = worker_id
        self.mean_: Optional[Dict[str, float]] = None
        self.std_: Optional[Dict[str, float]] = None
        self.n_observations_: int = 0
        self.fitted_ = False
    
    def fit(self, data: pd.DataFrame, feature_columns: List[str]) -> "WorkerProfile":
        """
        Compute statistics from worker's EMA data.
        
        Args:
            data: Worker's EMA data
            feature_columns: List of feature column names to compute stats for
            
        Returns:
            Self for method chaining
        """
        self.mean_ = {}
        self.std_ = {}
        
        for col in feature_columns:
            if col in data.columns:
                self.mean_[col] = float(data[col].mean())
                self.std_[col] = float(data[col].std())
            else:
                logger.warning(f"Column {col} not found for worker {self.worker_id}")
        
        self.n_observations_ = len(data)
        self.fitted_ = True
        
        logger.debug(
            f"Worker {self.worker_id}: fitted with {self.n_observations_} observations"
        )
        
        return self
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get computed statistics.
        
        Returns:
            Dictionary with 'mean' and 'std' statistics
        """
        if not self.fitted_:
            raise ValueError("Profile must be fitted before accessing stats")
        
        return {
            'mean': self.mean_,
            'std': self.std_,
            'n_observations': self.n_observations_
        }
    
    def normalize_value(self, feature: str, value: float) -> float:
        """
        Normalize a value using worker-specific statistics.
        
        Args:
            feature: Feature name
            value: Raw value to normalize
            
        Returns:
            Normalized value (z-score)
        """
        if not self.fitted_:
            raise ValueError("Profile must be fitted before normalization")
        
        if feature not in self.mean_:
            raise ValueError(f"Feature {feature} not in profile")
        
        mean = self.mean_[feature]
        std = self.std_[feature]
        
        # Avoid division by zero
        if std < 1e-8:
            return 0.0
        
        return (value - mean) / std


class WorkerProfileManager:
    """Manages profiles for multiple workers."""
    
    def __init__(self):
        """Initialize the profile manager."""
        self.profiles_: Dict[str, WorkerProfile] = {}
        self.feature_columns_: Optional[List[str]] = None
    
    def fit(
        self,
        data: pd.DataFrame,
        worker_id_column: str = "worker_id",
        feature_columns: Optional[List[str]] = None
    ) -> "WorkerProfileManager":
        """
        Compute profiles for all workers in the dataset.
        
        Args:
            data: EMA data with worker IDs
            worker_id_column: Name of the column containing worker IDs
            feature_columns: List of feature columns to compute stats for.
                           If None, uses all numeric columns except worker_id.
            
        Returns:
            Self for method chaining
        """
        if worker_id_column not in data.columns:
            raise ValueError(f"Column {worker_id_column} not found in data")
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [
                col for col in data.select_dtypes(include=[np.number]).columns
                if col != worker_id_column
            ]
        
        self.feature_columns_ = feature_columns
        
        # Group by worker and compute profiles
        unique_workers = data[worker_id_column].unique()
        logger.info(f"Computing profiles for {len(unique_workers)} workers")
        
        for worker_id in unique_workers:
            worker_data = data[data[worker_id_column] == worker_id]
            profile = WorkerProfile(str(worker_id))
            profile.fit(worker_data, feature_columns)
            self.profiles_[str(worker_id)] = profile
        
        logger.info(f"Fitted {len(self.profiles_)} worker profiles")
        
        return self
    
    def get_profile(self, worker_id: str) -> Optional[WorkerProfile]:
        """
        Get profile for a specific worker.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            WorkerProfile or None if not found
        """
        return self.profiles_.get(worker_id)
    
    def get_all_profiles(self) -> Dict[str, WorkerProfile]:
        """
        Get all worker profiles.
        
        Returns:
            Dictionary mapping worker_id to WorkerProfile
        """
        return self.profiles_
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics across all workers.
        
        Returns:
            DataFrame with aggregated statistics
        """
        if not self.profiles_:
            return pd.DataFrame()
        
        summary_data = []
        
        for worker_id, profile in self.profiles_.items():
            stats = profile.get_stats()
            row = {'worker_id': worker_id, 'n_observations': stats['n_observations']}
            
            # Add mean and std for each feature
            for feature in self.feature_columns_:
                if feature in stats['mean']:
                    row[f'{feature}_mean'] = stats['mean'][feature]
                    row[f'{feature}_std'] = stats['std'][feature]
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save(self, path: Path) -> None:
        """
        Save all worker profiles to disk.
        
        Args:
            path: Path to save the profiles (pickle format)
        """
        state = {
            'profiles': self.profiles_,
            'feature_columns': self.feature_columns_
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved {len(self.profiles_)} profiles to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "WorkerProfileManager":
        """
        Load worker profiles from disk.
        
        Args:
            path: Path to saved profiles
            
        Returns:
            Loaded WorkerProfileManager instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        manager = cls()
        manager.profiles_ = state['profiles']
        manager.feature_columns_ = state['feature_columns']
        
        logger.info(f"Loaded {len(manager.profiles_)} profiles from {path}")
        return manager


def main():
    """Main entry point for worker profile CLI."""
    parser = argparse.ArgumentParser(
        description="Compute EMA statistics (μ/σ) per worker"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input EMA data (CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/worker_profiles.pkl",
        help="Path to save worker profiles"
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="data/processed/worker_profiles_summary.csv",
        help="Path to save summary statistics CSV"
    )
    parser.add_argument(
        "--worker-id-column",
        type=str,
        default="worker_id",
        help="Name of the worker ID column"
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="+",
        default=None,
        help="Feature columns to compute stats for (default: all numeric)"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    data = pd.read_csv(args.input)
    
    # Compute worker profiles
    manager = WorkerProfileManager()
    manager.fit(
        data,
        worker_id_column=args.worker_id_column,
        feature_columns=args.feature_columns
    )
    
    # Save profiles
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manager.save(output_path)
    
    # Save summary statistics
    summary = manager.summary_statistics()
    summary_path = Path(args.summary_output)
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    logger.info("Worker profile computation complete!")


if __name__ == "__main__":
    main()
