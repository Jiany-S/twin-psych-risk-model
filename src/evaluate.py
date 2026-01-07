"""
Model evaluation module with AUROC and ECE (Expected Calibration Error) metrics.

This module provides:
- AUROC (Area Under ROC Curve) computation
- ECE (Expected Calibration Error) computation
- Model evaluation utilities
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model predictions using AUROC and ECE metrics."""
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize the evaluator.
        
        Args:
            n_bins: Number of bins for ECE calculation
        """
        self.n_bins = n_bins
    
    def compute_auroc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute Area Under ROC Curve.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred_proba: Predicted probabilities
            
        Returns:
            AUROC score
        """
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in y_true, AUROC is undefined")
            return np.nan
        
        auroc = roc_auc_score(y_true, y_pred_proba)
        return float(auroc)
    
    def compute_ece(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures the difference between predicted probabilities and
        actual outcomes, binned by confidence level.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred_proba: Predicted probabilities
            
        Returns:
            ECE score (lower is better)
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(y_pred_proba)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                # Compute accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Compute average confidence in this bin
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                # Add weighted difference to ECE
                ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
        
        return float(ece)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with AUROC and ECE scores
        """
        auroc = self.compute_auroc(y_true, y_pred_proba)
        ece = self.compute_ece(y_true, y_pred_proba)
        
        # Also compute accuracy at 0.5 threshold
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        accuracy = (y_pred_binary == y_true).mean()
        
        metrics = {
            'auroc': auroc,
            'ece': ece,
            'accuracy': float(accuracy)
        }
        
        logger.info("Evaluation metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return metrics
    
    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve data.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (mean_predicted_prob, fraction_of_positives) for each bin
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mean_predicted = []
        fraction_positive = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                mean_predicted.append(y_pred_proba[in_bin].mean())
                fraction_positive.append(y_true[in_bin].mean())
            else:
                mean_predicted.append((bin_lower + bin_upper) / 2)
                fraction_positive.append(0.0)
        
        return np.array(mean_predicted), np.array(fraction_positive)


def load_predictions(
    predictions_path: Path,
    labels_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and labels from disk.
    
    Args:
        predictions_path: Path to predicted probabilities (.npy)
        labels_path: Path to true labels (.npy)
        
    Returns:
        Tuple of (y_true, y_pred_proba)
    """
    y_pred_proba = np.load(predictions_path)
    y_true = np.load(labels_path)
    
    # Ensure predictions are probabilities (between 0 and 1)
    if y_pred_proba.min() < 0 or y_pred_proba.max() > 1:
        logger.warning("Predictions not in [0, 1] range, clipping values")
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Ensure labels are binary
    unique_labels = np.unique(y_true)
    if not set(unique_labels).issubset({0, 1}):
        logger.warning(f"Labels contain non-binary values: {unique_labels}")
    
    logger.info(f"Loaded {len(y_true)} samples")
    logger.info(f"Positive class proportion: {y_true.mean():.3f}")
    
    return y_true, y_pred_proba


def evaluate_model(
    predictions_path: Path,
    labels_path: Path,
    output_path: Optional[Path] = None,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate model predictions and optionally save results.
    
    Args:
        predictions_path: Path to predicted probabilities
        labels_path: Path to true labels
        output_path: Optional path to save evaluation results (JSON)
        n_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load data
    y_true, y_pred_proba = load_predictions(predictions_path, labels_path)
    
    # Evaluate
    evaluator = ModelEvaluator(n_bins=n_bins)
    metrics = evaluator.evaluate(y_true, y_pred_proba)
    
    # Save results if output path provided
    if output_path is not None:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return metrics


def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions using AUROC and ECE metrics"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predicted probabilities (.npy file)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to true labels (.npy file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON file)"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for ECE calculation"
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(
        predictions_path=Path(args.predictions),
        labels_path=Path(args.labels),
        output_path=Path(args.output) if args.output else None,
        n_bins=args.n_bins
    )
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
