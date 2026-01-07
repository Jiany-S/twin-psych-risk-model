"""
Configuration management for the risk forecasting project.

This module provides default configurations and utilities for managing
experiment settings.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json
import yaml


@dataclass
class DataConfig:
    """Configuration for data processing."""
    window_size: int = 30
    forecast_horizon: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    

@dataclass
class ModelConfig:
    """Configuration for TFT model."""
    hidden_size: int = 64
    num_attention_heads: int = 4
    dropout: float = 0.1
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'seed': self.seed
        }
    
    def save(self, path: Path) -> None:
        """Save configuration to file (JSON or YAML)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from file."""
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            seed=config_dict.get('seed', 42)
        )
    
    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Create default configuration."""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig()
        )


# Create a default config instance
DEFAULT_CONFIG = ExperimentConfig.default()
