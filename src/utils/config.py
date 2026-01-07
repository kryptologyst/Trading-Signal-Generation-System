"""Configuration management for the trading signal generation system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"
    data_source: str = "yfinance"
    cache_dir: str = "data/cache"
    features_cache: str = "data/features.parquet"
    labels_cache: str = "data/labels.parquet"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    macd_params: Dict[str, int] = field(default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9})
    rsi_params: Dict[str, int] = field(default_factory=lambda: {"window": 14, "overbought": 70, "oversold": 30})
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    bollinger_params: Dict[str, Union[int, float]] = field(default_factory=lambda: {"window": 20, "std": 2.0})
    
    # Advanced features
    volume_features: bool = True
    volatility_features: bool = True
    momentum_features: bool = True
    microstructure_features: bool = False
    
    # Feature selection
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    
    # Model types
    models: List[str] = field(default_factory=lambda: ["technical", "xgboost", "lightgbm"])
    
    # Cross-validation
    cv_method: str = "time_series_split"
    n_splits: int = 5
    test_size: float = 0.2
    purged_pct: float = 0.01  # Purge percentage for overlapping samples
    embargo_pct: float = 0.01  # Embargo percentage for overlapping samples
    
    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    
    # XGBoost parameters
    xgb_params: Dict[str, Union[str, int, float]] = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    })
    
    # LightGBM parameters
    lgb_params: Dict[str, Union[str, int, float]] = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    })


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.1  # 10% max position per asset
    rebalance_frequency: str = "daily"
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_drawdown: Optional[float] = 0.2  # 20% max drawdown


@dataclass
class Config:
    """Main configuration class."""
    
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # General settings
    output_dir: str = "outputs"
    assets_dir: str = "assets"
    log_level: str = "INFO"
    device: str = "auto"  # auto, cpu, cuda, mps


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file or return default config.
    
    Args:
        config_path: Path to YAML configuration file. If None, returns default config.
        
    Returns:
        Config object with loaded or default settings.
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf for better handling
    omega_conf = OmegaConf.create(config_dict)
    
    # Convert back to our dataclass structure
    return OmegaConf.to_object(omega_conf)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        config_path: Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save as YAML
    config_dict = OmegaConf.structured(config)
    with open(config_path, 'w') as f:
        yaml.dump(OmegaConf.to_yaml(config_dict), f, default_flow_style=False)
