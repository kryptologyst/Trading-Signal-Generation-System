"""Utility functions for label generation and validation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def create_labels(
    data: pd.DataFrame,
    symbols: List[str],
    method: str = "macd_rsi",
    label_type: str = "discrete",
    **kwargs
) -> pd.DataFrame:
    """Create trading labels for multiple symbols.
    
    Args:
        data: Multi-index DataFrame with OHLCV data.
        symbols: List of symbols to create labels for.
        method: Labeling method ("macd_rsi", "ma_crossover", "bollinger_bands", "momentum", "triple_barrier", "fixed_horizon").
        label_type: Type of labels ("discrete", "continuous").
        **kwargs: Additional arguments for specific methods.
        
    Returns:
        DataFrame with labels for all symbols.
    """
    logger.info(f"Creating {method} labels for {len(symbols)} symbols")
    
    from .signal_generation import SignalGenerator, TripleBarrierMethod
    
    signal_generator = SignalGenerator()
    triple_barrier = TripleBarrierMethod()
    
    all_labels = []
    
    for symbol in symbols:
        if (symbol, 'Close') not in data.columns:
            logger.warning(f"No close price data for {symbol}")
            continue
        
        if method in ["macd_rsi", "ma_crossover", "bollinger_bands", "momentum"]:
            # Generate signals using technical indicators
            if method == "macd_rsi":
                symbol_labels = signal_generator.generate_macd_rsi_signals(data, symbol, **kwargs)
            elif method == "ma_crossover":
                symbol_labels = signal_generator.generate_moving_average_signals(data, symbol, **kwargs)
            elif method == "bollinger_bands":
                symbol_labels = signal_generator.generate_bollinger_bands_signals(data, symbol, **kwargs)
            elif method == "momentum":
                symbol_labels = signal_generator.generate_momentum_signals(data, symbol, **kwargs)
            
            # Add symbol prefix to column names
            symbol_labels.columns = [f"{symbol}_{col}" for col in symbol_labels.columns]
            all_labels.append(symbol_labels)
            
        elif method == "triple_barrier":
            # Generate labels using triple barrier method
            symbol_labels = triple_barrier.create_triple_barrier_labels(data, symbol, **kwargs)
            symbol_labels.columns = [f"{symbol}_{col}" for col in symbol_labels.columns]
            all_labels.append(symbol_labels)
            
        elif method == "fixed_horizon":
            # Generate labels using fixed horizon method
            symbol_labels = triple_barrier.create_fixed_horizon_labels(data, symbol, **kwargs)
            symbol_labels.columns = [f"{symbol}_{col}" for col in symbol_labels.columns]
            all_labels.append(symbol_labels)
        
        else:
            logger.error(f"Unknown labeling method: {method}")
            continue
    
    if all_labels:
        labels_df = pd.concat(all_labels, axis=1)
        logger.info(f"Created labels for {len(symbols)} symbols. Shape: {labels_df.shape}")
        return labels_df
    else:
        logger.error("No labels created")
        return pd.DataFrame(index=data.index)


def validate_labels(
    labels: pd.DataFrame,
    min_samples: int = 10,
    max_imbalance: float = 0.9
) -> Dict[str, any]:
    """Validate generated labels.
    
    Args:
        labels: DataFrame with labels.
        min_samples: Minimum number of samples per class.
        max_imbalance: Maximum class imbalance ratio.
        
    Returns:
        Dictionary with validation results.
    """
    logger.info("Validating labels")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Check for empty labels
    if labels.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Empty labels DataFrame")
        return validation_results
    
    # Check for all NaN labels
    if labels.isnull().all().all():
        validation_results['is_valid'] = False
        validation_results['issues'].append("All labels are NaN")
        return validation_results
    
    # Check each label column
    for col in labels.columns:
        if 'Label' in col or 'Signal' in col:
            col_data = labels[col].dropna()
            
            if len(col_data) == 0:
                validation_results['issues'].append(f"Column {col} has no valid data")
                continue
            
            # Check for sufficient samples
            if len(col_data) < min_samples:
                validation_results['issues'].append(f"Column {col} has insufficient samples: {len(col_data)}")
            
            # Check class distribution
            value_counts = col_data.value_counts()
            total_samples = len(col_data)
            
            for value, count in value_counts.items():
                imbalance = count / total_samples
                if imbalance > max_imbalance:
                    validation_results['issues'].append(
                        f"Column {col} has imbalanced class {value}: {imbalance:.2%}"
                    )
            
            # Store statistics
            validation_results['stats'][col] = {
                'total_samples': total_samples,
                'unique_values': len(value_counts),
                'value_counts': value_counts.to_dict(),
                'imbalance_ratio': value_counts.max() / total_samples
            }
    
    # Determine overall validity
    if validation_results['issues']:
        validation_results['is_valid'] = False
        logger.warning(f"Label validation found {len(validation_results['issues'])} issues")
    else:
        logger.info("Label validation passed")
    
    return validation_results


def balance_labels(
    labels: pd.DataFrame,
    method: str = "undersample",
    random_state: int = 42
) -> pd.DataFrame:
    """Balance class distribution in labels.
    
    Args:
        labels: DataFrame with labels.
        method: Balancing method ("undersample", "oversample", "smote").
        random_state: Random state for reproducibility.
        
    Returns:
        Balanced DataFrame.
    """
    logger.info(f"Balancing labels using {method} method")
    
    if method == "undersample":
        return _undersample_labels(labels, random_state)
    elif method == "oversample":
        return _oversample_labels(labels, random_state)
    elif method == "smote":
        return _smote_labels(labels, random_state)
    else:
        logger.error(f"Unknown balancing method: {method}")
        return labels


def _undersample_labels(labels: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Undersample majority class to balance labels."""
    from sklearn.utils import resample
    
    np.random.seed(random_state)
    
    balanced_labels = labels.copy()
    
    for col in labels.columns:
        if 'Label' in col or 'Signal' in col:
            col_data = labels[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Find minority class size
            value_counts = col_data.value_counts()
            minority_size = value_counts.min()
            
            # Undersample each class
            balanced_indices = []
            for value in value_counts.index:
                class_indices = col_data[col_data == value].index
                if len(class_indices) > minority_size:
                    undersampled_indices = resample(
                        class_indices,
                        n_samples=minority_size,
                        random_state=random_state
                    )
                    balanced_indices.extend(undersampled_indices)
                else:
                    balanced_indices.extend(class_indices)
            
            # Create balanced column
            balanced_col = pd.Series(index=labels.index, dtype=col_data.dtype)
            balanced_col.loc[balanced_indices] = col_data.loc[balanced_indices]
            balanced_labels[col] = balanced_col
    
    logger.info(f"Undersampled labels. Original shape: {labels.shape}, Balanced shape: {balanced_labels.shape}")
    return balanced_labels


def _oversample_labels(labels: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Oversample minority class to balance labels."""
    from sklearn.utils import resample
    
    np.random.seed(random_state)
    
    balanced_labels = labels.copy()
    
    for col in labels.columns:
        if 'Label' in col or 'Signal' in col:
            col_data = labels[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Find majority class size
            value_counts = col_data.value_counts()
            majority_size = value_counts.max()
            
            # Oversample each class
            balanced_indices = []
            for value in value_counts.index:
                class_indices = col_data[col_data == value].index
                if len(class_indices) < majority_size:
                    oversampled_indices = resample(
                        class_indices,
                        n_samples=majority_size,
                        random_state=random_state,
                        replace=True
                    )
                    balanced_indices.extend(oversampled_indices)
                else:
                    balanced_indices.extend(class_indices)
            
            # Create balanced column
            balanced_col = pd.Series(index=labels.index, dtype=col_data.dtype)
            balanced_col.loc[balanced_indices] = col_data.loc[balanced_indices]
            balanced_labels[col] = balanced_col
    
    logger.info(f"Oversampled labels. Original shape: {labels.shape}, Balanced shape: {balanced_labels.shape}")
    return balanced_labels


def _smote_labels(labels: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Apply SMOTE to balance labels."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.error("SMOTE not available. Install imbalanced-learn: pip install imbalanced-learn")
        return labels
    
    logger.info("SMOTE balancing not implemented for this use case. Using oversampling instead.")
    return _oversample_labels(labels, random_state)


def create_synthetic_labels(
    data: pd.DataFrame,
    symbols: List[str],
    n_samples: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """Create synthetic labels for testing purposes.
    
    Args:
        data: Multi-index DataFrame with OHLCV data.
        symbols: List of symbols to create labels for.
        n_samples: Number of samples to generate.
        random_state: Random state for reproducibility.
        
    Returns:
        DataFrame with synthetic labels.
    """
    logger.info(f"Creating synthetic labels for {len(symbols)} symbols")
    
    np.random.seed(random_state)
    
    synthetic_labels = pd.DataFrame(index=data.index)
    
    for symbol in symbols:
        if (symbol, 'Close') in data.columns:
            # Generate synthetic signals based on price movements
            close = data[(symbol, 'Close')]
            returns = close.pct_change()
            
            # Create synthetic signals with some correlation to actual returns
            noise = np.random.normal(0, 0.1, len(returns))
            synthetic_signal = returns + noise
            
            # Convert to discrete signals
            buy_threshold = np.percentile(synthetic_signal.dropna(), 70)
            sell_threshold = np.percentile(synthetic_signal.dropna(), 30)
            
            synthetic_labels[f"{symbol}_Signal"] = 0
            synthetic_labels.loc[synthetic_signal > buy_threshold, f"{symbol}_Signal"] = 1
            synthetic_labels.loc[synthetic_signal < sell_threshold, f"{symbol}_Signal"] = -1
            
            # Add some additional synthetic features
            synthetic_labels[f"{symbol}_Confidence"] = np.random.uniform(0, 1, len(data))
            synthetic_labels[f"{symbol}_Volatility"] = returns.rolling(20).std()
    
    logger.info(f"Created synthetic labels for {len(symbols)} symbols")
    return synthetic_labels
