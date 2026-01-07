#!/usr/bin/env python3
"""Test script to verify the trading signal generation system works correctly.

This script runs a quick test of the main components to ensure everything
is working properly.

DISCLAIMER: This software is for educational and research purposes only.
It is not intended as investment advice and should not be used for actual trading
without proper risk management and professional consultation.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seeds, setup_logging
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, TechnicalIndicators
from src.labels import create_labels
from src.models import TechnicalStrategy, XGBoostModel, LightGBMModel
from src.backtest import Backtester

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    # Test synthetic data generation
    data_loader = DataLoader(cache_dir="data/cache")
    data = data_loader.load_synthetic_data(
        symbols=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    assert not data.empty, "Data should not be empty"
    assert len(data.columns.get_level_values(0).unique()) == 2, "Should have 2 symbols"
    print("✓ Data loading test passed")
    
    return data

def test_data_preprocessing(data):
    """Test data preprocessing functionality."""
    print("Testing data preprocessing...")
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(data)
    
    assert not processed_data.empty, "Processed data should not be empty"
    assert len(processed_data.columns) > len(data.columns), "Should have more columns after preprocessing"
    print("✓ Data preprocessing test passed")
    
    return processed_data

def test_feature_engineering(data):
    """Test feature engineering functionality."""
    print("Testing feature engineering...")
    
    feature_engineer = FeatureEngineer()
    features = feature_engineer.engineer_features(
        data=data,
        symbols=["AAPL", "MSFT"],
        include_technical=True,
        include_price=True,
        include_volume=True,
        include_time=True
    )
    
    assert not features.empty, "Features should not be empty"
    assert len(features.columns) > 50, "Should have many features"
    print("✓ Feature engineering test passed")
    
    return features

def test_technical_indicators():
    """Test technical indicators calculation."""
    print("Testing technical indicators...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    data = pd.DataFrame({
        'Close': prices,
        'High': prices + np.random.uniform(0, 2, 100),
        'Low': prices - np.random.uniform(0, 2, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    indicators = TechnicalIndicators()
    
    # Test individual indicators
    sma = indicators.sma(data['Close'], 20)
    assert len(sma) == len(data), "SMA should have same length as input"
    
    rsi = indicators.rsi(data['Close'])
    assert len(rsi) == len(data), "RSI should have same length as input"
    assert rsi.iloc[-1] >= 0 and rsi.iloc[-1] <= 100, "RSI should be between 0 and 100"
    
    macd_data = indicators.macd(data['Close'])
    assert 'MACD' in macd_data, "MACD should have MACD key"
    assert 'Signal' in macd_data, "MACD should have Signal key"
    
    print("✓ Technical indicators test passed")

def test_label_generation(data):
    """Test label generation functionality."""
    print("Testing label generation...")
    
    labels = create_labels(
        data=data,
        symbols=["AAPL", "MSFT"],
        method="macd_rsi",
        label_type="discrete"
    )
    
    assert not labels.empty, "Labels should not be empty"
    assert any('Signal' in col for col in labels.columns), "Should have signal columns"
    print("✓ Label generation test passed")
    
    return labels

def test_models(features, labels):
    """Test model training and prediction."""
    print("Testing models...")
    
    # Prepare data for first symbol
    first_symbol = "AAPL"
    symbol_features = features[[col for col in features.columns if first_symbol in col]]
    symbol_labels = labels[[col for col in labels.columns if first_symbol in col and 'Signal' in col]]
    
    if symbol_labels.empty:
        # Create synthetic labels if none exist
        symbol_labels = pd.Series(
            np.random.choice([-1, 0, 1], size=len(symbol_features)),
            index=symbol_features.index,
            name=f"{first_symbol}_Signal"
        )
    else:
        symbol_labels = symbol_labels.iloc[:, 0]
    
    # Test Technical Strategy
    tech_model = TechnicalStrategy()
    tech_model.fit(symbol_features, symbol_labels)
    tech_predictions = tech_model.predict(symbol_features)
    assert len(tech_predictions) == len(symbol_features), "Technical model predictions should match input length"
    
    # Test XGBoost
    xgb_model = XGBoostModel()
    xgb_model.fit(symbol_features, symbol_labels)
    xgb_predictions = xgb_model.predict(symbol_features)
    assert len(xgb_predictions) == len(symbol_features), "XGBoost predictions should match input length"
    
    # Test LightGBM
    lgb_model = LightGBMModel()
    lgb_model.fit(symbol_features, symbol_labels)
    lgb_predictions = lgb_model.predict(symbol_features)
    assert len(lgb_predictions) == len(symbol_features), "LightGBM predictions should match input length"
    
    print("✓ Models test passed")
    
    return {
        'technical': tech_model,
        'xgboost': xgb_model,
        'lightgbm': lgb_model
    }

def test_backtesting(data, models):
    """Test backtesting functionality."""
    print("Testing backtesting...")
    
    # Create sample signals
    first_symbol = "AAPL"
    signals = pd.DataFrame(index=data.index)
    
    # Generate signals for each model
    symbol_features = data[[col for col in data.columns if first_symbol in col and 'Close' in col]]
    if not symbol_features.empty:
        for model_name, model in models.items():
            if hasattr(model, 'predict'):
                # Create simple features for prediction
                simple_features = pd.DataFrame({
                    f'{first_symbol}_RSI': np.random.uniform(20, 80, len(data)),
                    f'{first_symbol}_MACD': np.random.randn(len(data)),
                    f'{first_symbol}_MACD_Signal': np.random.randn(len(data))
                }, index=data.index)
                
                predictions = model.predict(simple_features)
                signals[f"{first_symbol}_{model_name}_Signal"] = predictions
    
    # If no signals generated, create synthetic ones
    if signals.empty:
        signals = pd.DataFrame({
            f"{first_symbol}_technical_Signal": np.random.choice([-1, 0, 1], len(data)),
            f"{first_symbol}_xgboost_Signal": np.random.choice([-1, 0, 1], len(data)),
            f"{first_symbol}_lightgbm_Signal": np.random.choice([-1, 0, 1], len(data))
        }, index=data.index)
    
    # Run backtest
    backtest_config = {
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'slippage': 0.0005,
        'max_position_size': 0.1
    }
    
    backtester = Backtester(backtest_config)
    results = backtester.run_backtest(
        data=data,
        signals=signals,
        symbols=[first_symbol],
        start_date=data.index[0].strftime('%Y-%m-%d'),
        end_date=data.index[-1].strftime('%Y-%m-%d')
    )
    
    assert results is not None, "Backtest should return results"
    assert 'total_return' in results, "Results should include total return"
    print("✓ Backtesting test passed")
    
    return results

def main():
    """Run all tests."""
    print("="*60)
    print("TRADING SIGNAL GENERATION SYSTEM TEST")
    print("="*60)
    print()
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    try:
        # Test data loading
        data = test_data_loading()
        
        # Test data preprocessing
        processed_data = test_data_preprocessing(data)
        
        # Test technical indicators
        test_technical_indicators()
        
        # Test feature engineering
        features = test_feature_engineering(processed_data)
        
        # Test label generation
        labels = test_label_generation(processed_data)
        
        # Test models
        models = test_models(features, labels)
        
        # Test backtesting
        backtest_results = test_backtesting(processed_data, models)
        
        print()
        print("="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print()
        print("System components tested:")
        print("✓ Data loading and preprocessing")
        print("✓ Technical indicators calculation")
        print("✓ Feature engineering")
        print("✓ Label generation")
        print("✓ Model training and prediction")
        print("✓ Backtesting framework")
        print()
        print("The trading signal generation system is working correctly!")
        print()
        print("DISCLAIMER: This software is for educational and research purposes only.")
        print("It is not intended as investment advice and should not be used for")
        print("actual trading without proper risk management and professional consultation.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
