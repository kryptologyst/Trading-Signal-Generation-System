#!/usr/bin/env python3
"""Main script for trading signal generation system.

This script demonstrates the complete pipeline for generating trading signals
using technical indicators and machine learning models.

DISCLAIMER: This software is for educational and research purposes only.
It is not intended as investment advice and should not be used for actual trading
without proper risk management and professional consultation.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import Config, load_config, setup_logging, set_seeds
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, FeatureSelector
from src.labels import create_labels
from src.models import ModelFactory, TechnicalStrategy, XGBoostModel, LightGBMModel
from src.backtest import Backtester

# Register models
ModelFactory.register_model("technical", TechnicalStrategy)
ModelFactory.register_model("xgboost", XGBoostModel)
ModelFactory.register_model("lightgbm", LightGBMModel)


def main():
    """Main function to run the trading signal generation pipeline."""
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting trading signal generation pipeline")
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Load configuration
    config = load_config("configs/config.yaml")
    
    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    data_loader = DataLoader(cache_dir="data/cache")
    
    # Load market data
    market_data = data_loader.load_yfinance_data(
        symbols=config.data.symbols,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        use_cache=True
    )
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(
        market_data,
        add_returns=True,
        add_volatility=True,
        add_volume_features=True
    )
    
    logger.info(f"Loaded data shape: {processed_data.shape}")
    
    # Step 2: Feature engineering
    logger.info("Step 2: Feature engineering")
    feature_engineer = FeatureEngineer(config.features.__dict__)
    
    features = feature_engineer.engineer_features(
        data=processed_data,
        symbols=config.data.symbols,
        include_technical=True,
        include_price=True,
        include_volume=True,
        include_time=True,
        include_cross_asset=True,
        include_lags=True
    )
    
    logger.info(f"Created features shape: {features.shape}")
    
    # Step 3: Generate labels
    logger.info("Step 3: Generating labels")
    labels = create_labels(
        data=processed_data,
        symbols=config.data.symbols,
        method="macd_rsi",
        label_type="discrete"
    )
    
    logger.info(f"Created labels shape: {labels.shape}")
    
    # Step 4: Feature selection
    logger.info("Step 4: Feature selection")
    feature_selector = FeatureSelector()
    
    # Select features for the first symbol
    first_symbol = config.data.symbols[0]
    symbol_features = features[[col for col in features.columns if first_symbol in col]]
    symbol_labels = labels[[col for col in labels.columns if first_symbol in col and 'Signal' in col]]
    
    if not symbol_labels.empty:
        selected_features = feature_selector.select_features(
            X=symbol_features,
            y=symbol_labels.iloc[:, 0],
            method="mutual_info",
            n_features=50
        )
        logger.info(f"Selected {len(selected_features.columns)} features")
    else:
        selected_features = symbol_features
        logger.warning("No labels found, using all features")
    
    # Step 5: Train models
    logger.info("Step 5: Training models")
    
    # Split data for training and testing
    split_point = int(len(selected_features) * 0.8)
    X_train = selected_features.iloc[:split_point]
    X_test = selected_features.iloc[split_point:]
    
    if not symbol_labels.empty:
        y_train = symbol_labels.iloc[:split_point, 0]
        y_test = symbol_labels.iloc[split_point:, 0]
    else:
        # Create synthetic labels for demonstration
        y_train = pd.Series(np.random.choice([-1, 0, 1], size=len(X_train)), index=X_train.index)
        y_test = pd.Series(np.random.choice([-1, 0, 1], size=len(X_test)), index=X_test.index)
    
    # Train multiple models
    models = {}
    model_configs = {
        "technical": {},
        "xgboost": {"xgb_params": config.model.xgb_params},
        "lightgbm": {"lgb_params": config.model.lgb_params}
    }
    
    for model_name in config.model.models:
        if model_name in model_configs:
            logger.info(f"Training {model_name} model")
            model = ModelFactory.create_model(model_name, model_configs[model_name])
            model.fit(X_train, y_train)
            models[model_name] = model
            
            # Evaluate model
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            
            logger.info(f"{model_name} - Train metrics: {train_metrics}")
            logger.info(f"{model_name} - Test metrics: {test_metrics}")
    
    # Step 6: Backtesting
    logger.info("Step 6: Running backtests")
    
    backtest_config = {
        'initial_capital': config.backtest.initial_capital,
        'transaction_cost': config.backtest.transaction_cost,
        'slippage': config.backtest.slippage,
        'max_position_size': config.backtest.max_position_size
    }
    
    backtester = Backtester(backtest_config)
    
    # Generate signals for backtesting
    test_signals = pd.DataFrame(index=X_test.index)
    for model_name, model in models.items():
        predictions = model.predict(X_test)
        test_signals[f"{first_symbol}_{model_name}_Signal"] = predictions
    
    # Run backtest
    backtest_results = backtester.run_backtest(
        data=processed_data,
        signals=test_signals,
        symbols=[first_symbol],
        start_date=X_test.index[0].strftime('%Y-%m-%d'),
        end_date=X_test.index[-1].strftime('%Y-%m-%d')
    )
    
    # Step 7: Display results
    logger.info("Step 7: Displaying results")
    
    print("\n" + "="*60)
    print("TRADING SIGNAL GENERATION RESULTS")
    print("="*60)
    
    print(f"\nData Summary:")
    print(f"  Total data points: {len(processed_data)}")
    print(f"  Features created: {len(features.columns)}")
    print(f"  Selected features: {len(selected_features.columns)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    print(f"\nModel Performance:")
    for model_name, model in models.items():
        test_metrics = model.evaluate(X_test, y_test)
        print(f"  {model_name.upper()}:")
        for metric, value in test_metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    print(f"\nBacktest Results:")
    if backtest_results:
        trading_metrics = backtest_results.get('trading_metrics', {})
        print(f"  Total Return: {backtest_results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {backtest_results.get('win_rate', 0):.2%}")
        print(f"  Number of Trades: {backtest_results.get('num_trades', 0)}")
    
    print(f"\nDISCLAIMER:")
    print(f"  This software is for educational and research purposes only.")
    print(f"  It is not intended as investment advice and should not be used")
    print(f"  for actual trading without proper risk management and")
    print(f"  professional consultation.")
    
    print("\n" + "="*60)
    
    # Save results
    logger.info("Saving results")
    
    # Save features and labels
    features.to_parquet("outputs/features.parquet")
    labels.to_parquet("outputs/labels.parquet")
    
    # Save model predictions
    predictions_df = pd.DataFrame(index=X_test.index)
    for model_name, model in models.items():
        predictions_df[f"{model_name}_predictions"] = model.predict(X_test)
    predictions_df.to_parquet("outputs/predictions.parquet")
    
    # Save backtest results
    if backtest_results:
        backtester.save_results("outputs/backtest_results.json")
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
