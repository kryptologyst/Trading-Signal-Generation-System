"""Feature engineering pipeline for trading signals."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for trading signal generation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary for feature engineering.
        """
        self.config = config or {}
        self.technical_indicators = TechnicalIndicators()
        self.feature_names = []
    
    def create_technical_features(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """Create technical indicator features for all symbols.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbols: List of symbols to create features for.
            
        Returns:
            DataFrame with technical indicator features.
        """
        logger.info(f"Creating technical features for {len(symbols)} symbols")
        
        all_features = []
        
        for symbol in symbols:
            if (symbol, 'Close') in data.columns:
                symbol_features = self.technical_indicators.calculate_all_indicators(
                    data, symbol, self.config.get('technical_indicators')
                )
                
                # Add symbol prefix to column names
                symbol_features.columns = [f"{symbol}_{col}" for col in symbol_features.columns]
                all_features.append(symbol_features)
        
        if all_features:
            features_df = pd.concat(all_features, axis=1)
            self.feature_names.extend(features_df.columns.tolist())
            logger.info(f"Created {len(features_df.columns)} technical features")
            return features_df
        else:
            logger.warning("No technical features created")
            return pd.DataFrame(index=data.index)
    
    def create_price_features(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """Create price-based features.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbols: List of symbols to create features for.
            
        Returns:
            DataFrame with price features.
        """
        logger.info("Creating price features")
        
        price_features = []
        
        for symbol in symbols:
            if (symbol, 'Close') in data.columns:
                close = data[(symbol, 'Close')]
                high = data[(symbol, 'High')]
                low = data[(symbol, 'Low')]
                open_price = data[(symbol, 'Open')]
                
                symbol_features = pd.DataFrame(index=data.index)
                
                # Price ratios
                symbol_features[f"{symbol}_High_Low_Ratio"] = high / low
                symbol_features[f"{symbol}_Close_Open_Ratio"] = close / open_price
                symbol_features[f"{symbol}_High_Close_Ratio"] = high / close
                symbol_features[f"{symbol}_Low_Close_Ratio"] = low / close
                
                # Price changes
                for window in [1, 2, 5, 10, 20]:
                    symbol_features[f"{symbol}_Price_Change_{window}"] = close.pct_change(window)
                    symbol_features[f"{symbol}_Log_Return_{window}"] = np.log(close / close.shift(window))
                
                # Price volatility
                for window in [5, 10, 20]:
                    returns = close.pct_change()
                    symbol_features[f"{symbol}_Volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252)
                    symbol_features[f"{symbol}_Skewness_{window}"] = returns.rolling(window).skew()
                    symbol_features[f"{symbol}_Kurtosis_{window}"] = returns.rolling(window).kurt()
                
                # Price momentum
                for window in [5, 10, 20]:
                    symbol_features[f"{symbol}_Momentum_{window}"] = close / close.shift(window) - 1
                    symbol_features[f"{symbol}_ROC_{window}"] = (close - close.shift(window)) / close.shift(window)
                
                # Gap features
                symbol_features[f"{symbol}_Gap"] = open_price - close.shift(1)
                symbol_features[f"{symbol}_Gap_Ratio"] = symbol_features[f"{symbol}_Gap"] / close.shift(1)
                
                price_features.append(symbol_features)
        
        if price_features:
            features_df = pd.concat(price_features, axis=1)
            self.feature_names.extend(features_df.columns.tolist())
            logger.info(f"Created {len(features_df.columns)} price features")
            return features_df
        else:
            logger.warning("No price features created")
            return pd.DataFrame(index=data.index)
    
    def create_volume_features(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """Create volume-based features.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbols: List of symbols to create features for.
            
        Returns:
            DataFrame with volume features.
        """
        logger.info("Creating volume features")
        
        volume_features = []
        
        for symbol in symbols:
            if (symbol, 'Volume') in data.columns and (symbol, 'Close') in data.columns:
                volume = data[(symbol, 'Volume')]
                close = data[(symbol, 'Close')]
                
                symbol_features = pd.DataFrame(index=data.index)
                
                # Volume moving averages
                for window in [5, 10, 20, 50]:
                    symbol_features[f"{symbol}_Volume_MA_{window}"] = volume.rolling(window).mean()
                    symbol_features[f"{symbol}_Volume_MA_{window}_Ratio"] = volume / volume.rolling(window).mean()
                
                # Volume volatility
                for window in [5, 10, 20]:
                    symbol_features[f"{symbol}_Volume_Volatility_{window}"] = volume.rolling(window).std()
                    symbol_features[f"{symbol}_Volume_CV_{window}"] = (
                        volume.rolling(window).std() / volume.rolling(window).mean()
                    )
                
                # Price-volume features
                symbol_features[f"{symbol}_Price_Volume"] = close * volume
                symbol_features[f"{symbol}_Volume_Price_Change"] = volume * close.pct_change()
                
                # Volume momentum
                for window in [5, 10, 20]:
                    symbol_features[f"{symbol}_Volume_Momentum_{window}"] = volume / volume.shift(window) - 1
                
                # Volume patterns
                symbol_features[f"{symbol}_Volume_Spike"] = volume > volume.rolling(20).quantile(0.9)
                symbol_features[f"{symbol}_Volume_Dry"] = volume < volume.rolling(20).quantile(0.1)
                
                volume_features.append(symbol_features)
        
        if volume_features:
            features_df = pd.concat(volume_features, axis=1)
            self.feature_names.extend(features_df.columns.tolist())
            logger.info(f"Created {len(features_df.columns)} volume features")
            return features_df
        else:
            logger.warning("No volume features created")
            return pd.DataFrame(index=data.index)
    
    def create_time_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            
        Returns:
            DataFrame with time features.
        """
        logger.info("Creating time features")
        
        time_features = pd.DataFrame(index=data.index)
        
        # Basic time features
        time_features['Year'] = data.index.year
        time_features['Month'] = data.index.month
        time_features['Day'] = data.index.day
        time_features['DayOfWeek'] = data.index.dayofweek
        time_features['DayOfYear'] = data.index.dayofyear
        time_features['Quarter'] = data.index.quarter
        time_features['Week'] = data.index.isocalendar().week
        
        # Cyclical encoding
        time_features['Month_Sin'] = np.sin(2 * np.pi * time_features['Month'] / 12)
        time_features['Month_Cos'] = np.cos(2 * np.pi * time_features['Month'] / 12)
        time_features['DayOfWeek_Sin'] = np.sin(2 * np.pi * time_features['DayOfWeek'] / 7)
        time_features['DayOfWeek_Cos'] = np.cos(2 * np.pi * time_features['DayOfWeek'] / 7)
        time_features['DayOfYear_Sin'] = np.sin(2 * np.pi * time_features['DayOfYear'] / 365)
        time_features['DayOfYear_Cos'] = np.cos(2 * np.pi * time_features['DayOfYear'] / 365)
        
        # Market session features
        time_features['Is_Monday'] = (time_features['DayOfWeek'] == 0).astype(int)
        time_features['Is_Friday'] = (time_features['DayOfWeek'] == 4).astype(int)
        time_features['Is_Month_End'] = (data.index.day >= 28).astype(int)
        time_features['Is_Quarter_End'] = time_features['Is_Month_End'] & (time_features['Month'].isin([3, 6, 9, 12])).astype(int)
        
        # Holiday proximity (simplified)
        time_features['Days_From_Month_End'] = 31 - data.index.day
        time_features['Days_From_Quarter_End'] = np.where(
            time_features['Month'].isin([3, 6, 9, 12]),
            31 - data.index.day,
            np.nan
        )
        
        self.feature_names.extend(time_features.columns.tolist())
        logger.info(f"Created {len(time_features.columns)} time features")
        return time_features
    
    def create_cross_asset_features(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """Create cross-asset features.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbols: List of symbols to create features for.
            
        Returns:
            DataFrame with cross-asset features.
        """
        logger.info("Creating cross-asset features")
        
        if len(symbols) < 2:
            logger.warning("Need at least 2 symbols for cross-asset features")
            return pd.DataFrame(index=data.index)
        
        cross_features = pd.DataFrame(index=data.index)
        
        # Get close prices for all symbols
        close_prices = {}
        for symbol in symbols:
            if (symbol, 'Close') in data.columns:
                close_prices[symbol] = data[(symbol, 'Close')]
        
        if len(close_prices) < 2:
            logger.warning("Not enough close price data for cross-asset features")
            return pd.DataFrame(index=data.index)
        
        # Price ratios between assets
        symbol_list = list(close_prices.keys())
        for i, symbol1 in enumerate(symbol_list):
            for symbol2 in symbol_list[i+1:]:
                cross_features[f"{symbol1}_{symbol2}_Ratio"] = close_prices[symbol1] / close_prices[symbol2]
                cross_features[f"{symbol1}_{symbol2}_Spread"] = close_prices[symbol1] - close_prices[symbol2]
                cross_features[f"{symbol1}_{symbol2}_Log_Ratio"] = np.log(close_prices[symbol1] / close_prices[symbol2])
        
        # Market-wide features
        all_prices = pd.DataFrame(close_prices)
        cross_features['Market_Mean'] = all_prices.mean(axis=1)
        cross_features['Market_Std'] = all_prices.std(axis=1)
        cross_features['Market_Min'] = all_prices.min(axis=1)
        cross_features['Market_Max'] = all_prices.max(axis=1)
        
        # Relative strength
        for symbol in symbol_list:
            cross_features[f"{symbol}_Relative_Strength"] = close_prices[symbol] / cross_features['Market_Mean']
            cross_features[f"{symbol}_Market_Rank"] = all_prices.rank(axis=1)[symbol]
        
        self.feature_names.extend(cross_features.columns.tolist())
        logger.info(f"Created {len(cross_features.columns)} cross-asset features")
        return cross_features
    
    def create_lag_features(
        self,
        features: pd.DataFrame,
        lags: List[int] = [1, 2, 3, 5, 10, 20]
    ) -> pd.DataFrame:
        """Create lagged features.
        
        Args:
            features: Feature DataFrame.
            lags: List of lag periods.
            
        Returns:
            DataFrame with lagged features.
        """
        logger.info(f"Creating lag features for lags: {lags}")
        
        lag_features = features.copy()
        
        for lag in lags:
            for col in features.columns:
                lag_features[f"{col}_Lag_{lag}"] = features[col].shift(lag)
        
        self.feature_names.extend([f"{col}_Lag_{lag}" for col in features.columns for lag in lags])
        logger.info(f"Created {len(lag_features.columns) - len(features.columns)} lag features")
        return lag_features
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        include_technical: bool = True,
        include_price: bool = True,
        include_volume: bool = True,
        include_time: bool = True,
        include_cross_asset: bool = True,
        include_lags: bool = True,
        lags: List[int] = [1, 2, 3, 5, 10, 20]
    ) -> pd.DataFrame:
        """Complete feature engineering pipeline.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbols: List of symbols to create features for.
            include_technical: Whether to include technical indicators.
            include_price: Whether to include price features.
            include_volume: Whether to include volume features.
            include_time: Whether to include time features.
            include_cross_asset: Whether to include cross-asset features.
            include_lags: Whether to include lagged features.
            lags: List of lag periods.
            
        Returns:
            DataFrame with all engineered features.
        """
        logger.info("Starting feature engineering pipeline")
        
        all_features = []
        
        # Technical indicators
        if include_technical:
            tech_features = self.create_technical_features(data, symbols)
            all_features.append(tech_features)
        
        # Price features
        if include_price:
            price_features = self.create_price_features(data, symbols)
            all_features.append(price_features)
        
        # Volume features
        if include_volume:
            volume_features = self.create_volume_features(data, symbols)
            all_features.append(volume_features)
        
        # Time features
        if include_time:
            time_features = self.create_time_features(data)
            all_features.append(time_features)
        
        # Cross-asset features
        if include_cross_asset and len(symbols) > 1:
            cross_features = self.create_cross_asset_features(data, symbols)
            all_features.append(cross_features)
        
        # Combine all features
        if all_features:
            features_df = pd.concat(all_features, axis=1)
            
            # Add lag features
            if include_lags:
                features_df = self.create_lag_features(features_df, lags)
            
            # Remove any columns with all NaN values
            features_df = features_df.dropna(axis=1, how='all')
            
            logger.info(f"Feature engineering completed. Final shape: {features_df.shape}")
            return features_df
        else:
            logger.warning("No features created")
            return pd.DataFrame(index=data.index)
