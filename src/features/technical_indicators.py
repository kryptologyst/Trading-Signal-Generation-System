"""Technical indicators for trading signal generation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators calculator for financial data."""
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        pass
    
    def sma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average.
        
        Args:
            data: Price series.
            window: Window size.
            
        Returns:
            SMA series.
        """
        return data.rolling(window=window).mean()
    
    def ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Price series.
            window: Window size.
            
        Returns:
            EMA series.
        """
        return data.ewm(span=window, adjust=False).mean()
    
    def macd(
        self,
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line EMA period.
            
        Returns:
            Dictionary with MACD, signal line, and histogram.
        """
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index).
        
        Args:
            data: Price series.
            window: Window size.
            
        Returns:
            RSI series.
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def bollinger_bands(
        self,
        data: pd.Series,
        window: int = 20,
        std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            data: Price series.
            window: Window size.
            std: Standard deviation multiplier.
            
        Returns:
            Dictionary with upper, middle, and lower bands.
        """
        sma = self.sma(data, window)
        std_dev = data.rolling(window=window).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band
        }
    
    def stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3
    ) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator.
        
        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            k_window: %K window size.
            d_window: %D window size.
            
        Returns:
            Dictionary with %K and %D.
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'K': k_percent,
            'D': d_percent
        }
    
    def williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Williams %R.
        
        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            window: Window size.
            
        Returns:
            Williams %R series.
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Average True Range.
        
        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            window: Window size.
            
        Returns:
            ATR series.
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> Dict[str, pd.Series]:
        """Calculate ADX (Average Directional Index).
        
        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            window: Window size.
            
        Returns:
            Dictionary with ADX, +DI, and -DI.
        """
        # Calculate directional movements
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Calculate ATR
        atr = self.atr(high, low, close, window)
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        }
    
    def cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate Commodity Channel Index.
        
        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            window: Window size.
            
        Returns:
            CCI series.
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume.
        
        Args:
            close: Close price series.
            volume: Volume series.
            
        Returns:
            OBV series.
        """
        price_change = close.diff()
        obv = np.where(
            price_change > 0,
            volume,
            np.where(price_change < 0, -volume, 0)
        )
        
        return pd.Series(obv, index=close.index).cumsum()
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price.
        
        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            volume: Volume series.
            
        Returns:
            VWAP series.
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    def calculate_all_indicators(
        self,
        data: pd.DataFrame,
        symbol: str,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Calculate all technical indicators for a symbol.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to calculate indicators for.
            config: Configuration dictionary for indicator parameters.
            
        Returns:
            DataFrame with all technical indicators.
        """
        if config is None:
            config = {
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'rsi': {'window': 14},
                'bollinger': {'window': 20, 'std': 2.0},
                'stochastic': {'k_window': 14, 'd_window': 3},
                'williams_r': {'window': 14},
                'atr': {'window': 14},
                'adx': {'window': 14},
                'cci': {'window': 20}
            }
        
        logger.info(f"Calculating technical indicators for {symbol}")
        
        # Extract OHLCV data
        close = data[(symbol, 'Close')]
        high = data[(symbol, 'High')]
        low = data[(symbol, 'Low')]
        open_price = data[(symbol, 'Open')]
        volume = data[(symbol, 'Volume')]
        
        indicators = pd.DataFrame(index=data.index)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            indicators[f'SMA_{window}'] = self.sma(close, window)
            indicators[f'EMA_{window}'] = self.ema(close, window)
        
        # MACD
        macd_data = self.macd(
            close,
            config['macd']['fast'],
            config['macd']['slow'],
            config['macd']['signal']
        )
        indicators['MACD'] = macd_data['MACD']
        indicators['MACD_Signal'] = macd_data['Signal']
        indicators['MACD_Histogram'] = macd_data['Histogram']
        
        # RSI
        indicators['RSI'] = self.rsi(close, config['rsi']['window'])
        
        # Bollinger Bands
        bb_data = self.bollinger_bands(
            close,
            config['bollinger']['window'],
            config['bollinger']['std']
        )
        indicators['BB_Upper'] = bb_data['Upper']
        indicators['BB_Middle'] = bb_data['Middle']
        indicators['BB_Lower'] = bb_data['Lower']
        indicators['BB_Width'] = (bb_data['Upper'] - bb_data['Lower']) / bb_data['Middle']
        indicators['BB_Position'] = (close - bb_data['Lower']) / (bb_data['Upper'] - bb_data['Lower'])
        
        # Stochastic
        stoch_data = self.stochastic(
            high, low, close,
            config['stochastic']['k_window'],
            config['stochastic']['d_window']
        )
        indicators['Stoch_K'] = stoch_data['K']
        indicators['Stoch_D'] = stoch_data['D']
        
        # Williams %R
        indicators['Williams_R'] = self.williams_r(high, low, close, config['williams_r']['window'])
        
        # ATR
        indicators['ATR'] = self.atr(high, low, close, config['atr']['window'])
        
        # ADX
        adx_data = self.adx(high, low, close, config['adx']['window'])
        indicators['ADX'] = adx_data['ADX']
        indicators['Plus_DI'] = adx_data['Plus_DI']
        indicators['Minus_DI'] = adx_data['Minus_DI']
        
        # CCI
        indicators['CCI'] = self.cci(high, low, close, config['cci']['window'])
        
        # Volume indicators
        indicators['OBV'] = self.obv(close, volume)
        indicators['VWAP'] = self.vwap(high, low, close, volume)
        
        # Price ratios
        indicators['Price_SMA20_Ratio'] = close / indicators['SMA_20']
        indicators['Price_EMA20_Ratio'] = close / indicators['EMA_20']
        indicators['Price_VWAP_Ratio'] = close / indicators['VWAP']
        
        logger.info(f"Calculated {len(indicators.columns)} technical indicators for {symbol}")
        return indicators
