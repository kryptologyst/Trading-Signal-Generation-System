"""Trading signal generation and labeling methods."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals using various methods."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize signal generator.
        
        Args:
            config: Configuration dictionary for signal generation.
        """
        self.config = config or {}
    
    def generate_macd_rsi_signals(
        self,
        data: pd.DataFrame,
        symbol: str,
        macd_params: Optional[Dict] = None,
        rsi_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Generate signals using MACD and RSI indicators.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to generate signals for.
            macd_params: MACD parameters.
            rsi_params: RSI parameters.
            
        Returns:
            DataFrame with trading signals.
        """
        if macd_params is None:
            macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        if rsi_params is None:
            rsi_params = {'window': 14, 'overbought': 70, 'oversold': 30}
        
        logger.info(f"Generating MACD-RSI signals for {symbol}")
        
        close = data[(symbol, 'Close')]
        
        # Calculate MACD
        ema_fast = close.ewm(span=macd_params['fast'], adjust=False).mean()
        ema_slow = close.ewm(span=macd_params['slow'], adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=macd_params['signal'], adjust=False).mean()
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_params['window']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_params['window']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['MACD'] = macd
        signals['MACD_Signal'] = signal_line
        signals['RSI'] = rsi
        
        # Buy signal: MACD crosses above signal line AND RSI is oversold
        signals['Buy_Signal'] = (
            (macd > signal_line) & 
            (macd.shift(1) <= signal_line.shift(1)) & 
            (rsi < rsi_params['oversold'])
        ).astype(int)
        
        # Sell signal: MACD crosses below signal line AND RSI is overbought
        signals['Sell_Signal'] = (
            (macd < signal_line) & 
            (macd.shift(1) >= signal_line.shift(1)) & 
            (rsi > rsi_params['overbought'])
        ).astype(int)
        
        # Combined signal: 1 for buy, -1 for sell, 0 for hold
        signals['Signal'] = signals['Buy_Signal'] - signals['Sell_Signal']
        
        logger.info(f"Generated signals for {symbol}: {signals['Buy_Signal'].sum()} buys, {signals['Sell_Signal'].sum()} sells")
        return signals
    
    def generate_moving_average_signals(
        self,
        data: pd.DataFrame,
        symbol: str,
        short_window: int = 20,
        long_window: int = 50
    ) -> pd.DataFrame:
        """Generate signals using moving average crossover.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to generate signals for.
            short_window: Short moving average window.
            long_window: Long moving average window.
            
        Returns:
            DataFrame with trading signals.
        """
        logger.info(f"Generating MA crossover signals for {symbol}")
        
        close = data[(symbol, 'Close')]
        
        # Calculate moving averages
        sma_short = close.rolling(window=short_window).mean()
        sma_long = close.rolling(window=long_window).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['SMA_Short'] = sma_short
        signals['SMA_Long'] = sma_long
        
        # Buy signal: short MA crosses above long MA
        signals['Buy_Signal'] = (
            (sma_short > sma_long) & 
            (sma_short.shift(1) <= sma_long.shift(1))
        ).astype(int)
        
        # Sell signal: short MA crosses below long MA
        signals['Sell_Signal'] = (
            (sma_short < sma_long) & 
            (sma_short.shift(1) >= sma_long.shift(1))
        ).astype(int)
        
        # Combined signal
        signals['Signal'] = signals['Buy_Signal'] - signals['Sell_Signal']
        
        logger.info(f"Generated MA signals for {symbol}: {signals['Buy_Signal'].sum()} buys, {signals['Sell_Signal'].sum()} sells")
        return signals
    
    def generate_bollinger_bands_signals(
        self,
        data: pd.DataFrame,
        symbol: str,
        window: int = 20,
        std: float = 2.0
    ) -> pd.DataFrame:
        """Generate signals using Bollinger Bands.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to generate signals for.
            window: Window size for Bollinger Bands.
            std: Standard deviation multiplier.
            
        Returns:
            DataFrame with trading signals.
        """
        logger.info(f"Generating Bollinger Bands signals for {symbol}")
        
        close = data[(symbol, 'Close')]
        
        # Calculate Bollinger Bands
        sma = close.rolling(window=window).mean()
        std_dev = close.rolling(window=window).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['BB_Upper'] = upper_band
        signals['BB_Middle'] = sma
        signals['BB_Lower'] = lower_band
        signals['BB_Width'] = (upper_band - lower_band) / sma
        signals['BB_Position'] = (close - lower_band) / (upper_band - lower_band)
        
        # Buy signal: price touches lower band and starts to rise
        signals['Buy_Signal'] = (
            (close <= lower_band) & 
            (close.shift(1) > lower_band.shift(1))
        ).astype(int)
        
        # Sell signal: price touches upper band and starts to fall
        signals['Sell_Signal'] = (
            (close >= upper_band) & 
            (close.shift(1) < upper_band.shift(1))
        ).astype(int)
        
        # Combined signal
        signals['Signal'] = signals['Buy_Signal'] - signals['Sell_Signal']
        
        logger.info(f"Generated BB signals for {symbol}: {signals['Buy_Signal'].sum()} buys, {signals['Sell_Signal'].sum()} sells")
        return signals
    
    def generate_momentum_signals(
        self,
        data: pd.DataFrame,
        symbol: str,
        window: int = 20,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """Generate signals using momentum.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to generate signals for.
            window: Window size for momentum calculation.
            threshold: Momentum threshold for signals.
            
        Returns:
            DataFrame with trading signals.
        """
        logger.info(f"Generating momentum signals for {symbol}")
        
        close = data[(symbol, 'Close')]
        
        # Calculate momentum
        momentum = close / close.shift(window) - 1
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['Momentum'] = momentum
        
        # Buy signal: positive momentum above threshold
        signals['Buy_Signal'] = (momentum > threshold).astype(int)
        
        # Sell signal: negative momentum below threshold
        signals['Sell_Signal'] = (momentum < -threshold).astype(int)
        
        # Combined signal
        signals['Signal'] = signals['Buy_Signal'] - signals['Sell_Signal']
        
        logger.info(f"Generated momentum signals for {symbol}: {signals['Buy_Signal'].sum()} buys, {signals['Sell_Signal'].sum()} sells")
        return signals
    
    def generate_combined_signals(
        self,
        data: pd.DataFrame,
        symbol: str,
        methods: List[str] = None,
        weights: List[float] = None
    ) -> pd.DataFrame:
        """Generate combined signals from multiple methods.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to generate signals for.
            methods: List of signal generation methods.
            weights: Weights for each method.
            
        Returns:
            DataFrame with combined trading signals.
        """
        if methods is None:
            methods = ['macd_rsi', 'ma_crossover', 'bollinger_bands', 'momentum']
        if weights is None:
            weights = [0.3, 0.2, 0.2, 0.3]
        
        logger.info(f"Generating combined signals for {symbol} using {methods}")
        
        all_signals = []
        
        # Generate signals for each method
        for method in methods:
            if method == 'macd_rsi':
                method_signals = self.generate_macd_rsi_signals(data, symbol)
            elif method == 'ma_crossover':
                method_signals = self.generate_moving_average_signals(data, symbol)
            elif method == 'bollinger_bands':
                method_signals = self.generate_bollinger_bands_signals(data, symbol)
            elif method == 'momentum':
                method_signals = self.generate_momentum_signals(data, symbol)
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            all_signals.append(method_signals['Signal'])
        
        if not all_signals:
            logger.error("No valid signals generated")
            return pd.DataFrame(index=data.index)
        
        # Combine signals with weights
        combined_signals = pd.DataFrame(index=data.index)
        combined_signals['Individual_Signals'] = pd.concat(all_signals, axis=1)
        
        # Weighted average of signals
        weighted_signal = np.zeros(len(data))
        for i, (method, weight) in enumerate(zip(methods, weights)):
            if i < len(all_signals):
                weighted_signal += all_signals[i] * weight
        
        # Convert to discrete signals
        combined_signals['Weighted_Signal'] = weighted_signal
        combined_signals['Buy_Signal'] = (weighted_signal > 0.5).astype(int)
        combined_signals['Sell_Signal'] = (weighted_signal < -0.5).astype(int)
        combined_signals['Signal'] = combined_signals['Buy_Signal'] - combined_signals['Sell_Signal']
        
        logger.info(f"Generated combined signals for {symbol}: {combined_signals['Buy_Signal'].sum()} buys, {combined_signals['Sell_Signal'].sum()} sells")
        return combined_signals


class TripleBarrierMethod:
    """Triple barrier method for labeling trading opportunities."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize triple barrier method.
        
        Args:
            config: Configuration dictionary for triple barrier method.
        """
        self.config = config or {}
    
    def create_triple_barrier_labels(
        self,
        data: pd.DataFrame,
        symbol: str,
        upper_barrier: float = 0.02,
        lower_barrier: float = 0.02,
        max_holding_period: int = 20,
        min_holding_period: int = 1
    ) -> pd.DataFrame:
        """Create labels using triple barrier method.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to create labels for.
            upper_barrier: Upper barrier as fraction of price.
            lower_barrier: Lower barrier as fraction of price.
            max_holding_period: Maximum holding period in days.
            min_holding_period: Minimum holding period in days.
            
        Returns:
            DataFrame with triple barrier labels.
        """
        logger.info(f"Creating triple barrier labels for {symbol}")
        
        close = data[(symbol, 'Close')]
        high = data[(symbol, 'High')]
        low = data[(symbol, 'Low')]
        
        labels = pd.DataFrame(index=data.index)
        labels['Label'] = 0  # 0: no signal, 1: upper barrier hit, -1: lower barrier hit
        labels['Barrier_Hit_Date'] = pd.NaT
        labels['Holding_Period'] = 0
        
        for i in range(len(data) - min_holding_period):
            if pd.isna(close.iloc[i]):
                continue
            
            # Set barriers
            upper_price = close.iloc[i] * (1 + upper_barrier)
            lower_price = close.iloc[i] * (1 - lower_barrier)
            
            # Check barriers within holding period
            for j in range(i + min_holding_period, min(i + max_holding_period + 1, len(data))):
                if pd.isna(close.iloc[j]):
                    continue
                
                # Check if upper barrier hit
                if high.iloc[i+1:j+1].max() >= upper_price:
                    labels.iloc[i, labels.columns.get_loc('Label')] = 1
                    labels.iloc[i, labels.columns.get_loc('Barrier_Hit_Date')] = data.index[j]
                    labels.iloc[i, labels.columns.get_loc('Holding_Period')] = j - i
                    break
                
                # Check if lower barrier hit
                if low.iloc[i+1:j+1].min() <= lower_price:
                    labels.iloc[i, labels.columns.get_loc('Label')] = -1
                    labels.iloc[i, labels.columns.get_loc('Barrier_Hit_Date')] = data.index[j]
                    labels.iloc[i, labels.columns.get_loc('Holding_Period')] = j - i
                    break
            
            # If no barrier hit within max holding period, label as 0
            if labels.iloc[i, labels.columns.get_loc('Label')] == 0:
                labels.iloc[i, labels.columns.get_loc('Holding_Period')] = max_holding_period
        
        logger.info(f"Created triple barrier labels for {symbol}: {labels['Label'].sum()} upper hits, {(-labels['Label']).sum()} lower hits")
        return labels
    
    def create_fixed_horizon_labels(
        self,
        data: pd.DataFrame,
        symbol: str,
        horizon: int = 5,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """Create labels using fixed horizon method.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            symbol: Symbol to create labels for.
            horizon: Prediction horizon in days.
            threshold: Return threshold for labeling.
            
        Returns:
            DataFrame with fixed horizon labels.
        """
        logger.info(f"Creating fixed horizon labels for {symbol} (horizon={horizon})")
        
        close = data[(symbol, 'Close')]
        
        # Calculate future returns
        future_returns = close.shift(-horizon) / close - 1
        
        # Create labels
        labels = pd.DataFrame(index=data.index)
        labels['Future_Return'] = future_returns
        labels['Label'] = 0
        labels['Label'] = np.where(future_returns > threshold, 1, labels['Label'])
        labels['Label'] = np.where(future_returns < -threshold, -1, labels['Label'])
        
        logger.info(f"Created fixed horizon labels for {symbol}: {labels['Label'].sum()} positive, {(-labels['Label']).sum()} negative")
        return labels
