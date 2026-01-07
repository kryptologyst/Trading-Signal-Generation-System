"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.features.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test class for TechnicalIndicators."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Generate synthetic price data
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        highs = prices + np.random.uniform(0, 2, 100)
        lows = prices - np.random.uniform(0, 2, 100)
        volumes = np.random.randint(1000000, 10000000, 100)
        
        self.data = pd.DataFrame({
            'Close': prices,
            'High': highs,
            'Low': lows,
            'Volume': volumes
        }, index=dates)
        
        self.indicators = TechnicalIndicators()
    
    def test_sma(self):
        """Test Simple Moving Average calculation."""
        sma = self.indicators.sma(self.data['Close'], window=20)
        
        assert len(sma) == len(self.data)
        assert sma.iloc[19] == self.data['Close'].iloc[:20].mean()
        assert pd.isna(sma.iloc[0])  # First values should be NaN
    
    def test_ema(self):
        """Test Exponential Moving Average calculation."""
        ema = self.indicators.ema(self.data['Close'], window=20)
        
        assert len(ema) == len(self.data)
        assert not pd.isna(ema.iloc[0])  # EMA should not have NaN values
        assert ema.iloc[-1] > 0  # Should be positive
    
    def test_macd(self):
        """Test MACD calculation."""
        macd_data = self.indicators.macd(self.data['Close'])
        
        assert 'MACD' in macd_data
        assert 'Signal' in macd_data
        assert 'Histogram' in macd_data
        
        assert len(macd_data['MACD']) == len(self.data)
        assert not pd.isna(macd_data['MACD'].iloc[-1])
    
    def test_rsi(self):
        """Test RSI calculation."""
        rsi = self.indicators.rsi(self.data['Close'])
        
        assert len(rsi) == len(self.data)
        assert rsi.iloc[-1] >= 0
        assert rsi.iloc[-1] <= 100
        assert pd.isna(rsi.iloc[0])  # First values should be NaN
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_data = self.indicators.bollinger_bands(self.data['Close'])
        
        assert 'Upper' in bb_data
        assert 'Middle' in bb_data
        assert 'Lower' in bb_data
        
        assert len(bb_data['Upper']) == len(self.data)
        assert bb_data['Upper'].iloc[-1] > bb_data['Middle'].iloc[-1]
        assert bb_data['Middle'].iloc[-1] > bb_data['Lower'].iloc[-1]
    
    def test_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        stoch_data = self.indicators.stochastic(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close']
        )
        
        assert 'K' in stoch_data
        assert 'D' in stoch_data
        
        assert len(stoch_data['K']) == len(self.data)
        assert stoch_data['K'].iloc[-1] >= 0
        assert stoch_data['K'].iloc[-1] <= 100
    
    def test_williams_r(self):
        """Test Williams %R calculation."""
        williams_r = self.indicators.williams_r(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close']
        )
        
        assert len(williams_r) == len(self.data)
        assert williams_r.iloc[-1] <= 0
        assert williams_r.iloc[-1] >= -100
    
    def test_atr(self):
        """Test Average True Range calculation."""
        atr = self.indicators.atr(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close']
        )
        
        assert len(atr) == len(self.data)
        assert atr.iloc[-1] > 0
        assert pd.isna(atr.iloc[0])  # First values should be NaN
    
    def test_adx(self):
        """Test ADX calculation."""
        adx_data = self.indicators.adx(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close']
        )
        
        assert 'ADX' in adx_data
        assert 'Plus_DI' in adx_data
        assert 'Minus_DI' in adx_data
        
        assert len(adx_data['ADX']) == len(self.data)
        assert adx_data['ADX'].iloc[-1] >= 0
        assert adx_data['ADX'].iloc[-1] <= 100
    
    def test_cci(self):
        """Test Commodity Channel Index calculation."""
        cci = self.indicators.cci(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close']
        )
        
        assert len(cci) == len(self.data)
        assert not pd.isna(cci.iloc[-1])
    
    def test_obv(self):
        """Test On-Balance Volume calculation."""
        obv = self.indicators.obv(self.data['Close'], self.data['Volume'])
        
        assert len(obv) == len(self.data)
        assert not pd.isna(obv.iloc[-1])
    
    def test_vwap(self):
        """Test Volume Weighted Average Price calculation."""
        vwap = self.indicators.vwap(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            self.data['Volume']
        )
        
        assert len(vwap) == len(self.data)
        assert vwap.iloc[-1] > 0
        assert not pd.isna(vwap.iloc[-1])
    
    def test_calculate_all_indicators(self):
        """Test calculation of all indicators."""
        # Create multi-index data structure
        multi_data = pd.DataFrame({
            ('AAPL', 'Close'): self.data['Close'],
            ('AAPL', 'High'): self.data['High'],
            ('AAPL', 'Low'): self.data['Low'],
            ('AAPL', 'Volume'): self.data['Volume']
        })
        multi_data.columns.names = ['Symbol', 'Field']
        
        indicators_df = self.indicators.calculate_all_indicators(
            multi_data, 'AAPL'
        )
        
        assert len(indicators_df) == len(self.data)
        assert len(indicators_df.columns) > 0
        
        # Check that we have expected indicator columns
        expected_indicators = ['SMA_20', 'EMA_20', 'MACD', 'RSI', 'BB_Upper']
        for indicator in expected_indicators:
            assert any(indicator in col for col in indicators_df.columns)
