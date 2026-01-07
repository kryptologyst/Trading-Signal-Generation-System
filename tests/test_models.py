"""Tests for model implementations."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import TechnicalStrategy, XGBoostModel, LightGBMModel


class TestModels:
    """Test class for model implementations."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample features
        n_samples = 100
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create sample labels
        self.y = pd.Series(np.random.choice([-1, 0, 1], n_samples), name='signal')
        
        # Add some technical indicator features
        self.X['RSI'] = np.random.uniform(20, 80, n_samples)
        self.X['MACD'] = np.random.randn(n_samples)
        self.X['MACD_Signal'] = np.random.randn(n_samples)
    
    def test_technical_strategy(self):
        """Test TechnicalStrategy model."""
        model = TechnicalStrategy()
        
        # Test fitting
        model.fit(self.X, self.y)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.X)
        assert all(pred in [-1, 0, 1] for pred in predictions)
        
        # Test evaluation
        metrics = model.evaluate(self.X, self.y)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
    
    def test_xgboost_model(self):
        """Test XGBoostModel."""
        model = XGBoostModel()
        
        # Test fitting
        model.fit(self.X, self.y)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.X)
        
        # Test probability prediction
        probabilities = model.predict_proba(self.X)
        assert probabilities.shape[0] == len(self.X)
        assert probabilities.shape[1] >= 1
        
        # Test evaluation
        metrics = model.evaluate(self.X, self.y)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_lightgbm_model(self):
        """Test LightGBMModel."""
        model = LightGBMModel()
        
        # Test fitting
        model.fit(self.X, self.y)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.X)
        
        # Test probability prediction
        probabilities = model.predict_proba(self.X)
        assert probabilities.shape[0] == len(self.X)
        assert probabilities.shape[1] >= 1
        
        # Test evaluation
        metrics = model.evaluate(self.X, self.y)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = TechnicalStrategy()
        model.fit(self.X, self.y)
        
        # Save model
        model_path = "test_model.joblib"
        model.save(model_path)
        
        # Load model
        loaded_model = TechnicalStrategy()
        loaded_model.load(model_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.feature_names == model.feature_names
        
        # Test that loaded model makes same predictions
        original_predictions = model.predict(self.X)
        loaded_predictions = loaded_model.predict(self.X)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        
        # Clean up
        import os
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def test_model_with_missing_values(self):
        """Test model behavior with missing values."""
        # Create data with missing values
        X_with_nan = self.X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        X_with_nan.iloc[5, 2] = np.nan
        
        model = XGBoostModel()
        model.fit(X_with_nan, self.y)
        
        # Should handle missing values gracefully
        predictions = model.predict(X_with_nan)
        assert len(predictions) == len(X_with_nan)
        assert not np.any(np.isnan(predictions))
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation with different metrics."""
        model = TechnicalStrategy()
        model.fit(self.X, self.y)
        
        # Test with different metric sets
        metrics = model.evaluate(self.X, self.y, metrics=['accuracy', 'precision', 'recall', 'f1'])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # All metrics should be between 0 and 1
        for metric, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_model_feature_importance(self):
        """Test model feature importance."""
        model = XGBoostModel()
        model.fit(self.X, self.y)
        
        importance = model.get_feature_importance()
        
        if importance is not None:
            assert len(importance) == len(self.X.columns)
            assert all(importance['importance'] >= 0)
            assert importance['importance'].sum() > 0
