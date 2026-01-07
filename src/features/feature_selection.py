"""Feature selection utilities for trading signals."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    SelectKBest,
    SelectPercentile,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection for trading signal generation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature selector.
        
        Args:
            config: Configuration dictionary for feature selection.
        """
        self.config = config or {}
        self.selected_features = []
        self.feature_importance = {}
        self.scaler = StandardScaler()
    
    def remove_low_variance_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove features with low variance.
        
        Args:
            X: Feature matrix.
            threshold: Variance threshold.
            
        Returns:
            DataFrame with low variance features removed.
        """
        logger.info(f"Removing features with variance < {threshold}")
        
        # Calculate variance for each feature
        variances = X.var()
        
        # Select features above threshold
        high_var_features = variances[variances >= threshold].index
        
        logger.info(f"Removed {len(X.columns) - len(high_var_features)} low variance features")
        return X[high_var_features]
    
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features.
        
        Args:
            X: Feature matrix.
            threshold: Correlation threshold.
            
        Returns:
            DataFrame with correlated features removed.
        """
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Remove correlated features
        X_reduced = X.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return X_reduced
    
    def select_by_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 50,
        task_type: str = "regression"
    ) -> pd.DataFrame:
        """Select features using mutual information.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            k: Number of features to select.
            task_type: Type of task ("regression" or "classification").
            
        Returns:
            DataFrame with selected features.
        """
        logger.info(f"Selecting {k} features using mutual information ({task_type})")
        
        # Choose appropriate mutual information function
        if task_type == "regression":
            mi_func = mutual_info_regression
        else:
            mi_func = mutual_info_classif
        
        # Calculate mutual information
        mi_scores = mi_func(X.fillna(0), y.fillna(0))
        
        # Select top k features
        selector = SelectKBest(mi_func, k=min(k, len(X.columns)))
        X_selected = selector.fit_transform(X.fillna(0), y.fillna(0))
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance
        self.feature_importance.update(dict(zip(selected_features, mi_scores[selector.get_support()])))
        
        logger.info(f"Selected {len(selected_features)} features using mutual information")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        task_type: str = "regression"
    ) -> pd.DataFrame:
        """Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            n_features: Number of features to select.
            task_type: Type of task ("regression" or "classification").
            
        Returns:
            DataFrame with selected features.
        """
        logger.info(f"Selecting {n_features} features using RFE ({task_type})")
        
        # Choose appropriate estimator
        if task_type == "regression":
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Create RFE selector
        selector = RFE(estimator, n_features_to_select=min(n_features, len(X.columns)))
        
        # Fit selector
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        selector.fit(X_scaled, y.fillna(0))
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance
        self.feature_importance.update(dict(zip(selected_features, selector.estimator_.feature_importances_)))
        
        logger.info(f"Selected {len(selected_features)} features using RFE")
        return X[selected_features]
    
    def select_by_lasso(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """Select features using Lasso regularization.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            alpha: Lasso regularization parameter. If None, use cross-validation.
            
        Returns:
            DataFrame with selected features.
        """
        logger.info("Selecting features using Lasso regularization")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # Create Lasso model
        if alpha is None:
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=42)
        
        # Fit model
        lasso.fit(X_scaled, y.fillna(0))
        
        # Select features with non-zero coefficients
        selected_features = X.columns[lasso.coef_ != 0].tolist()
        
        # Store feature importance
        self.feature_importance.update(dict(zip(selected_features, np.abs(lasso.coef_[lasso.coef_ != 0]))))
        
        logger.info(f"Selected {len(selected_features)} features using Lasso")
        return X[selected_features]
    
    def select_by_elastic_net(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        l1_ratio: float = 0.5
    ) -> pd.DataFrame:
        """Select features using Elastic Net regularization.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            l1_ratio: L1 ratio for Elastic Net.
            
        Returns:
            DataFrame with selected features.
        """
        logger.info(f"Selecting features using Elastic Net (l1_ratio={l1_ratio})")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # Create Elastic Net model
        elastic_net = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=42, n_jobs=-1)
        
        # Fit model
        elastic_net.fit(X_scaled, y.fillna(0))
        
        # Select features with non-zero coefficients
        selected_features = X.columns[elastic_net.coef_ != 0].tolist()
        
        # Store feature importance
        self.feature_importance.update(dict(zip(selected_features, np.abs(elastic_net.coef_[elastic_net.coef_ != 0]))))
        
        logger.info(f"Selected {len(selected_features)} features using Elastic Net")
        return X[selected_features]
    
    def select_by_model_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        task_type: str = "regression"
    ) -> pd.DataFrame:
        """Select features using model-based importance.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            n_features: Number of features to select.
            task_type: Type of task ("regression" or "classification").
            
        Returns:
            DataFrame with selected features.
        """
        logger.info(f"Selecting {n_features} features using model importance ({task_type})")
        
        # Choose appropriate estimator
        if task_type == "regression":
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Create selector
        selector = SelectFromModel(estimator, max_features=min(n_features, len(X.columns)))
        
        # Fit selector
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        selector.fit(X_scaled, y.fillna(0))
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance
        self.feature_importance.update(dict(zip(selected_features, selector.estimator_.feature_importances_[selector.get_support()])))
        
        logger.info(f"Selected {len(selected_features)} features using model importance")
        return X[selected_features]
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
        task_type: str = "regression",
        n_features: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """Select features using specified method.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            method: Selection method ("mutual_info", "rfe", "lasso", "elastic_net", "model_importance").
            task_type: Type of task ("regression" or "classification").
            n_features: Number of features to select.
            **kwargs: Additional arguments for specific methods.
            
        Returns:
            DataFrame with selected features.
        """
        logger.info(f"Selecting features using {method} method")
        
        # Remove low variance features first
        X_clean = self.remove_low_variance_features(X, threshold=kwargs.get('variance_threshold', 0.01))
        
        # Remove highly correlated features
        X_clean = self.remove_correlated_features(X_clean, threshold=kwargs.get('correlation_threshold', 0.95))
        
        # Select features based on method
        if method == "mutual_info":
            X_selected = self.select_by_mutual_information(X_clean, y, n_features, task_type)
        elif method == "rfe":
            X_selected = self.select_by_rfe(X_clean, y, n_features, task_type)
        elif method == "lasso":
            X_selected = self.select_by_lasso(X_clean, y, kwargs.get('alpha'))
        elif method == "elastic_net":
            X_selected = self.select_by_elastic_net(X_clean, y, kwargs.get('l1_ratio', 0.5))
        elif method == "model_importance":
            X_selected = self.select_by_model_importance(X_clean, y, n_features, task_type)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Store selected features
        self.selected_features = X_selected.columns.tolist()
        
        logger.info(f"Feature selection completed. Selected {len(self.selected_features)} features")
        return X_selected
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores.
        """
        if not self.feature_importance:
            logger.warning("No feature importance scores available")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        return importance_df
