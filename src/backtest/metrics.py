"""Trading and risk metrics for backtesting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TradingMetrics:
    """Trading performance metrics calculator."""
    
    def __init__(self):
        """Initialize trading metrics calculator."""
        pass
    
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate comprehensive trading metrics.
        
        Args:
            equity_curve: Portfolio value over time.
            trades: List of trade dictionaries.
            initial_capital: Initial capital amount.
            
        Returns:
            Dictionary with trading metrics.
        """
        if equity_curve.empty:
            return {}
        
        metrics = {}
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        metrics['total_return'] = total_return
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        if years > 0:
            metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1
        else:
            metrics['annualized_return'] = 0
        
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) > 0:
            # Volatility metrics
            metrics['volatility'] = daily_returns.std() * np.sqrt(252)
            metrics['downside_volatility'] = self._calculate_downside_volatility(daily_returns)
            
            # Risk-adjusted returns
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
            
            if metrics['downside_volatility'] > 0:
                metrics['sortino_ratio'] = metrics['annualized_return'] / metrics['downside_volatility']
            else:
                metrics['sortino_ratio'] = 0
            
            # Drawdown metrics
            drawdown = self.calculate_drawdown(equity_curve)
            metrics['max_drawdown'] = drawdown.min()
            metrics['avg_drawdown'] = drawdown.mean()
            
            # Calmar ratio
            if abs(metrics['max_drawdown']) > 0:
                metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0
            
            # Additional metrics
            metrics['win_rate'] = self._calculate_win_rate(trades)
            metrics['avg_win'] = self._calculate_avg_win(trades)
            metrics['avg_loss'] = self._calculate_avg_loss(trades)
            metrics['profit_factor'] = self._calculate_profit_factor(trades)
            metrics['recovery_factor'] = self._calculate_recovery_factor(equity_curve, initial_capital)
            metrics['expectancy'] = self._calculate_expectancy(trades)
            metrics['kelly_percentage'] = self._calculate_kelly_percentage(trades)
        
        return metrics
    
    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """Calculate downside volatility."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0
        return negative_returns.std() * np.sqrt(252)
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series.
        
        Args:
            equity_curve: Portfolio value over time.
            
        Returns:
            Series with drawdown percentages.
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0
        
        closed_trades = [t for t in trades if 'pnl' in t]
        if not closed_trades:
            return 0
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        return len(winning_trades) / len(closed_trades)
    
    def _calculate_avg_win(self, trades: List[Dict]) -> float:
        """Calculate average winning trade."""
        if not trades:
            return 0
        
        closed_trades = [t for t in trades if 'pnl' in t]
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        
        if not winning_trades:
            return 0
        
        return np.mean([t['pnl'] for t in winning_trades])
    
    def _calculate_avg_loss(self, trades: List[Dict]) -> float:
        """Calculate average losing trade."""
        if not trades:
            return 0
        
        closed_trades = [t for t in trades if 'pnl' in t]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        if not losing_trades:
            return 0
        
        return np.mean([t['pnl'] for t in losing_trades])
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0
        
        closed_trades = [t for t in trades if 'pnl' in t]
        if not closed_trades:
            return 0
        
        gross_profit = sum([t['pnl'] for t in closed_trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in closed_trades if t['pnl'] < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, equity_curve: pd.Series, initial_capital: float) -> float:
        """Calculate recovery factor."""
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        drawdown = self.calculate_drawdown(equity_curve)
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0
        
        return total_return / max_drawdown
    
    def _calculate_expectancy(self, trades: List[Dict]) -> float:
        """Calculate expectancy."""
        if not trades:
            return 0
        
        closed_trades = [t for t in trades if 'pnl' in t]
        if not closed_trades:
            return 0
        
        return np.mean([t['pnl'] for t in closed_trades])
    
    def _calculate_kelly_percentage(self, trades: List[Dict]) -> float:
        """Calculate Kelly percentage."""
        if not trades:
            return 0
        
        closed_trades = [t for t in trades if 'pnl' in t]
        if not closed_trades:
            return 0
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        if not winning_trades or not losing_trades:
            return 0
        
        win_rate = len(winning_trades) / len(closed_trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades])
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
        
        if avg_loss == 0:
            return 0
        
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        return max(0, min(kelly, 1))  # Clamp between 0 and 1


class RiskMetrics:
    """Risk metrics calculator."""
    
    def __init__(self):
        """Initialize risk metrics calculator."""
        pass
    
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics.
        
        Args:
            equity_curve: Portfolio value over time.
            initial_capital: Initial capital amount.
            
        Returns:
            Dictionary with risk metrics.
        """
        if equity_curve.empty:
            return {}
        
        metrics = {}
        
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) > 0:
            # Basic risk metrics
            metrics['volatility'] = daily_returns.std() * np.sqrt(252)
            metrics['skewness'] = daily_returns.skew()
            metrics['kurtosis'] = daily_returns.kurtosis()
            
            # VaR calculations
            metrics['var_95'] = np.percentile(daily_returns, 5)
            metrics['var_99'] = np.percentile(daily_returns, 1)
            metrics['cvar_95'] = daily_returns[daily_returns <= metrics['var_95']].mean()
            metrics['cvar_99'] = daily_returns[daily_returns <= metrics['var_99']].mean()
            
            # Drawdown metrics
            drawdown = self.calculate_drawdown(equity_curve)
            metrics['max_drawdown'] = drawdown.min()
            metrics['avg_drawdown'] = drawdown.mean()
            metrics['drawdown_std'] = drawdown.std()
            
            # Drawdown duration
            drawdown_duration = self._calculate_drawdown_duration(drawdown)
            metrics['max_drawdown_duration'] = drawdown_duration.max()
            metrics['avg_drawdown_duration'] = drawdown_duration.mean()
            
            # Tail risk metrics
            metrics['tail_ratio'] = self._calculate_tail_ratio(daily_returns)
            metrics['common_sense_ratio'] = self._calculate_common_sense_ratio(daily_returns)
            
            # Stability metrics
            metrics['stability'] = self._calculate_stability(daily_returns)
            metrics['calmar_ratio'] = self._calculate_calmar_ratio(equity_curve, initial_capital)
        
        return metrics
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series.
        
        Args:
            equity_curve: Portfolio value over time.
            
        Returns:
            Series with drawdown percentages.
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> pd.Series:
        """Calculate drawdown duration."""
        in_drawdown = drawdown < 0
        drawdown_duration = in_drawdown.groupby((~in_drawdown).cumsum()).cumsum()
        return drawdown_duration
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return float('inf') if p95 > 0 else 0
        
        return p95 / abs(p5)
    
    def _calculate_common_sense_ratio(self, returns: pd.Series) -> float:
        """Calculate common sense ratio."""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0
        
        avg_positive = positive_returns.mean()
        avg_negative = abs(negative_returns.mean())
        
        if avg_negative == 0:
            return float('inf') if avg_positive > 0 else 0
        
        return avg_positive / avg_negative
    
    def _calculate_stability(self, returns: pd.Series) -> float:
        """Calculate stability (1 - coefficient of variation)."""
        if returns.std() == 0:
            return 1.0
        
        cv = returns.std() / abs(returns.mean())
        return 1 - cv
    
    def _calculate_calmar_ratio(self, equity_curve: pd.Series, initial_capital: float) -> float:
        """Calculate Calmar ratio."""
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        drawdown = self.calculate_drawdown(equity_curve)
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0
        
        return annualized_return / max_drawdown
    
    def calculate_risk_adjusted_returns(
        self,
        equity_curve: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics.
        
        Args:
            equity_curve: Portfolio value over time.
            risk_free_rate: Risk-free rate (annual).
            
        Returns:
            Dictionary with risk-adjusted metrics.
        """
        if equity_curve.empty:
            return {}
        
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {}
        
        # Calculate annualized metrics
        annualized_return = daily_returns.mean() * 252
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        annualized_risk_free = risk_free_rate
        
        metrics = {}
        
        # Sharpe ratio
        if annualized_volatility > 0:
            metrics['sharpe_ratio'] = (annualized_return - annualized_risk_free) / annualized_volatility
        else:
            metrics['sharpe_ratio'] = 0
        
        # Information ratio (using benchmark return of 0)
        if annualized_volatility > 0:
            metrics['information_ratio'] = annualized_return / annualized_volatility
        else:
            metrics['information_ratio'] = 0
        
        # Treynor ratio (using beta of 1)
        beta = 1.0  # Assuming beta of 1 for simplicity
        if beta > 0:
            metrics['treynor_ratio'] = (annualized_return - annualized_risk_free) / beta
        else:
            metrics['treynor_ratio'] = 0
        
        return metrics
