"""Portfolio management for backtesting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio management class for backtesting."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize portfolio.
        
        Args:
            config: Configuration dictionary for portfolio.
        """
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: {'shares': float, 'price': float, 'date': timestamp, 'position_type': str, 'cost': float}}
        self.equity_history = []
        self.date_history = []
        
        # Portfolio metrics
        self.total_value = self.initial_capital
        self.total_return = 0.0
        self.daily_returns = []
    
    def reset(self, initial_capital: Optional[float] = None) -> None:
        """Reset portfolio to initial state.
        
        Args:
            initial_capital: Initial capital amount.
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_history = []
        self.date_history = []
        self.total_value = self.initial_capital
        self.total_return = 0.0
        self.daily_returns = []
        
        logger.info(f"Portfolio reset with initial capital: ${self.initial_capital:,.2f}")
    
    def add_position(
        self,
        symbol: str,
        shares: float,
        price: float,
        date: pd.Timestamp,
        position_type: str = 'long'
    ) -> None:
        """Add a position to the portfolio.
        
        Args:
            symbol: Symbol to add.
            shares: Number of shares.
            price: Price per share.
            date: Date of the position.
            position_type: Type of position ('long' or 'short').
        """
        cost = shares * price
        
        self.positions[symbol] = {
            'shares': shares,
            'price': price,
            'date': date,
            'position_type': position_type,
            'cost': cost
        }
        
        logger.debug(f"Added {position_type} position: {symbol} - {shares} shares at ${price:.2f}")
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position from the portfolio.
        
        Args:
            symbol: Symbol to remove.
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.debug(f"Removed position: {symbol}")
    
    def get_position_value(self, symbol: str) -> float:
        """Get current value of a position.
        
        Args:
            symbol: Symbol to get value for.
            
        Returns:
            Current position value.
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        return position['shares'] * position['price']
    
    def get_position_cost(self, symbol: str) -> float:
        """Get cost basis of a position.
        
        Args:
            symbol: Symbol to get cost for.
            
        Returns:
            Position cost basis.
        """
        if symbol not in self.positions:
            return 0.0
        
        return self.positions[symbol]['cost']
    
    def update_position_price(self, symbol: str, new_price: float) -> None:
        """Update the price of a position.
        
        Args:
            symbol: Symbol to update.
            new_price: New price.
        """
        if symbol in self.positions:
            self.positions[symbol]['price'] = new_price
    
    def update_portfolio_value(self, date: pd.Timestamp, market_data: Dict[str, Dict[str, float]]) -> None:
        """Update portfolio value based on current market data.
        
        Args:
            date: Current date.
            market_data: Dictionary with current market prices.
        """
        # Update position prices
        for symbol in self.positions:
            if symbol in market_data:
                self.update_position_price(symbol, market_data[symbol]['price'])
        
        # Calculate total portfolio value
        position_values = sum(self.get_position_value(symbol) for symbol in self.positions)
        self.total_value = self.cash + position_values
        
        # Calculate daily return
        if self.equity_history:
            previous_value = self.equity_history[-1]
            daily_return = (self.total_value - previous_value) / previous_value
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)
        
        # Update history
        self.equity_history.append(self.total_value)
        self.date_history.append(date)
        
        # Calculate total return
        self.total_return = (self.total_value - self.initial_capital) / self.initial_capital
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as a pandas Series.
        
        Returns:
            Series with portfolio values over time.
        """
        if not self.equity_history:
            return pd.Series(dtype=float)
        
        return pd.Series(self.equity_history, index=self.date_history)
    
    def get_daily_returns(self) -> pd.Series:
        """Get daily returns as a pandas Series.
        
        Returns:
            Series with daily returns.
        """
        if not self.daily_returns:
            return pd.Series(dtype=float)
        
        return pd.Series(self.daily_returns, index=self.date_history)
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions.
        
        Returns:
            DataFrame with position summary.
        """
        if not self.positions:
            return pd.DataFrame()
        
        summary_data = []
        for symbol, position in self.positions.items():
            current_value = self.get_position_value(symbol)
            cost = position['cost']
            pnl = current_value - cost if position['position_type'] == 'long' else cost - current_value
            pnl_pct = pnl / cost if cost > 0 else 0
            
            summary_data.append({
                'Symbol': symbol,
                'Shares': position['shares'],
                'Price': position['price'],
                'Cost': cost,
                'Current_Value': current_value,
                'P&L': pnl,
                'P&L_%': pnl_pct,
                'Position_Type': position['position_type'],
                'Date': position['date']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics.
        
        Returns:
            Dictionary with portfolio summary.
        """
        summary = {
            'Initial_Capital': self.initial_capital,
            'Cash': self.cash,
            'Total_Value': self.total_value,
            'Total_Return': self.total_return,
            'Total_Return_%': self.total_return * 100,
            'Num_Positions': len(self.positions),
            'Position_Value': sum(self.get_position_value(symbol) for symbol in self.positions),
            'Cash_%': (self.cash / self.total_value) * 100 if self.total_value > 0 else 0,
            'Position_%': (sum(self.get_position_value(symbol) for symbol in self.positions) / self.total_value) * 100 if self.total_value > 0 else 0
        }
        
        # Add position breakdown
        if self.positions:
            position_values = [self.get_position_value(symbol) for symbol in self.positions]
            position_pnls = []
            
            for symbol in self.positions:
                position = self.positions[symbol]
                current_value = self.get_position_value(symbol)
                cost = position['cost']
                pnl = current_value - cost if position['position_type'] == 'long' else cost - current_value
                position_pnls.append(pnl)
            
            summary['Total_Position_PnL'] = sum(position_pnls)
            summary['Total_Position_PnL_%'] = (sum(position_pnls) / self.initial_capital) * 100
            summary['Avg_Position_Value'] = np.mean(position_values)
            summary['Max_Position_Value'] = np.max(position_values)
            summary['Min_Position_Value'] = np.min(position_values)
        
        return summary
    
    def rebalance(self, target_weights: Dict[str, float], market_data: Dict[str, Dict[str, float]]) -> None:
        """Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dictionary with target weights for each symbol.
            market_data: Current market data.
        """
        if not target_weights:
            return
        
        # Calculate target values
        total_value = self.total_value
        target_values = {symbol: weight * total_value for symbol, weight in target_weights.items()}
        
        # Calculate current values
        current_values = {symbol: self.get_position_value(symbol) for symbol in target_weights.keys()}
        
        # Calculate rebalancing trades
        for symbol in target_weights.keys():
            if symbol not in market_data:
                continue
            
            current_value = current_values.get(symbol, 0)
            target_value = target_values[symbol]
            price = market_data[symbol]['price']
            
            if target_value > current_value:
                # Need to buy
                buy_value = target_value - current_value
                shares_to_buy = buy_value / price
                self.add_position(symbol, shares_to_buy, price, pd.Timestamp.now(), 'long')
                self.cash -= buy_value
            elif target_value < current_value:
                # Need to sell
                sell_value = current_value - target_value
                shares_to_sell = sell_value / price
                if symbol in self.positions:
                    shares_to_sell = min(shares_to_sell, self.positions[symbol]['shares'])
                    self.remove_position(symbol)
                    self.cash += sell_value
        
        logger.info("Portfolio rebalanced")
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get risk metrics for the portfolio.
        
        Returns:
            Dictionary with risk metrics.
        """
        if len(self.daily_returns) < 2:
            return {}
        
        returns = np.array(self.daily_returns)
        
        metrics = {
            'Volatility': np.std(returns) * np.sqrt(252),  # Annualized volatility
            'Skewness': self._calculate_skewness(returns),
            'Kurtosis': self._calculate_kurtosis(returns),
            'VaR_95': np.percentile(returns, 5),  # 95% VaR
            'VaR_99': np.percentile(returns, 1),  # 99% VaR
            'Max_Daily_Loss': np.min(returns),
            'Max_Daily_Gain': np.max(returns)
        }
        
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.mean(((returns - mean) / std) ** 4) - 3
