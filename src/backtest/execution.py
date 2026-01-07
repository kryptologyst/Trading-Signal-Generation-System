"""Execution engine for backtesting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Execution engine for simulating realistic trading."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize execution engine.
        
        Args:
            config: Configuration dictionary for execution.
        """
        self.config = config or {}
        
        # Execution parameters
        self.transaction_cost = self.config.get('transaction_cost', 0.001)  # 0.1% per trade
        self.slippage = self.config.get('slippage', 0.0005)  # 0.05% slippage
        self.min_trade_size = self.config.get('min_trade_size', 100)  # Minimum trade size in dollars
        self.max_trade_size = self.config.get('max_trade_size', 1000000)  # Maximum trade size in dollars
        
        # Market impact model parameters
        self.impact_linear = self.config.get('impact_linear', 0.0001)  # Linear impact coefficient
        self.impact_sqrt = self.config.get('impact_sqrt', 0.0002)  # Square root impact coefficient
        
        # Execution delays
        self.execution_delay = self.config.get('execution_delay', 0)  # Days to execute trade
        self.pending_orders = []  # List of pending orders
        
        # Volume constraints
        self.max_volume_pct = self.config.get('max_volume_pct', 0.1)  # Max 10% of daily volume
        self.min_volume_pct = self.config.get('min_volume_pct', 0.001)  # Min 0.1% of daily volume
    
    def calculate_cost(
        self,
        price: float,
        shares: float,
        transaction_cost: Optional[float] = None,
        slippage: Optional[float] = None,
        volume: Optional[float] = None
    ) -> float:
        """Calculate total execution cost.
        
        Args:
            price: Execution price.
            shares: Number of shares.
            transaction_cost: Transaction cost rate.
            slippage: Slippage rate.
            volume: Daily volume (for impact calculation).
            
        Returns:
            Total execution cost.
        """
        if transaction_cost is None:
            transaction_cost = self.transaction_cost
        if slippage is None:
            slippage = self.slippage
        
        trade_value = price * shares
        
        # Base transaction cost
        base_cost = trade_value * transaction_cost
        
        # Slippage cost
        slippage_cost = trade_value * slippage
        
        # Market impact cost
        impact_cost = self._calculate_market_impact(price, shares, volume)
        
        total_cost = base_cost + slippage_cost + impact_cost
        
        return total_cost
    
    def _calculate_market_impact(
        self,
        price: float,
        shares: float,
        volume: Optional[float] = None
    ) -> float:
        """Calculate market impact cost.
        
        Args:
            price: Execution price.
            shares: Number of shares.
            volume: Daily volume.
            
        Returns:
            Market impact cost.
        """
        trade_value = price * shares
        
        if volume is None or volume == 0:
            # Use linear impact model
            impact = trade_value * self.impact_linear
        else:
            # Use volume-adjusted impact model
            volume_ratio = shares / volume
            impact = trade_value * (self.impact_linear + self.impact_sqrt * np.sqrt(volume_ratio))
        
        return impact
    
    def execute_trade(
        self,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        date: pd.Timestamp,
        volume: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute a trade with realistic constraints.
        
        Args:
            symbol: Symbol to trade.
            side: 'buy' or 'sell'.
            shares: Number of shares.
            price: Execution price.
            date: Trading date.
            volume: Daily volume.
            
        Returns:
            Dictionary with execution details.
        """
        # Validate trade size
        trade_value = price * shares
        
        if trade_value < self.min_trade_size:
            logger.warning(f"Trade value ${trade_value:.2f} below minimum ${self.min_trade_size}")
            return None
        
        if trade_value > self.max_trade_size:
            logger.warning(f"Trade value ${trade_value:.2f} above maximum ${self.max_trade_size}")
            return None
        
        # Check volume constraints
        if volume is not None and volume > 0:
            volume_ratio = shares / volume
            if volume_ratio > self.max_volume_pct:
                # Reduce shares to meet volume constraint
                max_shares = volume * self.max_volume_pct
                shares = max_shares
                trade_value = price * shares
                logger.warning(f"Reduced trade size to meet volume constraint: {shares:.0f} shares")
            
            if volume_ratio < self.min_volume_pct:
                logger.warning(f"Trade size {volume_ratio:.4f} below minimum volume ratio {self.min_volume_pct}")
        
        # Calculate execution cost
        total_cost = self.calculate_cost(price, shares, volume=volume)
        
        # Apply execution delay if specified
        if self.execution_delay > 0:
            execution_date = date + pd.Timedelta(days=self.execution_delay)
            self.pending_orders.append({
                'symbol': symbol,
                'side': side,
                'shares': shares,
                'price': price,
                'date': execution_date,
                'cost': total_cost
            })
            return None
        
        # Execute immediately
        execution_result = {
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'price': price,
            'date': date,
            'cost': total_cost,
            'executed': True
        }
        
        return execution_result
    
    def process_pending_orders(self, current_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Process pending orders for a given date.
        
        Args:
            current_date: Current trading date.
            
        Returns:
            List of executed orders.
        """
        executed_orders = []
        remaining_orders = []
        
        for order in self.pending_orders:
            if order['date'] <= current_date:
                # Execute the order
                order['executed'] = True
                executed_orders.append(order)
            else:
                remaining_orders.append(order)
        
        self.pending_orders = remaining_orders
        return executed_orders
    
    def calculate_portfolio_turnover(
        self,
        trades: List[Dict[str, Any]],
        portfolio_value: float
    ) -> float:
        """Calculate portfolio turnover.
        
        Args:
            trades: List of executed trades.
            portfolio_value: Current portfolio value.
            
        Returns:
            Portfolio turnover rate.
        """
        if not trades or portfolio_value == 0:
            return 0
        
        total_trade_value = sum(abs(trade.get('shares', 0) * trade.get('price', 0)) for trade in trades)
        turnover = total_trade_value / portfolio_value
        
        return turnover
    
    def calculate_execution_quality(
        self,
        trades: List[Dict[str, Any]],
        benchmark_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate execution quality metrics.
        
        Args:
            trades: List of executed trades.
            benchmark_prices: Benchmark prices for each symbol.
            
        Returns:
            Dictionary with execution quality metrics.
        """
        if not trades:
            return {}
        
        metrics = {}
        
        # Calculate implementation shortfall
        implementation_shortfalls = []
        for trade in trades:
            symbol = trade.get('symbol')
            if symbol in benchmark_prices:
                benchmark_price = benchmark_prices[symbol]
                execution_price = trade.get('price', 0)
                shortfall = (execution_price - benchmark_price) / benchmark_price
                implementation_shortfalls.append(shortfall)
        
        if implementation_shortfalls:
            metrics['avg_implementation_shortfall'] = np.mean(implementation_shortfalls)
            metrics['std_implementation_shortfall'] = np.std(implementation_shortfalls)
        
        # Calculate cost analysis
        total_costs = [trade.get('cost', 0) for trade in trades]
        total_trade_values = [abs(trade.get('shares', 0) * trade.get('price', 0)) for trade in trades]
        
        if total_trade_values:
            total_cost_ratio = sum(total_costs) / sum(total_trade_values)
            metrics['total_cost_ratio'] = total_cost_ratio
        
        # Calculate execution efficiency
        if 'avg_implementation_shortfall' in metrics:
            metrics['execution_efficiency'] = 1 - abs(metrics['avg_implementation_shortfall'])
        else:
            metrics['execution_efficiency'] = 1.0
        
        return metrics
    
    def simulate_market_impact(
        self,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        volume: float
    ) -> float:
        """Simulate market impact on price.
        
        Args:
            symbol: Symbol to trade.
            side: 'buy' or 'sell'.
            shares: Number of shares.
            price: Current price.
            volume: Daily volume.
            
        Returns:
            Impacted price.
        """
        if volume == 0:
            return price
        
        # Calculate impact based on trade size relative to volume
        volume_ratio = shares / volume
        
        # Linear impact
        linear_impact = self.impact_linear * volume_ratio
        
        # Square root impact
        sqrt_impact = self.impact_sqrt * np.sqrt(volume_ratio)
        
        # Total impact
        total_impact = linear_impact + sqrt_impact
        
        # Apply impact based on trade direction
        if side == 'buy':
            impacted_price = price * (1 + total_impact)
        else:  # sell
            impacted_price = price * (1 - total_impact)
        
        return impacted_price
    
    def get_execution_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get execution summary statistics.
        
        Args:
            trades: List of executed trades.
            
        Returns:
            Dictionary with execution summary.
        """
        if not trades:
            return {}
        
        summary = {
            'total_trades': len(trades),
            'total_cost': sum(trade.get('cost', 0) for trade in trades),
            'total_volume': sum(abs(trade.get('shares', 0)) for trade in trades),
            'total_value': sum(abs(trade.get('shares', 0) * trade.get('price', 0)) for trade in trades),
            'avg_trade_size': np.mean([abs(trade.get('shares', 0) * trade.get('price', 0)) for trade in trades]),
            'max_trade_size': max([abs(trade.get('shares', 0) * trade.get('price', 0)) for trade in trades]),
            'min_trade_size': min([abs(trade.get('shares', 0) * trade.get('price', 0)) for trade in trades])
        }
        
        # Calculate cost breakdown
        total_value = summary['total_value']
        if total_value > 0:
            summary['cost_ratio'] = summary['total_cost'] / total_value
            summary['avg_cost_per_trade'] = summary['total_cost'] / summary['total_trades']
        else:
            summary['cost_ratio'] = 0
            summary['avg_cost_per_trade'] = 0
        
        return summary
