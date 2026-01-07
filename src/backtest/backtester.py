"""Backtesting engine for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from .portfolio import Portfolio
from .metrics import TradingMetrics, RiskMetrics
from .execution import ExecutionEngine

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize backtester.
        
        Args:
            config: Configuration dictionary for backtesting.
        """
        self.config = config or {}
        self.portfolio = Portfolio(config.get('portfolio', {}))
        self.execution_engine = ExecutionEngine(config.get('execution', {}))
        self.trading_metrics = TradingMetrics()
        self.risk_metrics = RiskMetrics()
        
        # Backtesting parameters
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.transaction_cost = self.config.get('transaction_cost', 0.001)
        self.slippage = self.config.get('slippage', 0.0005)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        
        # Results storage
        self.results = {}
        self.trades = []
        self.equity_curve = None
        self.drawdown_curve = None
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run backtest on historical data.
        
        Args:
            data: Multi-index DataFrame with OHLCV data.
            signals: DataFrame with trading signals.
            symbols: List of symbols to backtest.
            start_date: Start date for backtesting.
            end_date: End date for backtesting.
            
        Returns:
            Dictionary with backtest results.
        """
        logger.info(f"Starting backtest for {len(symbols)} symbols")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
            signals = signals[signals.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            signals = signals[signals.index <= end_date]
        
        # Initialize portfolio
        self.portfolio.reset(self.initial_capital)
        
        # Process each trading day
        for date in data.index:
            if date not in signals.index:
                continue
            
            # Get market data for this date
            market_data = {}
            for symbol in symbols:
                if (symbol, 'Close') in data.columns:
                    market_data[symbol] = {
                        'price': data.loc[date, (symbol, 'Close')],
                        'volume': data.loc[date, (symbol, 'Volume')] if (symbol, 'Volume') in data.columns else 0
                    }
            
            # Get signals for this date
            date_signals = signals.loc[date]
            
            # Process signals for each symbol
            for symbol in symbols:
                if symbol not in market_data:
                    continue
                
                # Get signal for this symbol
                signal_cols = [col for col in signals.columns if symbol in col and ('Signal' in col or 'Label' in col)]
                if not signal_cols:
                    continue
                
                signal_value = date_signals[signal_cols[0]]
                if pd.isna(signal_value):
                    continue
                
                # Execute trade
                self._execute_trade(
                    symbol=symbol,
                    signal=signal_value,
                    price=market_data[symbol]['price'],
                    volume=market_data[symbol]['volume'],
                    date=date
                )
            
            # Update portfolio value
            self.portfolio.update_portfolio_value(date, market_data)
        
        # Calculate results
        self.results = self._calculate_results(data, symbols)
        
        logger.info("Backtest completed successfully")
        return self.results
    
    def _execute_trade(
        self,
        symbol: str,
        signal: float,
        price: float,
        volume: float,
        date: pd.Timestamp
    ) -> None:
        """Execute a trade based on signal.
        
        Args:
            symbol: Symbol to trade.
            signal: Trading signal (-1, 0, 1).
            price: Current price.
            volume: Current volume.
            date: Trading date.
        """
        if pd.isna(signal) or signal == 0:
            return
        
        # Calculate position size
        current_value = self.portfolio.get_position_value(symbol)
        max_position_value = self.portfolio.total_value * self.max_position_size
        
        if signal > 0:  # Buy signal
            if current_value < 0:  # Close short position
                self._close_position(symbol, price, date, "close_short")
            
            # Open long position
            position_size = min(max_position_value, self.portfolio.cash * 0.95)  # Use 95% of cash
            if position_size > 0:
                self._open_position(symbol, price, position_size, date, "long")
        
        elif signal < 0:  # Sell signal
            if current_value > 0:  # Close long position
                self._close_position(symbol, price, date, "close_long")
            
            # Open short position
            position_size = min(max_position_value, self.portfolio.cash * 0.95)
            if position_size > 0:
                self._open_position(symbol, price, position_size, date, "short")
    
    def _open_position(
        self,
        symbol: str,
        price: float,
        position_size: float,
        date: pd.Timestamp,
        position_type: str
    ) -> None:
        """Open a new position.
        
        Args:
            symbol: Symbol to trade.
            price: Entry price.
            position_size: Position size in dollars.
            date: Trading date.
            position_type: Type of position ("long" or "short").
        """
        # Calculate shares
        shares = position_size / price
        
        # Apply transaction costs and slippage
        cost = self.execution_engine.calculate_cost(price, shares, self.transaction_cost, self.slippage)
        
        # Update portfolio
        self.portfolio.add_position(symbol, shares, price, date, position_type)
        self.portfolio.cash -= cost
        
        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'open',
            'position_type': position_type,
            'shares': shares,
            'price': price,
            'value': position_size,
            'cost': cost
        }
        self.trades.append(trade)
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        action: str
    ) -> None:
        """Close an existing position.
        
        Args:
            symbol: Symbol to trade.
            price: Exit price.
            date: Trading date.
            action: Action type ("close_long" or "close_short").
        """
        position = self.portfolio.positions.get(symbol)
        if not position:
            return
        
        # Calculate proceeds
        proceeds = position['shares'] * price
        
        # Apply transaction costs and slippage
        cost = self.execution_engine.calculate_cost(price, position['shares'], self.transaction_cost, self.slippage)
        net_proceeds = proceeds - cost
        
        # Calculate P&L
        if position['position_type'] == 'long':
            pnl = net_proceeds - position['cost']
        else:  # short
            pnl = position['cost'] - net_proceeds
        
        # Update portfolio
        self.portfolio.cash += net_proceeds
        self.portfolio.remove_position(symbol)
        
        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'close',
            'position_type': position['position_type'],
            'shares': position['shares'],
            'price': price,
            'value': proceeds,
            'cost': cost,
            'pnl': pnl
        }
        self.trades.append(trade)
    
    def _calculate_results(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Calculate backtest results.
        
        Args:
            data: Historical data.
            symbols: List of symbols.
            
        Returns:
            Dictionary with backtest results.
        """
        logger.info("Calculating backtest results")
        
        # Get equity curve
        self.equity_curve = self.portfolio.get_equity_curve()
        
        # Calculate trading metrics
        trading_results = self.trading_metrics.calculate_metrics(
            self.equity_curve,
            self.trades,
            self.initial_capital
        )
        
        # Calculate risk metrics
        risk_results = self.risk_metrics.calculate_metrics(
            self.equity_curve,
            self.initial_capital
        )
        
        # Calculate drawdown
        self.drawdown_curve = self.risk_metrics.calculate_drawdown(self.equity_curve)
        
        # Combine results
        results = {
            'trading_metrics': trading_results,
            'risk_metrics': risk_results,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'trades': self.trades,
            'final_portfolio_value': self.portfolio.total_value,
            'total_return': (self.portfolio.total_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in self.trades if t.get('pnl', 0) < 0])
        }
        
        # Calculate win rate
        if results['num_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['num_trades']
        else:
            results['win_rate'] = 0.0
        
        logger.info(f"Backtest results calculated: {results['total_return']:.2%} total return")
        return results
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades.
        
        Returns:
            DataFrame with trade summary.
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df = trades_df.sort_values('date')
        
        return trades_df
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Dictionary with performance summary.
        """
        if not self.results:
            return {}
        
        summary = {
            'Total Return': f"{self.results['total_return']:.2%}",
            'Final Portfolio Value': f"${self.results['final_portfolio_value']:,.2f}",
            'Number of Trades': self.results['num_trades'],
            'Win Rate': f"{self.results['win_rate']:.2%}",
            'Winning Trades': self.results['winning_trades'],
            'Losing Trades': self.results['losing_trades']
        }
        
        # Add trading metrics
        if 'trading_metrics' in self.results:
            trading_metrics = self.results['trading_metrics']
            summary.update({
                'Sharpe Ratio': f"{trading_metrics.get('sharpe_ratio', 0):.2f}",
                'Sortino Ratio': f"{trading_metrics.get('sortino_ratio', 0):.2f}",
                'Calmar Ratio': f"{trading_metrics.get('calmar_ratio', 0):.2f}",
                'Max Drawdown': f"{trading_metrics.get('max_drawdown', 0):.2%}"
            })
        
        return summary
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot backtest results.
        
        Args:
            save_path: Path to save the plot.
        """
        import matplotlib.pyplot as plt
        
        if self.equity_curve is None:
            logger.warning("No equity curve available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot equity curve
        ax1.plot(self.equity_curve.index, self.equity_curve.values, label='Portfolio Value', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        if self.drawdown_curve is not None:
            ax2.fill_between(self.drawdown_curve.index, self.drawdown_curve.values, 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax2.plot(self.drawdown_curve.index, self.drawdown_curve.values, color='red', linewidth=1)
            ax2.set_title('Drawdown Over Time')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str) -> None:
        """Save backtest results to file.
        
        Args:
            filepath: Path to save results.
        """
        import json
        from pathlib import Path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = self.results.copy()
        
        # Convert DataFrames to dict
        if 'equity_curve' in results_to_save and results_to_save['equity_curve'] is not None:
            results_to_save['equity_curve'] = results_to_save['equity_curve'].to_dict()
        
        if 'drawdown_curve' in results_to_save and results_to_save['drawdown_curve'] is not None:
            results_to_save['drawdown_curve'] = results_to_save['drawdown_curve'].to_dict()
        
        # Convert trades to list of dicts
        if 'trades' in results_to_save:
            results_to_save['trades'] = [
                {k: v.isoformat() if isinstance(v, pd.Timestamp) else v 
                 for k, v in trade.items()} 
                for trade in results_to_save['trades']
            ]
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
