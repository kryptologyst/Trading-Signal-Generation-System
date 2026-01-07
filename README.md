# Trading Signal Generation System

A research-focused trading signal generation system using technical indicators and machine learning for educational and research purposes only.

## ⚠️ DISCLAIMER

**This software is for educational and research purposes only. It is not intended as investment advice and should not be used for actual trading without proper risk management and professional consultation. Past performance does not guarantee future results, and trading involves substantial risk of loss.**

## Features

- **Technical Indicators**: MACD, RSI, Bollinger Bands, Stochastic, Williams %R, ATR, ADX, CCI, and more
- **Feature Engineering**: Comprehensive feature creation including price, volume, time, and cross-asset features
- **Machine Learning Models**: XGBoost, LightGBM, LSTM, Transformer, and ensemble methods
- **Backtesting**: Realistic backtesting with transaction costs, slippage, and risk management
- **Interactive Demo**: Streamlit-based web interface for exploration
- **Risk Management**: Position sizing, drawdown control, and risk metrics
- **Reproducible**: Deterministic seeding and comprehensive logging

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Trading-Signal-Generation-System.git
cd Trading-Signal-Generation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```

4. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

### Development Installation

For development with additional tools:

```bash
pip install -e ".[dev,full]"
```

## Usage

### Command Line Interface

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load market data for configured symbols
2. Generate technical indicators and features
3. Create trading signals using multiple methods
4. Train machine learning models
5. Run backtests with realistic costs
6. Display results and save outputs

### Configuration

Edit `configs/config.yaml` to customize:

- **Data sources**: Symbols, date ranges, data sources
- **Feature engineering**: Technical indicator parameters, feature selection
- **Models**: Model types, hyperparameters, cross-validation
- **Backtesting**: Capital, costs, risk management

### Interactive Demo

Launch the Streamlit demo for interactive exploration:

```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive parameter configuration
- Real-time data visualization
- Model performance comparison
- Backtest results analysis

## Project Structure

```
trading-signal-generation/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── features/                 # Feature engineering
│   ├── labels/                   # Label generation
│   ├── models/                   # Model implementations
│   ├── backtest/                 # Backtesting framework
│   └── utils/                    # Utilities and configuration
├── configs/                      # Configuration files
├── demo/                         # Streamlit demo application
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests
├── assets/                       # Generated assets and plots
├── outputs/                      # Analysis outputs
├── data/                         # Data cache
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Data Sources

The system supports multiple data sources:

- **Yahoo Finance**: Free market data via `yfinance`
- **Synthetic Data**: Generated data for testing and demonstration
- **Custom Sources**: Extensible for additional data providers

## Models

### Technical Strategies
- MACD + RSI crossover
- Moving average crossover
- Bollinger Bands mean reversion
- Momentum strategies

### Machine Learning Models
- **XGBoost**: Gradient boosting for tabular data
- **LightGBM**: Fast gradient boosting
- **LSTM**: Recurrent neural networks for time series
- **Transformer**: Attention-based models
- **Ensemble**: Voting and stacking methods

## Backtesting

The backtesting framework includes:

- **Realistic Execution**: Transaction costs, slippage, market impact
- **Risk Management**: Position sizing, drawdown control
- **Performance Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, VaR
- **Trade Analysis**: Win rate, profit factor, expectancy

## Risk Management

Built-in risk management features:

- Position sizing limits
- Maximum drawdown controls
- Stop-loss and take-profit levels
- Portfolio rebalancing
- Risk-adjusted performance metrics

## Evaluation Metrics

### Trading Performance
- Total return and annualized return
- Sharpe ratio and Sortino ratio
- Maximum drawdown and recovery factor
- Win rate and profit factor
- Kelly percentage and expectancy

### Risk Metrics
- Value at Risk (VaR) and Conditional VaR
- Tail ratio and common sense ratio
- Drawdown duration and stability
- Skewness and kurtosis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Code Quality

The project uses:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Pre-commit**: Git hooks for code quality
- **Type hints**: Static type checking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{trading_signal_generation,
  title={Trading Signal Generation System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Trading-Signal-Generation-System}
}
```

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

## Changelog

### v0.1.0 (2024-01-01)
- Initial release
- Basic technical indicators
- XGBoost and LightGBM models
- Streamlit demo interface
- Comprehensive backtesting framework

## Acknowledgments

- Built with Python, pandas, scikit-learn, XGBoost, LightGBM, PyTorch
- Inspired by quantitative finance research and best practices
- Thanks to the open-source community for excellent tools and libraries

---

**Remember: This software is for educational purposes only. Always consult with qualified financial professionals before making investment decisions.**
# Trading-Signal-Generation-System
