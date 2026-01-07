# PROJECT 502: TRADING SIGNAL GENERATION - MODERNIZED VERSION

## ⚠️ DISCLAIMER
**This software is for educational and research purposes only. It is not intended as investment advice and should not be used for actual trading without proper risk management and professional consultation.**

## MODERNIZED FEATURES

This project has been completely refactored and modernized with:

### ✅ Core Improvements
- **Modern Python 3.10+** with type hints and comprehensive docstrings
- **Production-ready architecture** with modular design and clean separation of concerns
- **Comprehensive testing** with unit tests and integration tests
- **Reproducible results** with deterministic seeding and proper data splits
- **Risk management** with position sizing, drawdown control, and transaction costs

### ✅ Advanced Features
- **Multiple ML Models**: XGBoost, LightGBM, LSTM, Transformer, and ensemble methods
- **Comprehensive Technical Indicators**: MACD, RSI, Bollinger Bands, Stochastic, Williams %R, ATR, ADX, CCI, OBV, VWAP
- **Feature Engineering**: Price, volume, time, and cross-asset features with lag features
- **Realistic Backtesting**: Transaction costs, slippage, market impact, and execution delays
- **Interactive Demo**: Streamlit-based web interface for exploration and analysis

### ✅ Professional Structure
```
trading-signal-generation/
├── src/                    # Source code modules
├── configs/               # Configuration files
├── demo/                  # Streamlit demo application
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter notebooks
├── assets/                # Generated plots and results
├── outputs/               # Analysis outputs
├── data/                  # Data cache
├── main.py               # Main execution script
├── test_system.py        # System test script
├── install.sh            # Installation script
├── requirements.txt      # Dependencies
├── pyproject.toml        # Project configuration
└── README.md             # Comprehensive documentation
```

## 🛠️ QUICK START

### Installation
```bash
# Make installation script executable and run
chmod +x install.sh
./install.sh
```

### Run the System
```bash
# Run complete pipeline
python3 main.py

# Run system tests
python3 test_system.py

# Launch interactive demo
streamlit run demo/app.py
```

## ORIGINAL SIMPLE VERSION (for reference)

The original simple implementation using MACD and RSI:

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 1. Download historical stock data
stock_data = yf.download("AAPL", start="2019-01-01", end="2021-01-01")

# 2. Calculate MACD
stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

# 3. Calculate RSI
delta = stock_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# 4. Generate trading signals
stock_data['Buy_Signal'] = np.where(
    (stock_data['MACD'] > stock_data['Signal_Line']) & (stock_data['RSI'] < 30), 1, 0
)
stock_data['Sell_Signal'] = np.where(
    (stock_data['MACD'] < stock_data['Signal_Line']) & (stock_data['RSI'] > 70), 1, 0
)

# 5. Visualize results
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(stock_data['Close'], label='Stock Price', color='blue')
plt.scatter(stock_data.index[stock_data['Buy_Signal'] == 1], 
           stock_data['Close'][stock_data['Buy_Signal'] == 1], 
           marker='^', color='green', label='Buy Signal')
plt.scatter(stock_data.index[stock_data['Sell_Signal'] == 1], 
           stock_data['Close'][stock_data['Sell_Signal'] == 1], 
           marker='v', color='red', label='Sell Signal')
plt.title('Stock Price with Buy and Sell Signals')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(stock_data['RSI'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('RSI (Relative Strength Index)')
plt.legend()

plt.tight_layout()
plt.show()
```

## WHAT THE MODERNIZED VERSION DOES

### Data Pipeline
- **Multi-source data loading** with caching and preprocessing
- **Comprehensive feature engineering** with 50+ technical indicators
- **Proper time series splits** to prevent data leakage
- **Synthetic data generation** for testing and demonstration

### Machine Learning
- **Multiple model types**: Technical strategies, tree-based models, neural networks
- **Ensemble methods** with weighted voting and stacking
- **Feature selection** using mutual information, RFE, and regularization
- **Cross-validation** with purged and embargoed splits

### Backtesting & Risk Management
- **Realistic execution** with transaction costs and slippage
- **Position sizing** and risk controls
- **Comprehensive metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, VaR
- **Drawdown analysis** and recovery metrics

### Evaluation & Visualization
- **Interactive Streamlit demo** with real-time parameter adjustment
- **Comprehensive performance metrics** for both ML and trading performance
- **Feature importance analysis** and model interpretability
- **Professional reporting** with charts and statistical analysis

## CONFIGURATION

Edit `configs/config.yaml` to customize:
- **Data sources**: Symbols, date ranges, data providers
- **Feature engineering**: Technical indicator parameters
- **Models**: Model types, hyperparameters, cross-validation
- **Backtesting**: Capital, costs, risk management

## USAGE EXAMPLES

### Basic Usage
```python
from src.utils import load_config
from src.data import DataLoader
from src.features import FeatureEngineer
from src.models import ModelFactory
from src.backtest import Backtester

# Load configuration
config = load_config("configs/config.yaml")

# Load and preprocess data
data_loader = DataLoader()
data = data_loader.load_yfinance_data(
    symbols=config.data.symbols,
    start_date=config.data.start_date,
    end_date=config.data.end_date
)

# Create features
feature_engineer = FeatureEngineer()
features = feature_engineer.engineer_features(data, config.data.symbols)

# Train models
model = ModelFactory.create_model("xgboost")
model.fit(features, labels)

# Run backtest
backtester = Backtester(config.backtest.__dict__)
results = backtester.run_backtest(data, signals, config.data.symbols)
```

### Interactive Demo
```bash
streamlit run demo/app.py
```

## TESTING

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run system test
python3 test_system.py
```

## DOCUMENTATION

- **README.md**: Comprehensive setup and usage guide
- **DISCLAIMER.md**: Legal disclaimers and risk warnings
- **notebooks/demo.ipynb**: Interactive Jupyter notebook
- **Code documentation**: Type hints and docstrings throughout

## LEGAL & COMPLIANCE

This software includes:
- **Prominent disclaimers** in all interfaces and documentation
- **Educational focus** with clear research-only purpose
- **Risk warnings** about trading and investment risks
- **Professional compliance** with financial software standards

## NEXT STEPS

1. **Install dependencies**: Run `./install.sh`
2. **Test the system**: Run `python3 test_system.py`
3. **Explore the demo**: Run `streamlit run demo/app.py`
4. **Read the documentation**: Check `README.md` for detailed usage
5. **Customize configuration**: Edit `configs/config.yaml`

---

**Remember: This software is for educational and research purposes only. Always consult with qualified financial professionals before making investment decisions.**

