"""Streamlit demo application for trading signal generation.

This demo provides an interactive interface for exploring the trading signal
generation system.

DISCLAIMER: This software is for educational and research purposes only.
It is not intended as investment advice and should not be used for actual trading
without proper risk management and professional consultation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import Config, load_config, setup_logging, set_seeds
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer, FeatureSelector
from src.labels import create_labels
from src.models import ModelFactory, TechnicalStrategy, XGBoostModel, LightGBMModel
from src.backtest import Backtester

# Register models
ModelFactory.register_model("technical", TechnicalStrategy)
ModelFactory.register_model("xgboost", XGBoostModel)
ModelFactory.register_model("lightgbm", LightGBMModel)

# Page configuration
st.set_page_config(
    page_title="Trading Signal Generation Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main demo application."""
    
    # Header
    st.markdown('<div class="main-header">Trading Signal Generation Demo</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This software is for educational and research purposes only. 
        It is not intended as investment advice and should not be used for actual trading 
        without proper risk management and professional consultation.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Data selection
        st.subheader("Data Selection")
        symbols = st.multiselect(
            "Select Symbols",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
            default=["AAPL", "MSFT", "GOOGL"]
        )
        
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
        
        # Model selection
        st.subheader("Model Selection")
        selected_models = st.multiselect(
            "Select Models",
            ["technical", "xgboost", "lightgbm"],
            default=["technical", "xgboost"]
        )
        
        # Feature selection
        st.subheader("Feature Engineering")
        include_technical = st.checkbox("Technical Indicators", value=True)
        include_price = st.checkbox("Price Features", value=True)
        include_volume = st.checkbox("Volume Features", value=True)
        include_time = st.checkbox("Time Features", value=True)
        
        # Backtest parameters
        st.subheader("Backtest Parameters")
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
        slippage = st.slider("Slippage (%)", 0.0, 1.0, 0.05) / 100
        max_position_size = st.slider("Max Position Size (%)", 1, 50, 10) / 100
    
    # Main content
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            try:
                # Run the analysis
                results = run_analysis(
                    symbols=symbols,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    selected_models=selected_models,
                    include_technical=include_technical,
                    include_price=include_price,
                    include_volume=include_volume,
                    include_time=include_time,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                    max_position_size=max_position_size
                )
                
                # Display results
                display_results(results)
                
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)
    
    # Display sample data if no analysis has been run
    else:
        st.info("Configure the parameters in the sidebar and click 'Run Analysis' to start.")
        
        # Show sample data
        if st.checkbox("Show Sample Data"):
            show_sample_data()


def run_analysis(symbols, start_date, end_date, selected_models, include_technical, 
                include_price, include_volume, include_time, initial_capital, 
                transaction_cost, slippage, max_position_size):
    """Run the complete analysis pipeline."""
    
    # Set random seeds
    set_seeds(42)
    
    # Load data
    data_loader = DataLoader(cache_dir="data/cache")
    market_data = data_loader.load_yfinance_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(market_data)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    features = feature_engineer.engineer_features(
        data=processed_data,
        symbols=symbols,
        include_technical=include_technical,
        include_price=include_price,
        include_volume=include_volume,
        include_time=include_time
    )
    
    # Generate labels
    labels = create_labels(
        data=processed_data,
        symbols=symbols,
        method="macd_rsi"
    )
    
    # Train models
    models = {}
    model_metrics = {}
    
    for symbol in symbols:
        if f"{symbol}_Signal" in labels.columns:
            symbol_features = features[[col for col in features.columns if symbol in col]]
            symbol_labels = labels[f"{symbol}_Signal"]
            
            # Split data
            split_point = int(len(symbol_features) * 0.8)
            X_train = symbol_features.iloc[:split_point]
            X_test = symbol_features.iloc[split_point:]
            y_train = symbol_labels.iloc[:split_point]
            y_test = symbol_labels.iloc[split_point:]
            
            symbol_models = {}
            symbol_metrics = {}
            
            for model_name in selected_models:
                model = ModelFactory.create_model(model_name)
                model.fit(X_train, y_train)
                symbol_models[model_name] = model
                
                # Evaluate model
                train_metrics = model.evaluate(X_train, y_train)
                test_metrics = model.evaluate(X_test, y_test)
                symbol_metrics[model_name] = {
                    'train': train_metrics,
                    'test': test_metrics
                }
            
            models[symbol] = symbol_models
            model_metrics[symbol] = symbol_metrics
    
    # Run backtests
    backtest_results = {}
    
    for symbol in symbols:
        if symbol in models:
            # Generate signals for backtesting
            symbol_features = features[[col for col in features.columns if symbol in col]]
            split_point = int(len(symbol_features) * 0.8)
            X_test = symbol_features.iloc[split_point:]
            
            test_signals = pd.DataFrame(index=X_test.index)
            for model_name, model in models[symbol].items():
                predictions = model.predict(X_test)
                test_signals[f"{symbol}_{model_name}_Signal"] = predictions
            
            # Run backtest
            backtest_config = {
                'initial_capital': initial_capital,
                'transaction_cost': transaction_cost,
                'slippage': slippage,
                'max_position_size': max_position_size
            }
            
            backtester = Backtester(backtest_config)
            backtest_results[symbol] = backtester.run_backtest(
                data=processed_data,
                signals=test_signals,
                symbols=[symbol],
                start_date=X_test.index[0].strftime('%Y-%m-%d'),
                end_date=X_test.index[-1].strftime('%Y-%m-%d')
            )
    
    return {
        'market_data': processed_data,
        'features': features,
        'labels': labels,
        'models': models,
        'model_metrics': model_metrics,
        'backtest_results': backtest_results
    }


def display_results(results):
    """Display analysis results."""
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Analysis", "Model Performance", "Backtest Results"])
    
    with tab1:
        st.header("Analysis Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", len(results['market_data']))
        
        with col2:
            st.metric("Features Created", len(results['features'].columns))
        
        with col3:
            st.metric("Symbols Analyzed", len(results['models']))
        
        with col4:
            total_trades = sum(
                result.get('num_trades', 0) 
                for result in results['backtest_results'].values()
            )
            st.metric("Total Trades", total_trades)
        
        # Model performance summary
        st.subheader("Model Performance Summary")
        
        performance_data = []
        for symbol, metrics in results['model_metrics'].items():
            for model_name, model_metrics in metrics.items():
                performance_data.append({
                    'Symbol': symbol,
                    'Model': model_name,
                    'Accuracy': model_metrics['test'].get('accuracy', 0),
                    'F1 Score': model_metrics['test'].get('f1', 0),
                    'Precision': model_metrics['test'].get('precision', 0),
                    'Recall': model_metrics['test'].get('recall', 0)
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
    
    with tab2:
        st.header("Data Analysis")
        
        # Price charts
        for symbol in results['market_data'].columns.get_level_values(0).unique():
            if (symbol, 'Close') in results['market_data'].columns:
                st.subheader(f"{symbol} Price Chart")
                
                close_prices = results['market_data'][(symbol, 'Close')]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=close_prices.index,
                    y=close_prices.values,
                    mode='lines',
                    name=f'{symbol} Close Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if results['models']:
            st.subheader("Feature Importance")
            
            # Get feature importance from the first model
            first_symbol = list(results['models'].keys())[0]
            first_model = list(results['models'][first_symbol].values())[0]
            
            importance_df = first_model.get_feature_importance()
            if importance_df is not None:
                fig = px.bar(
                    importance_df.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 20 Most Important Features"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Performance")
        
        # Model comparison
        for symbol in results['model_metrics']:
            st.subheader(f"{symbol} Model Performance")
            
            symbol_metrics = results['model_metrics'][symbol]
            
            # Create comparison chart
            models = list(symbol_metrics.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            fig = go.Figure()
            
            for metric in metrics:
                values = [symbol_metrics[model]['test'].get(metric, 0) for model in models]
                fig.add_trace(go.Bar(
                    name=metric.title(),
                    x=models,
                    y=values
                ))
            
            fig.update_layout(
                title=f"{symbol} Model Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Backtest Results")
        
        # Backtest summary
        backtest_summary = []
        for symbol, result in results['backtest_results'].items():
            if result:
                backtest_summary.append({
                    'Symbol': symbol,
                    'Total Return': f"{result.get('total_return', 0):.2%}",
                    'Sharpe Ratio': f"{result.get('trading_metrics', {}).get('sharpe_ratio', 0):.2f}",
                    'Max Drawdown': f"{result.get('trading_metrics', {}).get('max_drawdown', 0):.2%}",
                    'Win Rate': f"{result.get('win_rate', 0):.2%}",
                    'Number of Trades': result.get('num_trades', 0)
                })
        
        if backtest_summary:
            summary_df = pd.DataFrame(backtest_summary)
            st.dataframe(summary_df, use_container_width=True)
        
        # Equity curves
        for symbol, result in results['backtest_results'].items():
            if result and 'equity_curve' in result:
                st.subheader(f"{symbol} Equity Curve")
                
                equity_curve = result['equity_curve']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode='lines',
                    name=f'{symbol} Portfolio Value',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_sample_data():
    """Show sample data for demonstration."""
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'AAPL_Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
        'AAPL_Volume': np.random.randint(1000000, 10000000, len(dates)),
        'AAPL_RSI': np.random.uniform(20, 80, len(dates)),
        'AAPL_MACD': np.random.randn(len(dates)) * 2,
        'AAPL_Signal': np.random.choice([-1, 0, 1], len(dates), p=[0.2, 0.6, 0.2])
    })
    
    sample_data.set_index('Date', inplace=True)
    
    st.subheader("Sample Data")
    st.dataframe(sample_data.head(10))
    
    # Sample chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['AAPL_Close'],
        mode='lines',
        name='AAPL Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add buy/sell signals
    buy_signals = sample_data[sample_data['AAPL_Signal'] == 1]
    sell_signals = sample_data[sample_data['AAPL_Signal'] == -1]
    
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['AAPL_Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['AAPL_Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))
    
    fig.update_layout(
        title="Sample Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
