# Stock Screener

A Python-based stock screening tool to identify strong performing assets using technical analysis and momentum indicators.

## Features

- **Multi-factor Analysis**: Combines price momentum, volume trends, and technical indicators
- **Customizable Filters**: Set your own thresholds for each screening metric
- **Sector Comparison**: Compare performance relative to sector peers
- **Risk Metrics**: Calculate volatility and risk-adjusted returns
- **Export Results**: Save screening results to CSV or Excel
- **Visualization**: Generate charts for top performers

## Screening Metrics

### Momentum Indicators
- **Price Performance**: 1-month, 3-month, 6-month, and 1-year returns
- **Relative Strength**: Performance vs benchmark index
- **Volume Momentum**: Volume trend analysis

### Technical Indicators
- **RSI (Relative Strength Index)**: Overbought/oversold conditions
- **Moving Averages**: 50-day and 200-day SMAs
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility-based bands

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

## Data Sources

- **Primary**: Yahoo Finance (via yfinance)
- **Alternative**: Alpha Vantage, IEX Cloud (with API key)
