# Stock Screener

This project aims to rank and allocate investments among a list of ticker symbols using performance metrics. The pipeline is designed for extensibility so that new metrics can be added easily.

## Features

- Filter the top N tickers based on historical returns, technical indicators, or other metrics.
- Allocate a percentage of equity to each selected asset based on its ranking.
- Backtest the strategy using `backtesting.py`.
- Compare with a simple buy-and-hold strategy where the top N assets receive equal allocations.

## Getting Started

1. Provide a list of ticker symbols.
2. Define metrics for ranking performance.
3. Run the allocation and backtesting process.

## Future Work

- Integrate additional performance metrics (e.g., fundamental analysis).
- Expand the backtesting engine for more complex strategies.