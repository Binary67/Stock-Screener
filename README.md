**Architecture Overview**
- **Purpose:** End-to-end stock screening pipeline that downloads price data, computes risk/return metrics, ranks assets, allocates weights, backtests a simple strategy, runs Monte Carlo variants, and visualizes results.
- **Orchestrator:** `main.py` coordinates modules; business logic resides in dedicated modules per the AGENTS Coding Contract.

**Pipeline Flow**
- **Download:** `DataDownloader.DownloadTradingData` pulls OHLCV via `yfinance` with on-disk CSV caching under `Caches/`.
- **Metrics:** `EvaluationMetrics.EvaluateSingleTicker` selects a price series and computes metrics (e.g., `SortinoRatio`, `SharpeRatio`, `NegMaxDrawdown`, `CAGR`, `UlcerPerformanceIndex`).
- **Ranking:** `AssetRanking.RankAssets` z-scores metrics, applies YAML-configured weights, and produces `CompositeScore` and ordinal `Ranking` (1 = best).
- **Allocation:** `AssetAllocation.AllocateAssets` blends inverse-volatility weighting with ranking-based weights and normalizes to precise `AllocationPercent` values summing to 100.
- **Backtest:** `StrategyBacktest.RunSmaCrossoverBacktest` runs a long-only SMA crossover per allocated ticker using `backtesting.py`, then aggregates portfolio-level metrics.
- **Monte Carlo:** `MonteCarloSimulation.RunMonteCarloRandomEqualWeight` samples tickers, builds equal-weight allocations, backtests, and appends results.
- **Visualization:** `ResultVisualization.PlotBacktestColumn` plots a chosen column from `Outputs/BacktestSummary.csv` to `Outputs/Plots/`.

**Modules**
- `main.py`: Orchestrates the sequence: load config → download data → compute metrics → rank → allocate → backtest → Monte Carlo → visualize. Writes intermediate CSVs to `Outputs/`.
- `Config.py`: Loads YAML configuration (`config.yaml`). Central place for configuration access to keep `main.py` slim.
- `DataDownloader.py`: Fetches price data via `yfinance`; flattens any multi-index to standard OHLCV columns; deterministic cache keying for repeatable CSV caches.
- `EvaluationMetrics.py`: Core metric computations and shared helpers for price selection and returns; provides a default metric registry.
- `AssetRanking.py`: Normalizes metrics with z-scores, merges user weights from config with defaults, computes `CompositeScore` and `Ranking`.
- `AssetAllocation.py`: Converts rankings and lookback volatility into allocations; supports `TopN` filtering and ranking/volatility blend from config.
- `StrategyBacktest.py`: SMA crossover strategy per ticker sized by allocation; aggregates combined metrics (e.g., `TotalReturnPct`, average and weighted risk stats).
- `MonteCarloSimulation.py`: Repeats backtests on random subsets using equal weights; appends each run to `BacktestSummary.csv`.
- `ResultVisualization.py`: Produces simple bar charts for a selected backtest metric.

**Configuration Model**
- `config.yaml` is the single source of truth consumed by modules via `Config.LoadConfig`.
- **Download:** `TickerSymbols`, `StartDate`, `EndDate`, `Interval`.
- **Ranking:** `RankingWeights` mapping per metric; unknown metrics are ignored, missing weights fall back to defaults.
- **Allocation:** Either scalar `RankingWeight` in [0,1] or `AllocationWeights: { Ranking, Volatility }` (normalized to a blend). Optional `TopN` as a fraction (0–1) or percent (e.g., `20`).
- **Backtest/Plots:** `InitialCapital` (optional), `PlotColumn` for visualization.

**Data Contracts**
- `DownloadTradingData` returns a `pd.DataFrame` indexed by datetime with OHLCV columns; caching is transparent to callers.
- `EvaluateSingleTicker` returns a one-row `pd.DataFrame` with `TickerSymbol` plus metric columns.
- `RankAssets` returns `TickerSymbol`, `CompositeScore`, `Ranking`.
- `AllocateAssets` returns `TickerSymbol`, `AllocationPercent` summing exactly to 100.0.
- `RunSmaCrossoverBacktest` returns a `dict` of combined portfolio metrics.
- `RunMonteCarloRandomEqualWeight` returns a `list[dict]` and appends to `Outputs/BacktestSummary.csv`.

**Storage Layout**
- `Caches/`: On-disk CSV cache per ticker/date/interval for data downloads.
- `Outputs/MetricsSummary.csv`: Per-ticker metrics concatenated across the universe.
- `Outputs/RankingFrame.csv`: Rankings with composite scores.
- `Outputs/AssetAllocation.csv`: Final allocation percentages.
- `Outputs/BacktestSummary.csv`: Portfolio-level metrics; Monte Carlo appends additional rows.
- `Outputs/Plots/`: Saved images from visualization (e.g., `Backtest_TotalReturnPct.png`).

**Design Principles**
- **Orchestrator-Only `main.py`:** All business logic lives in modules; `main.py` wires steps and persists artifacts.
- **PascalCase Variables:** All newly declared variables (locals, params, attributes, module constants) use PascalCase.
- **DRY Utilities:** Shared helpers (e.g., price selection, return calculations, normalization) are reused across modules to avoid duplication.
- **Graceful Degradation:** Modules catch and report errors without crashing the pipeline; file outputs are written when meaningful results exist.

**Extension Points**
- **New Metrics:** Add functions to `EvaluationMetrics` and register them via a custom metrics map or extend `DefaultMetricsRegistry`.
- **Alternate Ranking Schemes:** Modify weighting in `config.yaml` or extend `AssetRanking` to support new normalization/scoring.
- **Allocation Logic:** Adjust blend/filters via `config.yaml` or implement alternative weighting in `AssetAllocation`.
- **Strategies:** Implement additional strategies in `StrategyBacktest` and add an orchestration step in `main.py`.
- **Experiments:** Add new Monte Carlo samplers in `MonteCarloSimulation` or different plot variants in `ResultVisualization`.

