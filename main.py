"""Main orchestrator for the stock-screener pipeline.

This file orchestrates modules. It stays slim and delegates business logic
to modules per the AGENTS Coding Contract.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union, Optional

import pandas as pd

import DataDownloader
import EvaluationMetrics
import Config
import AssetRanking
import AssetAllocation
import StrategyBacktest
import MonteCarloSimulation
import ResultVisualization
import os

os.chdir('/home/user/stock-screener/')

def RunPipeline(
    TickerSymbols: Sequence[str],
    StartDate: Union[str, pd.Timestamp],
    EndDate: Union[str, pd.Timestamp],
    Interval: str,
) -> Dict[str, pd.DataFrame]:
    """Run data downloads for multiple tickers and return a mapping.

    This function wires configuration into the DataDownloader module for each ticker.
    """
    Results: Dict[str, pd.DataFrame] = {}
    for TickerSymbol in TickerSymbols:
        DataFrame = DataDownloader.DownloadTradingData(
            TickerSymbol=TickerSymbol,
            StartDate=StartDate,
            EndDate=EndDate,
            Interval=Interval,
        )
        Results[TickerSymbol] = DataFrame
    return Results


def ComputeMetricsSummary(
    Results: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Evaluate per-ticker metrics and return a concatenated summary frame.

    Persists the summary to `Outputs/MetricsSummary.csv` when available.
    Returns None when no metrics could be computed.
    """
    MetricsFrames = []
    for TickerSymbol, DataFrame in Results.items():
        try:
            MetricFrame = EvaluationMetrics.EvaluateSingleTicker(
                TickerSymbol=TickerSymbol,
                TradingData=DataFrame,
            )
            MetricsFrames.append(MetricFrame)
        except Exception as Error:
            print(f"Failed to evaluate metrics for {TickerSymbol}: {Error}")

    if not MetricsFrames:
        print("No metrics computed; skipping ranking, allocation, and simulations.")
        return None

    MetricsSummary = pd.concat(MetricsFrames, ignore_index=True)
    MetricsSummary.to_csv('Outputs/MetricsSummary.csv', index=False)
    return MetricsSummary


def ComputeRanking(MetricsSummary: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Rank assets from metrics and persist the ranking frame."""
    try:
        RankingFrame = AssetRanking.RankAssets(MetricsSummary=MetricsSummary)
        RankingFrame.to_csv('Outputs/RankingFrame.csv', index=False)
        return RankingFrame
    except Exception as Error:
        print(f"Failed to rank assets: {Error}")
        return None


def ComputeAllocation(
    RankingFrame: pd.DataFrame,
    PriceDataByTicker: Dict[str, pd.DataFrame],
    LookbackPeriods: int = 60,
) -> Optional[pd.DataFrame]:
    """Allocate assets from ranking and price data and persist the allocation."""
    try:
        AllocationFrame = AssetAllocation.AllocateAssets(
            RankingFrame=RankingFrame,
            PriceDataByTicker=PriceDataByTicker,
            LookbackPeriods=LookbackPeriods,
        )
        AllocationFrame.to_csv('Outputs/AssetAllocation.csv', index=False)
        return AllocationFrame
    except Exception as Error:
        print(f"Failed to allocate assets: {Error}")
        return None


def RunBacktests(AllocationFrame: pd.DataFrame) -> None:
    """Run SMA crossover backtests and persist summary if available."""
    try:
        BacktestSummary = StrategyBacktest.RunSmaCrossoverBacktest(AllocationFrame=AllocationFrame)
        if isinstance(BacktestSummary, dict) and len(BacktestSummary) > 0:
            pd.DataFrame([BacktestSummary]).to_csv('Outputs/BacktestSummary.csv', index=False)
    except Exception as Error:
        print(f"Failed to run backtests: {Error}")


def RunMonteCarlo(RankingFrame: pd.DataFrame) -> None:
    """Run Monte Carlo simulations and append results to the backtest summary file."""
    try:
        MonteCarloSimulation.RunMonteCarloRandomEqualWeight(
            RankingFrame=RankingFrame,
            SampleFraction=0.5,
            Iterations=50,
            OutputPath='Outputs/BacktestSummary.csv',
            RandomSeed=None,
        )
    except Exception as Error:
        print(f"Failed to run Monte Carlo simulation: {Error}")


if __name__ == "__main__":
    # Drive parameters via YAML config for ad-hoc runs without embedding logic here.
    try:
        AppConfig = Config.LoadConfig()
    except Exception as Error:
        print(f"Failed to load configuration: {Error}")
        raise

    RawSymbols = AppConfig.get("TickerSymbols", [])
    if isinstance(RawSymbols, str):
        TickerSymbols = [RawSymbols]
    elif isinstance(RawSymbols, (list, tuple)):
        TickerSymbols = [str(Symbol) for Symbol in RawSymbols]
    else:
        raise ValueError("TickerSymbols must be a string or a list of strings in config.yaml")

    StartDate = AppConfig.get("StartDate", "2023-01-01")
    EndDate = AppConfig.get("EndDate", "2023-12-31")
    Interval = str(AppConfig.get("Interval", "1d"))

    Results = RunPipeline(TickerSymbols, StartDate, EndDate, Interval)

    # Evaluate metrics per ticker and persist summary
    MetricsSummary = ComputeMetricsSummary(Results)
    if MetricsSummary is None:
        raise SystemExit(0)

    # Rank assets and persist ranking
    RankingFrame = ComputeRanking(MetricsSummary)
    if RankingFrame is None:
        raise SystemExit(0)

    # Allocate assets and persist allocation
    AllocationFrame = ComputeAllocation(RankingFrame, Results, LookbackPeriods=60)
    if AllocationFrame is None:
        raise SystemExit(0)

    # Run backtests and Monte Carlo simulations
    RunBacktests(AllocationFrame)
    RunMonteCarlo(RankingFrame)

    # Plot result visualization for a selected column from BacktestSummary.csv
    try:
        SelectedPlotColumn = str(AppConfig.get("PlotColumn", "TotalReturnPct"))
        ResultVisualization.PlotBacktestColumn(ColumnName=SelectedPlotColumn)
    except Exception as Error:
        print(f"Failed to plot results: {Error}")
