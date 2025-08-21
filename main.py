"""Main orchestrator for the stock-screener pipeline.

This file orchestrates modules. It stays slim and delegates business logic
to modules per the AGENTS Coding Contract.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import pandas as pd

import DataDownloader
import Config


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
    for TickerSymbol, DataFrame in Results.items():
        print(f"{TickerSymbol}: {len(DataFrame)} rows; Columns: {list(DataFrame.columns)}")
        print(f"{TickerSymbol}: {DataFrame.head()}")
