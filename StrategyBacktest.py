"""StrategyBacktest module: run SMA crossover backtests per allocation.

Behavior
--------
- Accepts only the asset allocation DataFrame produced by `AssetAllocation.AllocateAssets`.
- For each allocated ticker, downloads OHLCV data using `DataDownloader` based on
  config.yaml (StartDate, EndDate, Interval).
- Uses `backtesting.py` to run a simple long-only SMA crossover strategy for each
  ticker, with initial cash sized by `AllocationPercent` of a total InitialCapital
  (default 100_000 if not provided in config).
- Prints the library-provided metrics for each ticker, and prints a portfolio
  total equity (sum of each ticker’s final equity) at the end.

Notes
-----
- Follows the AGENTS Coding Contract: PascalCase for variables, DRY by reusing
  existing modules, and no business logic in `main.py`.
- If `backtesting` is not installed, prints an actionable message to install it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import Config
import DataDownloader


def RunSmaCrossoverBacktest(AllocationFrame: pd.DataFrame) -> None:
    """Run SMA crossover backtests for each allocated ticker and print metrics.

    Parameters
    ----------
    AllocationFrame: pd.DataFrame
        Must contain columns `TickerSymbol` and `AllocationPercent`.
    """

    try:
        from backtesting import Backtest, Strategy  # type: ignore[import-not-found]
        from backtesting.lib import crossover  # type: ignore[import-not-found]
    except Exception as ImportError:  # pragma: no cover - runtime guidance only
        print(
            "backtesting.py is required. Install with: .venv/bin/python3 -m pip install backtesting"
        )
        return

    # Validate allocation input
    if AllocationFrame is None or AllocationFrame.empty:
        print("AllocationFrame is empty; nothing to backtest.")
        return

    RequiredCols = {"TickerSymbol", "AllocationPercent"}
    MissingCols = RequiredCols - set(AllocationFrame.columns)
    if MissingCols:
        raise ValueError(
            f"AllocationFrame must include columns {RequiredCols}; missing {MissingCols}"
        )

    # Load run configuration (dates/interval and optional initial capital)
    try:
        AppConfig = Config.LoadConfig()
    except Exception:
        AppConfig = {}

    StartDate: Optional[str] = AppConfig.get("StartDate", None)
    EndDate: Optional[str] = AppConfig.get("EndDate", None)
    Interval: str = str(AppConfig.get("Interval", "1d"))
    InitialCapital: float = float(AppConfig.get("InitialCapital", 100_000.0))

    # Normalize allocation percentages to [0, 100] and filter zeros
    CleanFrame = AllocationFrame.copy()
    CleanFrame["AllocationPercent"] = pd.to_numeric(
        CleanFrame["AllocationPercent"], errors="coerce"
    ).fillna(0.0)
    CleanFrame = CleanFrame[CleanFrame["AllocationPercent"] > 0.0]
    if CleanFrame.empty:
        print("No positive allocations; nothing to backtest.")
        return

    # Simple SMA helper to avoid extra imports; compatible with backtesting.I
    def _sma(Series: pd.Series, Window: int) -> pd.Series:
        return pd.Series(Series, dtype=float).rolling(int(Window)).mean()

    # Strategy: long-only SMA crossover (fast over slow => long; cross down => flat)
    class SmaCross(Strategy):  # type: ignore[misc, valid-type]
        FastPeriod: int = 10
        SlowPeriod: int = 50

        def init(self) -> None:  # noqa: D401
            self.SmaFast = self.I(_sma, self.data.Close, self.FastPeriod)
            self.SmaSlow = self.I(_sma, self.data.Close, self.SlowPeriod)

        def next(self) -> None:
            if crossover(self.SmaFast, self.SmaSlow):
                # Bullish cross: go long
                if not self.position.is_long:
                    self.position.close()
                    self.buy()
            elif crossover(self.SmaSlow, self.SmaFast):
                # Bearish cross: exit to flat (no shorts)
                if self.position:
                    self.position.close()

    TotalFinalEquity: float = 0.0
    ResultsByTicker: list[tuple[str, pd.Series]] = []

    for _, Row in CleanFrame.iterrows():
        TickerSymbol = str(Row["TickerSymbol"]).strip()
        AllocationPercent = float(Row["AllocationPercent"])  # 0–100

        if AllocationPercent <= 0.0:
            continue

        AllocatedCash = float(InitialCapital * (AllocationPercent / 100.0))
        if AllocatedCash <= 0.0:
            continue

        try:
            PriceData = DataDownloader.DownloadTradingData(
                TickerSymbol=TickerSymbol,
                StartDate=StartDate if StartDate is not None else "1900-01-01",
                EndDate=EndDate if EndDate is not None else "2100-01-01",
                Interval=Interval,
            )
        except Exception as Error:
            print(f"Failed to download data for {TickerSymbol}: {Error}")
            continue

        if PriceData is None or PriceData.empty:
            print(f"No data for {TickerSymbol}; skipping.")
            continue

        # Ensure required columns for backtesting.py
        RequiredOhlc = ["Open", "High", "Low", "Close"]
        MissingOhlc = [C for C in RequiredOhlc if C not in PriceData.columns]
        if MissingOhlc:
            # Derive from Close if missing
            if "Close" not in PriceData.columns:
                print(f"Missing OHLC for {TickerSymbol}; skipping.")
                continue
            for C in MissingOhlc:
                PriceData[C] = PriceData["Close"]

        # Instantiate and run backtest for this ticker
        try:
            BacktestEngine = Backtest(
                PriceData,
                SmaCross,
                cash=AllocatedCash,
                commission=0.001,
                exclusive_orders=True,
            )
            Stats = BacktestEngine.run()
        except Exception as Error:
            print(f"Backtest failed for {TickerSymbol}: {Error}")
            continue

        # Print per-ticker stats emitted by backtesting.py
        print("\n=== Backtest Metrics:", TickerSymbol, f"(Allocated ${AllocatedCash:,.2f}) ===")
        try:
            # Stats is typically a pandas Series
            print(Stats.to_string())
        except Exception:
            print(Stats)

        # Accumulate total final equity across tickers
        try:
            EquityFinal = float(Stats.get("Equity Final [$]", np.nan))
            if np.isfinite(EquityFinal):
                TotalFinalEquity += EquityFinal
        except Exception:
            pass

        try:
            ResultsByTicker.append((TickerSymbol, Stats))  # type: ignore[arg-type]
        except Exception:
            pass

    # Portfolio-level summary (total equity)
    print("\n=== Portfolio Summary ===")
    print(f"Initial Capital: ${InitialCapital:,.2f}")
    print(f"Total Final Equity (sum of buckets): ${TotalFinalEquity:,.2f}")
    try:
        TotalReturnPct = ((TotalFinalEquity / InitialCapital) - 1.0) * 100.0
        print(f"Total Return [%] (naive sum basis): {TotalReturnPct:,.2f}%")
    except Exception:
        pass


__all__ = [
    "RunSmaCrossoverBacktest",
]

