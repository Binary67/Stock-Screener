"""StrategyBacktest module: run SMA crossover backtests per allocation.

Behavior
--------
- Accepts only the asset allocation DataFrame produced by `AssetAllocation.AllocateAssets`.
- For each allocated ticker, downloads OHLCV data using `DataDownloader` based on
  config.yaml (StartDate, EndDate, Interval).
- Uses `backtesting.py` to run a simple long-only SMA crossover strategy for each
  ticker, with initial cash sized by `AllocationPercent` of a total InitialCapital
  (default 100_000 if not provided in config).
- Returns combined (portfolio-level) metrics only; no per-ticker printing.
  Combined metrics include total final equity, total return percent, averages
  and allocation-weighted averages of Sharpe, Sortino, Max Drawdown, and more.

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


def RunSmaCrossoverBacktest(AllocationFrame: pd.DataFrame) -> dict:
    """Run SMA crossover backtests for each allocated ticker and return metrics.

    Parameters
    ----------
    AllocationFrame: pd.DataFrame
        Must contain columns `TickerSymbol` and `AllocationPercent`.

    Returns
    -------
    dict
        Mapping of combined portfolio-level metrics, including:
        - `InitialCapital`, `TotalFinalEquity`, `TotalReturnPct`
        - `AverageSharpeRatio`, `AverageSortinoRatio`, `AverageMaxDrawdown`
        - `AverageWinRatePct`, `AverageProfitFactor`
        - `WeightedSharpeRatio`, `WeightedSortinoRatio`, `WeightedMaxDrawdown`, `WeightedWinRatePct`
        - `TotalTrades`, `NumTickers`, `StartDate`, `EndDate`, `Interval`
    """

    try:
        from backtesting import Backtest, Strategy  # type: ignore[import-not-found]
        from backtesting.lib import crossover  # type: ignore[import-not-found]
    except Exception as ImportError:  # pragma: no cover - runtime guidance only
        print(
            "backtesting.py is required. Install with: .venv/bin/python3 -m pip install backtesting"
        )
        return {}

    # Validate allocation input
    if AllocationFrame is None or AllocationFrame.empty:
        print("AllocationFrame is empty; nothing to backtest.")
        return {}

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
        return {}

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
    ResultsByTicker: list[tuple[str, pd.Series, float]] = []  # (Ticker, Stats, Weight)

    # Helpers to safely extract metrics from backtesting.py results
    def _get_float(Stats: pd.Series, Keys: list[str]) -> float:
        for Key in Keys:
            try:
                if Key in Stats:
                    Value = float(Stats.get(Key))
                    return Value
            except Exception:
                continue
        return float("nan")

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
                finalize_trades=True
            )
            Stats = BacktestEngine.run()
        except Exception as Error:
            print(f"Backtest failed for {TickerSymbol}: {Error}")
            continue

        # Accumulate total final equity across tickers
        try:
            EquityFinal = float(Stats.get("Equity Final [$]", np.nan))
            if np.isfinite(EquityFinal):
                TotalFinalEquity += EquityFinal
        except Exception:
            pass

        try:
            ResultsByTicker.append(
                (
                    TickerSymbol,
                    Stats,  # type: ignore[arg-type]
                    float(AllocationPercent) / 100.0,  # weight in [0,1]
                )
            )
        except Exception:
            pass

    # Compute combined metrics (averages and weighted averages)
    AverageSharpeRatio: float = float("nan")
    AverageSortinoRatio: float = float("nan")
    AverageMaxDrawdown: float = float("nan")  # percent, typically negative
    AverageWinRatePct: float = float("nan")
    AverageProfitFactor: float = float("nan")
    TotalTrades: float = 0.0

    WeightedSharpeRatio: float = float("nan")
    WeightedSortinoRatio: float = float("nan")
    WeightedMaxDrawdown: float = float("nan")
    WeightedWinRatePct: float = float("nan")

    if len(ResultsByTicker) > 0:
        # Gather vectors
        SharpeVals: list[float] = []
        SortinoVals: list[float] = []
        MaxDdVals: list[float] = []
        WinRateVals: list[float] = []
        ProfitFactorVals: list[float] = []
        Weights: list[float] = []

        for _, Stats, Weight in ResultsByTicker:
            Sharpe = _get_float(Stats, ["Sharpe Ratio", "SharpeRatio"])
            Sortino = _get_float(Stats, ["Sortino Ratio", "SortinoRatio"])
            MaxDd = _get_float(Stats, ["Max. Drawdown [%]", "Max Drawdown [%]", "MaxDrawdown[%]", "MaxDrawdownPct"])  # percent
            WinRate = _get_float(Stats, ["Win Rate [%]", "WinRate[%]", "WinRatePct"])  # percent
            ProfitFactor = _get_float(Stats, ["Profit Factor", "ProfitFactor"])  # unitless
            Trades = _get_float(Stats, ["# Trades", "Trades", "NumTrades"])  # count

            if np.isfinite(Trades):
                TotalTrades += float(Trades)

            if np.isfinite(Sharpe):
                SharpeVals.append(Sharpe)
            if np.isfinite(Sortino):
                SortinoVals.append(Sortino)
            if np.isfinite(MaxDd):
                MaxDdVals.append(MaxDd)
            if np.isfinite(WinRate):
                WinRateVals.append(WinRate)
            if np.isfinite(ProfitFactor):
                ProfitFactorVals.append(ProfitFactor)
            Weights.append(float(Weight))

        def _safe_mean(Vals: list[float]) -> float:
            Arr = np.array(Vals, dtype=float)
            Finite = Arr[np.isfinite(Arr)]
            return float(Finite.mean()) if Finite.size > 0 else float("nan")

        def _safe_weighted_mean(Vals: list[float], Weights: list[float]) -> float:
            Arr = np.array(Vals, dtype=float)
            W = np.array(Weights, dtype=float)
            Mask = np.isfinite(Arr) & np.isfinite(W)
            Arr = Arr[Mask]
            W = W[Mask]
            TotalW = float(W.sum())
            if Arr.size == 0 or TotalW <= 0.0 or not np.isfinite(TotalW):
                return float("nan")
            return float(np.average(Arr, weights=W))

        AverageSharpeRatio = _safe_mean(SharpeVals)
        AverageSortinoRatio = _safe_mean(SortinoVals)
        AverageMaxDrawdown = _safe_mean(MaxDdVals)
        AverageWinRatePct = _safe_mean(WinRateVals)
        AverageProfitFactor = _safe_mean(ProfitFactorVals)

        WeightedSharpeRatio = _safe_weighted_mean(SharpeVals, Weights)
        WeightedSortinoRatio = _safe_weighted_mean(SortinoVals, Weights)
        WeightedMaxDrawdown = _safe_weighted_mean(MaxDdVals, Weights)
        WeightedWinRatePct = _safe_weighted_mean(WinRateVals, Weights)

    # Portfolio-level totals
    try:
        TotalReturnPct = ((TotalFinalEquity / InitialCapital) - 1.0) * 100.0
    except Exception:
        TotalReturnPct = float("nan")

    CombinedMetrics = {
        "InitialCapital": float(InitialCapital),
        "TotalFinalEquity": float(TotalFinalEquity),
        "TotalReturnPct": float(TotalReturnPct),
        "AverageSharpeRatio": float(AverageSharpeRatio),
        "AverageSortinoRatio": float(AverageSortinoRatio),
        "AverageMaxDrawdown": float(AverageMaxDrawdown),
        "AverageWinRatePct": float(AverageWinRatePct),
        "AverageProfitFactor": float(AverageProfitFactor),
        "WeightedSharpeRatio": float(WeightedSharpeRatio),
        "WeightedSortinoRatio": float(WeightedSortinoRatio),
        "WeightedMaxDrawdown": float(WeightedMaxDrawdown),
        "WeightedWinRatePct": float(WeightedWinRatePct),
        "TotalTrades": float(TotalTrades),
        "NumTickers": int(len(ResultsByTicker)),
        "StartDate": StartDate,
        "EndDate": EndDate,
        "Interval": Interval,
    }

    return CombinedMetrics


__all__ = [
    "RunSmaCrossoverBacktest",
]
