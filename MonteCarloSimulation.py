"""MonteCarloSimulation module: random 50% selection with equal-weight backtest.

Behavior
--------
- Accepts a `RankingFrame` DataFrame (must include `TickerSymbol`).
- Randomly samples 50% of the tickers (minimum 1) regardless of ranking.
- Builds an equal-weight `AllocationFrame` that sums exactly to 100%.
- Runs the existing backtesting module with this allocation.
- Appends the resulting metrics row to `Outputs/BacktestSummary.csv`.

Notes
-----
- Follows the AGENTS Coding Contract: PascalCase variables and DRY. To ensure
  allocations sum exactly to 100.0, reuses the normalization helper from
  `AssetAllocation`.
- Keeps `main.py` slim by providing a single function to execute the full flow.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import os
import numpy as np
import pandas as pd

import AssetAllocation
import StrategyBacktest


def _unique_tickers(RankingFrame: pd.DataFrame) -> List[str]:
    """Extract unique ticker symbols from a RankingFrame."""
    if RankingFrame is None or RankingFrame.empty:
        return []
    if "TickerSymbol" not in RankingFrame.columns:
        raise ValueError("RankingFrame must include a 'TickerSymbol' column.")
    Series = RankingFrame["TickerSymbol"].astype(str).dropna()
    Unique = pd.Index(Series).astype(str).unique().tolist()
    return Unique


def _equal_weight_allocation(SelectedTickers: Iterable[str]) -> pd.DataFrame:
    """Return an equal-weight AllocationFrame for the provided tickers.

    - Uses `AssetAllocation._normalize_to_percent` for precise 100.0% sum.
    - Returns a DataFrame with columns: `TickerSymbol`, `AllocationPercent`.
    """
    Tickers = [str(T).strip() for T in SelectedTickers if str(T).strip()]
    if len(Tickers) == 0:
        return pd.DataFrame(columns=["TickerSymbol", "AllocationPercent"])

    # Score each ticker equally (1.0) then normalize to percentages
    ScoreByTicker = {T: 1.0 for T in Tickers}
    PercentByTicker = AssetAllocation._normalize_to_percent(ScoreByTicker)  # type: ignore[attr-defined]

    AllocationFrame = pd.DataFrame(
        {
            "TickerSymbol": list(PercentByTicker.keys()),
            "AllocationPercent": list(PercentByTicker.values()),
        }
    )
    # Stable ordering by ticker for determinism in saved outputs
    AllocationFrame = AllocationFrame.sort_values("TickerSymbol", ascending=True).reset_index(drop=True)
    return AllocationFrame


def RunMonteCarloRandomEqualWeight(
    RankingFrame: pd.DataFrame,
    SampleFraction: float = 0.5,
    Iterations: int = 1,
    OutputPath: str = "Outputs/BacktestSummary.csv",
    RandomSeed: Optional[int] = None,
) -> List[dict]:
    """Run Monte Carlo by sampling 50% tickers with equal weights and append metrics.

    Parameters
    ----------
    RankingFrame: pd.DataFrame
        DataFrame that includes a `TickerSymbol` column. Ranking values are ignored.
    SampleFraction: float
        Fraction of tickers to sample each iteration. Default 0.5.
    Iterations: int
        Number of independent Monte Carlo iterations to run. Default 1.
    OutputPath: str
        CSV path to append the backtesting metrics. Default `Outputs/BacktestSummary.csv`.
    RandomSeed: int | None
        Optional seed for reproducible sampling across runs.

    Returns
    -------
    list[dict]
        List of backtest metrics dicts (one per iteration) as returned by
        `StrategyBacktest.RunSmaCrossoverBacktest`.
    """
    Tickers = _unique_tickers(RankingFrame)
    TotalCount = int(len(Tickers))
    if TotalCount == 0:
        print("RankingFrame has no tickers; skipping Monte Carlo.")
        return []

    # Clamp sample fraction to (0, 1]
    SampleFraction = float(max(0.0, min(1.0, SampleFraction)))
    if SampleFraction <= 0.0:
        print("SampleFraction is 0; skipping Monte Carlo.")
        return []

    Iterations = max(1, int(Iterations))
    RandomGen = np.random.default_rng(RandomSeed)

    # Prepare CSV append semantics (write header only when creating file)
    FileExists = os.path.exists(OutputPath) and os.path.getsize(OutputPath) > 0

    Results: List[dict] = []
    for IterationIndex in range(Iterations):
        # Determine sample size (at least 1)
        SampleSize = max(1, int(np.ceil(SampleFraction * TotalCount)))

        # Sample without replacement
        try:
            Indices = RandomGen.choice(Tickers, size=SampleSize, replace=False)
        except Exception:
            # Fallback: if SampleSize exceeds, allow replacement
            Indices = RandomGen.choice(Tickers, size=SampleSize, replace=True)
        SelectedTickers = [str(T) for T in Indices]

        AllocationFrame = _equal_weight_allocation(SelectedTickers)
        if AllocationFrame.empty:
            continue

        CombinedMetrics = StrategyBacktest.RunSmaCrossoverBacktest(AllocationFrame=AllocationFrame)
        if isinstance(CombinedMetrics, dict) and len(CombinedMetrics) > 0:
            # Append to CSV; header only if file didn't exist before the first write
            HeaderFlag = not FileExists and (IterationIndex == 0)
            pd.DataFrame([CombinedMetrics]).to_csv(
                OutputPath,
                mode="a" if FileExists else ("w" if HeaderFlag else "a"),
                header=HeaderFlag,
                index=False,
            )
            FileExists = True  # After first successful write, switch to append
            Results.append(CombinedMetrics)

    return Results


__all__ = [
    "RunMonteCarloRandomEqualWeight",
]
