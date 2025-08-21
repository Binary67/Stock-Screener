"""AssetAllocation module to compute portfolio weights using ranking and inverse volatility.

Overview
--------
- Inputs a ranking DataFrame (from AssetRanking.RankAssets) and a mapping of
  ticker -> OHLCV DataFrame (from DataDownloader.DownloadTradingData).
- Computes volatility from the last `LookbackPeriods` returns (default: 60)
  using a preferred price column ("Adj Close" then "Close").
- Combines inverse volatility with ranking into allocation scores using a
  configurable weighted blend (default: 70% ranking, 30% volatility), then
  normalizes to percentage weights that sum exactly to 100.

Rules & Conventions
-------------------
- Follows the AGENTS Coding Contract: PascalCase for variables, DRY by reusing
  helpers from EvaluationMetrics where applicable, and keeping logic in modules
  while `main.py` orchestrates.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

import EvaluationMetrics
import Config


def _compute_lookback_volatility(
    TradingData: pd.DataFrame,
    LookbackPeriods: int,
    PriceColumnPriority: Sequence[str],
) -> float:
    """Compute stdev of returns over the last `LookbackPeriods`.

    - Selects a price series using shared logic from EvaluationMetrics.
    - Computes simple returns, then takes the last N periods for volatility.
    - Returns NaN if not computable.
    """
    try:
        PriceSeries = EvaluationMetrics._select_price_series(TradingData, PriceColumnPriority)  # type: ignore[attr-defined]
        ReturnsSeries = EvaluationMetrics._compute_returns(PriceSeries)  # type: ignore[attr-defined]
        if LookbackPeriods > 0 and len(ReturnsSeries) > LookbackPeriods:
            ReturnsSeries = ReturnsSeries.tail(LookbackPeriods)
        Volatility = float(ReturnsSeries.std(ddof=0)) if len(ReturnsSeries) > 0 else float("nan")
        return Volatility
    except Exception:
        return float("nan")


def _compute_inverse_volatility_weights(
    PriceDataByTicker: Mapping[str, pd.DataFrame],
    LookbackPeriods: int,
    PriceColumnPriority: Sequence[str],
) -> Dict[str, float]:
    """Return a mapping ticker -> inverse volatility score (unnormalized)."""
    Weights: Dict[str, float] = {}
    for TickerSymbol, TradingData in PriceDataByTicker.items():
        Volatility = _compute_lookback_volatility(
            TradingData=TradingData,
            LookbackPeriods=LookbackPeriods,
            PriceColumnPriority=PriceColumnPriority,
        )
        if np.isfinite(Volatility) and Volatility > 0.0:
            Weights[TickerSymbol] = 1.0 / float(Volatility)
        else:
            # Assign zero weight when volatility is undefined or zero
            Weights[TickerSymbol] = 0.0
    return Weights


def _ranking_weights(RankingFrame: pd.DataFrame) -> Dict[str, float]:
    """Return ticker -> rank-based weight using inverse rank (1/rank).

    Notes
    -----
    - Lower rank means better; rank 1 gets weight 1.0, rank 2 gets 0.5, etc.
    - Missing or non-positive ranks receive weight 0.0.
    """
    RequiredCols = {"TickerSymbol", "Ranking"}
    Missing = RequiredCols - set(RankingFrame.columns)
    if Missing:
        raise ValueError(f"RankingFrame must include columns {RequiredCols}; missing {Missing}")

    Weights: Dict[str, float] = {}
    for _, Row in RankingFrame.iterrows():
        Ticker = str(Row["TickerSymbol"]).strip()
        try:
            RankValue = int(Row["Ranking"]) if pd.notna(Row["Ranking"]) else 0
        except Exception:
            RankValue = 0
        Weight = 1.0 / float(RankValue) if RankValue > 0 else 0.0
        Weights[Ticker] = Weight
    return Weights

def _load_allocation_weights(ConfigPath: Optional[Union[str, Path]] = None) -> float:
    """Load ranking-vs-volatility blend from YAML and return RankingWeight.

    Semantics
    ---------
    - Accepts either a single `RankingWeight` in [0,1], or an object
      `AllocationWeights: { Ranking: <float>, Volatility: <float> }`.
    - When both Ranking and Volatility are provided, they are normalized so the
      effective RankingWeight = Ranking / (Ranking + Volatility).
    - Defaults to 0.7 (i.e., 70% ranking, 30% volatility) if not set.
    """
    DefaultRankingWeight = 0.7
    try:
        AppConfig = Config.LoadConfig(ConfigPath)
    except Exception:
        return DefaultRankingWeight

    # Option A: Direct scalar
    try:
        if "RankingWeight" in AppConfig:
            RankingWeight = float(AppConfig.get("RankingWeight"))
            if np.isfinite(RankingWeight):
                return float(max(0.0, min(1.0, RankingWeight)))
    except Exception:
        pass

    # Option B: Two-part weights
    try:
        Raw = AppConfig.get("AllocationWeights", None)
        if isinstance(Raw, dict):
            RankingPart = float(Raw.get("Ranking", float("nan")))
            VolPart = float(Raw.get("Volatility", float("nan")))
            if np.isfinite(RankingPart) and np.isfinite(VolPart):
                Total = RankingPart + VolPart
                if Total > 0:
                    return float(max(0.0, min(1.0, RankingPart / Total)))
    except Exception:
        pass

    return DefaultRankingWeight

def _normalize_distribution(
    WeightsByTicker: Dict[str, float],
    Tickers: Sequence[str],
) -> Dict[str, float]:
    """Normalize a weights mapping over a provided ticker sequence.

    - Returns a dict of the same tickers with values summing to 1.0.
    - If the total weight is zero or invalid, returns equal distribution.
    """
    Values = np.array([max(0.0, float(WeightsByTicker.get(T, 0.0))) for T in Tickers], dtype=float)
    Total = float(Values.sum())
    if Total <= 0.0 or not np.isfinite(Total):
        if len(Tickers) == 0:
            return {}
        EqualShare = 1.0 / float(len(Tickers))
        return {T: EqualShare for T in Tickers}
    Dist = Values / Total
    return {Tickers[i]: float(Dist[i]) for i in range(len(Tickers))}

def _load_top_n_percent(ConfigPath: Optional[Union[str, Path]] = None) -> float:
    """Load `TopN` as a fraction from YAML; clamp to [0.0, 1.0].

    - New convention: TopN is a fraction (e.g., 0.2 keeps top 20%).
    - Backward compatibility: If value > 1.0 and <= 100, treat as percent and divide by 100.
    - Returns 1.0 when not set or invalid to preserve current behavior (keep all).
    """
    try:
        AppConfig = Config.LoadConfig(ConfigPath)
        Raw = AppConfig.get("TopN", 1)
        TopNFraction = float(Raw)
    except Exception:
        TopNFraction = 1.0

    if not np.isfinite(TopNFraction):
        return 1.0

    # Back-compat: interpret values in (1, 100] as percent
    if TopNFraction > 1.0 and TopNFraction <= 100.0:
        TopNFraction = TopNFraction / 100.0

    # Clamp to [0.0, 1.0]
    return float(max(0.0, min(1.0, TopNFraction)))


def _normalize_to_percent(ScoreByTicker: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to percentages summing exactly to 100.0.

    - If total score is zero, allocates equally among tickers.
    - Uses a residual adjustment on the largest allocation to ensure exact 100.0
      after rounding to 6 decimals.
    """
    Tickers = list(ScoreByTicker.keys())
    Scores = np.array([float(ScoreByTicker[T]) for T in Tickers], dtype=float)
    Total = float(Scores.sum())

    if Total <= 0.0 or not np.isfinite(Total):
        # Equal allocation if no signal
        if len(Tickers) == 0:
            return {}
        Equal = 100.0 / float(len(Tickers))
        return {T: Equal for T in Tickers}

    Percents = (Scores / Total) * 100.0

    # Round to avoid floating noise but keep precision
    Rounded = np.round(Percents, 6)
    Diff = 100.0 - float(Rounded.sum())
    if abs(Diff) > 1e-9:
        # Adjust the largest allocation by the residual to make exact 100.0
        MaxIdx = int(np.argmax(Rounded))
        Rounded[MaxIdx] = Rounded[MaxIdx] + Diff
    return {Tickers[i]: float(Rounded[i]) for i in range(len(Tickers))}


def AllocateAssets(
    RankingFrame: pd.DataFrame,
    PriceDataByTicker: Mapping[str, pd.DataFrame],
    LookbackPeriods: int = 60,
    PriceColumnPriority: Sequence[str] = EvaluationMetrics.DefaultPriceColumns,
    ConfigPath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Allocate assets by combining ranking with inverse volatility weighting.

    Parameters
    ----------
    RankingFrame: pd.DataFrame
        DataFrame with at least `TickerSymbol` and `Ranking` columns, where 1 is best.
    PriceDataByTicker: Mapping[str, pd.DataFrame]
        Mapping from ticker to its OHLCV DataFrame.
    LookbackPeriods: int
        Number of most recent return periods used to compute volatility (default 60).
    PriceColumnPriority: Sequence[str]
        Preference order for selecting price column when computing returns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: `TickerSymbol`, `AllocationPercent`.
        Percentages sum exactly to 100.0.
    """
    if RankingFrame is None or RankingFrame.empty:
        return pd.DataFrame(columns=["TickerSymbol", "AllocationPercent"])

    # Filter to the top N fraction of tickers by ranking if configured
    TopNPercent = _load_top_n_percent(ConfigPath)
    EffectiveFrame = RankingFrame
    try:
        if TopNPercent < 1.0 and "Ranking" in RankingFrame.columns:
            TotalCount = int(len(RankingFrame))
            if TotalCount > 0:
                KeepCount = int(np.ceil((TopNPercent) * TotalCount))
                KeepCount = max(1, min(TotalCount, KeepCount))
                EffectiveFrame = (
                    RankingFrame.sort_values(["Ranking", "TickerSymbol"], ascending=[True, True])
                    .head(KeepCount)
                    .reset_index(drop=True)
                )
    except Exception:
        # On any failure, fall back to using the full frame
        EffectiveFrame = RankingFrame

    # Build rank-based weights
    RankWeights = _ranking_weights(EffectiveFrame)

    # Compute inverse volatility weights for the same tickers only
    FilteredPriceMap: Dict[str, pd.DataFrame] = {T: DF for T, DF in PriceDataByTicker.items() if T in RankWeights}
    InverseVolWeights = _compute_inverse_volatility_weights(
        PriceDataByTicker=FilteredPriceMap,
        LookbackPeriods=LookbackPeriods,
        PriceColumnPriority=PriceColumnPriority,
    )

    # Load blend configuration (default 70% ranking, 30% volatility)
    RankingBlendWeight = _load_allocation_weights(ConfigPath)

    # Prepare normalized distributions so that the blend reflects the intended
    # contribution shares across the universe.
    TickersOrdered = list(RankWeights.keys())
    RankDist = _normalize_distribution(RankWeights, TickersOrdered)
    InvVolDist = _normalize_distribution(InverseVolWeights, TickersOrdered)

    # Combine via additive blend: Score = a*RankDist + (1-a)*InvVolDist
    Alpha = float(max(0.0, min(1.0, RankingBlendWeight)))
    CombinedScores: Dict[str, float] = {}
    for TickerSymbol in TickersOrdered:
        CombinedScores[TickerSymbol] = (
            Alpha * float(RankDist.get(TickerSymbol, 0.0))
            + (1.0 - Alpha) * float(InvVolDist.get(TickerSymbol, 0.0))
        )

    PercentByTicker = _normalize_to_percent(CombinedScores)

    # Assemble output, preserve ranking order (rank asc, then ticker)
    Ordered = (
        EffectiveFrame[["TickerSymbol", "Ranking"]]
        .sort_values(["Ranking", "TickerSymbol"], ascending=[True, True])
        .reset_index(drop=True)
    )
    Ordered["AllocationPercent"] = Ordered["TickerSymbol"].map(PercentByTicker).astype(float)
    Output = Ordered[["TickerSymbol", "AllocationPercent"]]
    return Output


__all__ = [
    "AllocateAssets",
]
