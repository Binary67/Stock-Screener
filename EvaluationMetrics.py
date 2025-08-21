"""EvaluationMetrics module for computing performance metrics on a single ticker.

Design goals:
- Accept a single trading DataFrame for one ticker and compute metrics.
- Start with Sortino Ratio and keep the API flexible to add more metrics later.
- Adds Sharpe Ratio and Max Drawdown metrics.
- Follow AGENTS Coding Contract: PascalCase for newly declared variables, keep logic
  inside modules, and allow `main.py` to orchestrate.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Module-level constants (PascalCase per repo contract)
DefaultPriceColumns: Tuple[str, ...] = ("Adj Close", "Close")


def _select_price_series(
    TradingData: pd.DataFrame, PriceColumnPriority: Sequence[str]
) -> pd.Series:
    """Select a price series from a trading DataFrame.

    Preference order is controlled by `PriceColumnPriority`. Falls back to the first
    numeric column if none of the preferred columns are present.
    """
    for Column in PriceColumnPriority:
        if Column in TradingData.columns:
            Series = TradingData[Column].dropna()
            if not Series.empty:
                return Series

    NumericOnly = TradingData.select_dtypes(include=["number"]) if len(TradingData.columns) else TradingData
    if NumericOnly.shape[1] > 0:
        return NumericOnly.iloc[:, 0].dropna()

    raise ValueError("No suitable price column found to compute returns.")


def _compute_returns(PriceSeries: pd.Series) -> pd.Series:
    """Compute simple returns from a price series, cleaning inf/nan values."""
    Returns = PriceSeries.pct_change()
    Returns = Returns.replace([np.inf, -np.inf], np.nan).dropna()
    return Returns


def ComputeSortinoRatio(
    Returns: pd.Series,
    TargetReturn: float = 0.0,
    PeriodsPerYear: Optional[int] = None,
) -> float:
    """Compute the Sortino Ratio for a series of returns.

    Parameters
    ----------
    Returns: pd.Series
        Periodic returns (e.g., daily returns).
    TargetReturn: float
        Minimal acceptable return (MAR) per period. Default 0.0.
    PeriodsPerYear: int | None
        If provided, annualizes the numerator and denominator using this factor.

    Returns
    -------
    float
        The Sortino Ratio. Returns NaN if not computable, or +inf if there is
        no downside volatility.
    """
    if Returns is None or len(Returns) == 0:
        return float("nan")

    Excess = Returns - TargetReturn
    Downside = Excess[Excess < 0]

    if Downside.empty:
        return float("inf")

    DownsideStd = float(Downside.std(ddof=0))
    MeanReturn = float(Returns.mean())

    if PeriodsPerYear is not None and PeriodsPerYear > 0:
        MeanReturn = MeanReturn * PeriodsPerYear
        DownsideStd = DownsideStd * (PeriodsPerYear ** 0.5)

    if DownsideStd == 0:
        return float("inf")

    return MeanReturn / DownsideStd


# Flexible metric type: function accepts (Returns, TradingData) and returns float
MetricFunc = Callable[[pd.Series, pd.DataFrame], float]


def ComputeSharpeRatio(
    Returns: pd.Series,
    RiskFreeRate: float = 0.0,
    PeriodsPerYear: Optional[int] = None,
) -> float:
    """Compute the Sharpe Ratio for a series of returns.

    Parameters
    ----------
    Returns: pd.Series
        Periodic returns (e.g., daily returns).
    RiskFreeRate: float
        Risk-free rate per period. Default 0.0.
    PeriodsPerYear: int | None
        If provided, annualizes the numerator and denominator using this factor.

    Returns
    -------
    float
        The Sharpe Ratio. Returns NaN if not computable or stdev is zero.
    """
    if Returns is None or len(Returns) == 0:
        return float("nan")

    Excess = Returns - RiskFreeRate
    MeanExcess = float(Excess.mean())
    StdExcess = float(Excess.std(ddof=0))

    if PeriodsPerYear is not None and PeriodsPerYear > 0:
        MeanExcess = MeanExcess * PeriodsPerYear
        StdExcess = StdExcess * (PeriodsPerYear ** 0.5)

    if StdExcess == 0:
        return float("nan")

    return MeanExcess / StdExcess


def ComputeMaxDrawdown(Returns: pd.Series) -> float:
    """Compute the maximum drawdown from a series of returns.

    Uses cumulative returns to form an equity curve: `Equity = (1 + R).cumprod()`.
    Drawdown is `Equity / cummax(Equity) - 1`. Returns the minimum drawdown
    (a non-positive number; e.g., -0.35 for a 35% max drawdown).
    """
    if Returns is None or len(Returns) == 0:
        return float("nan")

    EquityCurve = (1.0 + Returns).cumprod()
    RollingPeak = EquityCurve.cummax()
    Drawdown = (EquityCurve / RollingPeak) - 1.0
    return float(Drawdown.min())


def DefaultMetricsRegistry(
    TargetReturn: float = 0.0,
    PeriodsPerYear: Optional[int] = None,
    RiskFreeRate: Optional[float] = None,
) -> Dict[str, MetricFunc]:
    """Return a registry of default metrics. Starts with SortinoRatio.

    New metrics can be added by extending this mapping or passing a custom mapping
    to `EvaluateSingleTicker`.
    """

    def Sortino(Returns: pd.Series, TradingData: pd.DataFrame) -> float:  # noqa: ARG001
        return ComputeSortinoRatio(
            Returns=Returns, TargetReturn=TargetReturn, PeriodsPerYear=PeriodsPerYear
        )

    EffectiveRf = TargetReturn if RiskFreeRate is None else RiskFreeRate

    def Sharpe(Returns: pd.Series, TradingData: pd.DataFrame) -> float:  # noqa: ARG001
        return ComputeSharpeRatio(
            Returns=Returns, RiskFreeRate=EffectiveRf, PeriodsPerYear=PeriodsPerYear
        )

    def MaxDD(Returns: pd.Series, TradingData: pd.DataFrame) -> float:  # noqa: ARG001
        return ComputeMaxDrawdown(Returns=Returns)

    return {
        "SortinoRatio": Sortino,
        "SharpeRatio": Sharpe,
        "MaxDrawdown": MaxDD,
    }


def EvaluateSingleTicker(
    TickerSymbol: str,
    TradingData: pd.DataFrame,
    Metrics: Optional[Dict[str, MetricFunc]] = None,
    PriceColumnPriority: Sequence[str] = DefaultPriceColumns,
    TargetReturn: float = 0.0,
    PeriodsPerYear: Optional[int] = None,
    RiskFreeRate: Optional[float] = None,
) -> pd.DataFrame:
    """Evaluate metrics for a single ticker's trading DataFrame.

    Parameters
    ----------
    TickerSymbol: str
        The ticker symbol associated with `TradingData`.
    TradingData: pd.DataFrame
        OHLCV-like data for a single ticker. Must include a price column such as
        "Adj Close" or "Close".
    Metrics: dict | None
        Optional mapping of metric name to metric function with signature
        `(Returns, TradingData) -> float`. If omitted, uses `DefaultMetricsRegistry`.
    PriceColumnPriority: Sequence[str]
        Preference order for price columns when computing returns.
    TargetReturn: float
        Minimal acceptable return (MAR) per period for Sortino.
    PeriodsPerYear: int | None
        Annualization factor for metrics that support it (e.g., Sortino).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including: `TickerSymbol`, `SortinoRatio`,
        `SharpeRatio`, and `MaxDrawdown` (and any future metrics added).
    """
    PriceSeries = _select_price_series(TradingData, PriceColumnPriority)
    ReturnsSeries = _compute_returns(PriceSeries)

    EffectiveMetrics = (
        Metrics
        if Metrics is not None
        else DefaultMetricsRegistry(
            TargetReturn=TargetReturn,
            PeriodsPerYear=PeriodsPerYear,
            RiskFreeRate=RiskFreeRate,
        )
    )

    MetricsValues: Dict[str, float] = {}
    for MetricName, Metric in EffectiveMetrics.items():
        try:
            MetricsValues[MetricName] = float(Metric(ReturnsSeries, TradingData))
        except Exception:
            MetricsValues[MetricName] = float("nan")

    OutputRow = {"TickerSymbol": TickerSymbol}
    OutputRow.update(MetricsValues)
    OutputFrame = pd.DataFrame([OutputRow])
    return OutputFrame


__all__ = [
    "ComputeSortinoRatio",
    "ComputeSharpeRatio",
    "ComputeMaxDrawdown",
    "EvaluateSingleTicker",
    "DefaultMetricsRegistry",
]
