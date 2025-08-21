"""AssetRanking module to compute weighted rankings from evaluation metrics.

Workflow
--------
- Accept a `MetricsSummary` DataFrame that includes `TickerSymbol` and metric columns
  (e.g., SortinoRatio, SharpeRatio, NegMaxDrawdown, CAGR, UlcerPerformanceIndex).
- Apply z-score normalization to each metric column to keep scales comparable.
- Read per-metric weights from `config.yaml` (top-level key: `RankingWeights`).
- Compute a weighted composite score and convert it to an ordinal `Ranking` where 1 is best.
- Return a DataFrame with two columns: `TickerSymbol` and `Ranking`.

Notes
-----
- Follows the AGENTS Coding Contract: PascalCase naming and keeping logic in modules.
- Defaults to `DefaultRankingWeights` if configuration is missing or incomplete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

import Config


# Default weights used when `config.yaml` has no `RankingWeights` or is incomplete.
DefaultRankingWeights: Dict[str, float] = {
    "SortinoRatio": 0.35,
    "SharpeRatio": 0.25,
    "NegMaxDrawdown": 0.15,
    "CAGR": 0.15,
    "UlcerPerformanceIndex": 0.10,
}


def _zscore_series(Values: pd.Series) -> pd.Series:
    """Return z-scored series with non-finite treated as neutral (0).

    - Computes mean/std over finite entries only.
    - When std == 0 or no finite entries, returns zeros.
    - Non-finite inputs map to 0 to avoid skewing the composite score.
    """
    Numeric = pd.to_numeric(Values, errors="coerce")
    Numeric = Numeric.replace([np.inf, -np.inf], np.nan)
    Mean = float(Numeric.mean(skipna=True)) if Numeric.notna().any() else 0.0
    Std = float(Numeric.std(skipna=True, ddof=0)) if Numeric.notna().sum() > 1 else 0.0
    if not np.isfinite(Std) or Std == 0.0:
        return pd.Series(0.0, index=Values.index)
    Z = (Numeric - Mean) / Std
    return Z.fillna(0.0)


def _load_ranking_weights(ConfigPath: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """Load ranking weights from YAML config with safe defaults.

    The config should contain a top-level mapping `RankingWeights` where keys are
    metric names and values are numeric weights. Missing or invalid entries fall
    back to `DefaultRankingWeights`.
    """
    try:
        AppConfig = Config.LoadConfig(ConfigPath)
    except Exception:
        return dict(DefaultRankingWeights)

    Raw = AppConfig.get("RankingWeights")
    if not isinstance(Raw, Mapping):
        return dict(DefaultRankingWeights)

    Weights: Dict[str, float] = {}
    for Key, Val in Raw.items():
        try:
            Weights[str(Key)] = float(Val)
        except Exception:
            # Skip non-numeric weights
            continue

    # If nothing valid was provided, use defaults
    if not Weights:
        return dict(DefaultRankingWeights)

    # Merge with defaults for any known metrics not supplied by user
    Merged: Dict[str, float] = dict(DefaultRankingWeights)
    Merged.update(Weights)
    return Merged


def RankAssets(
    MetricsSummary: pd.DataFrame,
    ConfigPath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Compute weighted composite rankings for assets.

    Parameters
    ----------
    MetricsSummary: pd.DataFrame
        DataFrame containing `TickerSymbol` and one or more metric columns.
    ConfigPath: str | Path | None
        Optional path to YAML config. If not provided, defaults to repo-level
        `config.yaml` via `Config.LoadConfig`.

    Returns
    -------
    pd.DataFrame
        DataFrame with `TickerSymbol`, numeric `CompositeScore`, and integer
        `Ranking` (1 = best).
    """
    if MetricsSummary is None or MetricsSummary.empty:
        return pd.DataFrame(columns=["TickerSymbol", "Ranking"])

    if "TickerSymbol" not in MetricsSummary.columns:
        raise ValueError("MetricsSummary must include a 'TickerSymbol' column.")

    # Identify metric columns (exclude identifier columns)
    MetricColumns = [
        Col for Col in MetricsSummary.columns if Col != "TickerSymbol"
    ]
    if not MetricColumns:
        # No metrics to rank on
        Count = len(MetricsSummary)
        Output = pd.DataFrame(
            {
                "TickerSymbol": MetricsSummary["TickerSymbol"].astype(str).tolist(),
                "CompositeScore": [0.0] * Count,
                "Ranking": [1] * Count,
            }
        )
        return Output

    # Z-score normalization per metric
    ZScores = pd.DataFrame(index=MetricsSummary.index)
    for Col in MetricColumns:
        ZScores[Col] = _zscore_series(MetricsSummary[Col])

    # Load weights and compute weighted composite score
    Weights = _load_ranking_weights(ConfigPath)

    # Use only metrics present; ignore unknown weights
    EffectiveWeights: Dict[str, float] = {
        Col: float(Weights.get(Col, 0.0)) for Col in MetricColumns
    }

    # If all weights are zero, fall back to equal weighting across available metrics
    if all((Weight == 0.0 for Weight in EffectiveWeights.values())) and len(EffectiveWeights) > 0:
        EqualWeight = 1.0 / float(len(EffectiveWeights))
        EffectiveWeights = {Key: EqualWeight for Key in EffectiveWeights}

    # Compute weighted sum (composite score)
    Score = pd.Series(0.0, index=ZScores.index)
    for Col, Weight in EffectiveWeights.items():
        Score = Score + (ZScores.get(Col, 0.0) * float(Weight))

    # Convert composite score to ordinal ranking (1 = best)
    Ranking = Score.rank(ascending=False, method="min").astype(int)

    Output = pd.DataFrame(
        {
            "TickerSymbol": MetricsSummary["TickerSymbol"].astype(str),
            "CompositeScore": Score.astype(float),
            "Ranking": Ranking,
        }
    ).reset_index(drop=True)

    # Sort by ranking (best first), then by CompositeScore desc, then ticker for stability
    Output = Output.sort_values(
        by=["Ranking", "CompositeScore", "TickerSymbol"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    return Output


__all__ = [
    "RankAssets",
]
