"""ResultVisualization module: plot selected metric from BacktestSummary.csv.

Behavior
--------
- Loads `Outputs/BacktestSummary.csv` and plots a bar chart of a selected
  column versus the row index (each row corresponds to one strategy/backtest).
- Uses a qualitative colormap to color bars distinctly across indices.
- Saves the figure under `Outputs/Plots/Backtest_<ColumnName>.png`.

Notes
-----
- Follows the AGENTS Coding Contract: PascalCase variables, DRY, and keeps
  `main.py` as orchestrator-only. This module exposes a single function for use
  by `main.py`.
"""

from __future__ import annotations

from typing import Optional

import os
import math
import pandas as pd


def PlotBacktestColumn(
    ColumnName: str,
    CsvPath: str = "Outputs/BacktestSummary.csv",
    OutputPath: Optional[str] = None,
    MaxLegendEntries: int = 20,
) -> Optional[str]:
    """Plot the selected column as a colored bar chart by strategy index.

    Parameters
    ----------
    ColumnName: str
        Column in the CSV to plot on the Y-axis against the row index (X-axis).
    CsvPath: str
        Path to the `BacktestSummary.csv` file.
    OutputPath: Optional[str]
        Optional explicit path to save the plot image. If not provided, saves to
        `Outputs/Plots/Backtest_<ColumnName>.png`.
    MaxLegendEntries: int
        Maximum number of legend entries to display (to avoid overcrowding
        when there are many rows). Legend entries are sampled evenly across
        the index range.

    Returns
    -------
    Optional[str]
        File path of the saved figure, or None if plotting failed.
    """
    # Lazy import to avoid requiring matplotlib for non-plot flows
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import matplotlib.cm as cm  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        print(
            "matplotlib and numpy are required for plotting. Install with: "
            ".venv/bin/python3 -m pip install matplotlib numpy"
        )
        return None

    if not os.path.exists(CsvPath) or os.path.getsize(CsvPath) == 0:
        print(f"CSV not found or empty: {CsvPath}")
        return None

    try:
        DataFrame = pd.read_csv(CsvPath)
    except Exception as Error:
        print(f"Failed to read CSV '{CsvPath}': {Error}")
        return None

    if ColumnName not in DataFrame.columns:
        Available = ", ".join([str(C) for C in DataFrame.columns])
        print(
            f"Column '{ColumnName}' not found in CSV. Available columns: {Available}"
        )
        return None

    # Prepare series and indices for bar chart
    Series = pd.to_numeric(DataFrame[ColumnName], errors="coerce")
    ValidMask = Series.notna()
    if ValidMask.sum() < 1:
        print(f"No valid numeric values to plot for column '{ColumnName}'.")
        return None

    # Use compact, contiguous X positions only for valid rows
    Y = Series[ValidMask].to_numpy(dtype=float)
    X = np.arange(len(Y), dtype=float)

    # Set default output path if needed
    if OutputPath is None:
        SafeName = "".join(
            Ch if Ch.isalnum() or Ch in ("-", "_") else "_" for Ch in str(ColumnName)
        )
        OutputPath = os.path.join("Outputs", "Plots", f"Backtest_{SafeName}.png")

    # Ensure output directory exists
    OutputDir = os.path.dirname(OutputPath)
    if OutputDir:
        os.makedirs(OutputDir, exist_ok=True)

    # Create the plot: draw bars with index-based colors
    plt.figure(figsize=(10, 5), dpi=120)
    Axis = plt.gca()

    # Choose a qualitative colormap with many distinct colors
    ColorMap = cm.get_cmap("tab20", max(2, int(len(X))))
    Colors = [ColorMap(int(Index) % ColorMap.N) for Index in range(len(X))]

    Axis.bar(X, Y, color=Colors, edgecolor="none")

    Axis.set_title(f"Backtest Summary: {ColumnName}")
    Axis.set_xlabel("Strategy Index")
    Axis.set_ylabel(str(ColumnName))
    Axis.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    # Keep ticks reasonable for large data by sampling based on MaxLegendEntries
    if MaxLegendEntries is not None and MaxLegendEntries > 0 and len(X) > MaxLegendEntries:
        Step = max(1, int(math.ceil(len(X) / float(MaxLegendEntries))))
        TickIndices = np.arange(0, len(X), Step)
        Axis.set_xticks(TickIndices)
        Axis.set_xticklabels([str(int(I)) for I in TickIndices])

    plt.tight_layout()
    try:
        plt.savefig(OutputPath)
        plt.close()
    except Exception as Error:
        print(f"Failed to save figure to '{OutputPath}': {Error}")
        return None

    print(f"Saved plot to: {OutputPath}")
    return OutputPath


__all__ = [
    "PlotBacktestColumn",
]
