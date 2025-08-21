"""DataDownloader module for fetching trading data via yfinance with CSV caching.

Contract notes:
- Uses PascalCase for all newly declared variables (params + locals).
- Caches downloads under `Caches/` to avoid repeated network calls.
- Returns a DataFrame with non-multi-index columns limited to OHLCV.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import yfinance as yfinance  # external identifier kept as-is
except Exception as ImportError:  # type: ignore[misc]
    # Defer import error until function call to avoid import-time crashes in environments
    # without the dependency installed.
    yfinance = None  # type: ignore[assignment]


# Module-level constants (PascalCase per repo contract)
CacheRoot = Path(__file__).resolve().parent / "Caches"
OhlcvOrder = ["Open", "High", "Low", "Close", "Volume"]


def _ensure_cache_dir() -> None:
    CacheRoot.mkdir(parents=True, exist_ok=True)


def _normalize_date_component(DateValue: Union[str, pd.Timestamp, pd.DatetimeTZDtype, None]) -> str:
    if DateValue is None:
        return "None"
    if isinstance(DateValue, pd.Timestamp):
        return DateValue.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S") if DateValue.tz is not None else DateValue.strftime("%Y-%m-%d %H:%M:%S")
    try:
        Parsed = pd.to_datetime(DateValue, utc=False)
        return Parsed.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S") if getattr(Parsed, "tz", None) is not None else Parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(DateValue)


def _build_cache_path(TickerSymbol: str, StartDate: Union[str, pd.Timestamp], EndDate: Union[str, pd.Timestamp], Interval: str) -> Path:
    StartKey = _normalize_date_component(StartDate)
    EndKey = _normalize_date_component(EndDate)
    CacheKey = f"{TickerSymbol}|{StartKey}|{EndKey}|{Interval}"
    CacheHash = hashlib.md5(CacheKey.encode("utf-8")).hexdigest()[:12]
    SafeTicker = "".join(c for c in TickerSymbol if c.isalnum() or c in ("-", "_")) or "TICKER"
    FileName = f"{SafeTicker}_{StartKey.replace(' ', 'T').replace(':', '-')}_{EndKey.replace(' ', 'T').replace(':', '-')}_{Interval}_{CacheHash}.csv"
    return CacheRoot / FileName


def _select_ticker_from_multiindex(Data: pd.DataFrame, TickerSymbol: str) -> pd.DataFrame:
    Columns = Data.columns
    if not isinstance(Columns, pd.MultiIndex):
        return Data
    # Try selecting by any level that contains the ticker symbol
    for Level in range(Columns.nlevels):
        try:
            LevelValues = Columns.get_level_values(Level)
        except Exception:
            continue
        if TickerSymbol in set(LevelValues):
            try:
                Selected = Data.xs(TickerSymbol, axis=1, level=Level, drop_level=True)
                return Selected
            except Exception:
                pass
    # Fallback: no direct match; flatten and later filter
    try:
        FlatCols = ["_".join([str(p) for p in Tup if str(p) != ""]).strip("_") for Tup in Columns.to_flat_index()]
        Data.columns = FlatCols
    except Exception:
        Data.columns = [str(c) for c in Data.columns]
    return Data


def _ensure_ohlcv_columns(Data: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names for robust selection
    NormalizedMap = {}
    for Col in Data.columns:
        Key = str(Col).strip()
        Lower = Key.lower()
        if Lower.endswith("adj close") or Lower == "adj close":
            NormalizedMap[Col] = "Adj Close"
        elif Lower.endswith("open") or Lower == "open":
            NormalizedMap[Col] = "Open"
        elif Lower.endswith("high") or Lower == "high":
            NormalizedMap[Col] = "High"
        elif Lower.endswith("low") or Lower == "low":
            NormalizedMap[Col] = "Low"
        elif Lower.endswith("close") or Lower == "close":
            NormalizedMap[Col] = "Close"
        elif Lower.endswith("volume") or Lower == "volume":
            NormalizedMap[Col] = "Volume"
        else:
            NormalizedMap[Col] = Key

    Renamed = Data.rename(columns=NormalizedMap)
    # Keep only OHLCV in canonical order when present
    Present = [Col for Col in OhlcvOrder if Col in Renamed.columns]
    if not Present:
        return Renamed
    return Renamed.loc[:, Present]


def _read_cache(CachePath: Path) -> Optional[pd.DataFrame]:
    try:
        if CachePath.exists():
            Data = pd.read_csv(CachePath, index_col=0, parse_dates=[0])
            return Data
    except Exception:
        # Cache read failures should fall back to re-download
        return None
    return None


def _write_cache(CachePath: Path, Data: pd.DataFrame) -> None:
    try:
        Data.to_csv(CachePath)
    except Exception:
        # Silently ignore cache write errors to not break downloads
        pass


def DownloadTradingData(
    TickerSymbol: str,
    StartDate: Union[str, pd.Timestamp],
    EndDate: Union[str, pd.Timestamp],
    Interval: str,
    ForceRefresh: bool = False,
) -> pd.DataFrame:
    """Download trading data for a ticker with caching and OHLCV flattening.

    Parameters
    ----------
    TickerSymbol: str
        The ticker symbol, e.g., "AAPL".
    StartDate: str | pd.Timestamp
        Start date (inclusive). Accepts string or pandas Timestamp.
    EndDate: str | pd.Timestamp
        End date (exclusive). Accepts string or pandas Timestamp.
    Interval: str
        Data interval, e.g., "1d", "1h", "5m".
    ForceRefresh: bool
        If True, bypass cache and re-download.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with OHLCV columns and no MultiIndex.
    """
    if yfinance is None:
        raise ImportError("yfinance is required but not installed. Please install `yfinance`. ")

    _ensure_cache_dir()
    CachePath = _build_cache_path(TickerSymbol, StartDate, EndDate, Interval)

    if not ForceRefresh:
        Cached = _read_cache(CachePath)
        if Cached is not None and not Cached.empty:
            return Cached

    # Download from yfinance
    Data = yfinance.download(
        tickers=TickerSymbol,
        start=StartDate,
        end=EndDate,
        interval=Interval,
        group_by="column",  # helps avoid MultiIndex for single ticker
        auto_adjust=False,
        threads=False,
        progress=False,
    )

    if Data is None or len(Data) == 0:
        # Ensure we always return a DataFrame
        return pd.DataFrame(columns=OhlcvOrder)

    # Flatten any MultiIndex and extract the selected ticker's columns
    Data = _select_ticker_from_multiindex(Data, TickerSymbol)
    Data = _ensure_ohlcv_columns(Data)

    # Ensure index is a DatetimeIndex and sorted
    try:
        if not isinstance(Data.index, pd.DatetimeIndex):
            Data.index = pd.to_datetime(Data.index)
        Data = Data.sort_index()
    except Exception:
        pass

    # Persist to cache for future calls
    _write_cache(CachePath, Data)
    return Data

