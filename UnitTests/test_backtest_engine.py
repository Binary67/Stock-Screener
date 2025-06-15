import os
import sys
import pandas as pd
from unittest.mock import patch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from BacktestEngine import BacktestEngine


def _mock_download(*args, **kwargs):
    dates = pd.date_range('2024-01-01', periods=3)
    df_a = pd.DataFrame({
        'Open': [100, 110, 120],
        'High': [100, 110, 120],
        'Low': [100, 110, 120],
        'Close': [100, 110, 120],
        'Volume': [1, 1, 1]
    }, index=dates)
    df_a['Ticker'] = 'AAA'

    df_b = pd.DataFrame({
        'Open': [50, 55, 60],
        'High': [50, 55, 60],
        'Low': [50, 55, 60],
        'Close': [50, 55, 60],
        'Volume': [1, 1, 1]
    }, index=dates)
    df_b['Ticker'] = 'BBB'
    return pd.concat([df_a, df_b])


def test_run_backtest(monkeypatch):
    alloc = pd.DataFrame({'Allocation': [0.6, 0.4]}, index=['AAA', 'BBB'])
    with patch('BacktestEngine.YFinanceDownloader.DownloadData', side_effect=_mock_download):
        engine = BacktestEngine(alloc)
        stats_weighted, stats_equal = engine.RunBacktest()
        assert 'Return [%]' in stats_weighted
        assert 'Return [%]' in stats_equal
