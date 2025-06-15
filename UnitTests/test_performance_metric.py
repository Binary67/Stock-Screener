import pandas as pd
import numpy as np
from PerformanceMetric import PerformanceMetric


def test_generate_metrics():
    Dates = pd.date_range('2022-01-01', periods=3)
    Data = pd.DataFrame({
        'Close': [100, 105, 102.9],
        'Ticker': ['AAPL'] * 3
    }, index=Dates)

    Metrics = PerformanceMetric(Data).GenerateMetrics()

    Days = (Dates[-1] - Dates[0]).days
    Cagr = (102.9 / 100) ** (365 / Days) - 1
    Returns = pd.Series([0.05, -0.02])
    Vol = Returns.std() * np.sqrt(252)
    Cumulative = Data['Close'] / Data['Close'].iloc[0]
    RunningMax = Cumulative.cummax()
    Drawdown = (Cumulative - RunningMax) / RunningMax
    MaxDd = Drawdown.min()
    Excess = Returns - 0 / 252
    Sharpe = Excess.mean() / Returns.std() * np.sqrt(252)

    Expected = pd.DataFrame({
        'CAGR': [Cagr],
        'Volatility': [Vol],
        'MaxDrawdown': [MaxDd],
        'SharpeRatio': [Sharpe]
    }, index=pd.Index(['AAPL'], name='Ticker'))

    pd.testing.assert_frame_equal(Metrics, Expected)

