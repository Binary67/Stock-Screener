import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from PerformanceRanking import PerformanceRanking


def test_generate_ranking():
    Metrics = pd.DataFrame({
        'CAGR': [0.1, 0.2],
        'Volatility': [0.2, 0.15],
        'MaxDrawdown': [-0.1, -0.2],
        'SharpeRatio': [1.0, 1.2],
        'SortinoRatio': [1.5, 2.0]
    }, index=pd.Index(['AAPL', 'MSFT'], name='Ticker'))

    Ranking = PerformanceRanking(Metrics).GenerateRanking()

    Expected = pd.DataFrame({
        'CAGR': [2, 1],
        'Volatility': [2, 1],
        'MaxDrawdown': [1, 2],
        'SharpeRatio': [2, 1],
        'SortinoRatio': [2, 1]
    }, index=Metrics.index)

    pd.testing.assert_frame_equal(Ranking, Expected)


def test_generate_composite_ranking():
    Metrics = pd.DataFrame({
        'CAGR': [0.1, 0.2],
        'Volatility': [0.2, 0.15],
        'MaxDrawdown': [-0.1, -0.2],
        'SharpeRatio': [1.0, 1.2],
        'SortinoRatio': [1.5, 2.0]
    }, index=pd.Index(['AAPL', 'MSFT'], name='Ticker'))

    CompositeList = ['CAGR', 'Volatility', 'MaxDrawdown', 'SharpeRatio', 'SortinoRatio']
    Ranking = PerformanceRanking(Metrics).GenerateCompositeRanking(CompositeList)

    Expected = pd.DataFrame({
        'CAGR': [2, 1],
        'Volatility': [2, 1],
        'MaxDrawdown': [1, 2],
        'SharpeRatio': [2, 1],
        'SortinoRatio': [2, 1],
        'CompositeRank': [2, 1]
    }, index=Metrics.index)

    pd.testing.assert_frame_equal(Ranking, Expected)
