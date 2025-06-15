import pandas as pd
import numpy as np

class PerformanceMetric:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame.copy()

    def _GroupByTicker(self):
        if 'Ticker' not in self.DataFrame.columns:
            raise ValueError('Ticker column missing from DataFrame')
        return self.DataFrame.groupby('Ticker')

    @staticmethod
    def _CalculateDailyReturns(Series):
        Returns = Series.pct_change().dropna()
        return Returns

    def CalculateCompoundAnnualGrowthRate(self):
        Results = {}
        for Ticker, Group in self._GroupByTicker():
            FirstPrice = Group['Close'].iloc[0]
            LastPrice = Group['Close'].iloc[-1]
            Days = (Group.index[-1] - Group.index[0]).days or 1
            Cagr = (LastPrice / FirstPrice) ** (365 / Days) - 1
            Results[Ticker] = Cagr
        return pd.Series(Results)

    def CalculateVolatility(self):
        Results = {}
        for Ticker, Group in self._GroupByTicker():
            Returns = self._CalculateDailyReturns(Group['Close'])
            Vol = Returns.std() * np.sqrt(252)
            Results[Ticker] = Vol
        return pd.Series(Results)

    def CalculateMaxDrawdown(self):
        Results = {}
        for Ticker, Group in self._GroupByTicker():
            Cumulative = Group['Close'] / Group['Close'].iloc[0]
            RunningMax = Cumulative.cummax()
            Drawdown = (Cumulative - RunningMax) / RunningMax
            Results[Ticker] = Drawdown.min()
        return pd.Series(Results)

    def CalculateSharpeRatio(self, RiskFreeRate=0.0):
        Results = {}
        for Ticker, Group in self._GroupByTicker():
            Returns = self._CalculateDailyReturns(Group['Close'])
            Excess = Returns - RiskFreeRate / 252
            Sharpe = Excess.mean() / Returns.std() * np.sqrt(252)
            Results[Ticker] = Sharpe
        return pd.Series(Results)

    def GenerateMetrics(self):
        Cagr = self.CalculateCompoundAnnualGrowthRate()
        Vol = self.CalculateVolatility()
        Mdd = self.CalculateMaxDrawdown()
        Sharpe = self.CalculateSharpeRatio()
        MetricsDf = pd.concat([
            Cagr.rename('CAGR'),
            Vol.rename('Volatility'),
            Mdd.rename('MaxDrawdown'),
            Sharpe.rename('SharpeRatio')
        ], axis=1)
        MetricsDf.index.name = 'Ticker'
        return MetricsDf
