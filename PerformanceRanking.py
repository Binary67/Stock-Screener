import pandas as pd

class PerformanceRanking:
    def __init__(self, MetricsDataFrame):
        self.MetricsDataFrame = MetricsDataFrame.copy()

    def _GetRankOrders(self):
        return {
            "CAGR": False,
            "Volatility": True,
            "MaxDrawdown": False,
            "SharpeRatio": False
        }

    def GenerateRanking(self):
        RankingDf = pd.DataFrame(index=self.MetricsDataFrame.index)
        Orders = self._GetRankOrders()
        for Column in self.MetricsDataFrame.columns:
            Ascending = Orders.get(Column, False)
            RankingDf[Column] = (
                self.MetricsDataFrame[Column]
                .rank(ascending=Ascending, method="min")
                .astype(int)
            )
        RankingDf.index.name = self.MetricsDataFrame.index.name
        return RankingDf
