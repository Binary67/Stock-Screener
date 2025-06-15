import pandas as pd

class PerformanceRanking:
    def __init__(self, MetricsDataFrame):
        self.MetricsDataFrame = MetricsDataFrame.copy()

    def _GetRankOrders(self):
        return {
            "CAGR": False,
            "Volatility": True,
            "MaxDrawdown": False,
            "SharpeRatio": False,
            "SortinoRatio": False
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

    def GenerateCompositeRanking(self, MetricsList):
        RankingDf = self.GenerateRanking()
        if not MetricsList:
            raise ValueError("MetricsList cannot be empty")
        for Metric in MetricsList:
            if Metric not in RankingDf.columns:
                raise ValueError(f"Metric {Metric} not found in ranking data")
        Product = RankingDf[MetricsList].prod(axis=1)
        Root = len(MetricsList)
        CompositeScore = Product ** (1 / Root)
        RankingDf["CompositeRank"] = (
            CompositeScore.rank(ascending=True, method="min").astype(int)
        )
        return RankingDf
