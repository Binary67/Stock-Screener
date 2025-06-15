import pandas as pd
import numpy as np

class AssetAllocation:
    def __init__(self, RankingDataFrame, Config):
        self.RankingDataFrame = RankingDataFrame.copy()
        self.Config = Config
        self.TopNPercent = float(self.Config.GetParameter("TopN", 10))
        self.Alpha = float(self.Config.GetParameter("Alpha", 1))

    def GenerateAllocations(self):
        if "CompositeRank" not in self.RankingDataFrame.columns:
            raise ValueError("CompositeRank not found in ranking data")

        SortedDf = self.RankingDataFrame.sort_values("CompositeRank")
        Count = len(SortedDf)
        TopCount = max(int(np.ceil(Count * self.TopNPercent / 100)), 1)
        TopDf = SortedDf.head(TopCount).copy()

        Ranks = np.arange(TopCount)
        Weights = np.exp(-self.Alpha * Ranks)
        Weights /= Weights.sum()
        TopDf["Allocation"] = Weights

        return TopDf[["Allocation"]]
