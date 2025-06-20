from DataDownloader import YFinanceDownloader
from ConfigManager import ConfigManager
from LoggingManager import LoggingManager
from PerformanceMetric import PerformanceMetric
from PerformanceRanking import PerformanceRanking
from AssetAllocation import AssetAllocation
from BacktestEngine import BacktestEngine
import logging

if __name__ == "__main__":
    LoggingManager()
    Manager = ConfigManager()
    Downloader = YFinanceDownloader(
        Manager.GetParameter("Ticker"),
        Manager.GetParameter("StartDate"),
        Manager.GetParameter("EndDate"),
        Manager.GetParameter("Interval"),
        CacheDir=Manager.GetParameter("CacheDir")
    )
    CsvPath, _ = Downloader.GetCachePaths()
    logging.getLogger(__name__).info("Using cache file %s", CsvPath)
    Tickers = Manager.GetParameter("Tickers")
    Data = Downloader.DownloadData(Tickers)
    logging.getLogger(__name__).info("Data downloaded: %d rows", len(Data))

    Metrics = PerformanceMetric(Data).GenerateMetrics()
    logging.getLogger(__name__).info("\n%s", Metrics)

    CompositeMetrics = Manager.GetParameter(
        "CompositeMetrics",
        ["CAGR", "Volatility", "MaxDrawdown", "SharpeRatio", "SortinoRatio"],
    )
    Ranking = PerformanceRanking(Metrics).GenerateCompositeRanking(CompositeMetrics)
    logging.getLogger(__name__).info("\n%s", Ranking)

    Allocation = AssetAllocation(Ranking, Manager).GenerateAllocations()
    logging.getLogger(__name__).info("\n%s", Allocation)

    Engine = BacktestEngine(Allocation)
    WeightedStats, EqualStats = Engine.RunBacktest()
    logging.getLogger(__name__).info("Weighted Allocation Stats:\n%s", WeightedStats)
    logging.getLogger(__name__).info("Equal Allocation Stats:\n%s", EqualStats)
