from DataDownloader import YFinanceDownloader
from ConfigManager import ConfigManager
from LoggingManager import LoggingManager
from PerformanceMetric import PerformanceMetric
from PerformanceRanking import PerformanceRanking
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
    Tickers = Manager.GetParameter("Tickers")
    Data = Downloader.DownloadData(Tickers)
    logging.getLogger(__name__).info("Data downloaded: %d rows", len(Data))

    Metrics = PerformanceMetric(Data).GenerateMetrics()
    logging.getLogger(__name__).info("\n%s", Metrics)

    CompositeMetrics = Manager.GetParameter(
        "CompositeMetrics",
        ["CAGR", "Volatility", "MaxDrawdown", "SharpeRatio"],
    )
    Ranking = PerformanceRanking(Metrics).GenerateCompositeRanking(CompositeMetrics)
    logging.getLogger(__name__).info("\n%s", Ranking)
