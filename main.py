from DataDownloader import YFinanceDownloader
from ConfigManager import ConfigManager
from LoggingManager import LoggingManager
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
