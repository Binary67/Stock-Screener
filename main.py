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
    Data = Downloader.DownloadData()
