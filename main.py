from DataDownloader import YFinanceDownloader
from ConfigManager import ConfigManager

if __name__ == "__main__":
    Manager = ConfigManager()
    Downloader = YFinanceDownloader(
        Manager.GetParameter("Ticker"),
        Manager.GetParameter("StartDate"),
        Manager.GetParameter("EndDate"),
        Manager.GetParameter("Interval"),
        CacheDir=Manager.GetParameter("CacheDir")
    )
    Data = Downloader.DownloadData()
    print(Data.head())
