from DataDownloader import YFinanceDownloader

if __name__ == "__main__":
    Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-10", "1d")
    Data = Downloader.DownloadData()
    print(Data.head())
