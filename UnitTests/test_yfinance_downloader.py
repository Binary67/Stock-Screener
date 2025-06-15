import os
import json
import pandas as pd
from unittest.mock import patch
from DataDownloader import YFinanceDownloader


def test_cache_usage(tmp_path):
    CachePath = tmp_path / "Cache"
    CachePath.mkdir()
    TestDf = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3))

    with patch("DataDownloader.yf.download", return_value=TestDf) as MockDownload:
        Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-03", "1d", CacheDir=str(CachePath))
        Data1 = Downloader.DownloadData()
        assert MockDownload.call_count == 1
        CsvFile = CachePath / "AAPL.csv"
        MetaFile = CachePath / "AAPL_meta.json"
        assert CsvFile.exists() and MetaFile.exists()
        assert Data1.equals(TestDf)

    with patch("DataDownloader.yf.download", return_value=TestDf) as MockDownload:
        Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-03", "1d", CacheDir=str(CachePath))
        Data2 = Downloader.DownloadData()
        assert MockDownload.call_count == 0
        assert Data2.equals(TestDf)

    NewDf = pd.DataFrame({"Close": [4, 5, 6]}, index=pd.date_range("2022-01-01", periods=3))
    with patch("DataDownloader.yf.download", return_value=NewDf) as MockDownload:
        Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-03", "1h", CacheDir=str(CachePath))
        Data3 = Downloader.DownloadData()
        assert MockDownload.call_count == 1
        with open(CachePath / "AAPL_meta.json") as MetaFile:
            Meta = json.load(MetaFile)
        assert Meta["Interval"] == "1h"
        assert Data3.equals(NewDf)
