import os
import sys
import json
import pandas as pd
from unittest.mock import patch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
        Expected = TestDf.copy()
        Expected["Ticker"] = "AAPL"
        assert Data1.equals(Expected)

    with patch("DataDownloader.yf.download", return_value=TestDf) as MockDownload:
        Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-03", "1d", CacheDir=str(CachePath))
        Data2 = Downloader.DownloadData()
        assert MockDownload.call_count == 0
        Expected = TestDf.copy()
        Expected["Ticker"] = "AAPL"
        assert Data2.equals(Expected)

    NewDf = pd.DataFrame({"Close": [4, 5, 6]}, index=pd.date_range("2022-01-01", periods=3))
    with patch("DataDownloader.yf.download", return_value=NewDf) as MockDownload:
        Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-03", "1h", CacheDir=str(CachePath))
        Data3 = Downloader.DownloadData()
        assert MockDownload.call_count == 1
        with open(CachePath / "AAPL_meta.json") as MetaFile:
            Meta = json.load(MetaFile)
        assert Meta["Interval"] == "1h"
        Expected = NewDf.copy()
        Expected["Ticker"] = "AAPL"
        assert Data3.equals(Expected)


def test_multi_ticker_download(tmp_path):
    CachePath = tmp_path / "Cache"
    CachePath.mkdir()

    DfA = pd.DataFrame({"Close": [1]}, index=pd.date_range("2022-01-01", periods=1))
    DfB = pd.DataFrame({"Close": [2]}, index=pd.date_range("2022-01-01", periods=1))

    with patch("DataDownloader.yf.download", side_effect=[DfA, DfB]) as MockDownload:
        Downloader = YFinanceDownloader("AAPL", "2022-01-01", "2022-01-01", "1d", CacheDir=str(CachePath))
        Result = Downloader.DownloadData(["AAPL", "MSFT"])
        assert MockDownload.call_count == 2
        ExpectedA = DfA.copy()
        ExpectedA["Ticker"] = "AAPL"
        ExpectedB = DfB.copy()
        ExpectedB["Ticker"] = "MSFT"
        Expected = pd.concat([ExpectedA, ExpectedB])
        assert Result.equals(Expected)
