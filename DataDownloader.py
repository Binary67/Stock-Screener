import os
import json
import logging
import yfinance as yf
import pandas as pd

class YFinanceDownloader:
    def __init__(self, Ticker, StartDate, EndDate, Interval, CacheDir="Cache"):
        self.Ticker = Ticker
        self.StartDate = pd.to_datetime(StartDate)
        self.EndDate = pd.to_datetime(EndDate)
        self.Interval = Interval
        self.CacheDir = CacheDir
        os.makedirs(self.CacheDir, exist_ok=True)
        logging.getLogger(__name__).info(
            "Initialized YFinanceDownloader for %s", self.Ticker
        )

    def _GetCachePaths(self):
        CsvFilePath = os.path.join(self.CacheDir, f"{self.Ticker}.csv")
        MetaFilePath = os.path.join(self.CacheDir, f"{self.Ticker}_meta.json")
        return CsvFilePath, MetaFilePath

    def _LoadFromCache(self, CsvFilePath, MetaFilePath):
        if os.path.exists(CsvFilePath) and os.path.exists(MetaFilePath):
            with open(MetaFilePath, "r") as File:
                Meta = json.load(File)
            if (
                Meta.get("StartDate") == self.StartDate.strftime("%Y-%m-%d")
                and Meta.get("EndDate") == self.EndDate.strftime("%Y-%m-%d")
                and Meta.get("Interval") == self.Interval
            ):
                logging.getLogger(__name__).info("Loading data for %s from cache", self.Ticker)
                return pd.read_csv(CsvFilePath, index_col=0, parse_dates=True)
        return None

    def _SaveToCache(self, DataFrame, CsvFilePath, MetaFilePath):
        DataFrame.to_csv(CsvFilePath)
        Meta = {
            "StartDate": self.StartDate.strftime("%Y-%m-%d"),
            "EndDate": self.EndDate.strftime("%Y-%m-%d"),
            "Interval": self.Interval,
        }
        with open(MetaFilePath, "w") as File:
            json.dump(Meta, File)
        logging.getLogger(__name__).info("Saved data for %s to cache", self.Ticker)

    def DownloadData(self):
        CsvFilePath, MetaFilePath = self._GetCachePaths()
        CachedDf = self._LoadFromCache(CsvFilePath, MetaFilePath)
        if CachedDf is not None:
            logging.getLogger(__name__).info("Data for %s loaded from cache", self.Ticker)
            return CachedDf

        # For hourly data, yfinance limits the period that can be downloaded.
        # We define common hourly interval labels.
        HourlyIntervals = ['60m', '1h', 'hourly']
        if self.Interval in HourlyIntervals:
            # If the total period exceeds 2 weeks (14 days), split into 2-week segments.
            if (self.EndDate - self.StartDate).days > 14:
                DataFrames = []
                CurrentStart = self.StartDate
                while CurrentStart < self.EndDate:
                    CurrentEnd = CurrentStart + pd.Timedelta(days=14)

                    if CurrentEnd > self.EndDate:
                        CurrentEnd = self.EndDate

                    TempData = yf.download(
                        self.Ticker,
                        start=CurrentStart.strftime('%Y-%m-%d'),
                        end=CurrentEnd.strftime('%Y-%m-%d'),
                        interval=self.Interval,
                        progress=False
                    )

                    if isinstance(TempData.columns, pd.MultiIndex):
                        TempData.columns = TempData.columns.droplevel(1)

                    DataFrames.append(TempData)
                    CurrentStart = CurrentEnd

                if DataFrames:
                    FinalDf = pd.concat(DataFrames)
                    self._SaveToCache(FinalDf, CsvFilePath, MetaFilePath)
                    logging.getLogger(__name__).info("Downloaded data for %s in segments", self.Ticker)
                    return FinalDf
                
        # For non-hourly data or periods within the 2-week limit, download directly.
        FinalDf = yf.download(
            self.Ticker,
            start=self.StartDate.strftime('%Y-%m-%d'),
            end=self.EndDate.strftime('%Y-%m-%d'),
            interval=self.Interval,
            progress=False
        )

        if isinstance(FinalDf.columns, pd.MultiIndex):
            FinalDf.columns = FinalDf.columns.droplevel(1)

        FinalDf.columns.name = None
        self._SaveToCache(FinalDf, CsvFilePath, MetaFilePath)
        logging.getLogger(__name__).info("Downloaded data for %s", self.Ticker)

        return FinalDf
