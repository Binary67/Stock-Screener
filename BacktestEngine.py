import pandas as pd
from backtesting import Backtest, Strategy
from DataDownloader import YFinanceDownloader

class BuyAndHoldStrategy(Strategy):
    def init(self):
        self.buy()

    def next(self):
        pass

class BacktestEngine:
    def __init__(self, AllocationDataFrame, StartDate="2024-01-01", EndDate="2024-12-31", Interval="1d", CacheDir="Cache"):
        self.AllocationDataFrame = AllocationDataFrame.copy()
        self.StartDate = StartDate
        self.EndDate = EndDate
        self.Interval = Interval
        self.CacheDir = CacheDir

    def _DownloadPrices(self, Tickers):
        Downloader = YFinanceDownloader(Tickers[0], self.StartDate, self.EndDate, self.Interval, CacheDir=self.CacheDir)
        Data = Downloader.DownloadData(Tickers)
        return Data

    @staticmethod
    def _BuildPortfolioSeries(Data, Weights):
        Data = Data.reset_index().rename(columns={Data.index.name or 'index': 'Date'})
        PriceDf = Data.pivot(index='Date', columns='Ticker', values='Close')
        PriceDf.sort_index(inplace=True)
        PriceDf.fillna(method='ffill', inplace=True)
        PriceDf.dropna(inplace=True)
        Normalized = PriceDf / PriceDf.iloc[0]
        Portfolio = (Normalized * Weights).sum(axis=1)
        return Portfolio

    @staticmethod
    def _RunSingleBacktest(Series):
        DataFrame = pd.DataFrame({'Open': Series, 'High': Series, 'Low': Series, 'Close': Series, 'Volume': 0})
        Bt = Backtest(DataFrame, BuyAndHoldStrategy, cash=10000, commission=0, exclusive_orders=True)
        Stats = Bt.run()
        return Stats

    def RunBacktest(self):
        Tickers = list(self.AllocationDataFrame.index)
        Prices = self._DownloadPrices(Tickers)

        WeightSeries = self.AllocationDataFrame['Allocation']
        PortfolioWeighted = self._BuildPortfolioSeries(Prices, WeightSeries)
        WeightedStats = self._RunSingleBacktest(PortfolioWeighted)

        EqualWeights = pd.Series(1.0 / len(Tickers), index=Tickers)
        PortfolioEqual = self._BuildPortfolioSeries(Prices, EqualWeights)
        BaselineStats = self._RunSingleBacktest(PortfolioEqual)

        return WeightedStats, BaselineStats
