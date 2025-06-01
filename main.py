from DataDownloader import DownloadData
from StockScreener import RankAssets

if __name__ == "__main__":
    SampleTickerSymbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    MyStartDate = "2023-01-01"
    MyEndDate = "2023-12-01" # Using a fixed recent past date
    MyInterval = "1d"

    print(f"Starting stock screener for: {SampleTickerSymbols}")
    print(f"Data period: {MyStartDate} to {MyEndDate}\n")

    # 1. Download Data
    print("Downloading stock data...")
    StockDataDictionary = DownloadData(TickerSymbols=SampleTickerSymbols, StartDate=MyStartDate, EndDate=MyEndDate, Interval=MyInterval)
    print("Data download complete.\n")

    # 2. Optional: Print head of one stock's data if downloaded
    TestTicker = 'AAPL'
    if TestTicker in StockDataDictionary and not StockDataDictionary[TestTicker].empty:
        print(f"Sample data for {TestTicker}:")
        print(StockDataDictionary[TestTicker].head())
        print("\n")
    elif TestTicker in StockDataDictionary and StockDataDictionary[TestTicker].empty:
        print(f"No data was downloaded for {TestTicker} for the specified period.\n")
    else:
        print(f"{TestTicker} was not in the download list or an error occurred.\n")

    # 3. Rank Assets
    print("Ranking assets based on technical indicators...")
    RankedAssetsDf = RankAssets(StockDataDictionary)
    print("Asset ranking complete.\n")

    # 4. Print Ranked Assets
    print("Ranked Assets:")
    if not RankedAssetsDf.empty:
        print(RankedAssetsDf.to_string())
    else:
        print("No assets were ranked (possibly due to data issues for all tickers).")

    print("\nStock screener run finished.")