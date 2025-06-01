import yfinance as yf
import pandas as pd

def DownloadData(TickerSymbols: list[str], StartDate: str, EndDate: str, Interval: str) -> dict[str, pd.DataFrame]:
    """
    Downloads historical stock data for a list of ticker symbols.
    """
    Data = {}
    for TickerSymbol in TickerSymbols:
        try:
            TickerData = yf.download(TickerSymbol, start=StartDate, end=EndDate, interval=Interval, progress=False)
            if TickerData.empty:
                print(f"No data found for {TickerSymbol} for the given period.")
                Data[TickerSymbol] = pd.DataFrame()
            else:
                Data[TickerSymbol] = TickerData
        except Exception as E:
            print(f"Could not download data for {TickerSymbol}: {E}")
            Data[TickerSymbol] = pd.DataFrame()
    return Data