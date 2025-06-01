"""
This module implements a stock screener that downloads trading data using yfinance,
calculates various technical indicators, and ranks assets based on their performance.
"""
import yfinance as yf
import pandas as pd
import numpy as np

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

def CalculateRSI(Data: pd.DataFrame, Window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    """
    Delta = Data['Close'].diff()
    Gain = (Delta.where(Delta > 0, 0)).rolling(window=Window).mean()
    Loss = (-Delta.where(Delta < 0, 0)).rolling(window=Window).mean()

    RS = Gain / Loss
    RSI = 100 - (100 / (1 + RS))
    # Data[f'RSI_{Window}'] = RSI # Optional: Add as a column directly
    return RSI

def CalculateMACD(Data: pd.DataFrame, FastPeriod: int = 12, SlowPeriod: int = 26, SignalPeriod: int = 9) -> pd.DataFrame:
    """
    Calculates Moving Average Convergence Divergence (MACD).
    """
    EMAfast = Data['Close'].ewm(span=FastPeriod, adjust=False).mean()
    EMAslow = Data['Close'].ewm(span=SlowPeriod, adjust=False).mean()
    MACDLine = EMAfast - EMAslow # Corrected: Fast - Slow
    SignalLine = MACDLine.ewm(span=SignalPeriod, adjust=False).mean()
    MACDHistogram = MACDLine - SignalLine

    ReturnDf = pd.DataFrame(index=Data.index) # Initialize with Data's index
    ReturnDf['MACD'] = MACDLine
    ReturnDf['SignalLine'] = SignalLine
    ReturnDf['MACDHistogram'] = MACDHistogram

    return ReturnDf

def CalculateMovingAverages(Data: pd.DataFrame, Windows: list[int] = [20, 50]) -> pd.DataFrame:
    """
    Calculates Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
    """
    MA_Data = Data.copy() # Avoid modifying the original DataFrame directly if it's passed around
    for Window in Windows:
        MA_Data[f'SMA_{Window}'] = MA_Data['Close'].rolling(window=Window).mean()
        MA_Data[f'EMA_{Window}'] = MA_Data['Close'].ewm(span=Window, adjust=False).mean()
    return MA_Data

def CalculateBollingerBands(Data: pd.DataFrame, Window: int = 20, NumStdDev: int = 2) -> pd.DataFrame:
    """
    Calculates Bollinger Bands.
    """
    BB_Data = Data.copy()
    MiddleBand = BB_Data['Close'].rolling(window=Window).mean()
    StdDev = BB_Data['Close'].rolling(window=Window).std()
    UpperBand = MiddleBand + (StdDev * NumStdDev)
    LowerBand = MiddleBand - (StdDev * NumStdDev)

    ReturnDf = pd.DataFrame(index=Data.index) # Initialize with Data's index
    ReturnDf['MiddleBand'] = MiddleBand
    ReturnDf['UpperBand'] = UpperBand
    ReturnDf['LowerBand'] = LowerBand

    return ReturnDf

def CalculatePriceMomentum(Data: pd.DataFrame, Period: int = 10) -> pd.Series:
    """
    Calculates Price Momentum.
    Momentum = Price - Price `Period` days ago.
    """
    Momentum = Data['Close'].diff(Period)
    # Data[f'Momentum_{Period}'] = Momentum # Optional: Add as a column directly
    return Momentum

def RankAssets(StockDataDict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Ranks assets based on a composite score from technical indicators.
    """
    RankingList = []

    for TickerSymbol, StockDf in StockDataDict.items():
        if StockDf.empty or len(StockDf) < 20: # Need enough data for indicators
            RankingList.append({
                'Ticker': TickerSymbol,
                'CompositeScore': -1,
                'RSI': np.nan,
                'MACD_Above_Signal': 0,
                'Close_Above_SMA50': 0,
                'Close_Above_EMA50': 0, # Added for EMA50
                'Close_Above_BB_Middle': 0,
                'Positive_Momentum': 0,
                'Error': 'Insufficient data'
            })
            continue

        CurrentError = None
        try:
            # Calculate indicators & add them as columns to StockDf
            # It's often better to calculate indicators on a copy if StockDf is reused elsewhere,
            # but here we are processing it per ticker.
            StockDf.loc[:, 'RSI'] = CalculateRSI(StockDf)

            MacdDf = CalculateMACD(StockDf)
            StockDf.loc[:, 'MACD'] = MacdDf['MACD']
            StockDf.loc[:, 'SignalLine'] = MacdDf['SignalLine']

            # CalculateMovingAverages returns a DataFrame with new columns
            # Use default windows [20, 50]
            StockDf = CalculateMovingAverages(StockDf, Windows=[20, 50])

            BbDf = CalculateBollingerBands(StockDf)
            StockDf.loc[:, 'MiddleBand'] = BbDf['MiddleBand']

            StockDf.loc[:, 'Momentum'] = CalculatePriceMomentum(StockDf)

            # Get the latest values
            # Use .iloc[-1] for the last row, handle potential NaNs by filling or checking

            # RSI
            RawLatestRSI = StockDf['RSI'].iloc[-1]
            ScalarRSI = np.nan
            if isinstance(RawLatestRSI, pd.Series):
                if len(RawLatestRSI) == 1: ScalarRSI = RawLatestRSI.item()
                else: raise ValueError(f"For {TickerSymbol}, 'RSI' column's last value is a multi-item Series: {RawLatestRSI}. Cannot process.")
            elif isinstance(RawLatestRSI, pd.DataFrame): raise ValueError(f"For {TickerSymbol}, 'RSI' column's last value is a DataFrame: {RawLatestRSI}. Cannot process.")
            else: ScalarRSI = RawLatestRSI
            LatestRSI = ScalarRSI if pd.notna(ScalarRSI) else 0

            # MACD
            RawLatestMACD = StockDf['MACD'].iloc[-1]
            ScalarMACD = np.nan
            if isinstance(RawLatestMACD, pd.Series):
                if len(RawLatestMACD) == 1: ScalarMACD = RawLatestMACD.item()
                else: raise ValueError(f"For {TickerSymbol}, 'MACD' column's last value is a multi-item Series: {RawLatestMACD}. Cannot process.")
            elif isinstance(RawLatestMACD, pd.DataFrame): raise ValueError(f"For {TickerSymbol}, 'MACD' column's last value is a DataFrame: {RawLatestMACD}. Cannot process.")
            else: ScalarMACD = RawLatestMACD
            LatestMACD = ScalarMACD if pd.notna(ScalarMACD) else 0

            # SignalLine
            RawLatestSignalLine = StockDf['SignalLine'].iloc[-1]
            ScalarSignalLine = np.nan
            if isinstance(RawLatestSignalLine, pd.Series):
                if len(RawLatestSignalLine) == 1: ScalarSignalLine = RawLatestSignalLine.item()
                else: raise ValueError(f"For {TickerSymbol}, 'SignalLine' column's last value is a multi-item Series: {RawLatestSignalLine}. Cannot process.")
            elif isinstance(RawLatestSignalLine, pd.DataFrame): raise ValueError(f"For {TickerSymbol}, 'SignalLine' column's last value is a DataFrame: {RawLatestSignalLine}. Cannot process.")
            else: ScalarSignalLine = RawLatestSignalLine
            LatestSignalLine = ScalarSignalLine if pd.notna(ScalarSignalLine) else 0

            # Close
            RawLatestClose = StockDf['Close'].iloc[-1]
            ScalarClose = np.nan # Default to NaN before assignment

            if isinstance(RawLatestClose, pd.Series):
                if len(RawLatestClose) == 1:
                    ScalarClose = RawLatestClose.item()
                else:
                    # This error will be caught by the outer try-except and logged by RankAssets
                    raise ValueError(f"For {TickerSymbol}, 'Close' column's last value is a multi-item Series: {RawLatestClose}. Cannot process.")
            elif isinstance(RawLatestClose, pd.DataFrame):
                # This error will be caught by the outer try-except and logged by RankAssets
                raise ValueError(f"For {TickerSymbol}, 'Close' column's last value is a DataFrame: {RawLatestClose}. Cannot process.")
            else: # Assumed scalar
                ScalarClose = RawLatestClose

            LatestClose = ScalarClose if pd.notna(ScalarClose) else 0

            # Common function for MA and MiddleBand extraction
            def get_latest_indicator_value(column_name, default_value_for_nan):
                if column_name in StockDf.columns:
                    RawValue = StockDf[column_name].iloc[-1]
                    ScalarValue = np.nan
                    if isinstance(RawValue, pd.Series):
                        if len(RawValue) == 1: ScalarValue = RawValue.item()
                        else: raise ValueError(f"For {TickerSymbol}, '{column_name}' column's last value is a multi-item Series: {RawValue}. Cannot process.")
                    elif isinstance(RawValue, pd.DataFrame): raise ValueError(f"For {TickerSymbol}, '{column_name}' column's last value is a DataFrame: {RawValue}. Cannot process.")
                    else: ScalarValue = RawValue # Assumed scalar

                    return ScalarValue if pd.notna(ScalarValue) else default_value_for_nan
                return default_value_for_nan # If column doesn't exist, return the NaN default

            LatestSMA20 = get_latest_indicator_value('SMA_20', LatestClose)
            LatestSMA50 = get_latest_indicator_value('SMA_50', LatestClose)
            LatestEMA20 = get_latest_indicator_value('EMA_20', LatestClose)
            LatestEMA50 = get_latest_indicator_value('EMA_50', LatestClose)
            LatestMiddleBand = get_latest_indicator_value('MiddleBand', LatestClose)

            # Momentum
            RawLatestMomentum = StockDf['Momentum'].iloc[-1]
            ScalarMomentum = np.nan
            if isinstance(RawLatestMomentum, pd.Series):
                if len(RawLatestMomentum) == 1: ScalarMomentum = RawLatestMomentum.item()
                else: raise ValueError(f"For {TickerSymbol}, 'Momentum' column's last value is a multi-item Series: {RawLatestMomentum}. Cannot process.")
            elif isinstance(RawLatestMomentum, pd.DataFrame): raise ValueError(f"For {TickerSymbol}, 'Momentum' column's last value is a DataFrame: {RawLatestMomentum}. Cannot process.")
            else: ScalarMomentum = RawLatestMomentum
            LatestMomentum = ScalarMomentum if pd.notna(ScalarMomentum) else 0

            CompositeScore = 0

            # RSI: +1 if between 50 and 70
            RSI_Score = 0
            if 50 < LatestRSI < 70:
                RSI_Score = 1
            CompositeScore += RSI_Score

            # MACD: +1 if MACD > Signal
            MACD_Score = 0
            if LatestMACD > LatestSignalLine:
                MACD_Score = 1
            CompositeScore += MACD_Score

            # Moving Averages: +1 if Close > SMA_50 and +1 if Close > EMA_50
            SMA50_Score = 0
            if LatestClose > LatestSMA50:
                SMA50_Score = 1
            CompositeScore += SMA50_Score

            EMA50_Score = 0 # Score for EMA_50
            if LatestClose > LatestEMA50:
                EMA50_Score = 1
            CompositeScore += EMA50_Score

            # Bollinger Middle Band: +1 if Close > MiddleBand
            BB_Score = 0
            if LatestClose > LatestMiddleBand:
                BB_Score = 1
            CompositeScore += BB_Score

            # Momentum: +1 if Momentum > 0
            Momentum_Score = 0
            if LatestMomentum > 0:
                Momentum_Score = 1
            CompositeScore += Momentum_Score

            RankingList.append({
                'Ticker': TickerSymbol,
                'CompositeScore': CompositeScore,
                'RSI': LatestRSI,
                'MACD_Above_Signal': MACD_Score,
                'Close_Above_SMA50': SMA50_Score,
                'Close_Above_EMA50': EMA50_Score, # Added EMA50 score
                'Close_Above_BB_Middle': BB_Score,
                'Positive_Momentum': Momentum_Score,
                'Error': None
            })
        except Exception as E:
            CurrentError = str(E)
            print(f"Error processing {TickerSymbol}: {E}")
            RankingList.append({
                'Ticker': TickerSymbol,
                'CompositeScore': -1, # Low score due to error
                'RSI': np.nan,
                'MACD_Above_Signal': 0,
                'Close_Above_SMA50': 0,
                'Close_Above_EMA50': 0,
                'Close_Above_BB_Middle': 0,
                'Positive_Momentum': 0,
                'Error': CurrentError
            })


    RankedDf = pd.DataFrame(RankingList)
    RankedDf = RankedDf.sort_values(by='CompositeScore', ascending=False)
    return RankedDf

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
