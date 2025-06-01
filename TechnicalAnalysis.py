import pandas as pd
import numpy as np

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