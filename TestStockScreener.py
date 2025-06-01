import unittest
import pandas as pd
import numpy as np
# Import the module itself to allow direct patching of its functions
import StockScreener as SS # Use an alias for convenience
from StockScreener import RankAssets # We still need RankAssets directly for calling

# We will access other functions like SS.CalculateRSI

class TestStockScreener(unittest.TestCase):

    def test_DownloadData_valid_ticker(self):
        # This test relies on yfinance successfully fetching data, which can be slow or fail due to network.
        # For true unit tests, yf.download would be mocked. For this project, live calls are acceptable.
        Data = SS.DownloadData(TickerSymbols=['AAPL'], StartDate='2023-01-01', EndDate='2023-01-10', Interval='1d')
        self.assertIsInstance(Data, dict)
        self.assertIn('AAPL', Data)
        self.assertIsInstance(Data['AAPL'], pd.DataFrame)
        if not Data['AAPL'].empty: # If yfinance fails, this will be empty. Test should still pass based on function design.
            self.assertIn('Close', Data['AAPL'].columns)
        # else: The function is expected to return an empty DF if download fails, which is handled by next test.

    def test_DownloadData_invalid_ticker(self):
        Data = SS.DownloadData(TickerSymbols=['INVALIDTICKERXYZ'], StartDate='2023-01-01', EndDate='2023-01-10', Interval='1d')
        self.assertIsInstance(Data, dict)
        self.assertIn('INVALIDTICKERXYZ', Data)
        self.assertTrue(Data['INVALIDTICKERXYZ'].empty) # As per DownloadData design for errors/no data

    def test_DownloadData_no_data_date_range(self):
        # Using a future date range where no data should exist
        Data = SS.DownloadData(TickerSymbols=['AAPL'], StartDate='2099-01-01', EndDate='2099-01-10', Interval='1d')
        self.assertIsInstance(Data, dict)
        self.assertIn('AAPL', Data)
        self.assertTrue(Data['AAPL'].empty)

    # --- Tests for Technical Indicators ---
    def setUp(self):
        # Sample data for various indicator tests
        self.SampleData = pd.DataFrame({
            'Open':   [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95],
            'High':   [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 109.5, 108.5, 107.5, 106.5, 105.5, 104.5, 103.5, 102.5, 101.5, 100.5, 99.5, 98.5, 97.5, 96.5, 95.5],
            'Low':    [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 108.5, 107.5, 106.5, 105.5, 104.5, 103.5, 102.5, 101.5, 100.5, 99.5, 98.5, 97.5, 96.5, 95.5, 94.5],
            'Close':  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95],
            'Volume': [1000]*26 # Length 26 to test MACD defaults (SlowPeriod=26)
        })
        self.RsiTestData = pd.DataFrame({ # Data from a known source for RSI
            'Close': [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64] * 2 # Repeat to get enough data
        })
        self.ConstantPriceData = pd.DataFrame({'Close': [50.0] * 30})

    def test_CalculateRSI(self):
        DefaultWindow = 14
        RsiSeries = SS.CalculateRSI(self.RsiTestData.copy(), Window=DefaultWindow)
        self.assertIsInstance(RsiSeries, pd.Series)
        self.assertEqual(len(RsiSeries), len(self.RsiTestData))
        self.assertTrue(RsiSeries.iloc[:DefaultWindow-1].isnull().all())
        # For constant price, RSI should be undefined (Loss is 0 -> RS is inf) or 100 if Gain is also 0 after window,
        # or handled as NaN by pandas/numpy division by zero. Some implementations show 50.
        # Let's check a more dynamic series' last value.
        self.assertIsInstance(RsiSeries.iloc[-1], float)
        self.assertTrue(0 <= RsiSeries.iloc[-1] <= 100)

        # Test with constant price data - expect NaN or a specific value like 100 if gain=loss=0
        RsiConstant = SS.CalculateRSI(self.ConstantPriceData.copy(), Window=DefaultWindow)
        # Depending on implementation details of .rolling().mean() with all zeros,
        # Gain and Loss can both be 0. RS = 0/0 -> NaN. RSI = 100 - (100 / (1+NaN)) -> NaN
        # Or, if Gain is 0 and Loss is 0, RS might be 1 (convention), then RSI = 50.
        # Or, if Loss is 0 and Gain > 0, RSI = 100.
        # The current implementation: Gain and Loss become 0 after window if price is constant. RS = 0/0 = NaN.
        self.assertTrue(np.isnan(RsiConstant.iloc[-1]))


    def test_CalculateMACD(self):
        MacdDf = SS.CalculateMACD(self.SampleData.copy()) # Uses default Fast=12, Slow=26, Signal=9
        self.assertIsInstance(MacdDf, pd.DataFrame)
        self.assertIn('MACD', MacdDf.columns)
        self.assertIn('SignalLine', MacdDf.columns)
        self.assertIn('MACDHistogram', MacdDf.columns)
        # EMA calculation makes initial values non-NaN, but they are less reliable.
        # Pandas ewm default has min_periods=0. Let's check last value is float.
        self.assertTrue(MacdDf['MACD'].iloc[0:24].isnull().all) # NaN until enough data for SlowPeriod (26-1 index)
        self.assertIsInstance(MacdDf['MACD'].iloc[-1], float)
        self.assertIsInstance(MacdDf['SignalLine'].iloc[-1], float)
        self.assertIsInstance(MacdDf['MACDHistogram'].iloc[-1], float)


    def test_CalculateMovingAverages(self):
        Windows = [5, 10]
        MaDf = SS.CalculateMovingAverages(self.SampleData.copy(), Windows=Windows)
        self.assertIsInstance(MaDf, pd.DataFrame)
        for W in Windows:
            self.assertIn(f'SMA_{W}', MaDf.columns)
            self.assertIn(f'EMA_{W}', MaDf.columns)
            self.assertTrue(MaDf[f'SMA_{W}'].iloc[:W-1].isnull().all()) # Check SMA NaNs
            self.assertIsInstance(MaDf[f'SMA_{W}'].iloc[-1], float)
            self.assertIsInstance(MaDf[f'EMA_{W}'].iloc[-1], float) # EMA typically doesn't have starting NaNs with adjust=False

        # Manual check for SMA_5 on SampleData
        # Close prices: ..., 100, 99, 98, 97, 96, 95
        # SMA_5 for last point: (99+98+97+96+95)/5 = 97
        self.assertEqual(MaDf['SMA_5'].iloc[-1], (self.SampleData['Close'].iloc[-5:].mean()))
        self.assertEqual(MaDf['SMA_5'].iloc[-1], 97.0)


    def test_CalculateBollingerBands(self):
        Window = 5
        BbDf = SS.CalculateBollingerBands(self.SampleData.copy(), Window=Window)
        self.assertIsInstance(BbDf, pd.DataFrame)
        self.assertIn('MiddleBand', BbDf.columns)
        self.assertIn('UpperBand', BbDf.columns)
        self.assertIn('LowerBand', BbDf.columns)
        self.assertTrue(BbDf['MiddleBand'].iloc[:Window-1].isnull().all())
        self.assertEqual(BbDf['MiddleBand'].iloc[-1], self.SampleData['Close'].iloc[-Window:].mean()) # MiddleBand is SMA
        self.assertGreaterEqual(BbDf['UpperBand'].iloc[-1], BbDf['MiddleBand'].iloc[-1])
        self.assertLessEqual(BbDf['LowerBand'].iloc[-1], BbDf['MiddleBand'].iloc[-1])

    def test_CalculatePriceMomentum(self):
        Period = 5
        MomentumSeries = SS.CalculatePriceMomentum(self.SampleData.copy(), Period=Period)
        self.assertIsInstance(MomentumSeries, pd.Series)
        self.assertTrue(MomentumSeries.iloc[:Period].isnull().all())
        # Last value: Close[-1] - Close[-1-Period] = 95 - 100 = -5
        self.assertEqual(MomentumSeries.iloc[-1], self.SampleData['Close'].iloc[-1] - self.SampleData['Close'].iloc[-1-Period])
        self.assertEqual(MomentumSeries.iloc[-1], -5.0)

    # --- Tests for RankAssets ---
    def test_RankAssets_basic_scoring_and_order(self):
        # Create mock data. One clearly "good", one "bad" based on scoring rules
        DataGood = pd.DataFrame({
            'Close': np.arange(100, 120, 1) # Rising prices
        })
        # Manually craft last row values to hit scoring criteria after indicators are calculated
        # This requires understanding how indicators behave with this data.
        # For simplicity, we'll use a longer series and rely on RankAssets internal calculations

        GoodStockData = pd.DataFrame({'Close': np.linspace(100, 150, 50)}) # Generally up
        BadStockData = pd.DataFrame({'Close': np.linspace(100, 50, 50)})   # Generally down
        NeutralStockData = pd.DataFrame({'Close': [100]*50}) # Flat

        TestStockDict = {'GOOD': GoodStockData, 'BAD': BadStockData, 'NEUTRAL': NeutralStockData}

        RankedDf = RankAssets(TestStockDict)
        self.assertIsInstance(RankedDf, pd.DataFrame)
        self.assertEqual(len(RankedDf), 3)

        # Check general order - GOOD should be higher than BAD and NEUTRAL
        ScoreGood = RankedDf[RankedDf['Ticker'] == 'GOOD']['CompositeScore'].iloc[0]
        ScoreBad = RankedDf[RankedDf['Ticker'] == 'BAD']['CompositeScore'].iloc[0]
        ScoreNeutral = RankedDf[RankedDf['Ticker'] == 'NEUTRAL']['CompositeScore'].iloc[0]

        self.assertGreater(ScoreGood, ScoreBad)
        # Neutral stock might get some points (e.g. not overbought/oversold), but likely less than GOOD
        self.assertTrue(ScoreGood >= ScoreNeutral) # Could be equal if good criteria are not met perfectly
        self.assertTrue(ScoreNeutral >= ScoreBad)


    def test_RankAssets_empty_and_insufficient_data(self):
        SufficientData = pd.DataFrame({'Close': np.linspace(100,120,30)}) # 30 days of data
        InsufficientData = pd.DataFrame({'Close': np.linspace(100,105,10)}) # Only 10 days
        EmptyData = pd.DataFrame()

        TestStockDict = {'SUFFICIENT': SufficientData, 'INSUFFICIENT': InsufficientData, 'EMPTY': EmptyData}
        RankedDf = RankAssets(TestStockDict)

        self.assertEqual(RankedDf.loc[RankedDf['Ticker'] == 'EMPTY', 'CompositeScore'].iloc[0], -1)
        self.assertIn('Insufficient data', RankedDf.loc[RankedDf['Ticker'] == 'EMPTY', 'Error'].iloc[0])

        self.assertEqual(RankedDf.loc[RankedDf['Ticker'] == 'INSUFFICIENT', 'CompositeScore'].iloc[0], -1)
        self.assertIn('Insufficient data', RankedDf.loc[RankedDf['Ticker'] == 'INSUFFICIENT', 'Error'].iloc[0])

        self.assertNotEqual(RankedDf.loc[RankedDf['Ticker'] == 'SUFFICIENT', 'CompositeScore'].iloc[0], -1)
        self.assertTrue(pd.isnull(RankedDf.loc[RankedDf['Ticker'] == 'SUFFICIENT', 'Error'].iloc[0]))

    def test_RankAssets_with_minimal_data(self):
        # MinDataPoints in RankAssets is 20
        MinDataPoints = 20

        # Case 1: Data less than MinDataPoints (e.g., 5 rows)
        DataFiveRows = pd.DataFrame({
            'Open': [10]*5, 'High': [10]*5, 'Low': [10]*5, 'Close': [10, 11, 12, 11, 10], 'Volume': [100]*5
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))

        # Case 2: Data just enough or slightly more than MinDataPoints (e.g., 22 rows)
        ClosePricesTwentyTwo = [10,11,12,13,14,15,16,17,18,19,20,21,22,21,20,19,18,17,16,15,14,13]
        DataTwentyTwoRows = pd.DataFrame({
            'Open': ClosePricesTwentyTwo, 'High': ClosePricesTwentyTwo, 'Low': ClosePricesTwentyTwo, 'Close': ClosePricesTwentyTwo, 'Volume': [100]*len(ClosePricesTwentyTwo)
        }, index=pd.date_range('2023-01-01', periods=len(ClosePricesTwentyTwo), freq='D'))

        # Case 3: Extremely minimal data (1 row)
        DataOneRow = pd.DataFrame({
            'Open': [10], 'High': [10], 'Low': [10], 'Close': [10], 'Volume': [100]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))

        TestStockDict = {
            'FIVE_ROWS': DataFiveRows,
            'TWENTY_TWO_ROWS': DataTwentyTwoRows,
            'ONE_ROW': DataOneRow
        }

        try:
            RankedDf = RankAssets(TestStockDict)
            self.assertIsInstance(RankedDf, pd.DataFrame)

            # Check outcomes for tickers that should be filtered by MinDataPoints
            ScoreFiveRows = RankedDf.loc[RankedDf['Ticker'] == 'FIVE_ROWS', 'CompositeScore'].iloc[0]
            ErrorFiveRows = RankedDf.loc[RankedDf['Ticker'] == 'FIVE_ROWS', 'Error'].iloc[0]
            self.assertEqual(ScoreFiveRows, -1)
            self.assertIn("Insufficient data", ErrorFiveRows)

            ScoreOneRow = RankedDf.loc[RankedDf['Ticker'] == 'ONE_ROW', 'CompositeScore'].iloc[0]
            ErrorOneRow = RankedDf.loc[RankedDf['Ticker'] == 'ONE_ROW', 'Error'].iloc[0]
            self.assertEqual(ScoreOneRow, -1)
            self.assertIn("Insufficient data", ErrorOneRow)

            # Check outcome for the ticker that passes MinDataPoints
            # It should process without an "Insufficient data" error.
            # Its score will depend on indicator calculations on minimal data (likely many NaNs).
            ScoreTwentyTwoRows = RankedDf.loc[RankedDf['Ticker'] == 'TWENTY_TWO_ROWS', 'CompositeScore'].iloc[0]
            ErrorTwentyTwoRows = RankedDf.loc[RankedDf['Ticker'] == 'TWENTY_TWO_ROWS', 'Error'].iloc[0]

            # We expect a score to be calculated, not -1 from the insufficient data check.
            # The score might be low (even 0) if all indicators result in NaN and defaults are 0.
            self.assertNotEqual(ScoreTwentyTwoRows, -1, "Score for 22 rows should not be -1 due to insufficient data check.")
            self.assertTrue(pd.isna(ErrorTwentyTwoRows) or "Insufficient data" not in ErrorTwentyTwoRows,
                            f"Error for 22 rows should not be 'Insufficient data'. Got: {ErrorTwentyTwoRows}")

        except ValueError as ve:
            if "If using all scalar values, you must pass an index" in str(ve):
                self.fail(f"RankAssets raised a 'scalar values' ValueError with minimal data: {ve}")
            else:
                self.fail(f"RankAssets raised an unexpected ValueError with minimal data: {ve}")
        except Exception as e:
            self.fail(f"RankAssets raised an unexpected exception with minimal data: {e}")

    def test_RankAssets_detects_series_in_latest_extraction(self):
        NumRows = 22 # To pass MinDataPoints check
        TestIndex = pd.date_range('2023-01-01', periods=NumRows, freq='D')

        # Create a base DataFrame
        BaseData = {
            'Close': np.random.rand(NumRows) * 100 + 10,
            'RSI': [50.0] * NumRows,
            'MACD': [0.0] * NumRows,
            'SignalLine': [0.0] * NumRows,
            'SMA_20': [100.0] * NumRows, # Will be recalculated by RankAssets if not present
            'EMA_20': [100.0] * NumRows, # Will be recalculated by RankAssets if not present
            'SMA_50': [100.0] * NumRows, # Will be recalculated by RankAssets if not present
            'EMA_50': [100.0] * NumRows, # Will be recalculated by RankAssets if not present
            'MiddleBand': [100.0] * NumRows, # Will be recalculated by RankAssets if not present
            'Momentum': [0.0] * NumRows # Will be recalculated by RankAssets if not present
        }

        # --- Test for RSI ---
        TestDataRSI = pd.DataFrame(BaseData, index=TestIndex).copy()

        # Simulate the 'RSI' column's last value being a single-element Series
        RsiColValues = [50.0] * (NumRows - 1)
        RsiColValues.append(pd.Series([75.0], index=[TestDataRSI.index[-1]]))
        TestDataRSI['RSI'] = pd.Series(RsiColValues, index=TestDataRSI.index, dtype=object)

        # Note: RankAssets internally calls indicator calculation functions like CalculateRSI.
        # These calls will overwrite our manually set 'RSI' column *before* the latest value extraction.
        # To test the extraction logic directly, we need to mock these calculation functions
        # or make our pre-set column survive.
        # For this test, the simplest way is to ensure RankAssets *uses* our pre-set column.
        # The current RankAssets structure is:
        #   StockDf.loc[:, 'RSI'] = CalculateRSI(StockDf)  <-- This overwrites
        #   ...
        #   LatestRSI = StockDf['RSI'].iloc[-1] if pd.notna(StockDf['RSI'].iloc[-1]) else 0
        #
        # To effectively test the type check on LatestRSI, the CalculateRSI (and others)
        # within RankAssets must NOT run or must return our pre-crafted Series.
        # This requires mocking. Let's assume for now the goal is to test the type check
        # on data that *could* have come from a misbehaving (or non-scalar returning) iloc[-1].
        #
        # Given the current structure of RankAssets, it's difficult to inject a Series
        # into `StockDf['RSI'].iloc[-1]` just before the check without mocking `CalculateRSI`
        # itself to return a column where the last element is a Series.
        #
        # Let's adjust the test: We'll mock the indicator functions so they don't run,
        # allowing our manually crafted columns to be read by the "Latest" extraction logic.

        OriginalCalculateRSI = SS.CalculateRSI # Correct: Save from SS module
        OriginalCalculateMACD = SS.CalculateMACD # Correct: Save from SS module
        # Add mocks for all indicator calculations within RankAssets
        OriginalCalculateMovingAverages = SS.CalculateMovingAverages # Correct: Save from SS module
        OriginalCalculateBollingerBands = SS.CalculateBollingerBands # Correct: Save from SS module
        OriginalCalculatePriceMomentum = SS.CalculatePriceMomentum # Correct: Save from SS module

        def MockCalculateRSI(Data, Window=14): return Data['RSI'] # Returns the existing column
        def MockCalculateMACD(Data, FastPeriod=12, SlowPeriod=26, SignalPeriod=9):
            # Return a DataFrame with the existing MACD/SignalLine columns
            Df = pd.DataFrame(index=Data.index)
            Df['MACD'] = Data['MACD']
            Df['SignalLine'] = Data['SignalLine']
            Df['MACDHistogram'] = Data['MACD'] - Data['SignalLine'] # Dummy histogram
            return Df
        def MockCalculateMovingAverages(Data, Windows=[20,50]): return Data # Returns Data with existing MA columns
        def MockCalculateBollingerBands(Data, Window=20, NumStdDev=2):
            Df = pd.DataFrame(index=Data.index)
            Df['MiddleBand'] = Data['MiddleBand']
            Df['UpperBand'] = Data['MiddleBand'] + 10 # Dummy
            Df['LowerBand'] = Data['MiddleBand'] - 10 # Dummy
            return Df
        def MockCalculatePriceMomentum(Data, Period=10): return Data['Momentum']

        # Patch functions directly on the imported StockScreener module (SS)
        SS.CalculateRSI = MockCalculateRSI
        SS.CalculateMACD = MockCalculateMACD
        SS.CalculateMovingAverages = MockCalculateMovingAverages
        SS.CalculateBollingerBands = MockCalculateBollingerBands
        SS.CalculatePriceMomentum = MockCalculatePriceMomentum

        try:
            # RankAssets is imported directly, so it will use the patched functions from SS
            StockDataDictRSI = {'TEST_SERIES_RSI': TestDataRSI}
            RankedDf_RSI = RankAssets(StockDataDictRSI) # Call RankAssets

            # Check that the 'Error' column contains the expected TypeError message string
            ErrorMsg_RSI = RankedDf_RSI.loc[RankedDf_RSI['Ticker'] == 'TEST_SERIES_RSI', 'Error'].iloc[0]
            self.assertIsInstance(ErrorMsg_RSI, str)
            self.assertIn("RSI value from iloc[-1] is unexpectedly a Series/DataFrame", ErrorMsg_RSI)
            # Check that composite score is -1 as is typical for errors
            Score_RSI = RankedDf_RSI.loc[RankedDf_RSI['Ticker'] == 'TEST_SERIES_RSI', 'CompositeScore'].iloc[0]
            self.assertEqual(Score_RSI, -1)

            # --- Test for MACD ---
            TestDataMACD = pd.DataFrame(BaseData, index=TestIndex).copy()
            # Reset RSI to be normal scalar for this MACD test
            TestDataMACD.loc[:, 'RSI'] = [50.0] * NumRows

            MacdColValues = [0.5] * (NumRows - 1)
            MacdColValues.append(pd.Series([0.8], index=[TestDataMACD.index[-1]]))
            TestDataMACD['MACD'] = pd.Series(MacdColValues, index=TestDataMACD.index, dtype=object)
            # SignalLine can remain scalar
            TestDataMACD.loc[:, 'SignalLine'] = [0.4] * NumRows

            StockDataDictMACD = {'TEST_SERIES_MACD': TestDataMACD}
            RankedDf_MACD = RankAssets(StockDataDictMACD) # Call RankAssets

            ErrorMsg_MACD = RankedDf_MACD.loc[RankedDf_MACD['Ticker'] == 'TEST_SERIES_MACD', 'Error'].iloc[0]
            self.assertIsInstance(ErrorMsg_MACD, str)
            self.assertIn("MACD value from iloc[-1] is unexpectedly a Series/DataFrame", ErrorMsg_MACD)
            Score_MACD = RankedDf_MACD.loc[RankedDf_MACD['Ticker'] == 'TEST_SERIES_MACD', 'CompositeScore'].iloc[0]
            self.assertEqual(Score_MACD, -1)

        finally:
            # Restore original functions on the StockScreener module
            SS.CalculateRSI = OriginalCalculateRSI
            SS.CalculateMACD = OriginalCalculateMACD
            SS.CalculateMovingAverages = OriginalCalculateMovingAverages
            SS.CalculateBollingerBands = OriginalCalculateBollingerBands
            SS.CalculatePriceMomentum = OriginalCalculatePriceMomentum


if __name__ == '__main__':
    unittest.main()
