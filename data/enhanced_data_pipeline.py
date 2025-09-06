"""
Enhanced Data Pipeline for AlpacaBot Training
Comprehensive historical data fetching, validation, cleaning, and regime labeling
Author: AlpacaBot Training System
Date: 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDataPipeline:
    """
    Comprehensive data pipeline for fetching, validating, cleaning, and preparing
    balanced training data with market regime labels.
    """
    
    def __init__(self, 
                 start_date: str = "2020-01-01",
                 end_date: str = None,
                 cache_dir: str = "data/cache"):
        """
        Initialize enhanced data pipeline.
        
        Args:
            start_date: Start date for historical data (default: 2020-01-01 for COVID crash)
            end_date: End date for historical data (default: today)
            cache_dir: Directory for caching downloaded data
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Market event dates for regime labeling
        self.market_events = {
            'covid_crash_start': '2020-02-20',
            'covid_crash_bottom': '2020-03-23',
            'recovery_start': '2020-03-24',
            'tech_bubble_peak': '2021-11-19',
            'rate_hike_start': '2022-03-16',
            'bear_market_2022': '2022-06-13',
            'recovery_2023': '2023-01-01'
        }
        
        # Technical indicators for regime detection
        self.regime_indicators = {
            'sma_20': 20,
            'sma_50': 50,
            'sma_200': 200,
            'rsi_period': 14,
            'atr_period': 14,
            'bb_period': 20,
            'volume_ma': 20
        }
        
        logger.info(f"Enhanced Data Pipeline initialized: {start_date} to {end_date}")
    
    def fetch_historical_data(self, 
                            symbols: List[str], 
                            interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive historical data for multiple symbols with validation.
        
        Args:
            symbols: List of stock symbols to fetch
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dictionary of symbol: DataFrame with historical data
        """
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        data = {}
        failed_symbols = []
        
        def fetch_single_symbol(symbol):
            try:
                cache_file = os.path.join(self.cache_dir, f"{symbol}_{self.start_date}_{self.end_date}.csv")
                
                # Check cache first
                if os.path.exists(cache_file):
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    logger.info(f"Loaded {symbol} from cache")
                    return symbol, df
                
                # Fetch from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data retrieved for {symbol}")
                    return symbol, None
                
                # Add symbol column
                df['Symbol'] = symbol
                
                # Save to cache
                df.to_csv(cache_file)
                logger.info(f"Fetched and cached {symbol}: {len(df)} records")
                return symbol, df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                return symbol, None
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_single_symbol, symbol): symbol 
                      for symbol in symbols}
            
            for future in as_completed(futures):
                symbol, df = future.result()
                if df is not None and not df.empty:
                    data[symbol] = df
                else:
                    failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(data)} symbols")
        return data
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data validation and cleaning.
        
        Args:
            df: Raw DataFrame to validate and clean
            
        Returns:
            Cleaned DataFrame
        """
        initial_len = len(df)
        logger.info(f"Validating and cleaning data: {initial_len} records")
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Handle missing values
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col in df.columns:
                # Forward fill for small gaps (up to 2 days)
                df[col] = df[col].fillna(method='ffill', limit=2)
                # Backward fill for remaining
                df[col] = df[col].fillna(method='bfill', limit=1)
        
        # Remove rows with any remaining NaN in required columns
        df = df.dropna(subset=required_columns)
        
        # Validate price relationships
        invalid_prices = (
            (df['High'] < df['Low']) |
            (df['Close'] > df['High']) |
            (df['Close'] < df['Low']) |
            (df['Open'] > df['High']) |
            (df['Open'] < df['Low'])
        )
        
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} rows with invalid price relationships")
            df = df[~invalid_prices]
        
        # Handle stock splits and dividends
        df = self._adjust_for_splits(df)
        
        # Remove outliers (prices that change more than 50% in a day)
        if len(df) > 1:
            returns = df['Close'].pct_change()
            outliers = abs(returns) > 0.5
            if outliers.any():
                logger.warning(f"Removing {outliers.sum()} outlier records")
                df = df[~outliers]
        
        # Ensure chronological order
        df = df.sort_index()
        
        # Check for gaps in dates
        date_gaps = self._detect_date_gaps(df)
        if date_gaps:
            logger.warning(f"Detected {len(date_gaps)} date gaps in data")
        
        final_len = len(df)
        logger.info(f"Data cleaned: {initial_len} -> {final_len} records")
        
        return df
    
    def _adjust_for_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and adjust for stock splits.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Adjusted DataFrame
        """
        # Detect potential splits (price drops more than 40% with volume spike)
        if len(df) > 1:
            price_change = df['Close'].pct_change()
            volume_change = df['Volume'].pct_change()
            
            potential_splits = (price_change < -0.4) & (volume_change > 1.0)
            
            if potential_splits.any():
                split_dates = df[potential_splits].index
                logger.info(f"Detected potential splits on: {split_dates.tolist()}")
                
                # Adjust prices before split
                for split_date in split_dates:
                    split_ratio = abs(price_change[split_date])
                    adjustment_factor = 1 / (1 - split_ratio)
                    
                    # Adjust all prices before split
                    mask = df.index < split_date
                    df.loc[mask, ['Open', 'High', 'Low', 'Close']] *= adjustment_factor
                    df.loc[mask, 'Volume'] /= adjustment_factor
        
        return df
    
    def _detect_date_gaps(self, df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Detect gaps in trading dates.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of gap tuples (start, end)
        """
        gaps = []
        
        if len(df) > 1:
            # Expected trading days (excluding weekends)
            date_range = pd.bdate_range(start=df.index[0], end=df.index[-1])
            missing_dates = date_range.difference(df.index)
            
            if len(missing_dates) > 0:
                # Group consecutive missing dates
                missing_dates = missing_dates.sort_values()
                gap_start = missing_dates[0]
                
                for i in range(1, len(missing_dates)):
                    if (missing_dates[i] - missing_dates[i-1]).days > 1:
                        gaps.append((gap_start, missing_dates[i-1]))
                        gap_start = missing_dates[i]
                
                gaps.append((gap_start, missing_dates[-1]))
        
        return gaps
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for regime detection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Calculating technical indicators")
        
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # Bollinger Bands
        bb_sma = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = bb_sma + (bb_std * 2)
        df['BB_Lower'] = bb_sma - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (Average True Range)
        df['ATR'] = self._calculate_atr(df, 14)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_10'] = df['Close'].pct_change(10)
        df['Momentum_20'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
        df['Volatility_50'] = df['Close'].pct_change().rolling(window=50, min_periods=1).std()
        
        # Support and Resistance levels
        df['Resistance'] = df['High'].rolling(window=20, min_periods=1).max()
        df['Support'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['SR_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of prices
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            ATR values
        """
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def detect_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and label market regimes using multiple indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with regime labels
        """
        logger.info("Detecting market regimes")
        
        # Initialize regime column
        df['Market_Regime'] = 'normal'
        df['Regime_Confidence'] = 0.5
        
        # Trend detection
        df['Trend'] = 'sideways'
        bull_condition = (
            (df['SMA_20'] > df['SMA_50']) & 
            (df['SMA_50'] > df['SMA_200']) &
            (df['Close'] > df['SMA_20'])
        )
        bear_condition = (
            (df['SMA_20'] < df['SMA_50']) & 
            (df['SMA_50'] < df['SMA_200']) &
            (df['Close'] < df['SMA_20'])
        )
        
        df.loc[bull_condition, 'Trend'] = 'bull'
        df.loc[bear_condition, 'Trend'] = 'bear'
        
        # Volatility regimes
        volatility_percentile = df['Volatility_20'].rolling(window=252, min_periods=20).rank(pct=True)
        high_vol = volatility_percentile > 0.8
        low_vol = volatility_percentile < 0.2
        
        # Market regimes based on multiple factors
        # Bull market
        bull_market = (
            (df['Trend'] == 'bull') &
            (df['RSI'] > 50) &
            (df['MACD'] > 0)
        )
        df.loc[bull_market, 'Market_Regime'] = 'bull'
        df.loc[bull_market, 'Regime_Confidence'] = 0.8
        
        # Bear market
        bear_market = (
            (df['Trend'] == 'bear') &
            (df['RSI'] < 50) &
            (df['MACD'] < 0)
        )
        df.loc[bear_market, 'Market_Regime'] = 'bear'
        df.loc[bear_market, 'Regime_Confidence'] = 0.8
        
        # High volatility regime
        high_vol_regime = high_vol & (df['ATR'] > df['ATR'].rolling(window=50, min_periods=1).mean())
        df.loc[high_vol_regime, 'Market_Regime'] = 'high_volatility'
        df.loc[high_vol_regime, 'Regime_Confidence'] = 0.7
        
        # Low volatility regime
        low_vol_regime = low_vol & (df['ATR'] < df['ATR'].rolling(window=50, min_periods=1).mean())
        df.loc[low_vol_regime, 'Market_Regime'] = 'low_volatility'
        df.loc[low_vol_regime, 'Regime_Confidence'] = 0.7
        
        # Crash detection (rapid decline with high volume)
        crash_condition = (
            (df['Momentum_5'] < -0.1) &
            (df['Volume_Ratio'] > 2) &
            (df['RSI'] < 30)
        )
        df.loc[crash_condition, 'Market_Regime'] = 'crash'
        df.loc[crash_condition, 'Regime_Confidence'] = 0.9
        
        # Recovery detection (rapid rise after oversold)
        recovery_condition = (
            (df['Momentum_5'] > 0.1) &
            (df['RSI'].shift(5) < 30) &
            (df['Volume_Ratio'] > 1.5)
        )
        df.loc[recovery_condition, 'Market_Regime'] = 'recovery'
        df.loc[recovery_condition, 'Regime_Confidence'] = 0.85
        
        # Add regime duration
        df['Regime_Duration'] = 0
        current_regime = None
        duration = 0
        
        for i in range(len(df)):
            if df.iloc[i]['Market_Regime'] != current_regime:
                current_regime = df.iloc[i]['Market_Regime']
                duration = 1
            else:
                duration += 1
            df.iloc[i, df.columns.get_loc('Regime_Duration')] = duration
        
        logger.info(f"Market regime distribution:\n{df['Market_Regime'].value_counts()}")
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate balanced trading signals (buy/sell/hold) based on technical indicators and regimes.
        
        Args:
            df: DataFrame with indicators and regimes
            
        Returns:
            DataFrame with trading signals
        """
        logger.info("Generating balanced trading signals")
        
        # Initialize signals
        df['Signal'] = 0  # 0: hold, 1: buy, -1: sell
        df['Signal_Strength'] = 0.0
        df['Signal_Reason'] = ''
        
        # Buy signals
        buy_conditions = []
        
        # Strong buy: Golden cross with bull market
        buy_conditions.append({
            'condition': (
                (df['SMA_50'] > df['SMA_200']) &
                (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1)) &
                (df['Market_Regime'] == 'bull')
            ),
            'strength': 0.9,
            'reason': 'Golden cross in bull market'
        })
        
        # Buy: Oversold bounce
        buy_conditions.append({
            'condition': (
                (df['RSI'] < 30) &
                (df['RSI'] > df['RSI'].shift(1)) &
                (df['Close'] > df['BB_Lower'])
            ),
            'strength': 0.7,
            'reason': 'Oversold bounce'
        })
        
        # Buy: MACD bullish crossover
        buy_conditions.append({
            'condition': (
                (df['MACD'] > df['MACD_Signal']) &
                (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)) &
                (df['Trend'] != 'bear')
            ),
            'strength': 0.6,
            'reason': 'MACD bullish crossover'
        })
        
        # Buy: Support bounce
        buy_conditions.append({
            'condition': (
                (df['Close'] > df['Support']) &
                (df['Low'] <= df['Support'] * 1.01) &
                (df['RSI'] < 50)
            ),
            'strength': 0.65,
            'reason': 'Support bounce'
        })
        
        # Sell signals
        sell_conditions = []
        
        # Strong sell: Death cross with bear market
        sell_conditions.append({
            'condition': (
                (df['SMA_50'] < df['SMA_200']) &
                (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1)) &
                (df['Market_Regime'] == 'bear')
            ),
            'strength': 0.9,
            'reason': 'Death cross in bear market'
        })
        
        # Sell: Overbought reversal
        sell_conditions.append({
            'condition': (
                (df['RSI'] > 70) &
                (df['RSI'] < df['RSI'].shift(1)) &
                (df['Close'] < df['BB_Upper'])
            ),
            'strength': 0.7,
            'reason': 'Overbought reversal'
        })
        
        # Sell: MACD bearish crossover
        sell_conditions.append({
            'condition': (
                (df['MACD'] < df['MACD_Signal']) &
                (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)) &
                (df['Trend'] != 'bull')
            ),
            'strength': 0.6,
            'reason': 'MACD bearish crossover'
        })
        
        # Sell: Resistance rejection
        sell_conditions.append({
            'condition': (
                (df['Close'] < df['Resistance']) &
                (df['High'] >= df['Resistance'] * 0.99) &
                (df['RSI'] > 50)
            ),
            'strength': 0.65,
            'reason': 'Resistance rejection'
        })
        
        # Sell: Breakdown below support
        sell_conditions.append({
            'condition': (
                (df['Close'] < df['Support']) &
                (df['Volume_Ratio'] > 1.5)
            ),
            'strength': 0.75,
            'reason': 'Support breakdown'
        })
        
        # Apply buy conditions
        for buy_cond in buy_conditions:
            mask = buy_cond['condition']
            df.loc[mask, 'Signal'] = 1
            df.loc[mask, 'Signal_Strength'] = buy_cond['strength']
            df.loc[mask, 'Signal_Reason'] = buy_cond['reason']
        
        # Apply sell conditions (can override buy if stronger)
        for sell_cond in sell_conditions:
            mask = sell_cond['condition']
            # Only override if sell signal is stronger or no existing signal
            override_mask = mask & ((df['Signal'] == 0) | (sell_cond['strength'] > df['Signal_Strength']))
            df.loc[override_mask, 'Signal'] = -1
            df.loc[override_mask, 'Signal_Strength'] = sell_cond['strength']
            df.loc[override_mask, 'Signal_Reason'] = sell_cond['reason']
        
        # Add regime-based adjustments
        # Increase sell signals during crash
        crash_mask = (df['Market_Regime'] == 'crash') & (df['Signal'] == 0)
        df.loc[crash_mask, 'Signal'] = -1
        df.loc[crash_mask, 'Signal_Strength'] = 0.8
        df.loc[crash_mask, 'Signal_Reason'] = 'Crash regime sell'
        
        # Increase buy signals during recovery
        recovery_mask = (df['Market_Regime'] == 'recovery') & (df['Signal'] == 0) & (df['RSI'] < 50)
        df.loc[recovery_mask, 'Signal'] = 1
        df.loc[recovery_mask, 'Signal_Strength'] = 0.7
        df.loc[recovery_mask, 'Signal_Reason'] = 'Recovery regime buy'
        
        # Calculate signal distribution
        signal_dist = df['Signal'].value_counts()
        logger.info(f"Signal distribution:\n{signal_dist}")
        
        return df
    
    def create_balanced_dataset(self, df: pd.DataFrame, target_balance: Dict[int, float] = None) -> pd.DataFrame:
        """
        Create a balanced dataset with equal representation of buy/sell/hold signals.
        
        Args:
            df: DataFrame with signals
            target_balance: Target distribution {-1: 0.33, 0: 0.34, 1: 0.33}
            
        Returns:
            Balanced DataFrame
        """
        if target_balance is None:
            target_balance = {-1: 0.33, 0: 0.34, 1: 0.33}
        
        logger.info("Creating balanced dataset")
        
        # Get current distribution
        signal_counts = df['Signal'].value_counts()
        total_samples = len(df)
        
        # Calculate target counts
        target_counts = {
            signal: int(total_samples * ratio) 
            for signal, ratio in target_balance.items()
        }
        
        # Separate by signal type
        buy_df = df[df['Signal'] == 1]
        sell_df = df[df['Signal'] == -1]
        hold_df = df[df['Signal'] == 0]
        
        # Augment underrepresented classes
        balanced_dfs = []
        
        # Process buy signals
        if len(buy_df) < target_counts[1]:
            # Augment buy signals
            augmented_buy = self._augment_signals(buy_df, target_counts[1])
            balanced_dfs.append(augmented_buy)
        else:
            # Sample from buy signals
            balanced_dfs.append(buy_df.sample(n=target_counts[1], replace=False))
        
        # Process sell signals
        if len(sell_df) < target_counts[-1]:
            # Augment sell signals
            augmented_sell = self._augment_signals(sell_df, target_counts[-1])
            balanced_dfs.append(augmented_sell)
        else:
            # Sample from sell signals
            balanced_dfs.append(sell_df.sample(n=target_counts[-1], replace=False))
        
        # Process hold signals
        if len(hold_df) < target_counts[0]:
            # Augment hold signals
            augmented_hold = self._augment_signals(hold_df, target_counts[0])
            balanced_dfs.append(augmented_hold)
        else:
            # Sample from hold signals
            balanced_dfs.append(hold_df.sample(n=target_counts[0], replace=False))
        
        # Combine and shuffle
        balanced_df = pd.concat(balanced_dfs, ignore_index=False)
        balanced_df = balanced_df.sort_index()
        
        # Verify balance
        final_dist = balanced_df['Signal'].value_counts(normalize=True)
        logger.info(f"Final signal distribution:\n{final_dist}")
        
        return balanced_df
    
    def _augment_signals(self, df: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        Augment underrepresented signal classes using synthetic examples.
        
        Args:
            df: DataFrame with signals to augment
            target_count: Target number of samples
            
        Returns:
            Augmented DataFrame
        """
        current_count = len(df)
        if current_count == 0:
            logger.warning("Cannot augment empty DataFrame")
            return df
        
        needed = target_count - current_count
        logger.info(f"Augmenting {current_count} samples to {target_count} (need {needed} more)")
        
        augmented_samples = []
        
        while len(augmented_samples) < needed:
            # Sample a random row
            base_row = df.sample(n=1).iloc[0].copy()
            
            # Add small noise to numerical features
            noise_features = ['RSI', 'MACD', 'ATR', 'Volatility_20', 'Momentum_5']
            for feature in noise_features:
                if feature in base_row.index:
                    noise = np.random.normal(0, 0.05 * abs(base_row[feature]))
                    base_row[feature] += noise
            
            augmented_samples.append(base_row)
        
        # Create augmented DataFrame
        augmented_df = pd.DataFrame(augmented_samples)
        
        # Combine original and augmented
        result = pd.concat([df, augmented_df], ignore_index=False)
        
        return result.iloc[:target_count]
    
    def prepare_training_data(self, 
                            symbols: List[str],
                            feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare complete training data with features and labels.
        
        Args:
            symbols: List of symbols to process
            feature_columns: List of feature columns to use
            
        Returns:
            Tuple of (X_features, y_labels, metadata_df)
        """
        logger.info(f"Preparing training data for {len(symbols)} symbols")
        
        # Fetch historical data
        historical_data = self.fetch_historical_data(symbols)
        
        if not historical_data:
            raise ValueError("No historical data fetched")
        
        all_processed_data = []
        
        for symbol, df in historical_data.items():
            # Validate and clean
            df = self.validate_and_clean_data(df)
            
            # Calculate indicators
            df = self.calculate_technical_indicators(df)
            
            # Detect regimes
            df = self.detect_market_regimes(df)
            
            # Generate signals
            df = self.generate_trading_signals(df)
            
            # Create balanced dataset
            df = self.create_balanced_dataset(df)
            
            all_processed_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_processed_data, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        # Define feature columns if not provided
        if feature_columns is None:
            feature_columns = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Position', 'ATR', 'Volume_Ratio',
                'Momentum_5', 'Momentum_10', 'Momentum_20',
                'Volatility_20', 'Volatility_50', 'SR_Position',
                'Regime_Confidence', 'Regime_Duration'
            ]
        
        # Prepare features and labels
        X = combined_df[feature_columns].values
        y = combined_df['Signal'].values
        
        # Create metadata DataFrame
        metadata = combined_df[['Symbol', 'Market_Regime', 'Signal_Reason', 'Signal_Strength']]
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Label distribution: {np.unique(y, return_counts=True)}")
        
        return X, y, metadata
    
    def save_processed_data(self, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           metadata: pd.DataFrame,
                           output_dir: str = "data/processed"):
        """
        Save processed training data to disk.
        
        Args:
            X: Feature array
            y: Label array
            metadata: Metadata DataFrame
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(output_dir, 'X_features.npy'), X)
        np.save(os.path.join(output_dir, 'y_labels.npy'), y)
        
        # Save metadata
        metadata.to_csv(os.path.join(output_dir, 'metadata.csv'))
        
        # Save configuration
        config = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'label_distribution': {
                str(k): int(v) for k, v in 
                zip(*np.unique(y, return_counts=True))
            }
        }
        
        with open(os.path.join(output_dir, 'data_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Processed data saved to {output_dir}")


def main():
    """
    Main function to run the enhanced data pipeline.
    """
    # Initialize pipeline
    pipeline = EnhancedDataPipeline(
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    
    # Define symbols to process
    symbols = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE',
        'CRM', 'NFLX', 'PFE', 'TMO', 'CSCO', 'PEP', 'INTC', 'CMCSA', 'VZ', 'T'
    ]
    
    try:
        # Prepare training data
        X, y, metadata = pipeline.prepare_training_data(symbols)
        
        # Save processed data
        pipeline.save_processed_data(X, y, metadata)
        
        print("\n" + "="*60)
        print("Enhanced Data Pipeline Execution Complete!")
        print("="*60)
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Label distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}[label]
            percentage = (count / len(y)) * 100
            print(f"  {label_name}: {count} ({percentage:.1f}%)")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()