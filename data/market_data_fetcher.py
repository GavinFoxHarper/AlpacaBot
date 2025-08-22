#!/usr/bin/env python3
"""
Market Data Fetcher for LAEF Trading System
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class MarketDataFetcher:
    """Fetches market data from Alpaca API"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found")
            
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def fetch_stock_data(self, symbol: str, start_date: str = None, 
                        end_date: str = None, interval: str = '1Min') -> pd.DataFrame:
        """Fetch historical stock data"""
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            else:
                end_date = pd.to_datetime(end_date)
                
            if not start_date:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = pd.to_datetime(start_date)
            
            # Map interval to timeframe
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame.Minute,
                '15Min': TimeFrame.Minute,
                '1H': TimeFrame.Hour,
                '1D': TimeFrame.Day
            }
            
            timeframe = timeframe_map.get(interval, TimeFrame.Minute)
            
            # Create request parameters
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            # Fetch data
            bars = self.client.get_stock_bars(request_params)
            
            if symbol not in bars.data:
                self.logger.warning(f"No data returned for {symbol}")
                return None
                
            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'trade_count': getattr(bar, 'trade_count', 0),
                    'vwap': getattr(bar, 'vwap', None)
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
            
    def fetch_multiple_symbols(self, symbols: List[str], start_date: str = None,
                             end_date: str = None, interval: str = '1Min') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        result = {}
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, start_date, end_date, interval)
            if df is not None:
                result[symbol] = df
            
        return result
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            df = self.fetch_stock_data(symbol, interval='1Min')
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
        return None
        
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols"""
        result = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                result[symbol] = price
        return result

def fetch_stock_data(symbol: str, interval: str = '1Min', period: str = '1d') -> pd.DataFrame:
    """Convenience function for fetching data"""
    fetcher = MarketDataFetcher()
    return fetcher.fetch_stock_data(symbol, interval=interval)

def fetch_multiple_symbols(symbols: List[str], start_date: str = None,
                         end_date: str = None) -> Dict[str, pd.DataFrame]:
    """Convenience function for fetching multiple symbols"""
    fetcher = MarketDataFetcher()
    return fetcher.fetch_multiple_symbols(symbols, start_date, end_date)