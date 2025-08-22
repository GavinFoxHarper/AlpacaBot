"""
News Integration System for LAEF Trading Platform
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import threading
import time
import json
from dataclasses import dataclass
from textblob import TextBlob
import yfinance as yf
from newsapi import NewsApiClient
import feedparser
import re

@dataclass
class NewsArticle:
    headline: str
    content: str
    source: str
    timestamp: datetime
    symbol: str
    sentiment_score: float
    relevance_score: float
    market_impact: str  # 'bullish', 'bearish', 'neutral'
    urgency: str       # 'high', 'medium', 'low'

class NewsIntegrationEngine:
    """News integration engine with sentiment analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API configurations
        self.news_api_key = config.get('NEWS_API_KEY')
        self.alpha_vantage_key = config.get('ALPHA_VANTAGE_API_KEY')
        
        # News sources
        self.news_sources = config.get('news_sources', [
            'reuters', 'bloomberg', 'cnbc', 'marketwatch', 'yahoo-finance'
        ])
        
        # Sentiment analysis parameters
        self.sentiment_threshold_bullish = config.get('sentiment_threshold_bullish', 0.1)
        self.sentiment_threshold_bearish = config.get('sentiment_threshold_bearish', -0.1)
        
        # Market impact parameters
        self.high_impact_keywords = config.get('high_impact_keywords', [
            'earnings', 'acquisition', 'merger', 'bankruptcy', 'lawsuit',
            'fda approval', 'clinical trial', 'guidance', 'revenue',
            'partnership', 'contract', 'investigation', 'recall'
        ])
        
        # Initialize news clients
        self.news_client = None
        if self.news_api_key:
            try:
                self.news_client = NewsApiClient(api_key=self.news_api_key)
            except Exception as e:
                self.logger.warning(f"Failed to initialize NewsAPI client: {e}")
                
    def integrate_with_trading_decision(self, symbol: str, base_confidence: float, q_value: float) -> Tuple[float, float]:
        """Integrate news sentiment with trading decision"""
        try:
            # Get news impact
            news_impact = self._get_news_impact(symbol)
            
            # Adjust Q-value based on news sentiment
            if abs(news_impact) > 0.1:  # Significant news impact
                news_weight = self.config.get('news_weight', 0.2)  # Default 20% weight
                adjusted_q_value = q_value + (news_impact * news_weight)
            else:
                adjusted_q_value = q_value
            
            # Ensure values stay in valid ranges
            adjusted_q_value = max(0.0, min(1.0, adjusted_q_value))
            
            self.logger.debug(f"News integration for {symbol}: "
                          f"original_q={q_value:.3f}, adjusted_q={adjusted_q_value:.3f}, "
                          f"news_impact={news_impact:.3f}")
            
            return base_confidence, adjusted_q_value
            
        except Exception as e:
            self.logger.error(f"Failed to integrate news with trading decision for {symbol}: {e}")
            return base_confidence, q_value
            
    def _get_news_impact(self, symbol: str) -> float:
        """Get news impact score for a symbol"""
        # Stub implementation 
        return 0.0  # No impact