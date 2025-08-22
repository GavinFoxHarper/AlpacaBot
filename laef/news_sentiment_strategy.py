"""
News Sentiment Trading Strategy for LAEF Trading Platform

Analyzes news sentiment to generate trading signals:
1. Real-time news sentiment analysis
2. Headline impact scoring
3. Volume-sentiment correlation
4. Sentiment trend analysis
5. Event classification and weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

class NewsEventType(Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "ma_deal"
    PRODUCT_LAUNCH = "product"
    REGULATORY = "regulatory"
    MARKET_MOVING = "market_moving"
    GENERAL = "general"

@dataclass
class NewsSignal:
    """News sentiment trading signal"""
    symbol: str
    sentiment_score: float   # -1 to 1
    volume_score: float     # News volume score
    impact_score: float     # Expected price impact
    confidence: float       # Signal confidence
    event_types: Set[NewsEventType]
    relevant_headlines: List[str]
    suggested_holding_period: int  # minutes
    timestamp: datetime

class NewsSentimentStrategy:
    """News sentiment-based trading strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Configuration
        self.sentiment_window = config.get('sentiment_window', 120)  # 2 hours
        self.min_headlines = config.get('min_headlines', 3)
        self.sentiment_threshold = config.get('sentiment_threshold', 0.3)
        self.impact_threshold = config.get('impact_threshold', 0.5)
        
        # Keyword dictionaries
        self.impact_keywords = self._initialize_impact_keywords()
        self.event_keywords = self._initialize_event_keywords()
        
        # State tracking
        self.sentiment_history = {}  # symbol -> list of sentiment scores
        self.volume_history = {}     # symbol -> list of news volumes
        self.recent_signals = {}     # symbol -> last signal
        
    def analyze_news(self, symbol: str, news_data: List[Dict],
                    market_data: Optional[pd.DataFrame] = None,
                    timestamp: datetime = None) -> Optional[NewsSignal]:
        """
        Analyze news sentiment and generate trading signals
        
        Args:
            symbol: Trading symbol
            news_data: List of news articles/headlines
            market_data: Optional price/volume data
            timestamp: Current timestamp
            
        Returns:
            NewsSignal if significant sentiment detected, None otherwise
        """
        try:
            timestamp = timestamp or datetime.now()
            
            if not news_data or len(news_data) < self.min_headlines:
                return None
                
            # 1. Calculate sentiment scores
            sentiment_scores = self._calculate_sentiment_scores(news_data)
            
            # 2. Calculate volume score
            volume_score = self._calculate_news_volume_score(news_data)
            
            # 3. Identify event types
            event_types = self._classify_events(news_data)
            
            # 4. Calculate impact score
            impact_score = self._calculate_impact_score(
                sentiment_scores, volume_score, event_types
            )
            
            # 5. Validate with market data if available
            if market_data is not None:
                impact_score = self._validate_with_market_data(
                    impact_score, market_data
                )
                
            # Generate signal if significant
            if abs(impact_score) > self.impact_threshold:
                # Calculate aggregate sentiment
                avg_sentiment = np.mean(sentiment_scores)
                
                # Get relevant headlines
                relevant_headlines = self._get_relevant_headlines(
                    news_data, sentiment_scores
                )
                
                # Calculate holding period based on event types
                holding_period = self._calculate_holding_period(event_types)
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    sentiment_scores, volume_score, event_types
                )
                
                signal = NewsSignal(
                    symbol=symbol,
                    sentiment_score=avg_sentiment,
                    volume_score=volume_score,
                    impact_score=impact_score,
                    confidence=confidence,
                    event_types=event_types,
                    relevant_headlines=relevant_headlines,
                    suggested_holding_period=holding_period,
                    timestamp=timestamp
                )
                
                # Update tracking
                self._update_signal_history(symbol, signal)
                
                return signal
                
            return None
            
        except Exception as e:
            self.logger.error(f"News analysis failed for {symbol}: {e}")
            return None
            
    def _calculate_sentiment_scores(self, news_data: List[Dict]) -> List[float]:
        """Calculate sentiment scores for news items"""
        try:
            scores = []
            
            for item in news_data:
                # Combine headline and content
                text = f"{item.get('headline', '')} {item.get('content', '')}"
                
                if not text.strip():
                    continue
                    
                # VADER sentiment
                vader_scores = self.vader.polarity_scores(text)
                vader_compound = vader_scores['compound']
                
                # TextBlob sentiment
                blob = TextBlob(text)
                textblob_sentiment = blob.sentiment.polarity
                
                # Combine scores (VADER weighted more heavily)
                combined_score = (vader_compound * 0.7 + textblob_sentiment * 0.3)
                
                scores.append(combined_score)
                
            return scores
            
        except Exception as e:
            self.logger.error(f"Sentiment calculation failed: {e}")
            return []
            
    def _calculate_news_volume_score(self, news_data: List[Dict]) -> float:
        """Calculate news volume impact score"""
        try:
            # Count articles by source quality
            source_weights = {
                'reuters': 1.0,
                'bloomberg': 1.0,
                'wsj': 1.0,
                'financial-times': 0.9,
                'cnbc': 0.8,
                'seeking-alpha': 0.7,
                'default': 0.5
            }
            
            weighted_count = 0
            for item in news_data:
                source = item.get('source', '').lower()
                weight = source_weights.get(source, source_weights['default'])
                weighted_count += weight
                
            # Calculate relative volume score
            avg_volume = 5  # Typical news volume
            volume_score = (weighted_count - avg_volume) / avg_volume
            
            return min(1.0, max(-1.0, volume_score))
            
        except Exception as e:
            self.logger.error(f"Volume score calculation failed: {e}")
            return 0.0
            
    def _classify_events(self, news_data: List[Dict]) -> Set[NewsEventType]:
        """Classify news events by type"""
        try:
            events = set()
            
            for item in news_data:
                text = f"{item.get('headline', '')} {item.get('content', '')}"
                text = text.lower()
                
                # Check each event type
                for event_type, keywords in self.event_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        events.add(event_type)
                        
            return events
            
        except Exception as e:
            self.logger.error(f"Event classification failed: {e}")
            return {NewsEventType.GENERAL}
            
    def _calculate_impact_score(self, sentiment_scores: List[float],
                              volume_score: float,
                              event_types: Set[NewsEventType]) -> float:
        """Calculate expected price impact score"""
        try:
            if not sentiment_scores:
                return 0.0
                
            # Base impact on sentiment
            avg_sentiment = np.mean(sentiment_scores)
            impact = avg_sentiment
            
            # Adjust for volume
            impact *= (1 + volume_score)
            
            # Adjust for event types
            event_multipliers = {
                NewsEventType.EARNINGS: 2.0,
                NewsEventType.MERGER_ACQUISITION: 1.8,
                NewsEventType.REGULATORY: 1.5,
                NewsEventType.PRODUCT_LAUNCH: 1.3,
                NewsEventType.MARKET_MOVING: 1.4,
                NewsEventType.GENERAL: 1.0
            }
            
            max_multiplier = max(
                event_multipliers[event_type]
                for event_type in event_types
            ) if event_types else 1.0
            
            impact *= max_multiplier
            
            # Normalize to [-1, 1]
            return min(1.0, max(-1.0, impact))
            
        except Exception as e:
            self.logger.error(f"Impact score calculation failed: {e}")
            return 0.0
            
    def _validate_with_market_data(self, impact_score: float,
                                 market_data: pd.DataFrame) -> float:
        """Validate and adjust impact score using market data"""
        try:
            if market_data is None or len(market_data) < 2:
                return impact_score
                
            # Check if market is already moving
            returns = market_data['close'].pct_change()
            volatility = returns.std()
            current_return = returns.iloc[-1]
            
            # Reduce impact if market already moving in same direction
            if (impact_score > 0 and current_return > volatility) or \
               (impact_score < 0 and current_return < -volatility):
                impact_score *= 0.8
                
            # Increase impact if moving against market
            if (impact_score > 0 and current_return < -volatility) or \
               (impact_score < 0 and current_return > volatility):
                impact_score *= 1.2
                
            return min(1.0, max(-1.0, impact_score))
            
        except Exception as e:
            self.logger.error(f"Market validation failed: {e}")
            return impact_score
            
    def _get_relevant_headlines(self, news_data: List[Dict],
                              sentiment_scores: List[float]) -> List[str]:
        """Get most relevant headlines by sentiment impact"""
        try:
            # Sort news by absolute sentiment
            news_sentiment = list(zip(news_data, sentiment_scores))
            sorted_news = sorted(
                news_sentiment,
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top headlines
            headlines = []
            for news, _ in sorted_news[:3]:  # Top 3 headlines
                headline = news.get('headline', '').strip()
                if headline:
                    headlines.append(headline)
                    
            return headlines
            
        except Exception as e:
            self.logger.error(f"Headline selection failed: {e}")
            return []
            
    def _calculate_holding_period(self, event_types: Set[NewsEventType]) -> int:
        """Calculate suggested holding period based on event types"""
        try:
            # Base holding periods (in minutes)
            holding_periods = {
                NewsEventType.EARNINGS: 1440,        # 24 hours
                NewsEventType.MERGER_ACQUISITION: 2880,  # 48 hours
                NewsEventType.REGULATORY: 1440,      # 24 hours
                NewsEventType.PRODUCT_LAUNCH: 720,   # 12 hours
                NewsEventType.MARKET_MOVING: 360,    # 6 hours
                NewsEventType.GENERAL: 240           # 4 hours
            }
            
            if not event_types:
                return holding_periods[NewsEventType.GENERAL]
                
            # Use longest holding period from event types
            return max(
                holding_periods[event_type]
                for event_type in event_types
            )
            
        except Exception as e:
            self.logger.error(f"Holding period calculation failed: {e}")
            return 240  # Default 4 hours
            
    def _calculate_confidence(self, sentiment_scores: List[float],
                            volume_score: float,
                            event_types: Set[NewsEventType]) -> float:
        """Calculate signal confidence score"""
        try:
            # Start with base confidence from sentiment consistency
            if len(sentiment_scores) >= 2:
                sentiment_std = np.std(sentiment_scores)
                base_confidence = 1.0 - min(1.0, sentiment_std)
            else:
                base_confidence = 0.5
                
            # Adjust for volume
            volume_factor = (volume_score + 2) / 3  # Scale to 0-1
            confidence = base_confidence * volume_factor
            
            # Adjust for event types
            if NewsEventType.EARNINGS in event_types or \
               NewsEventType.MERGER_ACQUISITION in event_types:
                confidence *= 1.2
                
            # Ensure bounds
            return min(0.95, max(0.1, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
            
    def _update_signal_history(self, symbol: str, signal: NewsSignal):
        """Update signal history tracking"""
        try:
            # Update sentiment history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append(
                (signal.timestamp, signal.sentiment_score)
            )
            
            # Update volume history
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            self.volume_history[symbol].append(
                (signal.timestamp, signal.volume_score)
            )
            
            # Update recent signals
            self.recent_signals[symbol] = signal
            
            # Clean up old history
            cutoff = signal.timestamp - timedelta(hours=24)
            self.sentiment_history[symbol] = [
                (ts, score) for ts, score in self.sentiment_history[symbol]
                if ts > cutoff
            ]
            self.volume_history[symbol] = [
                (ts, score) for ts, score in self.volume_history[symbol]
                if ts > cutoff
            ]
            
        except Exception as e:
            self.logger.error(f"Signal history update failed: {e}")
            
    def _initialize_impact_keywords(self) -> Dict[str, float]:
        """Initialize keywords with impact weights"""
        return {
            'beats estimates': 1.5,
            'misses estimates': -1.5,
            'raises guidance': 1.3,
            'lowers guidance': -1.3,
            'acquires': 1.2,
            'acquired by': 1.2,
            'announces layoffs': -1.1,
            'fda approval': 1.4,
            'clinical trial success': 1.3,
            'clinical trial failure': -1.3,
            'sec investigation': -1.4,
            'class action lawsuit': -1.2,
            'patent grant': 1.1,
            'patent litigation': -1.1,
            'new contract': 1.1,
            'contract termination': -1.1
        }
        
    def _initialize_event_keywords(self) -> Dict[NewsEventType, List[str]]:
        """Initialize event classification keywords"""
        return {
            NewsEventType.EARNINGS: [
                'earnings', 'revenue', 'profit', 'eps', 'quarter',
                'guidance', 'outlook', 'forecast'
            ],
            NewsEventType.MERGER_ACQUISITION: [
                'merger', 'acquisition', 'acquire', 'takeover', 'bid',
                'deal', 'buy out', 'purchased'
            ],
            NewsEventType.PRODUCT_LAUNCH: [
                'launch', 'release', 'announce', 'unveil', 'introduce',
                'new product', 'new service'
            ],
            NewsEventType.REGULATORY: [
                'fda', 'sec', 'regulatory', 'approval', 'investigation',
                'patent', 'lawsuit', 'legal'
            ],
            NewsEventType.MARKET_MOVING: [
                'upgrade', 'downgrade', 'price target', 'rating',
                'analyst', 'recommendation'
            ]
        }