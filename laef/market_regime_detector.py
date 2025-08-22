"""
Market Regime Detection System for LAEF Trading Platform

Detects market regimes using multiple indicators:
1. Trend Analysis (ADX, Moving Averages)
2. Volatility Analysis (ATR, VIX)
3. Volume Analysis
4. Price Pattern Analysis
5. Market Structure Analysis (Support/Resistance)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from .technical_indicators import calculate_adx, calculate_atr
from multi_strategy_orchestrator import MarketRegime

@dataclass
class MarketState:
    """Current market state information"""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_trend: float
    support_level: float
    resistance_level: float
    confidence: float
    timestamp: datetime

class RegimeDetector:
    """Market regime detection system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.trend_period = config.get('trend_period', 20)
        self.vol_period = config.get('volatility_period', 14)
        self.volume_period = config.get('volume_period', 20)
        
        # Thresholds
        self.trend_threshold = config.get('trend_threshold', 25)
        self.vol_threshold = config.get('volatility_threshold', 0.015)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        
        # Regime transition smoothing
        self.regime_history = []
        self.regime_window = config.get('regime_window', 5)
        
        # Market state tracking
        self.current_state: Optional[MarketState] = None
        self.last_update = None
        
    def detect_regime(self, data: pd.DataFrame, timestamp: datetime = None) -> MarketState:
        """
        Detect current market regime using multiple indicators
        
        Args:
            data: DataFrame with OHLCV data
            timestamp: Current timestamp
            
        Returns:
            MarketState object with regime and metrics
        """
        try:
            timestamp = timestamp or datetime.now()
            
            if len(data) < self.trend_period:
                return self._create_default_state(timestamp)
            
            # 1. Trend Analysis
            trend_metrics = self._analyze_trend(data)
            trend_strength = trend_metrics['strength']
            trend_direction = trend_metrics['direction']
            
            # 2. Volatility Analysis
            volatility = self._analyze_volatility(data)
            
            # 3. Volume Analysis
            volume_trend = self._analyze_volume(data)
            
            # 4. Support/Resistance
            support, resistance = self._find_support_resistance(data)
            
            # 5. Determine Market Regime
            regime, confidence = self._determine_regime(
                trend_strength, trend_direction, volatility, volume_trend,
                support, resistance, data
            )
            
            # Create market state
            state = MarketState(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_trend=volume_trend,
                support_level=support,
                resistance_level=resistance,
                confidence=confidence,
                timestamp=timestamp
            )
            
            # Update tracking
            self.current_state = state
            self.last_update = timestamp
            self.regime_history.append(regime)
            if len(self.regime_history) > self.regime_window:
                self.regime_history.pop(0)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return self._create_default_state(timestamp)
            
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze trend strength and direction"""
        try:
            # Calculate ADX for trend strength
            adx = calculate_adx(data, period=self.trend_period)
            current_adx = adx[-1] if len(adx) > 0 else 0
            
            # Calculate moving averages
            ma20 = data['close'].rolling(20).mean()
            ma50 = data['close'].rolling(50).mean()
            
            # Determine trend direction
            current_price = data['close'].iloc[-1]
            direction = 1 if current_price > ma20.iloc[-1] else -1
            
            # Calculate trend strength (0 to 1)
            strength = min(1.0, current_adx / 100.0)
            
            return {
                'strength': strength,
                'direction': direction,
                'adx': current_adx,
                'ma20': ma20.iloc[-1],
                'ma50': ma50.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {'strength': 0, 'direction': 0}
            
    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """Analyze market volatility"""
        try:
            # Calculate ATR-based volatility
            atr = calculate_atr(data, period=self.vol_period)
            current_atr = atr[-1] if len(atr) > 0 else 0
            current_price = data['close'].iloc[-1]
            
            # Normalize ATR to percentage
            volatility = current_atr / current_price
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {e}")
            return 0.02  # Default moderate volatility
            
    def _analyze_volume(self, data: pd.DataFrame) -> float:
        """Analyze volume trend"""
        try:
            if 'volume' not in data.columns:
                return 1.0
                
            recent_volume = data['volume'].iloc[-self.volume_period:]
            volume_sma = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            
            # Calculate volume trend (-1 to 1)
            if volume_sma > 0:
                volume_trend = (current_volume / volume_sma) - 1
            else:
                volume_trend = 0
                
            return volume_trend
            
        except Exception as e:
            self.logger.error(f"Volume analysis failed: {e}")
            return 0.0
            
    def _find_support_resistance(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Find key support and resistance levels"""
        try:
            # Use recent high/low points
            recent_data = data.tail(50)
            
            # Simple support/resistance based on recent pivots
            resistance = recent_data['high'].max()
            support = recent_data['low'].min()
            
            current_price = data['close'].iloc[-1]
            
            # Adjust levels based on price location
            if current_price > resistance:
                support = resistance
                resistance = resistance * 1.02
            elif current_price < support:
                resistance = support
                support = support * 0.98
                
            return support, resistance
            
        except Exception as e:
            self.logger.error(f"Support/Resistance detection failed: {e}")
            current_price = data['close'].iloc[-1]
            return current_price * 0.98, current_price * 1.02
            
    def _determine_regime(self, trend_strength: float, trend_direction: int,
                         volatility: float, volume_trend: float,
                         support: float, resistance: float,
                         data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Determine market regime from indicators"""
        try:
            scores = {
                MarketRegime.TRENDING_UP: 0,
                MarketRegime.TRENDING_DOWN: 0,
                MarketRegime.RANGING: 0,
                MarketRegime.VOLATILE: 0,
                MarketRegime.LOW_VOLATILITY: 0
            }
            
            # Trend scoring
            if trend_strength > 0.6:  # Strong trend
                if trend_direction > 0:
                    scores[MarketRegime.TRENDING_UP] += 2
                else:
                    scores[MarketRegime.TRENDING_DOWN] += 2
            elif trend_strength < 0.2:  # Weak trend
                scores[MarketRegime.RANGING] += 1
                
            # Volatility scoring
            if volatility > self.vol_threshold * 1.5:  # High volatility
                scores[MarketRegime.VOLATILE] += 2
            elif volatility < self.vol_threshold * 0.5:  # Low volatility
                scores[MarketRegime.LOW_VOLATILITY] += 2
                
            # Volume scoring
            if abs(volume_trend) > self.volume_threshold:
                if trend_direction > 0:
                    scores[MarketRegime.TRENDING_UP] += 1
                else:
                    scores[MarketRegime.TRENDING_DOWN] += 1
                    
            # Range analysis
            price_range = (resistance - support) / support
            if price_range < 0.02:  # Tight range
                scores[MarketRegime.RANGING] += 1
                
            # Find regime with highest score
            regime = max(scores.items(), key=lambda x: x[1])[0]
            
            # Calculate confidence (0 to 1)
            max_score = max(scores.values())
            total_score = sum(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
            
            # Smooth regime transitions
            if self.regime_history:
                prev_regime = self.regime_history[-1]
                if prev_regime != regime and confidence < 0.7:
                    regime = prev_regime  # Stick to previous regime if not confident
                    
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"Regime determination failed: {e}")
            return MarketRegime.RANGING, 0.5
            
    def _create_default_state(self, timestamp: datetime) -> MarketState:
        """Create default market state when analysis fails"""
        return MarketState(
            regime=MarketRegime.RANGING,
            volatility=0.02,
            trend_strength=0.0,
            volume_trend=0.0,
            support_level=0.0,
            resistance_level=0.0,
            confidence=0.5,
            timestamp=timestamp
        )