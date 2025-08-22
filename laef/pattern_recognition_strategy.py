"""
Pattern Recognition Strategy for LAEF Trading Platform

Detects and trades based on chart patterns:
1. Classic Chart Patterns (Head & Shoulders, Double Top/Bottom, etc.)
2. Candlestick Patterns (Engulfing, Doji, etc.)
3. Harmonic Patterns (Gartley, Butterfly, etc.)
4. Support/Resistance Breakouts
5. Trendline Breaks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import talib

# Pattern Classifications
class PatternType(Enum):
    REVERSAL = "reversal"      # Trend reversal patterns
    CONTINUATION = "continuation"  # Trend continuation
    BREAKOUT = "breakout"      # Support/resistance breaks
    HARMONIC = "harmonic"      # Harmonic patterns
    CANDLESTICK = "candlestick"  # Candlestick patterns

@dataclass
class PatternSignal:
    """Pattern recognition signal"""
    pattern_type: PatternType
    pattern_name: str
    direction: int       # 1=bullish, -1=bearish
    strength: float     # Pattern strength/reliability
    completion: float   # Pattern completion percentage
    target_price: float
    stop_loss: float
    timeframe: str
    confidence: float

class PatternRecognitionStrategy:
    """Pattern recognition trading strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection settings
        self.min_pattern_bars = config.get('min_pattern_bars', 5)
        self.max_pattern_bars = config.get('max_pattern_bars', 50)
        self.completion_threshold = config.get('completion_threshold', 0.7)
        
        # Timeframe weights
        self.timeframe_weights = {
            '5m': 0.1,
            '15m': 0.2,
            '1h': 0.3,
            '4h': 0.4
        }
        
        # Initialize TA-Lib patterns
        self.candlestick_patterns = self._initialize_candlestick_patterns()
        
        # Pattern tracking
        self.active_patterns = {}  # symbol -> list of active patterns
        self.completed_patterns = []  # historical pattern record
        
    def analyze_patterns(self, symbol: str, data: Dict[str, pd.DataFrame],
                        timestamp: datetime = None) -> List[PatternSignal]:
        """
        Analyze price data for patterns across timeframes
        
        Args:
            symbol: Trading symbol
            data: Dict of DataFrames for each timeframe
            timestamp: Current timestamp
            
        Returns:
            List of detected pattern signals
        """
        try:
            timestamp = timestamp or datetime.now()
            all_patterns = []
            
            # Analyze each timeframe
            for timeframe, df in data.items():
                if len(df) >= self.min_pattern_bars:
                    # 1. Detect classic chart patterns
                    classic_patterns = self._detect_classic_patterns(df, timeframe)
                    all_patterns.extend(classic_patterns)
                    
                    # 2. Detect candlestick patterns
                    candle_patterns = self._detect_candlestick_patterns(df, timeframe)
                    all_patterns.extend(candle_patterns)
                    
                    # 3. Detect harmonic patterns
                    harmonic_patterns = self._detect_harmonic_patterns(df, timeframe)
                    all_patterns.extend(harmonic_patterns)
                    
                    # 4. Detect breakout patterns
                    breakout_patterns = self._detect_breakout_patterns(df, timeframe)
                    all_patterns.extend(breakout_patterns)
            
            # Filter and rank patterns
            valid_patterns = self._filter_patterns(all_patterns, timestamp)
            ranked_patterns = self._rank_patterns(valid_patterns)
            
            # Update pattern tracking
            self._update_pattern_tracking(symbol, ranked_patterns, timestamp)
            
            return ranked_patterns
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed for {symbol}: {e}")
            return []
            
    def _detect_classic_patterns(self, data: pd.DataFrame,
                               timeframe: str) -> List[PatternSignal]:
        """Detect classic chart patterns"""
        try:
            patterns = []
            
            # Head and Shoulders pattern
            hs_pattern = self._detect_head_shoulders(data)
            if hs_pattern:
                patterns.append(hs_pattern)
                
            # Double Top/Bottom pattern
            double_pattern = self._detect_double_pattern(data)
            if double_pattern:
                patterns.append(double_pattern)
                
            # Triangle patterns (Ascending, Descending, Symmetric)
            triangle_patterns = self._detect_triangle_patterns(data)
            patterns.extend(triangle_patterns)
            
            # Set timeframe for all patterns
            for pattern in patterns:
                pattern.timeframe = timeframe
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"Classic pattern detection failed: {e}")
            return []
            
    def _detect_head_shoulders(self, data: pd.DataFrame) -> Optional[PatternSignal]:
        """Detect head and shoulders pattern"""
        try:
            # Get recent highs and lows
            highs = data['high'].rolling(5).max()
            lows = data['low'].rolling(5).min()
            
            # Look for characteristic peaks
            left_shoulder = self._find_peak(highs, 0, 20)
            head = self._find_peak(highs, left_shoulder + 5, 20)
            right_shoulder = self._find_peak(highs, head + 5, 20)
            
            if all([left_shoulder, head, right_shoulder]):
                # Verify pattern geometry
                if (abs(highs[left_shoulder] - highs[right_shoulder]) < 
                    0.1 * highs[head]):  # Shoulders roughly equal
                    
                    # Calculate neckline
                    neckline = (lows[left_shoulder] + lows[right_shoulder]) / 2
                    
                    # Calculate pattern metrics
                    height = highs[head] - neckline
                    target = neckline - height  # Projected downside target
                    
                    return PatternSignal(
                        pattern_type=PatternType.REVERSAL,
                        pattern_name="Head and Shoulders",
                        direction=-1,  # Bearish
                        strength=0.8,
                        completion=self._calculate_completion(data, right_shoulder),
                        target_price=target,
                        stop_loss=highs[head],
                        timeframe="",  # Set by caller
                        confidence=0.7
                    )
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Head and shoulders detection failed: {e}")
            return None
            
    def _detect_candlestick_patterns(self, data: pd.DataFrame,
                                   timeframe: str) -> List[PatternSignal]:
        """Detect candlestick patterns using TA-Lib"""
        try:
            patterns = []
            
            for pattern_func, pattern_name in self.candlestick_patterns.items():
                # Get pattern recognition integers from TA-Lib
                pattern_ints = pattern_func(
                    data['open'], data['high'],
                    data['low'], data['close']
                )
                
                # Check last few bars for pattern
                recent_patterns = pattern_ints.tail(3)
                for i, value in enumerate(recent_patterns):
                    if value != 0:  # Pattern detected
                        direction = 1 if value > 0 else -1
                        
                        patterns.append(PatternSignal(
                            pattern_type=PatternType.CANDLESTICK,
                            pattern_name=pattern_name,
                            direction=direction,
                            strength=abs(value) / 100,
                            completion=1.0,  # Candlestick patterns are immediate
                            target_price=self._calculate_pattern_target(
                                data, direction, i
                            ),
                            stop_loss=self._calculate_pattern_stop(
                                data, direction, i
                            ),
                            timeframe=timeframe,
                            confidence=0.6  # Lower confidence for candlesticks
                        ))
                        
            return patterns
            
        except Exception as e:
            self.logger.error(f"Candlestick pattern detection failed: {e}")
            return []
            
    def _detect_harmonic_patterns(self, data: pd.DataFrame,
                                timeframe: str) -> List[PatternSignal]:
        """Detect harmonic price patterns"""
        try:
            patterns = []
            
            # Get swing points
            swings = self._find_swing_points(data)
            
            if len(swings) >= 5:  # Need 5 points for harmonic patterns
                # Check Gartley pattern
                gartley = self._check_gartley_pattern(swings)
                if gartley:
                    patterns.append(gartley)
                    
                # Check Butterfly pattern
                butterfly = self._check_butterfly_pattern(swings)
                if butterfly:
                    patterns.append(butterfly)
                    
                # Check Bat pattern
                bat = self._check_bat_pattern(swings)
                if bat:
                    patterns.append(bat)
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"Harmonic pattern detection failed: {e}")
            return []
            
    def _detect_breakout_patterns(self, data: pd.DataFrame,
                                timeframe: str) -> List[PatternSignal]:
        """Detect breakout patterns"""
        try:
            patterns = []
            
            # Calculate support/resistance levels
            levels = self._find_support_resistance_levels(data)
            
            # Check for breakouts
            current_price = data['close'].iloc[-1]
            
            for level in levels:
                # Breakout above resistance
                if current_price > level['price'] * 1.02:  # 2% breakout
                    patterns.append(PatternSignal(
                        pattern_type=PatternType.BREAKOUT,
                        pattern_name="Resistance Breakout",
                        direction=1,  # Bullish
                        strength=level['strength'],
                        completion=1.0,
                        target_price=level['price'] * 1.1,  # 10% target
                        stop_loss=level['price'],
                        timeframe=timeframe,
                        confidence=0.7
                    ))
                    
                # Breakdown below support
                elif current_price < level['price'] * 0.98:  # 2% breakdown
                    patterns.append(PatternSignal(
                        pattern_type=PatternType.BREAKOUT,
                        pattern_name="Support Breakdown",
                        direction=-1,  # Bearish
                        strength=level['strength'],
                        completion=1.0,
                        target_price=level['price'] * 0.9,  # 10% target
                        stop_loss=level['price'],
                        timeframe=timeframe,
                        confidence=0.7
                    ))
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"Breakout pattern detection failed: {e}")
            return []
            
    def _filter_patterns(self, patterns: List[PatternSignal],
                        timestamp: datetime) -> List[PatternSignal]:
        """Filter patterns for validity and relevance"""
        valid_patterns = []
        
        for pattern in patterns:
            # Check completion threshold
            if pattern.completion >= self.completion_threshold:
                # Check pattern hasn't expired
                if pattern.pattern_type == PatternType.CANDLESTICK:
                    # Candlestick patterns expire quickly
                    valid_patterns.append(pattern)
                else:
                    # Other patterns valid for longer
                    valid_patterns.append(pattern)
                    
        return valid_patterns
        
    def _rank_patterns(self, patterns: List[PatternSignal]) -> List[PatternSignal]:
        """Rank patterns by importance/reliability"""
        if not patterns:
            return []
            
        # Calculate ranking score for each pattern
        for pattern in patterns:
            timeframe_weight = self.timeframe_weights.get(pattern.timeframe, 0.1)
            
            # Base score on pattern metrics
            score = pattern.strength * pattern.completion * timeframe_weight
            
            # Adjust for pattern type
            type_weights = {
                PatternType.REVERSAL: 1.0,
                PatternType.CONTINUATION: 0.8,
                PatternType.BREAKOUT: 0.9,
                PatternType.HARMONIC: 0.7,
                PatternType.CANDLESTICK: 0.6
            }
            score *= type_weights.get(pattern.pattern_type, 0.5)
            
            # Store score
            pattern.confidence = min(0.95, score)
            
        # Sort by score
        ranked_patterns = sorted(
            patterns,
            key=lambda x: x.confidence,
            reverse=True
        )
        
        return ranked_patterns
        
    def _update_pattern_tracking(self, symbol: str,
                               patterns: List[PatternSignal],
                               timestamp: datetime):
        """Update pattern tracking data"""
        # Update active patterns
        self.active_patterns[symbol] = patterns
        
        # Move completed patterns to history
        for pattern in patterns:
            if pattern.completion >= 1.0:
                self.completed_patterns.append({
                    'symbol': symbol,
                    'pattern': pattern,
                    'timestamp': timestamp
                })
                
        # Limit history size
        if len(self.completed_patterns) > 1000:
            self.completed_patterns = self.completed_patterns[-1000:]
            
    def _initialize_candlestick_patterns(self) -> Dict:
        """Initialize TA-Lib candlestick functions"""
        return {
            talib.CDLENGULFING: "Engulfing",
            talib.CDLHAMMER: "Hammer",
            talib.CDLMORNINGSTAR: "Morning Star",
            talib.CDLEVENINGSTAR: "Evening Star",
            talib.CDLDOJI: "Doji",
            talib.CDLSHOOTINGSTAR: "Shooting Star"
        }
        
    # Additional helper methods would be implemented here:
    # - _find_peak()
    # - _calculate_completion()
    # - _calculate_pattern_target()
    # - _calculate_pattern_stop()
    # - _find_swing_points()
    # - _check_gartley_pattern()
    # - _check_butterfly_pattern()
    # - _check_bat_pattern()
    # - _find_support_resistance_levels()