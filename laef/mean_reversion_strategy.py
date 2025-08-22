"""
Mean Reversion Strategy for LAEF Trading Platform

Strategy Overview:
1. Bollinger Band-based mean reversion
2. RSI oversold/overbought confirmation
3. Volume-weighted price normalization
4. Dynamic mean calculation
5. Multi-timeframe confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .technical_indicators import calculate_bollinger_bands, calculate_rsi
from core.portfolio_manager import FIFOPortfolio

@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal"""
    action: int          # 0=hold, 1=buy, 2=sell
    confidence: float    # 0 to 1
    deviation: float     # Standard deviations from mean
    rsi: float          # Current RSI value
    volume_ratio: float # Current volume vs average
    mean_price: float   # Current mean price
    timeframe: str      # Signal timeframe

class MeanReversionStrategy:
    """Mean reversion trading strategy"""
    
    def __init__(self, portfolio: FIFOPortfolio, config: Dict = None):
        self.portfolio = portfolio
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.volume_period = config.get('volume_period', 20)
        
        # Entry/exit thresholds
        self.entry_threshold = config.get('entry_threshold', 2.0)  # Standard deviations
        self.exit_threshold = config.get('exit_threshold', 0.5)    # Standard deviations
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # Position sizing
        self.position_size = config.get('position_size', 0.05)  # 5% of portfolio
        self.max_positions = config.get('max_positions', 5)
        
        # Timeframes to analyze
        self.timeframes = ['5m', '15m', '1h', '4h']
        
        # Performance tracking
        self.trade_history = []
        self.mean_history = {}
        
    def analyze_opportunity(self, symbol: str, data: Dict[str, pd.DataFrame],
                          timestamp: datetime = None) -> MeanReversionSignal:
        """
        Analyze mean reversion opportunity across timeframes
        
        Args:
            symbol: Trading symbol
            data: Dict of DataFrames for each timeframe
            timestamp: Current timestamp
            
        Returns:
            MeanReversionSignal with trading decision
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Analyze each timeframe
            signals = []
            for timeframe in self.timeframes:
                if timeframe in data:
                    tf_data = data[timeframe]
                    if len(tf_data) >= self.bb_period:
                        signal = self._analyze_timeframe(symbol, tf_data, timeframe)
                        signals.append(signal)
            
            if not signals:
                return self._create_default_signal(timestamp)
            
            # Combine signals across timeframes
            final_signal = self._combine_timeframe_signals(signals)
            
            # Add position checks
            final_signal = self._apply_position_checks(symbol, final_signal)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Mean reversion analysis failed for {symbol}: {e}")
            return self._create_default_signal(timestamp)
            
    def _analyze_timeframe(self, symbol: str, data: pd.DataFrame,
                          timeframe: str) -> MeanReversionSignal:
        """Analyze single timeframe for mean reversion"""
        try:
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                data['close'], period=self.bb_period, std=self.bb_std
            )
            
            # Calculate RSI
            rsi = calculate_rsi(data['close'], period=self.rsi_period)
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Calculate deviation in standard deviations
            deviation = (current_price - bb_middle.iloc[-1]) / (bb_upper.iloc[-1] - bb_middle.iloc[-1])
            
            # Calculate volume ratio
            volume_sma = data['volume'].rolling(self.volume_period).mean()
            volume_ratio = data['volume'].iloc[-1] / volume_sma.iloc[-1]
            
            # Determine trading action
            action, confidence = self._evaluate_signal(
                deviation, current_rsi, volume_ratio
            )
            
            return MeanReversionSignal(
                action=action,
                confidence=confidence,
                deviation=deviation,
                rsi=current_rsi,
                volume_ratio=volume_ratio,
                mean_price=bb_middle.iloc[-1],
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.error(f"Timeframe analysis failed for {symbol} {timeframe}: {e}")
            return self._create_default_signal(datetime.now())
            
    def _evaluate_signal(self, deviation: float, rsi: float,
                        volume_ratio: float) -> Tuple[int, float]:
        """Evaluate trading signal from indicators"""
        try:
            # Initialize confidence
            confidence = 0.0
            action = 0  # Default to hold
            
            # CASE 1: Strong oversold - Buy signal
            if deviation <= -self.entry_threshold and rsi < self.rsi_oversold:
                action = 1  # Buy
                # Confidence based on oversold severity
                confidence = min(0.9, abs(deviation) / 4.0)
                # Boost confidence with high volume
                if volume_ratio > 1.5:
                    confidence = min(0.95, confidence * 1.2)
                    
            # CASE 2: Strong overbought - Sell signal
            elif deviation >= self.entry_threshold and rsi > self.rsi_overbought:
                action = 2  # Sell
                # Confidence based on overbought severity
                confidence = min(0.9, abs(deviation) / 4.0)
                # Boost confidence with high volume
                if volume_ratio > 1.5:
                    confidence = min(0.95, confidence * 1.2)
                    
            # CASE 3: Mean reversion exit
            elif abs(deviation) < self.exit_threshold:
                if deviation > 0:
                    action = 2  # Sell if above mean
                else:
                    action = 1  # Buy if below mean
                # Lower confidence for mean reversion exits
                confidence = 0.5 + (self.exit_threshold - abs(deviation)) * 0.5
                
            else:
                # No clear signal
                action = 0
                confidence = abs(deviation) / 10.0  # Low confidence
                
            return action, confidence
            
        except Exception as e:
            self.logger.error(f"Signal evaluation failed: {e}")
            return 0, 0.0
            
    def _combine_timeframe_signals(self, signals: List[MeanReversionSignal]) -> MeanReversionSignal:
        """Combine signals from multiple timeframes"""
        if not signals:
            return self._create_default_signal(datetime.now())
            
        # Weight signals by timeframe
        weights = {
            '5m': 0.2,
            '15m': 0.3,
            '1h': 0.3,
            '4h': 0.2
        }
        
        # Calculate weighted averages
        total_weight = 0
        weighted_action = 0
        weighted_confidence = 0
        weighted_deviation = 0
        
        for signal in signals:
            weight = weights.get(signal.timeframe, 0.1)
            weighted_action += signal.action * weight
            weighted_confidence += signal.confidence * weight
            weighted_deviation += signal.deviation * weight
            total_weight += weight
            
        if total_weight > 0:
            # Normalize weighted values
            weighted_action /= total_weight
            weighted_confidence /= total_weight
            weighted_deviation /= total_weight
            
            # Round action to nearest integer
            final_action = round(weighted_action)
            
            return MeanReversionSignal(
                action=final_action,
                confidence=weighted_confidence,
                deviation=weighted_deviation,
                rsi=signals[0].rsi,  # Use shortest timeframe RSI
                volume_ratio=signals[0].volume_ratio,
                mean_price=signals[0].mean_price,
                timeframe='multi'
            )
        else:
            return signals[0]  # Return shortest timeframe signal
            
    def _apply_position_checks(self, symbol: str,
                             signal: MeanReversionSignal) -> MeanReversionSignal:
        """Apply position-related checks to signal"""
        try:
            # Check current positions
            current_positions = len(self.portfolio.positions)
            has_position = self.portfolio.has_position(symbol)
            
            # Modify signal based on position checks
            if signal.action == 1:  # Buy signal
                if current_positions >= self.max_positions and not has_position:
                    # Block new positions if at limit
                    return MeanReversionSignal(
                        action=0,
                        confidence=0.2,
                        deviation=signal.deviation,
                        rsi=signal.rsi,
                        volume_ratio=signal.volume_ratio,
                        mean_price=signal.mean_price,
                        timeframe=signal.timeframe
                    )
                    
            elif signal.action == 2:  # Sell signal
                if not has_position:
                    # Can't sell what we don't have
                    return MeanReversionSignal(
                        action=0,
                        confidence=0.2,
                        deviation=signal.deviation,
                        rsi=signal.rsi,
                        volume_ratio=signal.volume_ratio,
                        mean_price=signal.mean_price,
                        timeframe=signal.timeframe
                    )
                    
            return signal
            
        except Exception as e:
            self.logger.error(f"Position check failed: {e}")
            return signal
            
    def calculate_position_size(self, symbol: str, current_price: float) -> Dict:
        """Calculate position size for mean reversion trade"""
        try:
            portfolio_value = self.portfolio.get_available_cash()
            position_value = portfolio_value * self.position_size
            
            shares = int(position_value / current_price)
            cost = shares * current_price
            
            if shares <= 0 or cost > self.portfolio.get_available_cash():
                return {
                    'can_trade': False,
                    'reason': f"Insufficient funds (need ${cost:.2f})",
                    'shares': 0,
                    'cost': 0
                }
                
            return {
                'can_trade': True,
                'shares': shares,
                'cost': cost,
                'price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return {
                'can_trade': False,
                'reason': f"Calculation error: {e}",
                'shares': 0,
                'cost': 0
            }
            
    def _create_default_signal(self, timestamp: datetime) -> MeanReversionSignal:
        """Create default signal when analysis fails"""
        return MeanReversionSignal(
            action=0,
            confidence=0.0,
            deviation=0.0,
            rsi=50.0,
            volume_ratio=1.0,
            mean_price=0.0,
            timeframe='none'
        )
        
    def update_trade_history(self, trade_data: Dict):
        """Update trade history for performance tracking"""
        self.trade_history.append(trade_data)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]