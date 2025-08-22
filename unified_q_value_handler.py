"""
Q-Value Handler for LAEF Trading System
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

class QValueMethod(Enum):
    DIFFERENCE_BASED = "difference_based"  # (q_buy - q_sell + 1) / 2
    MAX_BASED = "max_based"               # max(q_hold, q_buy, q_sell)
    SOFTMAX_BASED = "softmax_based"       # Apply softmax to all Q-values
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted combination of actions

@dataclass
class QValueOutput:
    q_hold: float
    q_buy: float
    q_sell: float
    directional_bias: float    # -1 (sell) to +1 (buy)
    confidence: float          # 0 to 1
    recommended_action: int    # 0=hold, 1=buy, 2=sell
    action_probabilities: Dict[int, float]  # Probability for each action
    reasoning: str

class UnifiedQValueHandler:
    """Unified Q-value processing with market regime adaptation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.q_value_method = QValueMethod(config.get('q_value_method', 'difference_based'))
        self.confidence_method = config.get('confidence_method', 'entropy_based')
        self.temperature = config.get('q_value_temperature', 1.0)  # For softmax
        
        # Thresholds
        self.buy_threshold = config.get('q_buy_threshold', 0.6)
        self.sell_threshold = config.get('q_sell_threshold', 0.4)
        self.hold_bias = config.get('hold_bias', 0.1)  # Slight bias toward holding
        
        # Risk adjustment parameters
        self.risk_adjustment_enabled = config.get('risk_adjustment_enabled', True)
        self.volatility_penalty = config.get('volatility_penalty', 0.1)
        self.momentum_bonus = config.get('momentum_bonus', 0.05)
        
        # Normalization parameters
        self.normalize_q_values = config.get('normalize_q_values', True)
        self.q_value_bounds = config.get('q_value_bounds', (-2.0, 2.0))
        
    def process_q_values(self, raw_q_values: np.ndarray, 
                        market_context: Optional[Dict] = None,
                        symbol: str = None) -> QValueOutput:
        """Process Q-values with market adaptation"""
        try:
            # Validate input
            if len(raw_q_values) != 3:
                raise ValueError(f"Expected 3 Q-values, got {len(raw_q_values)}")
            
            q_hold, q_buy, q_sell = raw_q_values
            
            # Apply normalization if enabled
            if self.normalize_q_values:
                q_hold, q_buy, q_sell = self._normalize_q_values(q_hold, q_buy, q_sell)
            
            # Apply market context adjustments
            if market_context and self.risk_adjustment_enabled:
                q_hold, q_buy, q_sell = self._apply_market_adjustments(
                    q_hold, q_buy, q_sell, market_context
                )
            
            # Calculate directional bias based on method
            directional_bias = self._calculate_directional_bias(q_hold, q_buy, q_sell)
            
            # Calculate confidence
            confidence = self._calculate_confidence(q_hold, q_buy, q_sell, market_context)
            
            # Determine recommended action
            action_probs = self._calculate_action_probabilities(q_hold, q_buy, q_sell)
            recommended_action = self._determine_action(q_hold, q_buy, q_sell, action_probs)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                q_hold, q_buy, q_sell, directional_bias, confidence, 
                recommended_action, market_context
            )
            
            result = QValueOutput(
                q_hold=q_hold,
                q_buy=q_buy,
                q_sell=q_sell,
                directional_bias=directional_bias,
                confidence=confidence,
                recommended_action=recommended_action,
                action_probabilities=action_probs,
                reasoning=reasoning
            )
            
            self.logger.debug(f"Q-values processed for {symbol}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing Q-values for {symbol}: {e}")
            return self._create_default_output()
    
    def _normalize_q_values(self, q_hold: float, q_buy: float, q_sell: float) -> Tuple[float, float, float]:
        """Normalize Q-values to a consistent range"""
        
        # Clip to bounds
        min_val, max_val = self.q_value_bounds
        q_hold = np.clip(q_hold, min_val, max_val)
        q_buy = np.clip(q_buy, min_val, max_val)
        q_sell = np.clip(q_sell, min_val, max_val)
        
        # Apply sigmoid normalization to [0, 1] range
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        if self.config.get('use_sigmoid_normalization', True):
            q_hold = sigmoid(q_hold)
            q_buy = sigmoid(q_buy)
            q_sell = sigmoid(q_sell)
        else:
            # Linear normalization to [0, 1]
            q_values = np.array([q_hold, q_buy, q_sell])
            q_values = (q_values - min_val) / (max_val - min_val)
            q_hold, q_buy, q_sell = q_values
        
        return q_hold, q_buy, q_sell
    
    def _apply_market_adjustments(self, q_hold: float, q_buy: float, q_sell: float,
                                 market_context: Dict) -> Tuple[float, float, float]:
        """Apply market context adjustments to Q-values"""
        
        # Volatility adjustment
        volatility = market_context.get('volatility', 0.2)
        if volatility > 0.4:  # High volatility
            # Reduce buy/sell signals, increase hold
            q_hold += self.volatility_penalty
            q_buy -= self.volatility_penalty * 0.5
            q_sell -= self.volatility_penalty * 0.5
        
        # Momentum adjustment
        momentum = market_context.get('momentum', 0.0)
        if abs(momentum) > 0.02:  # Strong momentum
            if momentum > 0:  # Positive momentum
                q_buy += self.momentum_bonus
            else:  # Negative momentum
                q_sell += self.momentum_bonus
        
        # Volume adjustment
        volume_ratio = market_context.get('volume_ratio', 1.0)
        if volume_ratio < 0.5:  # Low volume
            # Reduce buy/sell confidence
            q_buy *= 0.9
            q_sell *= 0.9
            q_hold += 0.05
        
        # Time of day adjustment
        time_of_day = market_context.get('time_of_day', 'mid_day')
        if time_of_day == 'market_close':
            # Bias toward closing positions
            q_hold += 0.1
        elif time_of_day == 'market_open':
            # Allow more aggressive trading
            q_buy += 0.05
            q_sell += 0.05
        
        # VIX adjustment (market fear)
        vix = market_context.get('vix', 20.0)
        if vix > 30:  # High fear
            q_hold += 0.15
            q_buy -= 0.1
        elif vix < 15:  # Low fear/complacency
            q_buy += 0.05
        
        return q_hold, q_buy, q_sell
    
    def _calculate_directional_bias(self, q_hold: float, q_buy: float, q_sell: float) -> float:
        """Calculate directional bias using the specified method"""
        
        if self.q_value_method == QValueMethod.DIFFERENCE_BASED:
            # Original LAEF method: (q_buy - q_sell + 1) / 2
            bias = (q_buy - q_sell + 1) / 2
            # Ensure it's in [0, 1] range, then convert to [-1, 1]
            bias = np.clip(bias, 0, 1)
            return (bias - 0.5) * 2  # Convert to [-1, 1]
            
        elif self.q_value_method == QValueMethod.MAX_BASED:
            # Find which action has highest Q-value
            max_q = max(q_hold, q_buy, q_sell)
            if max_q == q_buy:
                return 1.0
            elif max_q == q_sell:
                return -1.0
            else:
                return 0.0
                
        elif self.q_value_method == QValueMethod.SOFTMAX_BASED:
            # Use softmax probabilities
            q_values = np.array([q_hold, q_buy, q_sell])
            probs = self._softmax(q_values)
            return probs[1] - probs[2]  # buy_prob - sell_prob
            
        elif self.q_value_method == QValueMethod.WEIGHTED_AVERAGE:
            # Weighted average considering all actions
            total = q_hold + q_buy + q_sell
            if total == 0:
                return 0.0
            buy_weight = q_buy / total
            sell_weight = q_sell / total
            return buy_weight - sell_weight
        
        else:
            # Fallback to difference-based
            return (q_buy - q_sell) / (abs(q_buy) + abs(q_sell) + 1e-8)
    
    def _calculate_confidence(self, q_hold: float, q_buy: float, q_sell: float,
                            market_context: Optional[Dict] = None) -> float:
        """Calculate confidence in the Q-value prediction"""
        
        if self.confidence_method == 'entropy_based':
            # Use entropy to measure confidence (lower entropy = higher confidence)
            q_values = np.array([q_hold, q_buy, q_sell])
            probs = self._softmax(q_values)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            max_entropy = np.log(3)  # Maximum possible entropy for 3 actions
            confidence = 1.0 - (entropy / max_entropy)
            
        elif self.confidence_method == 'max_difference':
            # Confidence based on difference between max and second-max Q-values
            q_values = sorted([q_hold, q_buy, q_sell], reverse=True)
            max_diff = q_values[0] - q_values[1]
            confidence = np.tanh(max_diff)  # Sigmoid-like function
            
        elif self.confidence_method == 'variance_based':
            # Lower variance = higher confidence
            q_values = np.array([q_hold, q_buy, q_sell])
            variance = np.var(q_values)
            confidence = 1.0 / (1.0 + variance)
            
        else:  # 'simple' method
            # Simple method based on absolute Q-values
            max_q = max(abs(q_hold), abs(q_buy), abs(q_sell))
            confidence = np.tanh(max_q)
        
        # Apply market context adjustments to confidence
        if market_context:
            # Reduce confidence in high volatility
            volatility = market_context.get('volatility', 0.2)
            if volatility > 0.4:
                confidence *= 0.8
            
            # Reduce confidence with low volume
            volume_ratio = market_context.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                confidence *= 0.9
            
            # News uncertainty
            news_sentiment = market_context.get('news_sentiment', 0.0)
            if abs(news_sentiment) > 0.5:  # Strong news sentiment
                confidence *= 1.1  # Increase confidence
            
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_action_probabilities(self, q_hold: float, q_buy: float, q_sell: float) -> Dict[int, float]:
        """Calculate probability distribution over actions"""
        
        q_values = np.array([q_hold, q_buy, q_sell])
        probs = self._softmax(q_values)
        
        return {
            0: probs[0],  # hold
            1: probs[1],  # buy
            2: probs[2]   # sell
        }
    
    def _determine_action(self, q_hold: float, q_buy: float, q_sell: float,
                         action_probs: Dict[int, float]) -> int:
        """Determine the recommended action"""
        
        # Apply hold bias
        adjusted_q_hold = q_hold + self.hold_bias
        
        # Method 1: Simple maximum
        if self.config.get('use_threshold_decision', True):
            # Use thresholds for more conservative trading
            directional_bias = self._calculate_directional_bias(q_hold, q_buy, q_sell)
            
            if directional_bias > (self.buy_threshold - 0.5) * 2:  # Convert threshold to [-1,1] scale
                return 1  # Buy
            elif directional_bias < (self.sell_threshold - 0.5) * 2:
                return 2  # Sell
            else:
                return 0  # Hold
        else:
            # Method 2: Probability-based
            return max(action_probs, key=action_probs.get)
    
    def _softmax(self, q_values: np.ndarray) -> np.ndarray:
        """Apply softmax with temperature"""
        q_values = q_values / self.temperature
        exp_values = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        return exp_values / np.sum(exp_values)
    
    def _generate_reasoning(self, q_hold: float, q_buy: float, q_sell: float,
                          directional_bias: float, confidence: float,
                          recommended_action: int, market_context: Optional[Dict]) -> str:
        """Generate human-readable reasoning for the decision"""
        
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_name = action_names[recommended_action]
        
        # Build reasoning string
        reasons = []
        
        # Q-value analysis
        reasons.append(f"Q-values: Hold={q_hold:.3f}, Buy={q_buy:.3f}, Sell={q_sell:.3f}")
        
        # Directional bias
        if abs(directional_bias) > 0.1:
            direction = "bullish" if directional_bias > 0 else "bearish"
            reasons.append(f"Directional bias: {directional_bias:.3f} ({direction})")
        
        # Confidence level
        confidence_level = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        reasons.append(f"Confidence: {confidence:.3f} ({confidence_level})")
        
        # Market context
        if market_context:
            context_reasons = []
            
            volatility = market_context.get('volatility', 0.2)
            if volatility > 0.4:
                context_reasons.append("high volatility")
            elif volatility < 0.15:
                context_reasons.append("low volatility")
            
            volume_ratio = market_context.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                context_reasons.append("high volume")
            elif volume_ratio < 0.5:
                context_reasons.append("low volume")
            
            momentum = market_context.get('momentum', 0.0)
            if abs(momentum) > 0.02:
                direction = "positive" if momentum > 0 else "negative"
                context_reasons.append(f"{direction} momentum")
            
            if context_reasons:
                reasons.append(f"Market context: {', '.join(context_reasons)}")
        
        # Method used
        reasons.append(f"Method: {self.q_value_method.value}")
        
        return f"{action_name} - {' | '.join(reasons)}"
    
    def _create_default_output(self) -> QValueOutput:
        """Create default output for error cases"""
        return QValueOutput(
            q_hold=0.5,
            q_buy=0.5,
            q_sell=0.5,
            directional_bias=0.0,
            confidence=0.0,
            recommended_action=0,  # Hold
            action_probabilities={0: 1.0, 1: 0.0, 2: 0.0},
            reasoning="Error in Q-value processing - defaulting to HOLD"
        )