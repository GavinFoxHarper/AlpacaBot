"""
Multi-Strategy Orchestrator for LAEF Trading Platform
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

class StrategyType(Enum):
    MOMENTUM_SCALPING = "momentum_scalping"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    DUAL_MODEL_SWING = "dual_model_swing"
    PATTERN_RECOGNITION = "pattern_recognition"
    TIME_BASED_ALGORITHM = "time_based_algorithm"
    NEWS_SENTIMENT = "news_sentiment"
    HYBRID_ADAPTIVE = "hybrid_adaptive"
    REINFORCED_GRID_SEARCH = "reinforced_grid_search"

@dataclass
class StrategySignal:
    strategy_type: StrategyType
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    expected_return: float
    risk_score: float
    timeframe: str
    reasoning: str
    weight: float = 1.0

@dataclass
class MarketConditions:
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_trend: float
    time_of_day: str
    vix_level: float
    news_sentiment: float
    liquidity_score: float

class MultiStrategyOrchestrator:
    """Multi-strategy orchestration with market regime adaptation"""
    
    def __init__(self, config, strategies_dict, news_engine=None):
        self.config = config
        self.strategies = strategies_dict  # Dict of strategy_type -> strategy_instance
        self.news_engine = news_engine
        self.logger = logging.getLogger(__name__)
        
        # Strategy weights and conditions
        self.strategy_weights = config.get('strategy_weights', {
            StrategyType.MOMENTUM_SCALPING: 1.0,
            StrategyType.MEAN_REVERSION: 1.0,
            StrategyType.STATISTICAL_ARBITRAGE: 0.8,
            StrategyType.DUAL_MODEL_SWING: 1.2,
            StrategyType.PATTERN_RECOGNITION: 1.0,
            StrategyType.TIME_BASED_ALGORITHM: 0.6,
            StrategyType.NEWS_SENTIMENT: 0.8,
            StrategyType.HYBRID_ADAPTIVE: 1.5
        })
        
        # Market regime strategy preferences
        self.regime_strategy_preferences = {
            MarketRegime.TRENDING_UP: {
                StrategyType.MOMENTUM_SCALPING: 1.5,
                StrategyType.DUAL_MODEL_SWING: 1.3,
                StrategyType.PATTERN_RECOGNITION: 1.2,
                StrategyType.MEAN_REVERSION: 0.7
            },
            MarketRegime.TRENDING_DOWN: {
                StrategyType.MOMENTUM_SCALPING: 1.2,
                StrategyType.STATISTICAL_ARBITRAGE: 1.4,
                StrategyType.NEWS_SENTIMENT: 1.3,
                StrategyType.MEAN_REVERSION: 0.8
            },
            MarketRegime.RANGING: {
                StrategyType.MEAN_REVERSION: 1.6,
                StrategyType.STATISTICAL_ARBITRAGE: 1.4,
                StrategyType.PATTERN_RECOGNITION: 1.1,
                StrategyType.MOMENTUM_SCALPING: 0.6
            },
            MarketRegime.VOLATILE: {
                StrategyType.MOMENTUM_SCALPING: 1.8,
                StrategyType.NEWS_SENTIMENT: 1.5,
                StrategyType.TIME_BASED_ALGORITHM: 1.2,
                StrategyType.MEAN_REVERSION: 0.5
            },
            MarketRegime.LOW_VOLATILITY: {
                StrategyType.MEAN_REVERSION: 1.4,
                StrategyType.STATISTICAL_ARBITRAGE: 1.6,
                StrategyType.PATTERN_RECOGNITION: 1.1,
                StrategyType.MOMENTUM_SCALPING: 0.4
            }
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.recent_decisions = []
        self.decision_history_limit = 1000
        
        # Thresholds
        self.min_strategy_score = config.get('min_strategy_score', 0.6)
        self.max_active_strategies = config.get('max_active_strategies', 4)
        self.signal_aggregation_method = config.get('signal_aggregation_method', 'weighted_average')
    
    def make_trading_decision(self, market_conditions: MarketConditions, ml_q_value: float = None, ml_confidence: float = None) -> Dict[str, Any]:
        """Main method to make trading decisions using multiple strategies"""
        try:
            # Get signals from all strategies
            strategy_signals = []
            
            # Add ML signal if provided
            if ml_q_value is not None and ml_confidence is not None:
                ml_signal = self._create_ml_signal(ml_q_value, ml_confidence)
                if ml_signal:
                    strategy_signals.append(ml_signal)
            
            # Aggregate all signals
            decision = self._aggregate_signals(strategy_signals, market_conditions)
            
            # Add market context
            decision['market_conditions'] = {
                'regime': market_conditions.regime.value,
                'volatility': market_conditions.volatility,
                'trend_strength': market_conditions.trend_strength,
                'time_of_day': market_conditions.time_of_day,
                'news_sentiment': market_conditions.news_sentiment
            }
            
            # Store for performance tracking
            self._store_decision(decision, strategy_signals, market_conditions)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return {
                'action': 0,
                'confidence': 0.0,
                'expected_return': 0.0,
                'reasoning': f"Error in decision making: {str(e)}",
                'active_strategies': []
            }
            
    def _create_ml_signal(self, q_value: float, confidence: float) -> Optional[StrategySignal]:
        """Create a signal from ML predictions"""
        # Convert Q-value to action
        if q_value > 0.7:
            action = 1  # Buy
        elif q_value < 0.3:
            action = 2  # Sell
        else:
            action = 0  # Hold
            
        # Estimate expected return
        expected_return = (q_value - 0.5) * 0.02  # Scale to reasonable range
        
        return StrategySignal(
            strategy_type=StrategyType.DUAL_MODEL_SWING,
            action=action,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=1.0 - confidence,
            timeframe="1h",
            reasoning=f"ML Q-value: {q_value:.3f}, confidence: {confidence:.3f}",
            weight=1.2  # Give ML signals higher weight
        )
        
    def _aggregate_signals(self, signals: List[StrategySignal], conditions: MarketConditions) -> Dict[str, Any]:
        """Aggregate signals using specified method"""
        if not signals:
            return {
                'action': 0,
                'confidence': 0.0,
                'expected_return': 0.0,
                'reasoning': "No active strategies",
                'active_strategies': []
            }
            
        # Filter signals by minimum score
        qualified_signals = [s for s in signals if s.confidence >= self.min_strategy_score]
        
        if not qualified_signals:
            return {
                'action': 0,
                'confidence': 0.0,
                'expected_return': 0.0,
                'reasoning': "No strategies meet minimum confidence threshold",
                'active_strategies': [s.strategy_type.value for s in signals]
            }
            
        # Limit to top strategies
        qualified_signals.sort(key=lambda x: x.confidence * x.weight, reverse=True)
        top_signals = qualified_signals[:self.max_active_strategies]
        
        # Aggregate based on method
        if self.signal_aggregation_method == 'weighted_average':
            result = self._weighted_average_aggregation(top_signals)
        elif self.signal_aggregation_method == 'consensus':
            result = self._consensus_aggregation(top_signals)
        elif self.signal_aggregation_method == 'best_signal':
            result = self._best_signal_aggregation(top_signals)
        else:
            result = self._weighted_average_aggregation(top_signals)
            
        # Add metadata
        result['active_strategies'] = [s.strategy_type.value for s in top_signals]
        result['signal_count'] = len(top_signals)
        result['market_regime'] = conditions.regime.value
        
        return result
        
    def _weighted_average_aggregation(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Aggregate signals using weighted average"""
        if not signals:
            return {
                'action': 0,
                'confidence': 0.0,
                'expected_return': 0.0,
                'reasoning': "No signals to aggregate"
            }
            
        # Calculate weighted scores
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return {
                'action': 0,
                'confidence': 0.0,
                'expected_return': 0.0,
                'reasoning': "Zero total weight"
            }
            
        # Calculate weighted metrics
        weighted_confidence = sum(s.confidence * s.weight for s in signals) / total_weight
        weighted_return = sum(s.expected_return * s.weight for s in signals) / total_weight
        
        # Determine action based on weighted metrics
        if weighted_confidence > 0.7:
            action = 1  # Buy
        elif weighted_confidence < 0.3:
            action = 2  # Sell
        else:
            action = 0  # Hold
            
        return {
            'action': action,
            'confidence': weighted_confidence,
            'expected_return': weighted_return,
            'reasoning': f"Weighted average of {len(signals)} strategies"
        }
        
    def _store_decision(self, decision: Dict, signals: List[StrategySignal], conditions: MarketConditions):
        """Store decision for performance tracking"""
        self.recent_decisions.append({
            'action': decision['action'],
            'confidence': decision['confidence'],
            'expected_return': decision['expected_return'],
            'signals': [s.strategy_type.value for s in signals],
            'market_regime': conditions.regime.value
        })
        
        # Limit history size
        if len(self.recent_decisions) > self.decision_history_limit:
            self.recent_decisions = self.recent_decisions[-self.decision_history_limit:]