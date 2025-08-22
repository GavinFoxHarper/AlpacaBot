"""
Timeframe Selection System for LAEF Trading Platform

Dynamically selects optimal trading timeframes based on:
1. Market Volatility & Liquidity
2. Trading Strategy Requirements
3. Signal Strength Across Timeframes
4. Market Regime Adaptation
5. Historical Performance Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from multi_strategy_orchestrator import MarketRegime
from .market_regime_detector import MarketState

@dataclass
class TimeframeRecommendation:
    """Timeframe selection recommendation"""
    primary_timeframe: str
    secondary_timeframes: List[str]
    confidence: float
    reasoning: str
    volatility_score: float
    liquidity_score: float
    timestamp: datetime

class TimeframeSelector:
    """Dynamic timeframe selection system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Available timeframes
        self.timeframes = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '1d'
        ]
        
        # Strategy-specific preferences
        self.strategy_timeframes = {
            'momentum_scalping': ['1m', '3m', '5m'],
            'mean_reversion': ['15m', '30m', '1h'],
            'pattern_recognition': ['1h', '2h', '4h'],
            'statistical_arbitrage': ['1h', '4h', '1d'],
            'news_sentiment': ['5m', '15m', '30m']
        }
        
        # Market regime preferences
        self.regime_timeframes = {
            MarketRegime.TRENDING_UP: ['15m', '30m', '1h'],
            MarketRegime.TRENDING_DOWN: ['15m', '30m', '1h'],
            MarketRegime.RANGING: ['5m', '15m', '30m'],
            MarketRegime.VOLATILE: ['1m', '3m', '5m'],
            MarketRegime.LOW_VOLATILITY: ['30m', '1h', '2h']
        }
        
        # Thresholds
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.liquidity_threshold = config.get('liquidity_threshold', 100000)
        
        # Performance tracking
        self.timeframe_performance = {}  # timeframe -> performance metrics
        self.recommendation_history = []
        
    def select_timeframes(self, market_state: MarketState,
                         strategy_type: str,
                         market_data: Dict[str, pd.DataFrame],
                         timestamp: datetime = None) -> TimeframeRecommendation:
        """
        Select optimal trading timeframes
        
        Args:
            market_state: Current market state/regime
            strategy_type: Type of trading strategy
            market_data: Price/volume data for different timeframes
            timestamp: Current timestamp
            
        Returns:
            TimeframeRecommendation with optimal timeframes
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # 1. Calculate market condition scores
            volatility_score = self._calculate_volatility_score(market_data)
            liquidity_score = self._calculate_liquidity_score(market_data)
            
            # 2. Get strategy preferences
            strategy_preferences = self.strategy_timeframes.get(
                strategy_type, ['5m', '15m', '30m']
            )
            
            # 3. Get regime preferences
            regime_preferences = self.regime_timeframes.get(
                market_state.regime, ['15m', '30m', '1h']
            )
            
            # 4. Score timeframes
            timeframe_scores = self._score_timeframes(
                strategy_preferences,
                regime_preferences,
                volatility_score,
                liquidity_score,
                market_data
            )
            
            # 5. Select optimal timeframes
            primary, secondary = self._select_optimal_timeframes(timeframe_scores)
            
            # 6. Calculate confidence
            confidence = self._calculate_selection_confidence(
                primary, secondary, timeframe_scores,
                volatility_score, liquidity_score
            )
            
            # 7. Generate reasoning
            reasoning = self._generate_selection_reasoning(
                primary, secondary, market_state.regime,
                strategy_type, timeframe_scores
            )
            
            recommendation = TimeframeRecommendation(
                primary_timeframe=primary,
                secondary_timeframes=secondary,
                confidence=confidence,
                reasoning=reasoning,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                timestamp=timestamp
            )
            
            # Update tracking
            self._update_recommendation_history(recommendation)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Timeframe selection failed: {e}")
            return self._create_default_recommendation(timestamp)
            
    def _calculate_volatility_score(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate market volatility score"""
        try:
            # Use 5-minute timeframe for volatility calculation
            if '5m' in market_data:
                data = market_data['5m']
                returns = data['close'].pct_change()
                volatility = returns.std() * np.sqrt(252 * 78)  # Annualized
                
                # Normalize to 0-1 range
                score = min(1.0, volatility / self.volatility_threshold)
                return score
                
            return 0.5  # Default moderate volatility
            
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return 0.5
            
    def _calculate_liquidity_score(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate market liquidity score"""
        try:
            # Use 5-minute timeframe for liquidity calculation
            if '5m' in market_data:
                data = market_data['5m']
                avg_volume = data['volume'].mean() * data['close'].mean()
                
                # Normalize to 0-1 range
                score = min(1.0, avg_volume / self.liquidity_threshold)
                return score
                
            return 0.5  # Default moderate liquidity
            
        except Exception as e:
            self.logger.error(f"Liquidity calculation failed: {e}")
            return 0.5
            
    def _score_timeframes(self, strategy_preferences: List[str],
                         regime_preferences: List[str],
                         volatility_score: float,
                         liquidity_score: float,
                         market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Score each timeframe based on multiple factors"""
        try:
            scores = {}
            
            for timeframe in self.timeframes:
                if timeframe not in market_data:
                    continue
                    
                # Base score
                base_score = 0.5
                
                # Strategy preference score
                strategy_rank = strategy_preferences.index(timeframe) + 1 \
                    if timeframe in strategy_preferences else len(strategy_preferences) + 1
                strategy_score = 1.0 / strategy_rank
                
                # Regime preference score
                regime_rank = regime_preferences.index(timeframe) + 1 \
                    if timeframe in regime_preferences else len(regime_preferences) + 1
                regime_score = 1.0 / regime_rank
                
                # Volatility adjustment
                volatility_adjustment = self._get_volatility_adjustment(
                    timeframe, volatility_score
                )
                
                # Liquidity adjustment
                liquidity_adjustment = self._get_liquidity_adjustment(
                    timeframe, liquidity_score
                )
                
                # Performance adjustment
                performance_adjustment = self._get_performance_adjustment(timeframe)
                
                # Combine scores
                final_score = (
                    base_score * 0.2 +
                    strategy_score * 0.3 +
                    regime_score * 0.2 +
                    volatility_adjustment * 0.1 +
                    liquidity_adjustment * 0.1 +
                    performance_adjustment * 0.1
                )
                
                scores[timeframe] = final_score
                
            return scores
            
        except Exception as e:
            self.logger.error(f"Timeframe scoring failed: {e}")
            return {tf: 0.5 for tf in self.timeframes}
            
    def _select_optimal_timeframes(self, timeframe_scores: Dict[str, float]) -> Tuple[str, List[str]]:
        """Select optimal primary and secondary timeframes"""
        try:
            # Sort timeframes by score
            sorted_timeframes = sorted(
                timeframe_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select primary timeframe
            primary = sorted_timeframes[0][0]
            
            # Select secondary timeframes
            secondary = [
                tf[0] for tf in sorted_timeframes[1:4]  # Next 3 best timeframes
            ]
            
            return primary, secondary
            
        except Exception as e:
            self.logger.error(f"Optimal timeframe selection failed: {e}")
            return '15m', ['5m', '30m', '1h']
            
    def _calculate_selection_confidence(self, primary: str,
                                     secondary: List[str],
                                     scores: Dict[str, float],
                                     volatility_score: float,
                                     liquidity_score: float) -> float:
        """Calculate confidence in timeframe selection"""
        try:
            # Base confidence on primary timeframe score
            confidence = scores[primary]
            
            # Adjust for score differential
            if len(scores) > 1:
                next_best = max(scores[tf] for tf in secondary)
                score_diff = scores[primary] - next_best
                confidence *= (1 + score_diff)
                
            # Adjust for market conditions
            condition_modifier = (volatility_score + liquidity_score) / 2
            confidence *= condition_modifier
            
            return min(0.95, max(0.1, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
            
    def _generate_selection_reasoning(self, primary: str,
                                    secondary: List[str],
                                    regime: MarketRegime,
                                    strategy: str,
                                    scores: Dict[str, float]) -> str:
        """Generate explanation for timeframe selection"""
        try:
            reasons = []
            
            # Strategy alignment
            if primary in self.strategy_timeframes.get(strategy, []):
                reasons.append(f"Optimal for {strategy} strategy")
            
            # Regime alignment
            if primary in self.regime_timeframes.get(regime, []):
                reasons.append(f"Suitable for {regime.value} regime")
                
            # Score strength
            primary_score = scores[primary]
            if primary_score > 0.8:
                reasons.append("Strong signal alignment")
            elif primary_score > 0.6:
                reasons.append("Good signal alignment")
                
            # Create explanation
            reasoning = f"Selected {primary} timeframe: {', '.join(reasons)}"
            reasoning += f"\nSecondary timeframes {', '.join(secondary)} for confirmation"
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Reasoning generation failed: {e}")
            return "Default timeframe selection based on available data"
            
    def _get_volatility_adjustment(self, timeframe: str, volatility_score: float) -> float:
        """Get timeframe adjustment based on volatility"""
        try:
            # Higher volatility favors shorter timeframes
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            
            if volatility_score > 0.7:  # High volatility
                return 1.0 / timeframe_minutes
            elif volatility_score < 0.3:  # Low volatility
                return timeframe_minutes / 1440  # Normalize to daily
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Volatility adjustment failed: {e}")
            return 0.0
            
    def _get_liquidity_adjustment(self, timeframe: str, liquidity_score: float) -> float:
        """Get timeframe adjustment based on liquidity"""
        try:
            # Lower liquidity favors longer timeframes
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            
            if liquidity_score < 0.3:  # Low liquidity
                return timeframe_minutes / 1440  # Normalize to daily
            elif liquidity_score > 0.7:  # High liquidity
                return 1.0 / timeframe_minutes
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Liquidity adjustment failed: {e}")
            return 0.0
            
    def _get_performance_adjustment(self, timeframe: str) -> float:
        """Get adjustment based on historical performance"""
        try:
            if timeframe in self.timeframe_performance:
                metrics = self.timeframe_performance[timeframe]
                
                # Calculate score based on win rate and profit factor
                win_rate = metrics.get('win_rate', 0.5)
                profit_factor = metrics.get('profit_factor', 1.0)
                
                score = (win_rate * 0.7 + min(1.0, profit_factor / 3) * 0.3)
                return score
                
            return 0.0  # No performance history
            
        except Exception as e:
            self.logger.error(f"Performance adjustment failed: {e}")
            return 0.0
            
    def _update_recommendation_history(self, recommendation: TimeframeRecommendation):
        """Update recommendation history"""
        try:
            self.recommendation_history.append({
                'timestamp': recommendation.timestamp,
                'primary': recommendation.primary_timeframe,
                'secondary': recommendation.secondary_timeframes,
                'confidence': recommendation.confidence,
                'volatility': recommendation.volatility_score,
                'liquidity': recommendation.liquidity_score
            })
            
            # Limit history size
            if len(self.recommendation_history) > 1000:
                self.recommendation_history = self.recommendation_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"History update failed: {e}")
            
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        try:
            value = int(timeframe[:-1])
            unit = timeframe[-1]
            
            if unit == 'm':
                return value
            elif unit == 'h':
                return value * 60
            elif unit == 'd':
                return value * 1440
            else:
                return 60  # Default 1 hour
                
        except Exception as e:
            self.logger.error(f"Timeframe conversion failed: {e}")
            return 60
            
    def _create_default_recommendation(self, timestamp: datetime) -> TimeframeRecommendation:
        """Create default recommendation when selection fails"""
        return TimeframeRecommendation(
            primary_timeframe='15m',
            secondary_timeframes=['5m', '30m', '1h'],
            confidence=0.5,
            reasoning="Default timeframe selection",
            volatility_score=0.5,
            liquidity_score=0.5,
            timestamp=timestamp
        )