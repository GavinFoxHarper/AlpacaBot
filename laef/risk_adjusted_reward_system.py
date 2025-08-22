"""
Risk-Adjusted Reward System for LAEF Trading Platform
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class RewardType(Enum):
    SIMPLE_RETURN = "simple_return"
    SHARPE_ADJUSTED = "sharpe_adjusted"
    SORTINO_ADJUSTED = "sortino_adjusted"
    DRAWDOWN_PENALIZED = "drawdown_penalized"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    KELLY_OPTIMIZED = "kelly_optimized"

@dataclass
class TradeOutcome:
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    action: int  # 1=buy, 2=sell
    quantity: int
    raw_return: float
    fees: float
    slippage: float
    market_context: Dict

@dataclass
class RewardMetrics:
    raw_reward: float
    risk_adjusted_reward: float
    volatility_penalty: float
    drawdown_penalty: float
    consistency_bonus: float
    timing_bonus: float
    market_regime_adjustment: float
    final_reward: float
    components: Dict[str, float]

class RiskAdjustedRewardSystem:
    """Risk-adjusted reward calculation system"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Reward method configuration
        self.reward_type = RewardType(config.get('reward_type', 'sharpe_adjusted'))
        self.base_reward_weight = config.get('base_reward_weight', 1.0)
        
        # Risk penalties
        self.volatility_penalty_factor = config.get('volatility_penalty_factor', 0.2)
        self.drawdown_penalty_factor = config.get('drawdown_penalty_factor', 0.5)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.05)
        
        # Consistency rewards
        self.consistency_window = config.get('consistency_window', 10)
        self.consistency_bonus_factor = config.get('consistency_bonus_factor', 0.1)
        
        # Timing rewards
        self.timing_bonus_enabled = config.get('timing_bonus_enabled', True)
        self.market_timing_weight = config.get('market_timing_weight', 0.15)
        
        # Market regime adjustments
        self.regime_adjustments = config.get('regime_adjustments', {
            'trending_up': 1.1,
            'trending_down': 0.9,
            'ranging': 1.0,
            'volatile': 0.8,
            'low_volatility': 1.05
        })
        
        # Performance tracking
        self.performance_history = []
        self.reward_history = []
        
        # Risk-free rate (for Sharpe ratio calculation)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        
    def calculate_reward(self, trade_outcome: TradeOutcome) -> RewardMetrics:
        """Calculate risk-adjusted reward for a trade outcome"""
        try:
            # Calculate base reward
            raw_reward = self._calculate_base_reward(trade_outcome)
            
            # Calculate risk adjustments
            volatility_penalty = self._calculate_volatility_penalty(trade_outcome)
            drawdown_penalty = self._calculate_drawdown_penalty(trade_outcome)
            
            # Calculate bonuses
            consistency_bonus = self._calculate_consistency_bonus(trade_outcome)
            timing_bonus = self._calculate_timing_bonus(trade_outcome)
            
            # Market regime adjustment
            regime_adjustment = self._calculate_regime_adjustment(trade_outcome)
            
            # Combine all components
            risk_adjusted_reward = self._combine_reward_components(
                raw_reward, volatility_penalty, drawdown_penalty,
                consistency_bonus, timing_bonus, regime_adjustment
            )
            
            # Create reward metrics
            metrics = RewardMetrics(
                raw_reward=raw_reward,
                risk_adjusted_reward=risk_adjusted_reward,
                volatility_penalty=volatility_penalty,
                drawdown_penalty=drawdown_penalty,
                consistency_bonus=consistency_bonus,
                timing_bonus=timing_bonus,
                market_regime_adjustment=regime_adjustment,
                final_reward=risk_adjusted_reward,
                components={
                    'raw_reward': raw_reward,
                    'volatility_penalty': -volatility_penalty,
                    'drawdown_penalty': -drawdown_penalty,
                    'consistency_bonus': consistency_bonus,
                    'timing_bonus': timing_bonus,
                    'regime_adjustment': regime_adjustment
                }
            )
            
            # Update performance history
            self._update_performance_history(trade_outcome, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return self._create_default_metrics(trade_outcome.raw_return)
            
    def _calculate_base_reward(self, trade_outcome: TradeOutcome) -> float:
        """Calculate the base reward component"""
        
        if self.reward_type == RewardType.SIMPLE_RETURN:
            return trade_outcome.raw_return
            
        elif self.reward_type == RewardType.SHARPE_ADJUSTED:
            # Adjust return by risk-free rate
            risk_premium = trade_outcome.raw_return - (self.risk_free_rate / 252)  # Daily risk-free rate
            return risk_premium
            
        elif self.reward_type == RewardType.SORTINO_ADJUSTED:
            # Only penalize downside volatility
            if trade_outcome.raw_return < 0:
                return trade_outcome.raw_return * 1.5  # Increase penalty for losses
            else:
                return trade_outcome.raw_return
                
        elif self.reward_type == RewardType.DRAWDOWN_PENALIZED:
            # Base reward with immediate drawdown consideration
            base = trade_outcome.raw_return
            if base < 0:
                base *= (1 + self.drawdown_penalty_factor)
            return base
            
        elif self.reward_type == RewardType.VOLATILITY_ADJUSTED:
            # Adjust by estimated volatility
            volatility = trade_outcome.market_context.get('volatility', 0.2)
            return trade_outcome.raw_return / (1 + volatility)
            
        elif self.reward_type == RewardType.KELLY_OPTIMIZED:
            # Kelly criterion-inspired adjustment
            win_prob = self._estimate_win_probability(trade_outcome)
            kelly_fraction = self._calculate_kelly_fraction(trade_outcome, win_prob)
            return trade_outcome.raw_return * kelly_fraction
            
        else:
            return trade_outcome.raw_return
            
    def _calculate_volatility_penalty(self, trade_outcome: TradeOutcome) -> float:
        """Calculate penalty based on volatility"""
        volatility = trade_outcome.market_context.get('volatility', 0.2)
        
        # Apply penalty if volatility is above threshold
        vol_threshold = self.config.get('volatility_threshold', 0.3)
        if volatility > vol_threshold:
            excess_vol = volatility - vol_threshold
            penalty = excess_vol * self.volatility_penalty_factor * abs(trade_outcome.raw_return)
            return penalty
            
        return 0.0
        
    def _calculate_drawdown_penalty(self, trade_outcome: TradeOutcome) -> float:
        """Calculate penalty based on drawdown contribution"""
        if trade_outcome.raw_return < 0:
            # Simple penalty for losses
            return abs(trade_outcome.raw_return) * self.drawdown_penalty_factor
        return 0.0
        
    def _calculate_consistency_bonus(self, trade_outcome: TradeOutcome) -> float:
        """Calculate bonus for consistent performance"""
        if len(self.performance_history) < self.consistency_window:
            return 0.0
            
        # Look at recent performance
        recent_returns = [p['return'] for p in self.performance_history[-self.consistency_window:]]
        positive_trades = sum(1 for r in recent_returns if r > 0)
        consistency_ratio = positive_trades / len(recent_returns)
        
        # Reward high consistency
        if consistency_ratio > 0.7:  # 70% win rate
            bonus = self.consistency_bonus_factor * abs(trade_outcome.raw_return)
            return bonus
            
        return 0.0
        
    def _calculate_timing_bonus(self, trade_outcome: TradeOutcome) -> float:
        """Calculate bonus for good market timing"""
        if not self.timing_bonus_enabled:
            return 0.0
            
        # Time-based factors
        trade_duration = (trade_outcome.exit_time - trade_outcome.entry_time).total_seconds() / 60  # minutes
        
        # Bonus for quick profitable trades (scalping)
        if trade_outcome.raw_return > 0 and trade_duration < 60:  # Less than 1 hour
            bonus = self.market_timing_weight * trade_outcome.raw_return
            return bonus
            
        return 0.0
        
    def _calculate_regime_adjustment(self, trade_outcome: TradeOutcome) -> float:
        """Adjust reward based on market regime"""
        market_regime = trade_outcome.market_context.get('market_regime', 'ranging')
        adjustment_factor = self.regime_adjustments.get(market_regime, 1.0)
        
        # Apply adjustment to the absolute reward
        adjustment = (adjustment_factor - 1.0) * abs(trade_outcome.raw_return)
        
        # Only apply positive adjustments for profitable trades
        if trade_outcome.raw_return > 0:
            return adjustment
        elif trade_outcome.raw_return < 0 and adjustment_factor < 1.0:
            # Apply negative adjustment (increase penalty) for losses in difficult regimes
            return adjustment
            
        return 0.0
        
    def _combine_reward_components(self, raw_reward: float, volatility_penalty: float,
                                drawdown_penalty: float, consistency_bonus: float,
                                timing_bonus: float, regime_adjustment: float) -> float:
        """Combine all reward components into final risk-adjusted reward"""
        
        # Start with base reward
        adjusted_reward = raw_reward * self.base_reward_weight
        
        # Apply penalties (subtract)
        adjusted_reward -= volatility_penalty
        adjusted_reward -= drawdown_penalty
        
        # Apply bonuses (add)
        adjusted_reward += consistency_bonus
        adjusted_reward += timing_bonus
        adjusted_reward += regime_adjustment
        
        # Ensure the reward doesn't exceed reasonable bounds
        max_reward = abs(raw_reward) * 2  # Maximum 2x the raw reward
        min_reward = raw_reward * 3 if raw_reward < 0 else raw_reward * 0.1  # Minimum 10% of raw reward for profits
        
        return max(min_reward, min(max_reward, adjusted_reward))
        
    def _estimate_win_probability(self, trade_outcome: TradeOutcome) -> float:
        """Estimate win probability based on recent performance"""
        if len(self.performance_history) < 10:
            return 0.5  # Default 50%
            
        recent_wins = sum(1 for p in self.performance_history[-20:] if p['return'] > 0)
        return recent_wins / min(20, len(self.performance_history))
        
    def _calculate_kelly_fraction(self, trade_outcome: TradeOutcome, win_prob: float) -> float:
        """Calculate Kelly criterion fraction"""
        if win_prob <= 0 or win_prob >= 1:
            return 0.5
            
        # Simplified Kelly calculation
        avg_win = np.mean([p['return'] for p in self.performance_history if p['return'] > 0]) or 0.01
        avg_loss = np.mean([abs(p['return']) for p in self.performance_history if p['return'] < 0]) or 0.01
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Conservative Kelly (use fraction of Kelly)
        return max(0.1, min(0.9, kelly_fraction * 0.25))
        
    def _update_performance_history(self, trade_outcome: TradeOutcome, metrics: RewardMetrics):
        """Update performance history for tracking"""
        self.performance_history.append({
            'timestamp': trade_outcome.exit_time,
            'return': trade_outcome.raw_return,
            'risk_adjusted_return': metrics.final_reward,
            'action': trade_outcome.action
        })
        
        self.reward_history.append(metrics.final_reward)
        
        # Limit history size
        max_history = self.config.get('max_performance_history', 1000)
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            self.reward_history = self.reward_history[-max_history:]
            
    def _create_default_metrics(self, raw_return: float) -> RewardMetrics:
        """Create default metrics for error cases"""
        return RewardMetrics(
            raw_reward=raw_return,
            risk_adjusted_reward=raw_return,
            volatility_penalty=0.0,
            drawdown_penalty=0.0,
            consistency_bonus=0.0,
            timing_bonus=0.0,
            market_regime_adjustment=0.0,
            final_reward=raw_return,
            components={'raw_reward': raw_return}
        )