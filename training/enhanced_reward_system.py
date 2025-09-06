"""
Enhanced Reward System for AlpacaBot
Action-specific reward calculations with market regime awareness
Author: AlpacaBot Training System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading action enumeration."""
    HOLD = 0
    BUY = 1
    SELL = 2


class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"
    NORMAL = "normal"


@dataclass
class RewardComponents:
    """Structured representation of reward components."""
    base_reward: float
    regime_multiplier: float
    risk_penalty: float
    consistency_bonus: float
    total_reward: float
    explanation: str


class EnhancedRewardSystem:
    """
    Comprehensive reward system with action-specific calculations,
    market regime awareness, and multi-objective optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced reward system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Reward parameters
        self.action_weights = self.config['action_weights']
        self.regime_multipliers = self.config['regime_multipliers']
        self.risk_parameters = self.config['risk_parameters']
        
        # Performance tracking
        self.reward_history = []
        self.action_distribution = {Action.BUY: 0, Action.SELL: 0, Action.HOLD: 0}
        self.regime_performance = {regime: [] for regime in MarketRegime}
        
        logger.info("Enhanced Reward System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for reward system.
        
        Returns:
            Configuration dictionary
        """
        return {
            'action_weights': {
                Action.BUY: {
                    'profit_weight': 1.0,
                    'timing_weight': 0.3,
                    'risk_weight': 0.2
                },
                Action.SELL: {
                    'profit_weight': 1.2,  # Higher weight for sell signals (harder to learn)
                    'timing_weight': 0.4,
                    'risk_weight': 0.3
                },
                Action.HOLD: {
                    'stability_weight': 0.5,
                    'opportunity_cost_weight': 0.3
                }
            },
            'regime_multipliers': {
                MarketRegime.BULL: {
                    Action.BUY: 1.0,
                    Action.SELL: 0.8,
                    Action.HOLD: 0.9
                },
                MarketRegime.BEAR: {
                    Action.BUY: 0.7,
                    Action.SELL: 1.3,
                    Action.HOLD: 1.0
                },
                MarketRegime.CRASH: {
                    Action.BUY: 0.5,
                    Action.SELL: 1.5,
                    Action.HOLD: 1.1
                },
                MarketRegime.RECOVERY: {
                    Action.BUY: 1.4,
                    Action.SELL: 0.6,
                    Action.HOLD: 0.8
                },
                MarketRegime.HIGH_VOLATILITY: {
                    Action.BUY: 0.9,
                    Action.SELL: 1.1,
                    Action.HOLD: 1.2
                },
                MarketRegime.LOW_VOLATILITY: {
                    Action.BUY: 1.0,
                    Action.SELL: 1.0,
                    Action.HOLD: 0.8
                },
                MarketRegime.SIDEWAYS: {
                    Action.BUY: 0.8,
                    Action.SELL: 0.8,
                    Action.HOLD: 1.3
                },
                MarketRegime.NORMAL: {
                    Action.BUY: 1.0,
                    Action.SELL: 1.0,
                    Action.HOLD: 1.0
                }
            },
            'risk_parameters': {
                'max_drawdown_penalty': 0.5,
                'volatility_threshold': 0.02,
                'confidence_threshold': 0.6,
                'position_size_factor': 0.1
            },
            'consistency_parameters': {
                'streak_bonus_factor': 0.1,
                'min_streak_length': 3,
                'diversity_bonus': 0.05
            }
        }
    
    def calculate_enhanced_reward(self,
                                 action: int,
                                 actual_return: float,
                                 market_regime: str,
                                 volatility: float,
                                 confidence: float,
                                 position_size: float = 1.0,
                                 previous_actions: List[int] = None,
                                 portfolio_value: float = 10000.0,
                                 max_drawdown: float = 0.0) -> RewardComponents:
        """
        Calculate comprehensive reward with all enhancements.
        
        Args:
            action: Trading action (0: hold, 1: buy, 2: sell)
            actual_return: Actual return from the action
            market_regime: Current market regime
            volatility: Current market volatility
            confidence: Model confidence in the action
            position_size: Size of the position
            previous_actions: List of previous actions for consistency check
            portfolio_value: Current portfolio value
            max_drawdown: Current maximum drawdown
            
        Returns:
            RewardComponents with detailed breakdown
        """
        # Convert to enums
        action_enum = Action(action)
        regime_enum = MarketRegime(market_regime)
        
        # Track action distribution
        self.action_distribution[action_enum] += 1
        
        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action_enum, actual_return, volatility, position_size
        )
        
        # Apply market regime multiplier
        regime_multiplier = self._get_regime_multiplier(action_enum, regime_enum)
        
        # Calculate risk penalty
        risk_penalty = self._calculate_risk_penalty(
            volatility, confidence, max_drawdown, portfolio_value
        )
        
        # Calculate consistency bonus
        consistency_bonus = self._calculate_consistency_bonus(
            action, previous_actions, actual_return
        )
        
        # Calculate total reward
        total_reward = (base_reward * regime_multiplier * (1 - risk_penalty)) + consistency_bonus
        
        # Generate explanation
        explanation = self._generate_reward_explanation(
            action_enum, actual_return, regime_enum, 
            base_reward, regime_multiplier, risk_penalty, consistency_bonus
        )
        
        # Create reward components
        reward_components = RewardComponents(
            base_reward=base_reward,
            regime_multiplier=regime_multiplier,
            risk_penalty=risk_penalty,
            consistency_bonus=consistency_bonus,
            total_reward=total_reward,
            explanation=explanation
        )
        
        # Track reward history
        self.reward_history.append(reward_components)
        self.regime_performance[regime_enum].append(total_reward)
        
        return reward_components
    
    def _calculate_base_reward(self,
                              action: Action,
                              actual_return: float,
                              volatility: float,
                              position_size: float) -> float:
        """
        Calculate action-specific base reward.
        
        Args:
            action: Trading action
            actual_return: Actual return
            volatility: Market volatility
            position_size: Position size
            
        Returns:
            Base reward value
        """
        if action == Action.BUY:
            # Reward profitable buys, penalize losses
            if actual_return > 0:
                # Profitable buy
                base_reward = actual_return * position_size
                # Bonus for buying in low volatility (better entry)
                if volatility < self.config['risk_parameters']['volatility_threshold']:
                    base_reward *= 1.2
            else:
                # Loss on buy
                base_reward = actual_return * position_size * 1.5  # Higher penalty for losses
        
        elif action == Action.SELL:
            # Reward profitable sells (avoiding losses)
            if actual_return < 0:
                # Good sell (avoided loss)
                base_reward = abs(actual_return) * position_size * 1.3  # Higher reward for good sells
                # Bonus for selling in high volatility (risk avoidance)
                if volatility > self.config['risk_parameters']['volatility_threshold']:
                    base_reward *= 1.2
            else:
                # Missed opportunity (sold when would have gained)
                base_reward = -actual_return * position_size * 0.7  # Lower penalty for conservative sells
        
        else:  # HOLD
            # Reward holding during high volatility or small movements
            if abs(actual_return) < 0.01:
                # Good hold (avoided noise)
                base_reward = 0.002 * position_size
            elif volatility > self.config['risk_parameters']['volatility_threshold'] * 1.5:
                # Good hold during high volatility
                base_reward = 0.003 * position_size
            else:
                # Opportunity cost of holding
                base_reward = -abs(actual_return) * position_size * 0.3
        
        return base_reward
    
    def _get_regime_multiplier(self, action: Action, regime: MarketRegime) -> float:
        """
        Get regime-specific action multiplier.
        
        Args:
            action: Trading action
            regime: Market regime
            
        Returns:
            Regime multiplier
        """
        return self.regime_multipliers[regime][action]
    
    def _calculate_risk_penalty(self,
                               volatility: float,
                               confidence: float,
                               max_drawdown: float,
                               portfolio_value: float) -> float:
        """
        Calculate risk-adjusted penalty.
        
        Args:
            volatility: Market volatility
            confidence: Model confidence
            max_drawdown: Maximum drawdown
            portfolio_value: Portfolio value
            
        Returns:
            Risk penalty (0 to 1)
        """
        penalty = 0.0
        
        # Volatility penalty
        vol_threshold = self.config['risk_parameters']['volatility_threshold']
        if volatility > vol_threshold * 2:
            penalty += 0.2 * (volatility / (vol_threshold * 2) - 1)
        
        # Low confidence penalty
        conf_threshold = self.config['risk_parameters']['confidence_threshold']
        if confidence < conf_threshold:
            penalty += 0.15 * (1 - confidence / conf_threshold)
        
        # Drawdown penalty
        max_dd_penalty = self.config['risk_parameters']['max_drawdown_penalty']
        if abs(max_drawdown) > 0.1:  # More than 10% drawdown
            penalty += max_dd_penalty * min(abs(max_drawdown) / 0.3, 1.0)
        
        # Portfolio risk (position too large relative to portfolio)
        position_risk = 1.0 / portfolio_value if portfolio_value > 0 else 0
        if position_risk > 0.1:  # Position > 10% of portfolio
            penalty += 0.1 * (position_risk - 0.1) / 0.1
        
        return min(penalty, 0.8)  # Cap penalty at 80%
    
    def _calculate_consistency_bonus(self,
                                    current_action: int,
                                    previous_actions: List[int],
                                    actual_return: float) -> float:
        """
        Calculate bonus for consistent profitable actions.
        
        Args:
            current_action: Current action
            previous_actions: List of previous actions
            actual_return: Actual return
            
        Returns:
            Consistency bonus
        """
        if not previous_actions or len(previous_actions) < 3:
            return 0.0
        
        bonus = 0.0
        
        # Check for profitable streak
        min_streak = self.config['consistency_parameters']['min_streak_length']
        streak_bonus = self.config['consistency_parameters']['streak_bonus_factor']
        
        # Count consecutive successful actions
        if actual_return > 0 and current_action == Action.BUY.value:
            # Check for buy streak
            buy_streak = sum(1 for a in previous_actions[-min_streak:] if a == Action.BUY.value)
            if buy_streak >= min_streak:
                bonus += streak_bonus * buy_streak
        
        elif actual_return < 0 and current_action == Action.SELL.value:
            # Check for sell streak (successful risk avoidance)
            sell_streak = sum(1 for a in previous_actions[-min_streak:] if a == Action.SELL.value)
            if sell_streak >= min_streak:
                bonus += streak_bonus * sell_streak * 1.5  # Higher bonus for sell streaks
        
        # Diversity bonus (using all three actions effectively)
        recent_actions = previous_actions[-10:] if len(previous_actions) >= 10 else previous_actions
        unique_actions = len(set(recent_actions))
        if unique_actions == 3:
            bonus += self.config['consistency_parameters']['diversity_bonus']
        
        return bonus
    
    def _generate_reward_explanation(self,
                                    action: Action,
                                    actual_return: float,
                                    regime: MarketRegime,
                                    base_reward: float,
                                    regime_multiplier: float,
                                    risk_penalty: float,
                                    consistency_bonus: float) -> str:
        """
        Generate human-readable explanation of reward calculation.
        
        Args:
            action: Trading action
            actual_return: Actual return
            regime: Market regime
            base_reward: Base reward value
            regime_multiplier: Regime multiplier
            risk_penalty: Risk penalty
            consistency_bonus: Consistency bonus
            
        Returns:
            Explanation string
        """
        action_str = action.name
        regime_str = regime.value
        
        explanation = f"Action: {action_str}, Return: {actual_return:.4f}, Regime: {regime_str}\n"
        explanation += f"Base Reward: {base_reward:.4f}\n"
        explanation += f"Regime Multiplier: {regime_multiplier:.2f}\n"
        explanation += f"Risk Penalty: {risk_penalty:.2f}\n"
        explanation += f"Consistency Bonus: {consistency_bonus:.4f}\n"
        
        # Add action-specific explanation
        if action == Action.BUY:
            if actual_return > 0:
                explanation += "Successful buy - profit captured"
            else:
                explanation += "Failed buy - loss incurred"
        elif action == Action.SELL:
            if actual_return < 0:
                explanation += "Successful sell - loss avoided"
            else:
                explanation += "Premature sell - missed opportunity"
        else:
            if abs(actual_return) < 0.01:
                explanation += "Good hold - avoided noise trading"
            else:
                explanation += "Questionable hold - missed opportunity"
        
        return explanation
    
    def get_action_balance_reward(self, action_counts: Dict[int, int]) -> float:
        """
        Calculate reward adjustment for action balance.
        
        Args:
            action_counts: Dictionary of action counts
            
        Returns:
            Balance reward adjustment
        """
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return 0.0
        
        # Target distribution
        target_distribution = {0: 0.34, 1: 0.33, 2: 0.33}
        
        # Calculate distribution distance
        actual_distribution = {
            action: count / total_actions 
            for action, count in action_counts.items()
        }
        
        # Calculate KL divergence
        kl_divergence = 0.0
        for action in range(3):
            if action in actual_distribution and actual_distribution[action] > 0:
                kl_divergence += target_distribution[action] * np.log(
                    target_distribution[action] / actual_distribution[action]
                )
        
        # Convert to reward (inverse of divergence)
        balance_reward = max(0, 1.0 - kl_divergence)
        
        # Bonus for balanced distribution
        if all(0.25 < actual_distribution.get(i, 0) < 0.4 for i in range(3)):
            balance_reward += 0.1
        
        return balance_reward
    
    def update_reward_for_episode_end(self,
                                     episode_return: float,
                                     sharpe_ratio: float,
                                     max_drawdown: float,
                                     win_rate: float) -> float:
        """
        Calculate episode-end reward adjustment based on overall performance.
        
        Args:
            episode_return: Total episode return
            sharpe_ratio: Sharpe ratio for the episode
            max_drawdown: Maximum drawdown
            win_rate: Winning trade percentage
            
        Returns:
            Episode-end reward adjustment
        """
        episode_reward = 0.0
        
        # Return component
        if episode_return > 0:
            episode_reward += episode_return * 10
        else:
            episode_reward += episode_return * 5  # Less penalty for losses
        
        # Sharpe ratio component
        if sharpe_ratio > 1.0:
            episode_reward += (sharpe_ratio - 1.0) * 5
        elif sharpe_ratio < 0:
            episode_reward -= abs(sharpe_ratio) * 2
        
        # Drawdown component
        if abs(max_drawdown) < 0.1:
            episode_reward += 2.0
        elif abs(max_drawdown) > 0.2:
            episode_reward -= (abs(max_drawdown) - 0.2) * 10
        
        # Win rate component
        if win_rate > 0.6:
            episode_reward += (win_rate - 0.6) * 10
        elif win_rate < 0.4:
            episode_reward -= (0.4 - win_rate) * 5
        
        # Action balance component
        action_counts = {
            0: self.action_distribution[Action.HOLD],
            1: self.action_distribution[Action.BUY],
            2: self.action_distribution[Action.SELL]
        }
        balance_reward = self.get_action_balance_reward(action_counts)
        episode_reward += balance_reward * 5
        
        return episode_reward
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive reward statistics.
        
        Returns:
            Dictionary of reward statistics
        """
        if not self.reward_history:
            return {}
        
        total_rewards = [r.total_reward for r in self.reward_history]
        base_rewards = [r.base_reward for r in self.reward_history]
        risk_penalties = [r.risk_penalty for r in self.reward_history]
        
        # Action distribution percentages
        total_actions = sum(self.action_distribution.values())
        action_percentages = {
            action.name: (count / total_actions * 100) if total_actions > 0 else 0
            for action, count in self.action_distribution.items()
        }
        
        # Regime performance averages
        regime_avg_rewards = {}
        for regime, rewards in self.regime_performance.items():
            if rewards:
                regime_avg_rewards[regime.value] = np.mean(rewards)
        
        return {
            'total_rewards': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'min': np.min(total_rewards),
                'max': np.max(total_rewards),
                'cumulative': np.sum(total_rewards)
            },
            'base_rewards': {
                'mean': np.mean(base_rewards),
                'std': np.std(base_rewards)
            },
            'risk_penalties': {
                'mean': np.mean(risk_penalties),
                'max': np.max(risk_penalties)
            },
            'action_distribution': action_percentages,
            'regime_performance': regime_avg_rewards,
            'reward_count': len(self.reward_history)
        }
    
    def reset_statistics(self):
        """Reset all tracking statistics."""
        self.reward_history = []
        self.action_distribution = {Action.BUY: 0, Action.SELL: 0, Action.HOLD: 0}
        self.regime_performance = {regime: [] for regime in MarketRegime}
        logger.info("Reward statistics reset")


def test_reward_system():
    """Test the enhanced reward system with various scenarios."""
    reward_system = EnhancedRewardSystem()
    
    # Test scenarios
    test_cases = [
        # (action, return, regime, volatility, confidence, description)
        (1, 0.02, "bull", 0.01, 0.8, "Profitable buy in bull market"),
        (2, -0.015, "bear", 0.03, 0.7, "Good sell avoiding loss in bear market"),
        (0, 0.001, "sideways", 0.005, 0.6, "Hold during low volatility"),
        (2, -0.03, "crash", 0.05, 0.9, "Excellent sell during crash"),
        (1, -0.02, "bear", 0.04, 0.5, "Bad buy in bear market"),
        (1, 0.03, "recovery", 0.02, 0.85, "Great buy during recovery"),
    ]
    
    print("\n" + "="*80)
    print("ENHANCED REWARD SYSTEM TEST RESULTS")
    print("="*80)
    
    previous_actions = []
    for action, return_val, regime, vol, conf, desc in test_cases:
        reward = reward_system.calculate_enhanced_reward(
            action=action,
            actual_return=return_val,
            market_regime=regime,
            volatility=vol,
            confidence=conf,
            previous_actions=previous_actions[-5:] if previous_actions else None
        )
        
        print(f"\nScenario: {desc}")
        print(f"  Total Reward: {reward.total_reward:.4f}")
        print(f"  Base Reward: {reward.base_reward:.4f}")
        print(f"  Regime Multiplier: {reward.regime_multiplier:.2f}")
        print(f"  Risk Penalty: {reward.risk_penalty:.2f}")
        print(f"  Consistency Bonus: {reward.consistency_bonus:.4f}")
        
        previous_actions.append(action)
    
    # Print statistics
    stats = reward_system.get_reward_statistics()
    print("\n" + "="*80)
    print("REWARD SYSTEM STATISTICS")
    print("="*80)
    print(f"Total Rewards - Mean: {stats['total_rewards']['mean']:.4f}, Cumulative: {stats['total_rewards']['cumulative']:.4f}")
    print(f"Action Distribution: {stats['action_distribution']}")
    print(f"Regime Performance: {stats['regime_performance']}")
    print("="*80)


if __name__ == "__main__":
    test_reward_system()