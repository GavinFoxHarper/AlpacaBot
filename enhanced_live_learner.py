"""
Enhanced Live Market Learning System for LAEF Trading Platform
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

from laef.q_value_handler import UnifiedQValueHandler
from laef.reward_system import RiskAdjustedRewardSystem

@dataclass
class PredictionRecord:
    timestamp: datetime
    symbol: str
    action: int
    confidence: float
    q_value: float
    actual_return: float
    prediction_error: float
    market_conditions: Dict

@dataclass
class TradeOutcome:
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    action: int
    quantity: int
    raw_return: float
    fees: float
    slippage: float
    market_context: Dict

class EnhancedLiveMarketLearner:
    """Enhanced live market learning system"""
    
    def __init__(self, q_handler: UnifiedQValueHandler, reward_system: RiskAdjustedRewardSystem,
                 config: Dict):
        self.q_handler = q_handler
        self.reward_system = reward_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.95)  # Future reward discount
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.min_learning_rate = config.get('min_learning_rate', 0.0001)
        self.max_learning_rate = config.get('max_learning_rate', 0.01)
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'prediction_accuracy': 0.0,
            'avg_return': 0.0,
            'avg_prediction_error': 0.0
        }
    
    def process_trade_outcome(self, trade_outcome: TradeOutcome, prediction_record: PredictionRecord):
        """Process trade outcome and update learning"""
        try:
            # Calculate reward
            reward_metrics = self.reward_system.calculate_reward(trade_outcome)
            
            # Update performance tracking
            self._update_performance_metrics(trade_outcome, prediction_record, reward_metrics)
            
            # Update learning rate based on performance
            self._update_learning_rate()
            
            # Store prediction record
            self._store_prediction(prediction_record)
            
            # Update Q-values
            self._update_q_values(prediction_record, reward_metrics)
            
            return reward_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing trade outcome: {e}")
            return None
            
    def _update_performance_metrics(self, trade_outcome: TradeOutcome,
                                  prediction: PredictionRecord,
                                  reward_metrics: Dict):
        """Update performance tracking metrics"""
        # Update trade counts
        self.performance_metrics['total_trades'] += 1
        if trade_outcome.raw_return > 0:
            self.performance_metrics['successful_trades'] += 1
            
        # Update prediction accuracy
        predicted_direction = 1 if prediction.q_value > 0.5 else -1
        actual_direction = 1 if trade_outcome.raw_return > 0 else -1
        correct_prediction = predicted_direction == actual_direction
        
        total_predictions = len(self.prediction_history) + 1
        self.performance_metrics['prediction_accuracy'] = (
            (self.performance_metrics['prediction_accuracy'] * (total_predictions - 1) +
             (1.0 if correct_prediction else 0.0)) / total_predictions
        )
        
        # Update average return
        self.performance_metrics['avg_return'] = (
            (self.performance_metrics['avg_return'] * (total_predictions - 1) +
             trade_outcome.raw_return) / total_predictions
        )
        
        # Update prediction error
        prediction_error = abs(prediction.q_value - (trade_outcome.raw_return + 1) / 2)
        self.performance_metrics['avg_prediction_error'] = (
            (self.performance_metrics['avg_prediction_error'] * (total_predictions - 1) +
             prediction_error) / total_predictions
        )
        
    def _update_learning_rate(self):
        """Adapt learning rate based on performance"""
        # Base adjustment on prediction accuracy trend
        accuracy = self.performance_metrics['prediction_accuracy']
        error = self.performance_metrics['avg_prediction_error']
        
        if accuracy > 0.6 and error < 0.2:  # Good performance
            # Reduce learning rate to preserve good performance
            self.learning_rate *= 0.95
        elif accuracy < 0.4 or error > 0.4:  # Poor performance
            # Increase learning rate to learn faster
            self.learning_rate *= 1.05
            
        # Ensure learning rate stays within bounds
        self.learning_rate = max(self.min_learning_rate,
                              min(self.max_learning_rate, self.learning_rate))
                              
    def _store_prediction(self, prediction: PredictionRecord):
        """Store prediction for performance tracking"""
        self.prediction_history.append(prediction)
        
        # Limit history size
        max_history = self.config.get('max_prediction_history', 1000)
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
            
    def _update_q_values(self, prediction: PredictionRecord, reward_metrics: Dict):
        """Update Q-values based on actual outcome"""
        try:
            # Get next state Q-values (if available)
            next_q_value = self.q_handler.predict_q_values(
                self._get_current_state_features()
            )
            
            # Calculate target Q-value
            current_q = prediction.q_value
            reward = reward_metrics['final_reward']
            
            if next_q_value is not None:
                # Q-learning update
                target_q = reward + self.gamma * max(next_q_value)
            else:
                # Final state update
                target_q = reward
                
            # Update Q-value
            new_q = current_q + self.learning_rate * (target_q - current_q)
            
            # Apply the update
            self.q_handler.update_q_value(
                prediction.action,
                new_q,
                prediction.market_conditions
            )
            
        except Exception as e:
            self.logger.error(f"Error updating Q-values: {e}")
            
    def _get_current_state_features(self) -> np.ndarray:
        """Get current market state features"""
        # This would get current market data and convert to features
        # Stub implementation
        return np.zeros(12)  # 12 features