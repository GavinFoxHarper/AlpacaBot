"""
Enhanced Q-Learning Agent with Fixed Reward Structure
Addresses critical training biases and implements balanced action generation
Author: AlpacaBot Training System
Date: 2024
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import deque, defaultdict

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import configurations and reward system
try:
    from config import STATE_SIZE, LEARNING_RATE, MODEL_PATH
except ImportError:
    STATE_SIZE = 15
    LEARNING_RATE = 0.001
    MODEL_PATH = 'models/enhanced_q_agent.h5'

try:
    from training.enhanced_reward_system import EnhancedRewardSystem
except ImportError:
    class EnhancedRewardSystem:
        def calculate_enhanced_reward(self, *args, **kwargs):
            return type('obj', (object,), {'total_reward': kwargs.get('actual_return', 0.0)})
        def get_reward_statistics(self):
            return {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQLearningAgent:
    """
    Enhanced Q-Learning Agent with comprehensive fixes for training biases.
    
    Key improvements:
    - Action-specific reward calculations
    - Market regime awareness  
    - Balanced sell signal generation
    - Experience replay with prioritization
    - Comprehensive performance tracking
    """
    
    def __init__(self, 
                 state_size=None, 
                 model_path=None, 
                 config=None):
        
        self.state_size = state_size or STATE_SIZE
        self.model_path = model_path or MODEL_PATH
        self.config = config or self._get_default_config()
        
        # Enhanced reward system
        self.reward_system = EnhancedRewardSystem(self.config.get('reward_config', {}))
        
        # Experience replay components
        self.memory = deque(maxlen=self.config['memory_size'])
        self.priority_weights = deque(maxlen=self.config['memory_size'])
        
        # Performance tracking
        self.training_history = []
        self.action_distribution = defaultdict(int)
        self.reward_history = []
        self.regime_performance = defaultdict(list)
        
        # Training parameters
        self.epsilon = self.config['epsilon_start']
        self.epsilon_decay = self.config['epsilon_decay']
        self.epsilon_min = self.config['epsilon_min']
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.training_step = 0
        
        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        logger.info(f"Enhanced Q-Learning Agent initialized: state_size={self.state_size}")
        logger.info(f"Memory size: {self.config['memory_size']}, Epsilon: {self.epsilon}")
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'memory_size': 50000,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.01,
            'gamma': 0.95,
            'batch_size': 32,
            'target_update_frequency': 100,
            'learning_rate': LEARNING_RATE,
            'hidden_layers': [128, 64, 32],
            'dropout_rates': [0.3, 0.2, 0.1],
            'use_batch_norm': True,
            'sell_signal_boost': 2.0,
            'action_balance_threshold': 0.15,  # Minimum % for each action
            'regime_adjustment_factor': 0.2
        }
    
    def _build_model(self):
        """Build enhanced neural network."""
        layers = [Input(shape=(self.state_size,))]
        
        # Hidden layers with batch norm and dropout
        for i, (units, dropout) in enumerate(zip(self.config['hidden_layers'], self.config['dropout_rates'])):
            layers.append(Dense(units, activation='relu', name=f'dense_{i+1}'))
            if self.config['use_batch_norm']:
                layers.append(BatchNormalization())
            layers.append(Dropout(dropout))
        
        # Output layer: 3 Q-values [hold, buy, sell]
        layers.append(Dense(3, activation='linear', name='q_output'))
        
        model = Sequential(layers)
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        return model
    
    def update_target_model(self):
        """Update target network weights."""
        self.target_model.set_weights(self.model.get_weights())
    
    def predict_q_values(self, state, market_regime='normal'):
        """
        Predict Q-values with regime adjustments and sell signal boosting.
        
        Args:
            state: Input state array
            market_regime: Current market regime
            
        Returns:
            Array of Q-values [hold, buy, sell]
        """
        try:
            # Validate and prepare state
            state = np.array(state)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            # Basic validation
            if state.shape[1] != self.state_size:
                logger.error(f"State size mismatch: {state.shape[1]} vs {self.state_size}")
                return np.array([0.0, 0.0, 0.0])
            
            # Handle invalid values
            state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
            state = np.clip(state, -10, 10)
            
            # Get base Q-values
            q_values = self.model.predict(state, verbose=0)[0]
            
            # Apply regime adjustments
            q_values = self._apply_regime_adjustments(q_values, market_regime)
            
            # Apply sell signal boosting
            q_values = self._apply_sell_signal_boost(q_values, state[0])
            
            # Apply action balancing
            q_values = self._apply_action_balancing(q_values)
            
            return np.array(q_values, dtype=float)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _apply_regime_adjustments(self, q_values, market_regime):
        """Apply market regime-specific adjustments."""
        regime_multipliers = {
            'bull': [0.9, 1.1, 0.7],      # [hold, buy, sell] - favor buy
            'bear': [1.0, 0.6, 1.4],      # favor sell
            'crash': [1.1, 0.4, 1.6],     # strongly favor sell
            'recovery': [0.8, 1.5, 0.5],  # strongly favor buy
            'high_volatility': [1.2, 0.8, 1.2],  # favor hold and sell
            'low_volatility': [0.8, 1.1, 1.0],   # slight buy bias
            'sideways': [1.4, 0.7, 0.7],  # strongly favor hold
            'normal': [1.0, 1.0, 1.0]     # no adjustment
        }
        
        multipliers = regime_multipliers.get(market_regime, [1.0, 1.0, 1.0])
        adjustment_factor = self.config['regime_adjustment_factor']
        
        # Apply adjustments gradually
        adjusted_multipliers = [
            1.0 + (m - 1.0) * adjustment_factor for m in multipliers
        ]
        
        return q_values * np.array(adjusted_multipliers)
    
    def _apply_sell_signal_boost(self, q_values, state):
        """Boost sell signals to address training bias."""
        boost_factor = self.config['sell_signal_boost']
        
        # Technical condition boosting
        if len(state) >= 5:
            # Assume state contains technical indicators
            rsi_like = state[0] if len(state) > 0 else 0.5
            volatility_like = state[1] if len(state) > 1 else 0.02
            momentum_like = state[2] if len(state) > 2 else 0.0
            
            # Boost sell in potentially overbought conditions
            if rsi_like > 0.7:  # High RSI-like indicator
                q_values[2] *= boost_factor
            
            # Boost sell in high volatility
            if volatility_like > 0.04:
                q_values[2] *= 1.5
            
            # Boost sell on negative momentum
            if momentum_like < -0.05:
                q_values[2] *= 1.3
        
        return q_values
    
    def _apply_action_balancing(self, q_values):
        """Apply dynamic balancing to ensure diverse actions."""
        total_actions = sum(self.action_distribution.values())
        
        if total_actions > 100:  # Only after sufficient experience
            # Calculate action ratios
            action_ratios = []
            for action in range(3):
                ratio = self.action_distribution.get(action, 0) / total_actions
                action_ratios.append(ratio)
            
            # Find underrepresented actions
            threshold = self.config['action_balance_threshold']
            
            for action, ratio in enumerate(action_ratios):
                if ratio < threshold:
                    # Boost underrepresented actions
                    boost = 1.0 + (threshold - ratio) * 3.0
                    q_values[action] *= boost
                    
                    if action == 2:  # Extra boost for sell signals
                        q_values[action] *= 1.5
        
        return q_values
    
    def select_action(self, state, market_regime='normal', use_epsilon_greedy=True):
        """
        Select action with epsilon-greedy exploration and comprehensive tracking.
        
        Args:
            state: Current state
            market_regime: Market regime
            use_epsilon_greedy: Use exploration
            
        Returns:
            tuple: (action, metadata)
        """
        try:
            # Epsilon-greedy exploration
            if use_epsilon_greedy and np.random.random() < self.epsilon:
                # Weighted exploration to encourage sell exploration
                action_weights = [0.25, 0.35, 0.4]  # Favor sell in exploration
                action = np.random.choice(3, p=action_weights)
                method = 'exploration'
                q_values = [0.0, 0.0, 0.0]
            else:
                # Exploitation
                q_values = self.predict_q_values(state, market_regime)
                action = int(np.argmax(q_values))
                method = 'exploitation'
                q_values = q_values.tolist()
            
            # Update tracking
            self.action_distribution[action] += 1
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Create metadata
            metadata = {
                'method': method,
                'epsilon': self.epsilon,
                'q_values': q_values,
                'market_regime': market_regime,
                'action_distribution': dict(self.action_distribution)
            }
            
            return action, metadata
            
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            return 0, {'method': 'fallback', 'error': str(e)}
    
    def remember(self, state, action, reward, next_state, done, **kwargs):
        """
        Store experience with enhanced reward calculation.
        
        Args:
            state: Current state
            action: Action taken (0=hold, 1=buy, 2=sell)
            reward: Environment reward
            next_state: Next state
            done: Episode done flag
            **kwargs: Additional parameters for reward calculation
        """
        try:
            # Calculate enhanced reward
            market_regime = kwargs.get('market_regime', 'normal')
            
            # Convert action to reward system format (-1, 0, 1)
            reward_action = action - 1
            
            enhanced_reward_components = self.reward_system.calculate_enhanced_reward(
                action=reward_action,
                actual_return=reward,
                market_regime=market_regime,
                volatility=kwargs.get('volatility', 0.02),
                confidence=kwargs.get('confidence', 0.5),
                position_size=kwargs.get('position_size', 1.0),
                previous_actions=kwargs.get('previous_actions', []),
                portfolio_value=kwargs.get('portfolio_value', 10000.0),
                max_drawdown=kwargs.get('max_drawdown', 0.0)
            )
            
            enhanced_reward = enhanced_reward_components.total_reward
            
            # Calculate priority weight (higher for rare sell actions)
            priority = 1.0
            if action == 2:  # Sell action
                priority = 3.0  # Higher priority for sell experiences
            elif enhanced_reward > 0.01:  # High reward experiences
                priority = 2.0
            
            # Store experience
            experience = {
                'state': np.array(state),
                'action': action,
                'reward': enhanced_reward,
                'next_state': np.array(next_state),
                'done': done,
                'market_regime': market_regime,
                'raw_reward': reward,
                'priority': priority
            }
            
            self.memory.append(experience)
            self.priority_weights.append(priority)
            
            # Track statistics
            self.reward_history.append(enhanced_reward)
            self.regime_performance[market_regime].append(enhanced_reward)
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
    
    def replay(self, batch_size=None):
        """
        Experience replay training with prioritized sampling.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Training history or None if insufficient data
        """
        batch_size = batch_size or self.batch_size
        
        if len(self.memory) < batch_size:
            return None
        
        try:
            # Prioritized sampling
            batch_indices = self._sample_prioritized_indices(batch_size)
            batch = [self.memory[i] for i in batch_indices]
            
            # Prepare training data
            states = np.array([exp['state'] for exp in batch])
            actions = np.array([exp['action'] for exp in batch])
            rewards = np.array([exp['reward'] for exp in batch])
            next_states = np.array([exp['next_state'] for exp in batch])
            dones = np.array([exp['done'] for exp in batch])
            
            # Calculate target Q-values using target network
            current_q_values = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)
            
            # Update Q-values using Bellman equation
            targets = current_q_values.copy()
            
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                if done:
                    targets[i][action] = reward
                else:
                    targets[i][action] = reward + self.gamma * np.max(next_q_values[i])
            
            # Train with callbacks
            callbacks = [
                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3, min_lr=1e-7, verbose=0)
            ]
            
            history = self.model.fit(
                states, targets,
                batch_size=batch_size,
                epochs=1,
                verbose=0,
                callbacks=callbacks
            )
            
            # Update training statistics
            self.training_step += 1
            training_stats = {
                'step': self.training_step,
                'loss': history.history['loss'][0],
                'mae': history.history.get('mae', [0])[0],
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'action_distribution': dict(self.action_distribution)
            }
            self.training_history.append(training_stats)
            
            # Update target model periodically
            if self.training_step % self.config['target_update_frequency'] == 0:
                self.update_target_model()
                logger.info(f"Target model updated at step {self.training_step}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Replay training failed: {e}")
            return None
    
    def _sample_prioritized_indices(self, batch_size):
        """Sample batch indices with priority weighting."""
        if not self.priority_weights:
            return np.random.choice(len(self.memory), batch_size, replace=False)
        
        # Convert to numpy array for easier manipulation
        priorities = np.array(self.priority_weights)
        
        # Add small constant to avoid zero probabilities
        priorities = priorities + 1e-6
        
        # Calculate probabilities
        probabilities = priorities / np.sum(priorities)
        
        # Sample with replacement based on priorities
        indices = np.random.choice(
            len(self.memory), 
            size=batch_size, 
            replace=True,
            p=probabilities
        )
        
        return indices
    
    def get_training_stats(self):
        """Get comprehensive training statistics."""
        total_actions = sum(self.action_distribution.values())
        
        stats = {
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'total_actions': total_actions,
            'action_distribution': {
                'counts': dict(self.action_distribution),
                'percentages': {
                    action: (count / total_actions * 100) if total_actions > 0 else 0
                    for action, count in self.action_distribution.items()
                }
            },
            'recent_rewards': {
                'mean': np.mean(self.reward_history[-100:]) if self.reward_history else 0,
                'std': np.std(self.reward_history[-100:]) if self.reward_history else 0,
                'count': len(self.reward_history)
            },
            'regime_performance': {
                regime: {
                    'mean_reward': np.mean(rewards),
                    'count': len(rewards),
                    'std': np.std(rewards)
                } for regime, rewards in self.regime_performance.items()
            }
        }
        
        # Add recent training metrics
        if self.training_history:
            recent = self.training_history[-10:]
            stats['recent_training'] = {
                'mean_loss': np.mean([h['loss'] for h in recent]),
                'mean_mae': np.mean([h.get('mae', 0) for h in recent])
            }
        
        return stats
    
    def save_model(self, path=None, include_metadata=True):
        """Save model with comprehensive metadata."""
        try:
            save_path = path or self.model_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save main model
            self.model.save(save_path)
            
            # Save metadata
            if include_metadata:
                metadata_path = save_path.replace('.h5', '_metadata.json')
                metadata = {
                    'config': self.config,
                    'training_stats': self.get_training_stats(),
                    'training_history': self.training_history[-100:],
                    'reward_system_stats': self.reward_system.get_reward_statistics(),
                    'model_info': {
                        'state_size': self.state_size,
                        'total_params': self.model.count_params(),
                        'architecture': [layer.name for layer in self.model.layers]
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Enhanced Q-Learning Agent saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path=None):
        """Load model with metadata."""
        try:
            load_path = path or self.model_path
            
            if os.path.exists(load_path):
                self.model = load_model(load_path)
                self.update_target_model()
                
                # Load metadata if available
                metadata_path = load_path.replace('.h5', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Restore some training state
                    if 'training_stats' in metadata:
                        stats = metadata['training_stats']
                        self.epsilon = stats.get('epsilon', self.epsilon)
                        self.training_step = stats.get('training_step', 0)
                
                logger.info(f"Enhanced Q-Learning Agent loaded from {load_path}")
            else:
                logger.warning(f"Model file not found: {load_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Using fresh model instead")


def main():
    """Test the enhanced Q-learning agent."""
    # Create agent
    agent = EnhancedQLearningAgent()
    
    # Test basic functionality
    test_state = np.random.randn(15)
    
    # Test action selection
    for regime in ['bull', 'bear', 'crash', 'normal']:
        action, metadata = agent.select_action(test_state, regime)
        print(f"Regime: {regime}, Action: {action}, Q-values: {metadata['q_values']}")
    
    # Test experience storage and replay
    for i in range(100):
        state = np.random.randn(15)
        action = np.random.choice(3)
        reward = np.random.normal(0, 0.02)
        next_state = np.random.randn(15)
        regime = np.random.choice(['bull', 'bear', 'normal'])
        
        agent.remember(state, action, reward, next_state, False, 
                      market_regime=regime, volatility=np.random.random()*0.05)
    
    # Test replay training
    history = agent.replay()
    if history:
        print(f"Training completed: Loss = {history['loss'][0]:.4f}")
    
    # Print statistics
    stats = agent.get_training_stats()
    print(f"Training Statistics:")
    print(f"  Action Distribution: {stats['action_distribution']['percentages']}")
    print(f"  Memory Size: {stats['memory_size']}")
    print(f"  Epsilon: {stats['epsilon']:.4f}")


if __name__ == "__main__":
    main()