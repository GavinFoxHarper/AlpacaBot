"""
Prioritized Experience Replay Buffer for AlpacaBot
Enhanced experience buffer with prioritized sampling and importance weighting
Author: AlpacaBot Training System
Date: 2024
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any, NamedTuple
from collections import deque
import heapq
import logging
from dataclasses import dataclass
import pickle
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple with metadata."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    market_regime: str
    timestamp: float
    confidence: float
    volatility: float
    importance: float = 1.0
    td_error: float = 0.0


class PrioritizedExperienceBuffer:
    """
    Prioritized experience replay buffer with importance sampling,
    separate regime buffers, and quality scoring.
    """
    
    def __init__(self, 
                 max_size: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 regime_specific: bool = True):
        """
        Initialize prioritized experience buffer.
        
        Args:
            max_size: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sample
            epsilon: Small constant to prevent zero probabilities
            regime_specific: Whether to maintain separate buffers per regime
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.regime_specific = regime_specific
        
        # Main buffer
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.position = 0
        
        # Regime-specific buffers
        if regime_specific:
            self.regime_buffers = {
                'bull': deque(maxlen=max_size // 4),
                'bear': deque(maxlen=max_size // 4),
                'sideways': deque(maxlen=max_size // 4),
                'crash': deque(maxlen=max_size // 8),
                'recovery': deque(maxlen=max_size // 8),
                'high_volatility': deque(maxlen=max_size // 8),
                'low_volatility': deque(maxlen=max_size // 8),
                'normal': deque(maxlen=max_size // 4)
            }
            
            self.regime_priorities = {
                regime: deque(maxlen=buffer.maxlen)
                for regime, buffer in self.regime_buffers.items()
            }
        
        # Action-specific tracking for balance
        self.action_counts = {-1: 0, 0: 0, 1: 0}
        self.action_priorities = {-1: [], 0: [], 1: []}
        
        # Quality tracking
        self.quality_scores = deque(maxlen=max_size)
        self.age_weights = deque(maxlen=max_size)
        
        # Statistics
        self.total_added = 0
        self.sampling_stats = {'prioritized': 0, 'uniform': 0, 'balanced': 0}
        
        logger.info(f"Prioritized Experience Buffer initialized: max_size={max_size}, alpha={alpha}, beta={beta}")
    
    def add(self, experience: Experience):
        """
        Add experience to buffer with automatic priority calculation.
        
        Args:
            experience: Experience tuple to add
        """
        # Calculate initial priority
        priority = self._calculate_initial_priority(experience)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(experience)
        
        # Add to main buffer
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
            self.quality_scores.append(quality_score)
            self.age_weights.append(1.0)
        else:
            # Replace oldest experience
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.quality_scores[self.position] = quality_score
            self.age_weights[self.position] = 1.0
            self.position = (self.position + 1) % self.max_size
        
        # Add to regime-specific buffer
        if self.regime_specific and experience.market_regime in self.regime_buffers:
            regime_buffer = self.regime_buffers[experience.market_regime]
            regime_priorities = self.regime_priorities[experience.market_regime]
            
            regime_buffer.append(experience)
            regime_priorities.append(priority)
        
        # Update action tracking
        self.action_counts[experience.action] += 1
        self.action_priorities[experience.action].append(priority)
        
        # Keep only recent action priorities for balance calculation
        max_action_priorities = 1000
        for action in self.action_priorities:
            if len(self.action_priorities[action]) > max_action_priorities:
                self.action_priorities[action] = self.action_priorities[action][-max_action_priorities:]
        
        self.total_added += 1
        
        # Age existing experiences
        self._update_age_weights()
    
    def _calculate_initial_priority(self, experience: Experience) -> float:
        """
        Calculate initial priority for new experience.
        
        Args:
            experience: Experience to calculate priority for
            
        Returns:
            Initial priority value
        """
        base_priority = abs(experience.reward) + self.epsilon
        
        # Boost priority for rare actions (especially sell)
        action_boost = self._get_action_boost(experience.action)
        base_priority *= action_boost
        
        # Boost priority for extreme market conditions
        regime_boost = self._get_regime_boost(experience.market_regime)
        base_priority *= regime_boost
        
        # Boost priority for high-confidence decisions
        confidence_boost = 1.0 + (experience.confidence - 0.5)
        base_priority *= max(0.5, confidence_boost)
        
        # Boost priority for high volatility (more informative)
        volatility_boost = 1.0 + min(experience.volatility * 10, 1.0)
        base_priority *= volatility_boost
        
        return base_priority
    
    def _get_action_boost(self, action: int) -> float:
        """Get priority boost based on action rarity."""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return 1.0
        
        action_frequency = self.action_counts[action] / total_actions
        
        # Boost rare actions (especially sells)
        if action == -1:  # Sell - typically rarest
            return 3.0 / max(action_frequency, 0.1)
        elif action == 1:  # Buy
            return 2.0 / max(action_frequency, 0.1)
        else:  # Hold
            return 1.5 / max(action_frequency, 0.1)
    
    def _get_regime_boost(self, regime: str) -> float:
        """Get priority boost based on market regime importance."""
        regime_boosts = {
            'crash': 3.0,      # Extremely important to learn
            'recovery': 2.5,   # Critical transitions
            'bear': 2.0,       # Important for sell signals
            'high_volatility': 1.8,
            'bull': 1.0,       # Common condition
            'sideways': 1.2,   # Hold signal importance
            'low_volatility': 1.1,
            'normal': 1.0
        }
        return regime_boosts.get(regime, 1.0)
    
    def _calculate_quality_score(self, experience: Experience) -> float:
        """
        Calculate quality score for experience.
        
        Args:
            experience: Experience to evaluate
            
        Returns:
            Quality score (0 to 1)
        """
        score = 0.0
        
        # Reward magnitude component
        reward_score = min(abs(experience.reward) * 10, 1.0)
        score += reward_score * 0.3
        
        # Confidence component
        confidence_score = experience.confidence
        score += confidence_score * 0.2
        
        # Action-outcome consistency
        if experience.action == 1 and experience.reward > 0:  # Good buy
            score += 0.3
        elif experience.action == -1 and experience.reward > 0:  # Good sell (avoided loss)
            score += 0.4  # Higher weight for good sells
        elif experience.action == 0 and abs(experience.reward) < 0.01:  # Good hold
            score += 0.2
        
        # Market condition informativeness
        if experience.market_regime in ['crash', 'recovery']:
            score += 0.2
        elif experience.volatility > 0.04:  # High volatility
            score += 0.1
        
        return min(score, 1.0)
    
    def _update_age_weights(self):
        """Update age weights for all experiences."""
        if len(self.age_weights) == 0:
            return
        
        # Decay age weights
        decay_factor = 0.999
        for i in range(len(self.age_weights)):
            self.age_weights[i] *= decay_factor
    
    def sample(self, 
               batch_size: int,
               sampling_strategy: str = 'prioritized',
               regime_filter: Optional[str] = None,
               balance_actions: bool = True) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences with various strategies.
        
        Args:
            batch_size: Number of experiences to sample
            sampling_strategy: 'prioritized', 'uniform', or 'balanced'
            regime_filter: Optional regime to filter by
            balance_actions: Whether to balance action distribution
            
        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        self.sampling_stats[sampling_strategy] += 1
        
        # Choose sampling method
        if sampling_strategy == 'prioritized':
            experiences, indices, weights = self._sample_prioritized(batch_size, regime_filter)
        elif sampling_strategy == 'uniform':
            experiences, indices, weights = self._sample_uniform(batch_size, regime_filter)
        elif sampling_strategy == 'balanced':
            experiences, indices, weights = self._sample_balanced(batch_size, regime_filter)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Apply action balancing if requested
        if balance_actions and len(experiences) > 0:
            experiences, indices, weights = self._apply_action_balancing(
                experiences, indices, weights, batch_size
            )
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def _sample_prioritized(self, 
                           batch_size: int,
                           regime_filter: Optional[str] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample using prioritized experience replay."""
        if regime_filter and regime_filter in self.regime_buffers:
            buffer = list(self.regime_buffers[regime_filter])
            priorities = list(self.regime_priorities[regime_filter])
        else:
            buffer = list(self.buffer)
            priorities = list(self.priorities)
        
        if len(buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Apply age weighting
        if regime_filter:
            age_weights = [1.0] * len(buffer)  # Simplified for regime buffers
        else:
            age_weights = list(self.age_weights)
        
        # Calculate sampling probabilities
        priorities = np.array(priorities)
        age_weights = np.array(age_weights)
        
        # Combine priority and age weights
        combined_priorities = priorities * age_weights
        prob_powers = combined_priorities ** self.alpha
        probabilities = prob_powers / prob_powers.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(buffer), 
            size=min(batch_size, len(buffer)), 
            replace=False,
            p=probabilities
        )
        
        # Calculate importance sampling weights
        max_weight = (len(buffer) * probabilities.min()) ** (-self.beta)
        weights = ((len(buffer) * probabilities[indices]) ** (-self.beta)) / max_weight
        
        # Get experiences
        experiences = [buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def _sample_uniform(self, 
                       batch_size: int,
                       regime_filter: Optional[str] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample uniformly at random."""
        if regime_filter and regime_filter in self.regime_buffers:
            buffer = list(self.regime_buffers[regime_filter])
        else:
            buffer = list(self.buffer)
        
        if len(buffer) == 0:
            return [], np.array([]), np.array([])
        
        indices = np.random.choice(
            len(buffer),
            size=min(batch_size, len(buffer)),
            replace=False
        )
        
        experiences = [buffer[i] for i in indices]
        weights = np.ones(len(experiences))  # Uniform weights
        
        return experiences, indices, weights
    
    def _sample_balanced(self, 
                        batch_size: int,
                        regime_filter: Optional[str] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample with balanced action distribution."""
        if regime_filter and regime_filter in self.regime_buffers:
            buffer = list(self.regime_buffers[regime_filter])
        else:
            buffer = list(self.buffer)
        
        if len(buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Separate by action
        action_experiences = {-1: [], 0: [], 1: []}
        action_indices = {-1: [], 0: [], 1: []}
        
        for i, exp in enumerate(buffer):
            if exp.action in action_experiences:
                action_experiences[exp.action].append(exp)
                action_indices[exp.action].append(i)
        
        # Calculate samples per action
        samples_per_action = batch_size // 3
        remaining_samples = batch_size % 3
        
        experiences = []
        indices = []
        weights = []
        
        for action in [-1, 0, 1]:
            if not action_experiences[action]:
                continue
            
            n_samples = samples_per_action
            if remaining_samples > 0 and action == 1:  # Give extra to buy action
                n_samples += remaining_samples
                remaining_samples = 0
            
            # Sample from this action
            available_samples = len(action_experiences[action])
            if n_samples > available_samples:
                # Use all available samples
                sampled_exp = action_experiences[action]
                sampled_idx = action_indices[action]
            else:
                # Random sample
                sample_indices = np.random.choice(available_samples, n_samples, replace=False)
                sampled_exp = [action_experiences[action][i] for i in sample_indices]
                sampled_idx = [action_indices[action][i] for i in sample_indices]
            
            experiences.extend(sampled_exp)
            indices.extend(sampled_idx)
            weights.extend([1.0] * len(sampled_exp))
        
        return experiences, np.array(indices), np.array(weights)
    
    def _apply_action_balancing(self, 
                               experiences: List[Experience],
                               indices: np.ndarray,
                               weights: np.ndarray,
                               target_batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Apply action balancing to sampled experiences."""
        # Count actions in current sample
        action_counts = {-1: 0, 0: 0, 1: 0}
        for exp in experiences:
            action_counts[exp.action] += 1
        
        # Check if balancing is needed
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return experiences, indices, weights
        
        # Target distribution
        target_per_action = target_batch_size // 3
        
        balanced_experiences = []
        balanced_indices = []
        balanced_weights = []
        
        # Group experiences by action
        action_groups = {-1: [], 0: [], 1: []}
        action_group_indices = {-1: [], 0: [], 1: []}
        action_group_weights = {-1: [], 0: [], 1: []}
        
        for i, exp in enumerate(experiences):
            action_groups[exp.action].append(exp)
            action_group_indices[exp.action].append(indices[i])
            action_group_weights[exp.action].append(weights[i])
        
        # Sample target amount from each action
        for action in [-1, 0, 1]:
            if not action_groups[action]:
                continue
            
            available = len(action_groups[action])
            if available <= target_per_action:
                # Use all available
                balanced_experiences.extend(action_groups[action])
                balanced_indices.extend(action_group_indices[action])
                balanced_weights.extend(action_group_weights[action])
            else:
                # Randomly sample target amount
                sample_idx = np.random.choice(available, target_per_action, replace=False)
                for idx in sample_idx:
                    balanced_experiences.append(action_groups[action][idx])
                    balanced_indices.append(action_group_indices[action][idx])
                    balanced_weights.append(action_group_weights[action][idx])
        
        return balanced_experiences, np.array(balanced_indices), np.array(balanced_weights)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                new_priority = abs(td_error) + self.epsilon
                self.priorities[idx] = new_priority
                
                # Update experience TD error
                if idx < len(self.buffer):
                    self.buffer[idx].td_error = td_error
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        if len(self.buffer) == 0:
            return {'error': 'Empty buffer'}
        
        experiences = list(self.buffer)
        
        # Action distribution
        action_dist = {-1: 0, 0: 0, 1: 0}
        for exp in experiences:
            action_dist[exp.action] += 1
        
        # Regime distribution
        regime_dist = {}
        for exp in experiences:
            regime_dist[exp.market_regime] = regime_dist.get(exp.market_regime, 0) + 1
        
        # Reward statistics
        rewards = [exp.reward for exp in experiences]
        
        # Priority statistics
        priorities = list(self.priorities)
        
        # Quality statistics
        qualities = list(self.quality_scores)
        
        return {
            'buffer_size': len(self.buffer),
            'max_size': self.max_size,
            'total_added': self.total_added,
            'action_distribution': action_dist,
            'regime_distribution': regime_dist,
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'priority_stats': {
                'mean': np.mean(priorities),
                'std': np.std(priorities),
                'min': np.min(priorities),
                'max': np.max(priorities)
            },
            'quality_stats': {
                'mean': np.mean(qualities),
                'std': np.std(qualities),
                'min': np.min(qualities),
                'max': np.max(qualities)
            },
            'sampling_stats': self.sampling_stats,
            'current_beta': self.beta
        }
    
    def save(self, filepath: str):
        """Save buffer to disk."""
        buffer_data = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities),
            'quality_scores': list(self.quality_scores),
            'age_weights': list(self.age_weights),
            'action_counts': self.action_counts,
            'total_added': self.total_added,
            'position': self.position,
            'beta': self.beta,
            'config': {
                'max_size': self.max_size,
                'alpha': self.alpha,
                'beta_increment': self.beta_increment,
                'epsilon': self.epsilon,
                'regime_specific': self.regime_specific
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_data, f)
        
        logger.info(f"Buffer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            buffer_data = pickle.load(f)
        
        self.buffer = deque(buffer_data['buffer'], maxlen=self.max_size)
        self.priorities = deque(buffer_data['priorities'], maxlen=self.max_size)
        self.quality_scores = deque(buffer_data['quality_scores'], maxlen=self.max_size)
        self.age_weights = deque(buffer_data['age_weights'], maxlen=self.max_size)
        self.action_counts = buffer_data['action_counts']
        self.total_added = buffer_data['total_added']
        self.position = buffer_data['position']
        self.beta = buffer_data['beta']
        
        logger.info(f"Buffer loaded from {filepath}")


def test_prioritized_buffer():
    """Test the prioritized experience buffer."""
    buffer = PrioritizedExperienceBuffer(max_size=1000, regime_specific=True)
    
    # Create test experiences
    regimes = ['bull', 'bear', 'crash', 'recovery', 'normal']
    actions = [-1, 0, 1]
    
    print("\n" + "="*60)
    print("PRIORITIZED EXPERIENCE BUFFER TEST")
    print("="*60)
    
    # Add test experiences
    for i in range(500):
        exp = Experience(
            state=np.random.randn(10),
            action=np.random.choice(actions),
            reward=np.random.randn() * 0.02,  # Realistic reward scale
            next_state=np.random.randn(10),
            done=np.random.random() < 0.1,
            market_regime=np.random.choice(regimes),
            timestamp=i,
            confidence=np.random.random(),
            volatility=np.random.random() * 0.05
        )
        buffer.add(exp)
    
    # Test different sampling strategies
    strategies = ['prioritized', 'uniform', 'balanced']
    
    for strategy in strategies:
        experiences, indices, weights = buffer.sample(
            batch_size=64,
            sampling_strategy=strategy,
            balance_actions=True
        )
        
        action_dist = {-1: 0, 0: 0, 1: 0}
        for exp in experiences:
            action_dist[exp.action] += 1
        
        print(f"\n{strategy.upper()} SAMPLING:")
        print(f"  Batch size: {len(experiences)}")
        print(f"  Action distribution: {action_dist}")
        print(f"  Weight stats: mean={np.mean(weights):.3f}, max={np.max(weights):.3f}")
    
    # Buffer statistics
    stats = buffer.get_statistics()
    print(f"\nBUFFER STATISTICS:")
    print(f"  Size: {stats['buffer_size']}/{stats['max_size']}")
    print(f"  Action distribution: {stats['action_distribution']}")
    print(f"  Regime distribution: {stats['regime_distribution']}")
    print(f"  Reward stats: {stats['reward_stats']}")
    print(f"  Sampling stats: {stats['sampling_stats']}")
    print("="*60)


if __name__ == "__main__":
    test_prioritized_buffer()