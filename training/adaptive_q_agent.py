"""
Adaptive Q-Learning Agent with Market Regime Specialists
Ensemble approach with attention mechanism and adaptive learning
Author: AlpacaBot Training System
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import random
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for adaptive Q-learning agent."""
    state_size: int
    hidden_sizes: List[int]
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: int
    target_update_frequency: int
    device: str


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for market condition awareness.
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        """
        Initialize attention mechanism.
        
        Args:
            input_dim: Input feature dimension
            attention_dim: Attention layer dimension
        """
        super(AttentionMechanism, self).__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        # Output projection
        self.output_proj = nn.Linear(attention_dim, input_dim)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention weights."""
        for module in [self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of attention mechanism.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            mask: Optional attention mask
            
        Returns:
            Attended output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, values
        Q = self.query(x)  # (batch_size, seq_len, attention_dim)
        K = self.key(x)    # (batch_size, seq_len, attention_dim)
        V = self.value(x)  # (batch_size, seq_len, attention_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.output_proj(attended_values)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output


class RegimeSpecialistNetwork(nn.Module):
    """
    Specialist network for specific market regime.
    """
    
    def __init__(self, 
                 state_size: int, 
                 hidden_sizes: List[int],
                 regime_name: str,
                 dropout_rate: float = 0.3):
        """
        Initialize regime specialist network.
        
        Args:
            state_size: Input state size
            hidden_sizes: List of hidden layer sizes
            regime_name: Name of the market regime
            dropout_rate: Dropout rate for regularization
        """
        super(RegimeSpecialistNetwork, self).__init__()
        
        self.regime_name = regime_name
        self.state_size = state_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size
        
        # Output layer (3 actions: sell, hold, buy)
        layers.append(nn.Linear(input_size, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Regime-specific parameters
        self.confidence_threshold = self._get_confidence_threshold(regime_name)
        self.risk_tolerance = self._get_risk_tolerance(regime_name)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_confidence_threshold(self, regime: str) -> float:
        """Get confidence threshold for regime."""
        thresholds = {
            'bull': 0.6,
            'bear': 0.7,
            'crash': 0.8,
            'recovery': 0.65,
            'high_volatility': 0.75,
            'low_volatility': 0.55,
            'sideways': 0.6,
            'normal': 0.6
        }
        return thresholds.get(regime, 0.6)
    
    def _get_risk_tolerance(self, regime: str) -> float:
        """Get risk tolerance for regime."""
        tolerances = {
            'bull': 0.8,
            'bear': 0.3,
            'crash': 0.1,
            'recovery': 0.7,
            'high_volatility': 0.2,
            'low_volatility': 0.9,
            'sideways': 0.5,
            'normal': 0.6
        }
        return tolerances.get(regime, 0.6)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with confidence estimation.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (q_values, confidence)
        """
        x = self.network(state)
        
        # Calculate confidence based on Q-value distribution
        q_values = x
        confidence = self._calculate_confidence(q_values)
        
        return q_values, confidence
    
    def _calculate_confidence(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate prediction confidence based on Q-value distribution.
        
        Args:
            q_values: Q-values tensor
            
        Returns:
            Confidence scores
        """
        # Method 1: Softmax entropy (lower entropy = higher confidence)
        probs = F.softmax(q_values, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        max_entropy = math.log(q_values.shape[-1])  # Maximum possible entropy
        entropy_confidence = 1.0 - (entropy / max_entropy)
        
        # Method 2: Max-second max difference
        sorted_q, _ = torch.sort(q_values, dim=-1, descending=True)
        max_diff = sorted_q[:, 0] - sorted_q[:, 1]
        max_diff_confidence = torch.sigmoid(max_diff)
        
        # Combine both methods
        confidence = (entropy_confidence + max_diff_confidence) / 2.0
        
        return confidence


class MetaRegimeDetector(nn.Module):
    """
    Meta-network for regime detection and specialist selection.
    """
    
    def __init__(self, state_size: int, num_regimes: int):
        """
        Initialize meta regime detector.
        
        Args:
            state_size: Input state size
            num_regimes: Number of market regimes
        """
        super(MetaRegimeDetector, self).__init__()
        
        self.state_size = state_size
        self.num_regimes = num_regimes
        
        # Regime detection network
        self.regime_detector = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_regimes)
        )
        
        # Attention weights for specialist combination
        self.attention_weights = nn.Sequential(
            nn.Linear(state_size + num_regimes, 64),
            nn.ReLU(),
            nn.Linear(64, num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for regime detection and attention weights.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (regime_probabilities, attention_weights)
        """
        # Detect market regime
        regime_logits = self.regime_detector(state)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Calculate attention weights
        combined_input = torch.cat([state, regime_probs], dim=-1)
        attention_weights = self.attention_weights(combined_input)
        
        return regime_probs, attention_weights


class AdaptiveQLearningAgent:
    """
    Adaptive Q-Learning Agent with ensemble of market regime specialists.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize adaptive Q-learning agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Market regimes
        self.regimes = ['bull', 'bear', 'crash', 'recovery', 'high_volatility', 
                       'low_volatility', 'sideways', 'normal']
        self.num_regimes = len(self.regimes)
        self.regime_to_idx = {regime: idx for idx, regime in enumerate(self.regimes)}
        
        # Create specialist networks for each regime
        self.specialists = {}
        self.target_specialists = {}
        
        for regime in self.regimes:
            specialist = RegimeSpecialistNetwork(
                config.state_size,
                config.hidden_sizes,
                regime
            ).to(self.device)
            
            target_specialist = RegimeSpecialistNetwork(
                config.state_size,
                config.hidden_sizes,
                regime
            ).to(self.device)
            
            # Copy weights to target network
            target_specialist.load_state_dict(specialist.state_dict())
            
            self.specialists[regime] = specialist
            self.target_specialists[regime] = target_specialist
        
        # Meta-network for regime detection and specialist selection
        self.meta_network = MetaRegimeDetector(
            config.state_size, 
            self.num_regimes
        ).to(self.device)
        
        # Attention mechanism for market context
        self.attention = AttentionMechanism(config.state_size).to(self.device)
        
        # Optimizers with adaptive learning rates
        self.optimizers = {}
        for regime in self.regimes:
            self.optimizers[regime] = optim.Adam(
                self.specialists[regime].parameters(),
                lr=config.learning_rate
            )
        
        self.meta_optimizer = optim.Adam(
            self.meta_network.parameters(),
            lr=config.learning_rate
        )
        
        self.attention_optimizer = optim.Adam(
            self.attention.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate schedulers
        self.schedulers = {}
        for regime in self.regimes:
            self.schedulers[regime] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[regime],
                mode='min',
                factor=0.8,
                patience=100,
                verbose=False
            )
        
        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.steps_done = 0
        
        # Performance tracking
        self.regime_performance = defaultdict(list)
        self.specialist_usage = defaultdict(int)
        self.confidence_history = []
        
        # Training state
        self.training_step = 0
        self.target_update_frequency = config.target_update_frequency
        
        logger.info(f"Adaptive Q-Learning Agent initialized with {len(self.regimes)} specialists")
    
    def get_current_epsilon(self) -> float:
        """Get current exploration rate."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1. * self.steps_done / self.epsilon_decay)
    
    def select_action(self, 
                     state: np.ndarray, 
                     market_regime: Optional[str] = None,
                     use_epsilon_greedy: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using ensemble of specialists with attention.
        
        Args:
            state: Current market state
            market_regime: Optional known market regime
            use_epsilon_greedy: Whether to use epsilon-greedy exploration
            
        Returns:
            Tuple of (action, metadata)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Apply attention mechanism
        attended_state = self.attention(state_tensor.unsqueeze(1)).squeeze(1)
        
        # Get regime predictions and attention weights
        regime_probs, attention_weights = self.meta_network(attended_state)
        
        # Get predictions from all specialists
        specialist_outputs = {}
        all_confidences = []
        
        for regime in self.regimes:
            q_values, confidence = self.specialists[regime](attended_state)
            specialist_outputs[regime] = {
                'q_values': q_values,
                'confidence': confidence
            }
            all_confidences.append(confidence.item())
        
        # Ensemble combination using attention weights
        ensemble_q_values = torch.zeros(3).to(self.device)
        total_confidence = 0.0
        
        for i, regime in enumerate(self.regimes):
            weight = attention_weights[0, i].item()
            q_vals = specialist_outputs[regime]['q_values'][0]
            confidence = specialist_outputs[regime]['confidence'].item()
            
            # Weight by attention and confidence
            combined_weight = weight * (1.0 + confidence)
            ensemble_q_values += combined_weight * q_vals
            total_confidence += combined_weight
            
            self.specialist_usage[regime] += weight
        
        # Normalize
        if total_confidence > 0:
            ensemble_q_values /= total_confidence
        
        # Action selection
        if use_epsilon_greedy and random.random() < self.get_current_epsilon():
            # Exploration: random action
            action = random.randint(0, 2)
            selection_method = 'exploration'
        else:
            # Exploitation: best action
            action = ensemble_q_values.argmax().item()
            selection_method = 'exploitation'
        
        self.steps_done += 1
        
        # Calculate overall confidence
        overall_confidence = np.mean(all_confidences)
        self.confidence_history.append(overall_confidence)
        
        # Metadata
        metadata = {
            'ensemble_q_values': ensemble_q_values.cpu().numpy(),
            'regime_probabilities': regime_probs[0].cpu().numpy(),
            'attention_weights': attention_weights[0].cpu().numpy(),
            'specialist_confidences': {regime: specialist_outputs[regime]['confidence'].item() 
                                     for regime in self.regimes},
            'overall_confidence': overall_confidence,
            'selection_method': selection_method,
            'epsilon': self.get_current_epsilon(),
            'predicted_regime': self.regimes[regime_probs.argmax().item()]
        }
        
        # Convert action to trading action (-1, 0, 1)
        trading_action = action - 1
        
        return trading_action, metadata
    
    def train_step(self, 
                   batch_experiences: List[Any],
                   batch_indices: np.ndarray,
                   importance_weights: np.ndarray) -> Dict[str, float]:
        """
        Perform one training step with prioritized experience replay.
        
        Args:
            batch_experiences: Batch of experiences
            batch_indices: Batch indices for priority updates
            importance_weights: Importance sampling weights
            
        Returns:
            Training metrics dictionary
        """
        if not batch_experiences:
            return {}
        
        # Prepare batch tensors
        states = torch.FloatTensor([exp.state for exp in batch_experiences]).to(self.device)
        actions = torch.LongTensor([exp.action + 1 for exp in batch_experiences]).to(self.device)  # Convert to 0,1,2
        rewards = torch.FloatTensor([exp.reward for exp in batch_experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch_experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch_experiences]).to(self.device)
        regimes = [exp.market_regime for exp in batch_experiences]
        weights = torch.FloatTensor(importance_weights).to(self.device)
        
        batch_size = len(batch_experiences)
        
        # Apply attention to states
        attended_states = self.attention(states.unsqueeze(1)).squeeze(1)
        attended_next_states = self.attention(next_states.unsqueeze(1)).squeeze(1)
        
        # Get regime predictions
        regime_probs, attention_weights = self.meta_network(attended_states)
        next_regime_probs, next_attention_weights = self.meta_network(attended_next_states)
        
        # Train each specialist
        total_loss = 0.0
        regime_losses = {}
        td_errors = []
        
        for regime_idx, regime in enumerate(self.regimes):
            # Get current Q-values
            current_q_values, current_confidence = self.specialists[regime](attended_states)
            current_q_values_selected = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Get target Q-values
            with torch.no_grad():
                next_q_values, _ = self.target_specialists[regime](attended_next_states)
                next_q_values_max = next_q_values.max(1)[0]
                target_q_values = rewards + (self.config.gamma * next_q_values_max * ~dones)
            
            # Calculate TD errors
            td_error = target_q_values - current_q_values_selected
            
            # Weight by attention and importance sampling
            regime_attention = attention_weights[:, regime_idx]
            weighted_td_error = td_error * regime_attention * weights
            
            # Calculate loss
            loss = F.mse_loss(
                current_q_values_selected * regime_attention * weights,
                target_q_values * regime_attention * weights,
                reduction='mean'
            )
            
            # Confidence loss (encourage high confidence for correct predictions)
            confidence_target = torch.where(
                torch.abs(td_error) < 0.01,  # Small TD error = good prediction
                torch.ones_like(current_confidence),
                torch.zeros_like(current_confidence)
            )
            confidence_loss = F.mse_loss(current_confidence, confidence_target)
            
            total_regime_loss = loss + 0.1 * confidence_loss
            
            # Backpropagation
            self.optimizers[regime].zero_grad()
            total_regime_loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.specialists[regime].parameters(), max_norm=1.0)
            
            self.optimizers[regime].step()
            self.schedulers[regime].step(total_regime_loss.item())
            
            regime_losses[regime] = total_regime_loss.item()
            total_loss += total_regime_loss.item()
            
            # Store TD errors for this regime
            if regime_idx == 0:  # Use first regime's TD errors for priority updates
                td_errors = td_error.abs().cpu().numpy()
        
        # Train meta-network (regime detection)
        # Use regime labels from experiences if available
        regime_targets = torch.zeros(batch_size, self.num_regimes).to(self.device)
        for i, regime in enumerate(regimes):
            if regime in self.regime_to_idx:
                regime_targets[i, self.regime_to_idx[regime]] = 1.0
        
        regime_loss = F.cross_entropy(
            regime_probs.view(-1, self.num_regimes),
            regime_targets.argmax(dim=1)
        )
        
        self.meta_optimizer.zero_grad()
        regime_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.meta_network.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        
        # Train attention mechanism
        # Attention should focus on important features for regime detection
        attention_loss = F.mse_loss(attended_states, states)  # Simplified attention loss
        
        self.attention_optimizer.zero_grad()
        attention_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.attention.parameters(), max_norm=1.0)
        self.attention_optimizer.step()
        
        # Update target networks
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self._update_target_networks()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * 0.995)
        
        return {
            'total_loss': total_loss / len(self.regimes),
            'regime_losses': regime_losses,
            'regime_loss': regime_loss.item(),
            'attention_loss': attention_loss.item(),
            'mean_td_error': np.mean(td_errors) if len(td_errors) > 0 else 0.0,
            'td_errors': td_errors,
            'epsilon': self.epsilon,
            'mean_confidence': np.mean(self.confidence_history[-100:]) if self.confidence_history else 0.0
        }
    
    def _update_target_networks(self):
        """Update target networks with current network weights."""
        for regime in self.regimes:
            self.target_specialists[regime].load_state_dict(
                self.specialists[regime].state_dict()
            )
        logger.debug("Target networks updated")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_usage = sum(self.specialist_usage.values())
        
        usage_distribution = {}
        if total_usage > 0:
            usage_distribution = {
                regime: count / total_usage 
                for regime, count in self.specialist_usage.items()
            }
        
        return {
            'specialist_usage': dict(self.specialist_usage),
            'usage_distribution': usage_distribution,
            'regime_performance': dict(self.regime_performance),
            'confidence_stats': {
                'mean': np.mean(self.confidence_history) if self.confidence_history else 0.0,
                'std': np.std(self.confidence_history) if self.confidence_history else 0.0,
                'recent_mean': np.mean(self.confidence_history[-100:]) if len(self.confidence_history) >= 100 else 0.0
            },
            'exploration_stats': {
                'current_epsilon': self.epsilon,
                'steps_done': self.steps_done
            },
            'training_stats': {
                'training_step': self.training_step,
                'target_updates': self.training_step // self.target_update_frequency
            }
        }
    
    def save_model(self, filepath: str):
        """Save all model components."""
        checkpoint = {
            'config': self.config.__dict__,
            'regimes': self.regimes,
            'specialists': {regime: net.state_dict() for regime, net in self.specialists.items()},
            'target_specialists': {regime: net.state_dict() for regime, net in self.target_specialists.items()},
            'meta_network': self.meta_network.state_dict(),
            'attention': self.attention.state_dict(),
            'optimizers': {regime: opt.state_dict() for regime, opt in self.optimizers.items()},
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'attention_optimizer': self.attention_optimizer.state_dict(),
            'training_state': {
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'training_step': self.training_step
            },
            'performance_stats': self.get_performance_statistics()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load all model components."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        for regime in self.regimes:
            if regime in checkpoint['specialists']:
                self.specialists[regime].load_state_dict(checkpoint['specialists'][regime])
                self.target_specialists[regime].load_state_dict(checkpoint['target_specialists'][regime])
        
        self.meta_network.load_state_dict(checkpoint['meta_network'])
        self.attention.load_state_dict(checkpoint['attention'])
        
        # Load optimizer states
        for regime in self.regimes:
            if regime in checkpoint['optimizers']:
                self.optimizers[regime].load_state_dict(checkpoint['optimizers'][regime])
        
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.attention_optimizer.load_state_dict(checkpoint['attention_optimizer'])
        
        # Load training state
        training_state = checkpoint['training_state']
        self.epsilon = training_state['epsilon']
        self.steps_done = training_state['steps_done']
        self.training_step = training_state['training_step']
        
        logger.info(f"Model loaded from {filepath}")


def create_test_agent() -> AdaptiveQLearningAgent:
    """Create a test adaptive Q-learning agent."""
    config = AgentConfig(
        state_size=15,
        hidden_sizes=[128, 64, 32],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        target_update_frequency=100,
        device='cpu'
    )
    
    return AdaptiveQLearningAgent(config)


def test_adaptive_agent():
    """Test the adaptive Q-learning agent."""
    agent = create_test_agent()
    
    print("\n" + "="*60)
    print("ADAPTIVE Q-LEARNING AGENT TEST")
    print("="*60)
    
    # Test action selection
    test_state = np.random.randn(15)
    
    for i in range(5):
        action, metadata = agent.select_action(test_state)
        
        print(f"\nTest {i+1}:")
        print(f"  Action: {action}")
        print(f"  Confidence: {metadata['overall_confidence']:.3f}")
        print(f"  Predicted Regime: {metadata['predicted_regime']}")
        print(f"  Selection Method: {metadata['selection_method']}")
        print(f"  Epsilon: {metadata['epsilon']:.3f}")
    
    # Test training (with mock experiences)
    from training.prioritized_experience_buffer import Experience
    
    mock_experiences = []
    for i in range(10):
        exp = Experience(
            state=np.random.randn(15),
            action=np.random.choice([-1, 0, 1]),
            reward=np.random.randn() * 0.02,
            next_state=np.random.randn(15),
            done=False,
            market_regime=np.random.choice(['bull', 'bear', 'normal']),
            timestamp=i,
            confidence=np.random.random(),
            volatility=np.random.random() * 0.05
        )
        mock_experiences.append(exp)
    
    # Mock training step
    batch_indices = np.arange(10)
    importance_weights = np.ones(10)
    
    train_metrics = agent.train_step(mock_experiences, batch_indices, importance_weights)
    
    print(f"\nTraining Metrics:")
    print(f"  Total Loss: {train_metrics.get('total_loss', 0):.4f}")
    print(f"  Mean TD Error: {train_metrics.get('mean_td_error', 0):.4f}")
    print(f"  Mean Confidence: {train_metrics.get('mean_confidence', 0):.3f}")
    
    # Performance statistics
    stats = agent.get_performance_statistics()
    print(f"\nPerformance Statistics:")
    print(f"  Specialist Usage: {stats['usage_distribution']}")
    print(f"  Confidence Stats: {stats['confidence_stats']}")
    print("="*60)


if __name__ == "__main__":
    test_adaptive_agent()