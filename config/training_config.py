"""
Centralized Training Configuration for AlpacaBot
Comprehensive configuration management for all training components
Author: AlpacaBot Training System
Date: 2024
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass
class DataPipelineConfig:
    """Configuration for enhanced data pipeline."""
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    cache_dir: str = "data/cache"
    symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE',
        'CRM', 'NFLX', 'PFE', 'TMO', 'CSCO', 'PEP', 'INTC', 'CMCSA', 'VZ', 'T'
    ])
    validation_split: float = 0.2
    test_split: float = 0.1
    balance_actions: bool = True
    target_distribution: Dict[int, float] = field(default_factory=lambda: {
        -1: 0.33, 0: 0.34, 1: 0.33
    })
    feature_columns: List[str] = field(default_factory=lambda: [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'ATR', 'Volume_Ratio',
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Volatility_20', 'Volatility_50', 'SR_Position',
        'Regime_Confidence', 'Regime_Duration'
    ])


@dataclass
class RewardSystemConfig:
    """Configuration for enhanced reward system."""
    action_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'buy': {'profit_weight': 1.0, 'timing_weight': 0.3, 'risk_weight': 0.2},
        'sell': {'profit_weight': 1.2, 'timing_weight': 0.4, 'risk_weight': 0.3},
        'hold': {'stability_weight': 0.5, 'opportunity_cost_weight': 0.3}
    })
    
    regime_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'bull': {'buy': 1.0, 'sell': 0.8, 'hold': 0.9},
        'bear': {'buy': 0.7, 'sell': 1.3, 'hold': 1.0},
        'crash': {'buy': 0.5, 'sell': 1.5, 'hold': 1.1},
        'recovery': {'buy': 1.4, 'sell': 0.6, 'hold': 0.8},
        'high_volatility': {'buy': 0.9, 'sell': 1.1, 'hold': 1.2},
        'low_volatility': {'buy': 1.0, 'sell': 1.0, 'hold': 0.8},
        'sideways': {'buy': 0.8, 'sell': 0.8, 'hold': 1.3},
        'normal': {'buy': 1.0, 'sell': 1.0, 'hold': 1.0}
    })
    
    risk_parameters: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown_penalty': 0.5,
        'volatility_threshold': 0.02,
        'confidence_threshold': 0.6,
        'position_size_factor': 0.1
    })
    
    consistency_parameters: Dict[str, float] = field(default_factory=lambda: {
        'streak_bonus_factor': 0.1,
        'min_streak_length': 3,
        'diversity_bonus': 0.05
    })


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    state_size: int = 15
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    attention_dim: int = 64
    num_regimes: int = 8
    use_batch_norm: bool = True
    activation: str = 'relu'
    output_size: int = 3  # sell, hold, buy


@dataclass
class TrainingPhaseConfig:
    """Configuration for individual training phases."""
    name: str
    duration_epochs: int
    difficulty_level: str
    regime_filter: Optional[str]
    action_balance: Dict[int, float]
    learning_rate: float
    batch_size: int
    description: str


@dataclass
class ComprehensiveTrainingConfig:
    """Configuration for comprehensive training pipeline."""
    phases: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            'name': 'foundation_easy',
            'duration_epochs': 50,
            'difficulty_level': 'easy',
            'regime_filter': None,
            'action_balance': {0: 0.34, 1: 0.33, -1: 0.33},
            'learning_rate': 0.001,
            'batch_size': 64,
            'description': 'Foundation training with easy patterns'
        },
        {
            'name': 'bull_market_specialization',
            'duration_epochs': 30,
            'difficulty_level': 'medium',
            'regime_filter': 'bull',
            'action_balance': {0: 0.25, 1: 0.60, -1: 0.15},
            'learning_rate': 0.0008,
            'batch_size': 48,
            'description': 'Bull market pattern specialization'
        },
        {
            'name': 'bear_market_specialization',
            'duration_epochs': 40,
            'difficulty_level': 'medium',
            'regime_filter': 'bear',
            'action_balance': {0: 0.25, 1: 0.15, -1: 0.60},
            'learning_rate': 0.0008,
            'batch_size': 48,
            'description': 'Bear market and sell signal specialization'
        },
        {
            'name': 'crash_recovery_training',
            'duration_epochs': 35,
            'difficulty_level': 'hard',
            'regime_filter': 'crash',
            'action_balance': {0: 0.20, 1: 0.20, -1: 0.60},
            'learning_rate': 0.0006,
            'batch_size': 32,
            'description': 'Extreme market conditions training'
        },
        {
            'name': 'balanced_integration',
            'duration_epochs': 40,
            'difficulty_level': 'medium',
            'regime_filter': None,
            'action_balance': {0: 0.34, 1: 0.33, -1: 0.33},
            'learning_rate': 0.0005,
            'batch_size': 64,
            'description': 'Balanced integration of all patterns'
        },
        {
            'name': 'adversarial_hardening',
            'duration_epochs': 25,
            'difficulty_level': 'extreme',
            'regime_filter': None,
            'action_balance': {0: 0.30, 1: 0.35, -1: 0.35},
            'learning_rate': 0.0003,
            'batch_size': 32,
            'description': 'Final hardening with extreme conditions'
        }
    ])
    
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        'patience': 10,
        'min_delta': 0.001,
        'monitor_metric': 'accuracy'
    })
    
    checkpoint_every: int = 5
    save_dir: str = 'models/checkpoints'
    max_epochs: int = 200
    validation_frequency: int = 1


@dataclass
class ExperienceBufferConfig:
    """Configuration for prioritized experience buffer."""
    max_size: int = 100000
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4   # Importance sampling exponent
    beta_increment: float = 0.001
    epsilon: float = 1e-6
    regime_specific: bool = True
    
    # Buffer size allocation by regime
    regime_buffer_sizes: Dict[str, int] = field(default_factory=lambda: {
        'bull': 25000,
        'bear': 25000,
        'sideways': 25000,
        'crash': 12500,
        'recovery': 12500,
        'high_volatility': 12500,
        'low_volatility': 12500,
        'normal': 25000
    })
    
    # Priority boosts by action and regime
    action_priority_boosts: Dict[int, float] = field(default_factory=lambda: {
        -1: 3.0,  # Sell signals get highest priority
        1: 2.0,   # Buy signals
        0: 1.5    # Hold signals
    })
    
    regime_priority_boosts: Dict[str, float] = field(default_factory=lambda: {
        'crash': 3.0,
        'recovery': 2.5,
        'bear': 2.0,
        'high_volatility': 1.8,
        'bull': 1.0,
        'sideways': 1.2,
        'low_volatility': 1.1,
        'normal': 1.0
    })


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    n_splits: int = 5
    test_size: float = 0.2
    gap_days: int = 5
    min_train_size: int = 252
    stress_test_enabled: bool = True
    regime_stratified: bool = True
    
    performance_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'sharpe_ratio', 
        'max_drawdown', 'total_return', 'win_rate', 'profit_factor', 
        'calmar_ratio', 'balance_score'
    ])
    
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        'market_crash', 'high_volatility', 'sideways_market',
        'regime_transition', 'low_confidence'
    ])
    
    benchmark_models: List[str] = field(default_factory=lambda: [
        'random', 'majority_class', 'buy_and_hold'
    ])


@dataclass
class AdaptiveAgentConfig:
    """Configuration for adaptive Q-learning agent."""
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    target_update_frequency: int = 100
    device: str = 'cuda'
    
    # Specialist network configurations
    specialist_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'bull': {'confidence_threshold': 0.6, 'risk_tolerance': 0.8},
        'bear': {'confidence_threshold': 0.7, 'risk_tolerance': 0.3},
        'crash': {'confidence_threshold': 0.8, 'risk_tolerance': 0.1},
        'recovery': {'confidence_threshold': 0.65, 'risk_tolerance': 0.7},
        'high_volatility': {'confidence_threshold': 0.75, 'risk_tolerance': 0.2},
        'low_volatility': {'confidence_threshold': 0.55, 'risk_tolerance': 0.9},
        'sideways': {'confidence_threshold': 0.6, 'risk_tolerance': 0.5},
        'normal': {'confidence_threshold': 0.6, 'risk_tolerance': 0.6}
    })


@dataclass
class MasterTrainingConfig:
    """Master configuration containing all training configurations."""
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    reward_system: RewardSystemConfig = field(default_factory=RewardSystemConfig)
    model_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    comprehensive_training: ComprehensiveTrainingConfig = field(default_factory=ComprehensiveTrainingConfig)
    experience_buffer: ExperienceBufferConfig = field(default_factory=ExperienceBufferConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    adaptive_agent: AdaptiveAgentConfig = field(default_factory=AdaptiveAgentConfig)
    
    # Global settings
    random_seed: int = 42
    logging_level: str = 'INFO'
    output_dir: str = 'training_outputs'
    experiment_name: str = 'alpacabot_training_fix'
    save_intermediate_results: bool = True
    create_visualizations: bool = True
    
    # Hardware settings
    use_gpu: bool = True
    num_workers: int = 4
    memory_limit_gb: int = 16
    
    # Success criteria
    target_accuracy: float = 0.65
    target_sharpe_ratio: float = 1.0
    target_max_drawdown: float = -0.15
    target_balance_score: float = 0.8
    min_sell_signal_ratio: float = 0.25


class ConfigManager:
    """
    Configuration manager for loading, saving, and validating configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> MasterTrainingConfig:
        """Load configuration from file or create default."""
        if self.config_path and os.path.exists(self.config_path):
            return self.load_config(self.config_path)
        else:
            return MasterTrainingConfig()
    
    def load_config(self, config_path: str) -> MasterTrainingConfig:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dictionary to dataclass
        return self._dict_to_config(config_dict)
    
    def save_config(self, config: MasterTrainingConfig, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Convert dataclass to dictionary
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: MasterTrainingConfig) -> Dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        result = {}
        
        for field_name, field_value in config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Nested dataclass
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MasterTrainingConfig:
        """Convert dictionary to configuration dataclass."""
        # This is a simplified implementation
        # In practice, you'd want more robust conversion
        config = MasterTrainingConfig()
        
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config
    
    def validate_config(self, config: MasterTrainingConfig) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Validate data pipeline
        if not config.data_pipeline.symbols:
            issues.append("Data pipeline: No symbols specified")
        
        if config.data_pipeline.validation_split + config.data_pipeline.test_split >= 1.0:
            issues.append("Data pipeline: Validation and test splits sum to >= 1.0")
        
        # Validate model architecture
        if config.model_architecture.state_size != len(config.data_pipeline.feature_columns):
            issues.append(f"Model architecture: state_size ({config.model_architecture.state_size}) "
                         f"doesn't match feature_columns length ({len(config.data_pipeline.feature_columns)})")
        
        # Validate training phases
        for phase in config.comprehensive_training.phases:
            if phase['batch_size'] <= 0:
                issues.append(f"Training phase {phase['name']}: Invalid batch size")
            
            if phase['learning_rate'] <= 0:
                issues.append(f"Training phase {phase['name']}: Invalid learning rate")
        
        # Validate experience buffer
        if config.experience_buffer.max_size <= 0:
            issues.append("Experience buffer: Invalid max_size")
        
        # Validate success criteria
        if config.target_accuracy <= 0 or config.target_accuracy > 1:
            issues.append("Invalid target accuracy")
        
        return issues
    
    def get_config_summary(self, config: MasterTrainingConfig) -> str:
        """
        Generate a human-readable configuration summary.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Configuration summary string
        """
        summary = f"""
AlpacaBot Training Configuration Summary
=====================================

Data Pipeline:
- Symbols: {len(config.data_pipeline.symbols)} symbols
- Date range: {config.data_pipeline.start_date} to {config.data_pipeline.end_date}
- Features: {len(config.data_pipeline.feature_columns)} features
- Validation split: {config.data_pipeline.validation_split}
- Test split: {config.data_pipeline.test_split}

Model Architecture:
- State size: {config.model_architecture.state_size}
- Hidden layers: {config.model_architecture.hidden_sizes}
- Dropout rate: {config.model_architecture.dropout_rate}
- Attention dimension: {config.model_architecture.attention_dim}

Training Pipeline:
- Number of phases: {len(config.comprehensive_training.phases)}
- Total max epochs: {sum(phase['duration_epochs'] for phase in config.comprehensive_training.phases)}
- Early stopping patience: {config.comprehensive_training.early_stopping['patience']}

Experience Buffer:
- Max size: {config.experience_buffer.max_size:,}
- Prioritization alpha: {config.experience_buffer.alpha}
- Regime specific: {config.experience_buffer.regime_specific}

Validation:
- Cross-validation splits: {config.validation.n_splits}
- Test size: {config.validation.test_size}
- Stress testing: {config.validation.stress_test_enabled}
- Performance metrics: {len(config.validation.performance_metrics)} metrics

Success Criteria:
- Target accuracy: {config.target_accuracy}
- Target Sharpe ratio: {config.target_sharpe_ratio}
- Target max drawdown: {config.target_max_drawdown}
- Target balance score: {config.target_balance_score}

Global Settings:
- Random seed: {config.random_seed}
- Output directory: {config.output_dir}
- Experiment name: {config.experiment_name}
- Use GPU: {config.use_gpu}
        """
        
        return summary.strip()


def create_default_config() -> MasterTrainingConfig:
    """Create and return default training configuration."""
    return MasterTrainingConfig()


def load_config_from_file(config_path: str) -> MasterTrainingConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager()
    return manager.load_config(config_path)


def save_config_to_file(config: MasterTrainingConfig, config_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
    """
    manager = ConfigManager()
    manager.save_config(config, config_path)


def main():
    """Test configuration management."""
    # Create default configuration
    config = create_default_config()
    
    # Create config manager
    manager = ConfigManager()
    
    # Print configuration summary
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(manager.get_config_summary(config))
    
    # Validate configuration
    issues = manager.validate_config(config)
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration validation: PASSED")
    
    # Save configuration
    config_path = "config/default_training_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    manager.save_config(config, config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    print("="*60)


if __name__ == "__main__":
    main()