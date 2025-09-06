"""
Comprehensive Training Pipeline for AlpacaBot
Stratified training with balanced actions, regime-specific phases, and progressive difficulty
Author: AlpacaBot Training System
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import random
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingPhase:
    """Configuration for a training phase."""
    name: str
    duration_epochs: int
    difficulty_level: float
    regime_filter: Optional[str]
    action_balance: Dict[int, float]
    learning_rate: float
    batch_size: int
    description: str


@dataclass
class TrainingMetrics:
    """Training metrics for a phase or epoch."""
    phase: str
    epoch: int
    loss: float
    accuracy: float
    action_distribution: Dict[int, int]
    regime_performance: Dict[str, float]
    sharpe_ratio: float
    max_drawdown: float
    total_return: float


class ProgressiveDifficultyScheduler:
    """Manages progressive difficulty increases during training."""
    
    def __init__(self):
        self.difficulty_levels = {
            'easy': {
                'description': 'Clear trend signals, low noise',
                'noise_factor': 0.0,
                'trend_strength_threshold': 0.7,
                'volatility_threshold': 0.02
            },
            'medium': {
                'description': 'Mixed signals, moderate noise',
                'noise_factor': 0.1,
                'trend_strength_threshold': 0.5,
                'volatility_threshold': 0.04
            },
            'hard': {
                'description': 'Weak signals, high noise',
                'noise_factor': 0.2,
                'trend_strength_threshold': 0.3,
                'volatility_threshold': 0.06
            },
            'extreme': {
                'description': 'Market crashes, extreme volatility',
                'noise_factor': 0.3,
                'trend_strength_threshold': 0.1,
                'volatility_threshold': 0.1
            }
        }
    
    def filter_data_by_difficulty(self, 
                                 data: pd.DataFrame, 
                                 difficulty: str) -> pd.DataFrame:
        """
        Filter data based on difficulty level.
        
        Args:
            data: Training data DataFrame
            difficulty: Difficulty level
            
        Returns:
            Filtered DataFrame
        """
        if difficulty not in self.difficulty_levels:
            return data
        
        config = self.difficulty_levels[difficulty]
        
        # Filter by volatility
        vol_threshold = config['volatility_threshold']
        if difficulty == 'easy':
            filtered_data = data[data['Volatility_20'] <= vol_threshold]
        elif difficulty == 'extreme':
            filtered_data = data[data['Volatility_20'] >= vol_threshold]
        else:
            # Include all volatility levels for medium/hard
            filtered_data = data.copy()
        
        # Add noise for higher difficulties
        if config['noise_factor'] > 0:
            noise_cols = ['RSI', 'MACD', 'BB_Position']
            for col in noise_cols:
                if col in filtered_data.columns:
                    noise = np.random.normal(0, config['noise_factor'], len(filtered_data))
                    filtered_data[col] += noise
        
        return filtered_data


class StratifiedTrainer:
    """
    Comprehensive trainer with stratified sampling, regime awareness,
    and progressive difficulty training.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 reward_system: Any,
                 config: Dict[str, Any] = None):
        """
        Initialize comprehensive trainer.
        
        Args:
            model: Neural network model
            reward_system: Enhanced reward system
            config: Training configuration
        """
        self.model = model
        self.reward_system = reward_system
        self.config = config or self._get_default_config()
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.loss_function = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
        
        # Progressive difficulty
        self.difficulty_scheduler = ProgressiveDifficultyScheduler()
        
        # Training state
        self.training_history = []
        self.phase_metrics = defaultdict(list)
        self.current_phase = None
        
        # Data splits by regime
        self.regime_data = {}
        self.action_samplers = {}
        
        # Performance tracking
        self.best_model_state = None
        self.best_metrics = {'accuracy': 0, 'sharpe_ratio': 0}
        
        logger.info("Comprehensive Trainer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'phases': [
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
            ],
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.001,
                'monitor_metric': 'accuracy'
            },
            'validation_split': 0.2,
            'test_split': 0.1,
            'checkpoint_every': 5,
            'save_dir': 'models/checkpoints'
        }
    
    def prepare_stratified_data(self, 
                               X: np.ndarray, 
                               y: np.ndarray, 
                               metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare stratified data splits by regime and action.
        
        Args:
            X: Feature array
            y: Label array
            metadata: Metadata with regime information
            
        Returns:
            Dictionary with prepared data splits
        """
        logger.info("Preparing stratified data splits")
        
        # Create combined dataset
        data = pd.DataFrame(X)
        data['label'] = y
        data['regime'] = metadata['Market_Regime'].values
        data['symbol'] = metadata['Symbol'].values
        
        # Split data by regime
        self.regime_data = {}
        for regime in data['regime'].unique():
            regime_mask = data['regime'] == regime
            regime_data = data[regime_mask]
            
            # Further split by action for balanced sampling
            action_splits = {}
            for action in [-1, 0, 1]:
                action_mask = regime_data['label'] == action
                action_data = regime_data[action_mask]
                if len(action_data) > 0:
                    action_splits[action] = action_data
            
            self.regime_data[regime] = {
                'full_data': regime_data,
                'action_splits': action_splits,
                'size': len(regime_data)
            }
        
        # Create overall train/val/test splits
        train_data, temp_data = train_test_split(
            data, 
            test_size=self.config['validation_split'] + self.config['test_split'],
            stratify=data['label'],
            random_state=42
        )
        
        val_size = self.config['validation_split'] / (
            self.config['validation_split'] + self.config['test_split']
        )
        val_data, test_data = train_test_split(
            temp_data,
            test_size=1-val_size,
            stratify=temp_data['label'],
            random_state=42
        )
        
        # Prepare features and labels
        feature_cols = [col for col in data.columns if col not in ['label', 'regime', 'symbol']]
        
        X_train = train_data[feature_cols].values
        y_train = train_data['label'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['label'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['label'].values
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'train': {
                'X': X_train_scaled,
                'y': y_train,
                'metadata': train_data[['regime', 'symbol']]
            },
            'val': {
                'X': X_val_scaled,
                'y': y_val,
                'metadata': val_data[['regime', 'symbol']]
            },
            'test': {
                'X': X_test_scaled,
                'y': y_test,
                'metadata': test_data[['regime', 'symbol']]
            },
            'feature_columns': feature_cols,
            'regime_stats': {regime: info['size'] for regime, info in self.regime_data.items()}
        }
    
    def create_balanced_batch(self, 
                             data_splits: Dict[str, Any],
                             phase_config: Dict[str, Any],
                             batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a balanced batch according to phase configuration.
        
        Args:
            data_splits: Data splits dictionary
            phase_config: Phase configuration
            batch_size: Batch size
            
        Returns:
            Batch features and labels
        """
        target_balance = phase_config['action_balance']
        regime_filter = phase_config.get('regime_filter')
        
        batch_X = []
        batch_y = []
        
        # Calculate samples per action
        samples_per_action = {}
        for action, proportion in target_balance.items():
            samples_per_action[action] = int(batch_size * proportion)
        
        # Adjust for rounding
        total_samples = sum(samples_per_action.values())
        if total_samples < batch_size:
            # Add remaining samples to largest category
            max_action = max(target_balance, key=target_balance.get)
            samples_per_action[max_action] += batch_size - total_samples
        
        # Sample from training data
        train_data = pd.DataFrame(data_splits['train']['X'])
        train_data['label'] = data_splits['train']['y']
        train_data['regime'] = data_splits['train']['metadata']['regime'].values
        
        # Apply regime filter if specified
        if regime_filter:
            train_data = train_data[train_data['regime'] == regime_filter]
        
        # Sample for each action
        for action, n_samples in samples_per_action.items():
            action_data = train_data[train_data['label'] == action]
            
            if len(action_data) > 0:
                if n_samples > len(action_data):
                    # Oversample if needed
                    sampled = action_data.sample(n=n_samples, replace=True)
                else:
                    sampled = action_data.sample(n=n_samples, replace=False)
                
                feature_cols = [col for col in sampled.columns if col not in ['label', 'regime']]
                batch_X.append(sampled[feature_cols].values)
                batch_y.extend([action] * len(sampled))
        
        if batch_X:
            batch_X = np.vstack(batch_X)
            batch_y = np.array(batch_y)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(batch_X)
            y_tensor = torch.LongTensor(batch_y + 1)  # Convert -1,0,1 to 0,1,2
            
            return X_tensor, y_tensor
        else:
            # Return empty tensors if no data available
            empty_X = torch.FloatTensor(0, data_splits['train']['X'].shape[1])
            empty_y = torch.LongTensor(0)
            return empty_X, empty_y
    
    def train_phase(self, 
                    phase_config: Dict[str, Any],
                    data_splits: Dict[str, Any]) -> Dict[str, float]:
        """
        Train a single phase with specific configuration.
        
        Args:
            phase_config: Phase configuration
            data_splits: Data splits
            
        Returns:
            Phase metrics
        """
        phase_name = phase_config['name']
        logger.info(f"Starting training phase: {phase_name}")
        
        # Update optimizer for this phase
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=phase_config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=5,
            verbose=True
        )
        
        phase_metrics = []
        best_phase_metric = 0
        patience_counter = 0
        
        for epoch in range(phase_config['duration_epochs']):
            # Training
            self.model.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            
            # Calculate number of batches
            train_size = len(data_splits['train']['X'])
            n_batches = max(1, train_size // phase_config['batch_size'])
            
            for batch_idx in range(n_batches):
                # Create balanced batch
                X_batch, y_batch = self.create_balanced_batch(
                    data_splits, phase_config, phase_config['batch_size']
                )
                
                if len(X_batch) == 0:
                    continue
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_function(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
            
            # Validation
            val_metrics = self._validate_model(data_splits['val'])
            
            # Calculate epoch metrics
            epoch_metrics = {
                'phase': phase_name,
                'epoch': epoch,
                'train_loss': np.mean(train_losses) if train_losses else 0.0,
                'train_accuracy': train_correct / train_total if train_total > 0 else 0.0,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_action_distribution': val_metrics['action_distribution'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            phase_metrics.append(epoch_metrics)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping check
            monitor_metric = val_metrics[self.config['early_stopping']['monitor_metric']]
            if monitor_metric > best_phase_metric + self.config['early_stopping']['min_delta']:
                best_phase_metric = monitor_metric
                patience_counter = 0
                
                # Save best model for this phase
                if monitor_metric > self.best_metrics['accuracy']:
                    self.best_metrics['accuracy'] = monitor_metric
                    self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(
                    f"Phase {phase_name}, Epoch {epoch}/{phase_config['duration_epochs']}: "
                    f"Train Acc: {epoch_metrics['train_accuracy']:.4f}, "
                    f"Val Acc: {epoch_metrics['val_accuracy']:.4f}, "
                    f"Val Loss: {epoch_metrics['val_loss']:.4f}"
                )
            
            # Early stopping
            if patience_counter >= self.config['early_stopping']['patience']:
                logger.info(f"Early stopping triggered for phase {phase_name}")
                break
            
            # Checkpoint saving
            if epoch % self.config['checkpoint_every'] == 0:
                self._save_checkpoint(phase_name, epoch, epoch_metrics)
        
        # Store phase metrics
        self.phase_metrics[phase_name] = phase_metrics
        
        # Return final metrics
        final_metrics = phase_metrics[-1] if phase_metrics else {}
        logger.info(f"Completed phase {phase_name}: Final accuracy: {final_metrics.get('val_accuracy', 0):.4f}")
        
        return final_metrics
    
    def _validate_model(self, val_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model on validation data.
        
        Args:
            val_data: Validation data dictionary
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        X_val = torch.FloatTensor(val_data['X'])
        y_val = torch.LongTensor(val_data['y'] + 1)  # Convert -1,0,1 to 0,1,2
        
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self.loss_function(outputs, y_val).item()
            
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_val).float().mean().item()
            
            # Action distribution
            predicted_actions = predicted.numpy() - 1  # Convert back to -1,0,1
            action_dist = {
                int(action): int(count) 
                for action, count in zip(*np.unique(predicted_actions, return_counts=True))
            }
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'action_distribution': action_dist
        }
    
    def train_comprehensive(self, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive training pipeline with all phases.
        
        Args:
            X: Feature array
            y: Label array
            metadata: Metadata DataFrame
            
        Returns:
            Complete training results
        """
        logger.info("Starting comprehensive training pipeline")
        
        # Prepare stratified data
        data_splits = self.prepare_stratified_data(X, y, metadata)
        
        # Create checkpoint directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Train each phase
        phase_results = {}
        
        for phase_config in self.config['phases']:
            self.current_phase = phase_config['name']
            
            # Apply difficulty filtering if needed
            difficulty = phase_config.get('difficulty_level', 'medium')
            if difficulty in self.difficulty_scheduler.difficulty_levels:
                # Filter training data by difficulty
                # This is a simplified implementation - full implementation would
                # modify the data preparation to include difficulty filtering
                pass
            
            # Train the phase
            phase_result = self.train_phase(phase_config, data_splits)
            phase_results[phase_config['name']] = phase_result
        
        # Final evaluation on test set
        test_results = self._evaluate_final_model(data_splits['test'])
        
        # Compile comprehensive results
        results = {
            'phase_results': phase_results,
            'test_results': test_results,
            'best_metrics': self.best_metrics,
            'training_history': self.phase_metrics,
            'regime_stats': data_splits['regime_stats'],
            'model_architecture': str(self.model),
            'config': self.config
        }
        
        # Save final model and results
        self._save_final_results(results)
        
        logger.info("Comprehensive training completed successfully")
        return results
    
    def _evaluate_final_model(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate final model on test set.
        
        Args:
            test_data: Test data dictionary
            
        Returns:
            Test evaluation metrics
        """
        logger.info("Evaluating final model on test set")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        self.model.eval()
        
        X_test = torch.FloatTensor(test_data['X'])
        y_test = test_data['y']
        y_test_tensor = torch.LongTensor(y_test + 1)
        
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            
            test_accuracy = (predicted == y_test_tensor).float().mean().item()
            predicted_labels = predicted.numpy() - 1  # Convert back to -1,0,1
        
        # Classification report
        class_report = classification_report(
            y_test, predicted_labels, 
            target_names=['Sell', 'Hold', 'Buy'],
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predicted_labels)
        
        # Action distribution analysis
        action_distribution = {
            'actual': {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
            'predicted': {int(k): int(v) for k, v in zip(*np.unique(predicted_labels, return_counts=True))}
        }
        
        # Calculate balance metrics
        total_predictions = len(predicted_labels)
        predicted_dist = action_distribution['predicted']
        balance_score = self._calculate_balance_score(predicted_dist, total_predictions)
        
        return {
            'accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'action_distribution': action_distribution,
            'balance_score': balance_score,
            'n_samples': len(y_test)
        }
    
    def _calculate_balance_score(self, action_counts: Dict[int, int], total: int) -> float:
        """
        Calculate how balanced the action distribution is.
        
        Args:
            action_counts: Count of each action
            total: Total number of predictions
            
        Returns:
            Balance score (1.0 = perfectly balanced, 0.0 = completely imbalanced)
        """
        target_prop = 1/3  # Ideal proportion for each action
        
        balance_score = 0.0
        for action in [-1, 0, 1]:
            actual_prop = action_counts.get(action, 0) / total if total > 0 else 0
            balance_score += 1 - abs(actual_prop - target_prop) / target_prop
        
        return balance_score / 3  # Average across all actions
    
    def _save_checkpoint(self, phase: str, epoch: int, metrics: Dict[str, Any]):
        """Save training checkpoint."""
        checkpoint = {
            'phase': phase,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.config['save_dir'], 
            f"checkpoint_{phase}_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final training results."""
        # Save model
        model_path = os.path.join(self.config['save_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': self.best_model_state or self.model.state_dict(),
            'model_architecture': str(self.model),
            'scaler_state': {
                'mean_': self.scaler.mean_.tolist(),
                'scale_': self.scaler.scale_.tolist()
            }
        }, model_path)
        
        # Save results
        results_path = os.path.join(self.config['save_dir'], 'training_results.json')
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save training plots
        self._create_training_plots(results)
        
        logger.info(f"Final results saved to {self.config['save_dir']}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _create_training_plots(self, results: Dict[str, Any]):
        """Create training visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Comprehensive Training Results', fontsize=16)
            
            # Plot 1: Accuracy over phases
            phase_names = []
            phase_accuracies = []
            
            for phase_name, metrics in results['phase_results'].items():
                phase_names.append(phase_name)
                phase_accuracies.append(metrics.get('val_accuracy', 0))
            
            axes[0, 0].bar(range(len(phase_names)), phase_accuracies)
            axes[0, 0].set_title('Validation Accuracy by Phase')
            axes[0, 0].set_xlabel('Training Phase')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(range(len(phase_names)))
            axes[0, 0].set_xticklabels(phase_names, rotation=45, ha='right')
            
            # Plot 2: Action distribution in test predictions
            if 'test_results' in results and 'action_distribution' in results['test_results']:
                test_dist = results['test_results']['action_distribution']['predicted']
                actions = list(test_dist.keys())
                counts = list(test_dist.values())
                
                axes[0, 1].pie(counts, labels=['Sell', 'Hold', 'Buy'], autopct='%1.1f%%')
                axes[0, 1].set_title('Test Set Action Distribution')
            
            # Plot 3: Training loss progression (if available)
            # This would require storing loss history across phases
            axes[1, 0].text(0.5, 0.5, 'Training Loss Plot\n(Implementation depends on loss history storage)', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Progress')
            
            # Plot 4: Confusion matrix
            if 'test_results' in results and 'confusion_matrix' in results['test_results']:
                cm = np.array(results['test_results']['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Sell', 'Hold', 'Buy'],
                           yticklabels=['Sell', 'Hold', 'Buy'],
                           ax=axes[1, 1])
                axes[1, 1].set_title('Test Set Confusion Matrix')
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Actual')
            
            plt.tight_layout()
            plot_path = os.path.join(self.config['save_dir'], 'training_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not create training plots: {str(e)}")


def create_test_model(input_size: int = 15) -> nn.Module:
    """Create a test neural network model."""
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 3)  # Output: sell, hold, buy
    )


def main():
    """Main function to test the comprehensive trainer."""
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], size=n_samples)
    
    # Create metadata
    regimes = ['bull', 'bear', 'sideways', 'high_volatility']
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    
    metadata = pd.DataFrame({
        'Market_Regime': np.random.choice(regimes, n_samples),
        'Symbol': np.random.choice(symbols, n_samples)
    })
    
    # Create model and reward system
    model = create_test_model(n_features)
    
    # Import reward system (would normally be from the enhanced_reward_system module)
    class MockRewardSystem:
        def __init__(self):
            pass
    
    reward_system = MockRewardSystem()
    
    # Create trainer
    trainer = StratifiedTrainer(model, reward_system)
    
    # Run training
    try:
        results = trainer.train_comprehensive(X, y, metadata)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE TRAINING RESULTS")
        print("="*60)
        print(f"Best Accuracy: {results['best_metrics']['accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
        print(f"Balance Score: {results['test_results']['balance_score']:.4f}")
        print(f"Action Distribution: {results['test_results']['action_distribution']['predicted']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()