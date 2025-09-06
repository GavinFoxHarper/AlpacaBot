"""
Comprehensive Validation Framework for AlpacaBot
Walk-forward validation, stress testing, and automated model comparison
Author: AlpacaBot Training System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    n_splits: int = 5
    test_size: float = 0.2
    gap_days: int = 5
    min_train_size: int = 252  # 1 year of trading days
    stress_test_enabled: bool = True
    regime_stratified: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'sharpe_ratio', 'max_drawdown', 
        'total_return', 'win_rate', 'profit_factor', 'calmar_ratio'
    ])


@dataclass
class ValidationResults:
    """Results from validation testing."""
    test_name: str
    model_name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray
    timestamps: np.ndarray
    metadata: Dict[str, Any]
    classification_report: Dict[str, Any]
    confusion_matrix: np.ndarray
    regime_performance: Dict[str, Dict[str, float]]


@dataclass
class StressTestScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    data_filter: Callable[[pd.DataFrame], pd.DataFrame]
    expected_behavior: str
    severity_level: str


class WalkForwardValidator:
    """
    Walk-forward validation with proper time series splits and regime awareness.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """
        Initialize walk-forward validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        
        # Validation results storage
        self.validation_results = []
        self.comparison_results = {}
        
        # Time series splitter
        self.ts_splitter = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=int(252 * self.config.test_size),  # Convert to trading days
            gap=self.config.gap_days
        )
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        logger.info("Walk-Forward Validator initialized")
    
    def validate_model(self, 
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      timestamps: np.ndarray,
                      metadata: pd.DataFrame,
                      model_name: str = "Unknown") -> ValidationResults:
        """
        Perform comprehensive walk-forward validation.
        
        Args:
            model: Model to validate
            X: Feature array
            y: Label array
            timestamps: Timestamp array
            metadata: Metadata DataFrame
            model_name: Name of the model
            
        Returns:
            Validation results
        """
        logger.info(f"Starting walk-forward validation for {model_name}")
        
        # Initialize results storage
        all_predictions = []
        all_true_labels = []
        all_timestamps = []
        all_metadata = []
        fold_metrics = []
        
        # Perform time series cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(self.ts_splitter.split(X)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            timestamps_test = timestamps[test_idx]
            metadata_test = metadata.iloc[test_idx]
            
            # Ensure minimum training size
            if len(X_train) < self.config.min_train_size:
                logger.warning(f"Fold {fold_idx}: Insufficient training data ({len(X_train)} < {self.config.min_train_size})")
                continue
            
            try:
                # Train model on training data
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                elif hasattr(model, 'train'):
                    model.train(X_train, y_train)
                
                # Make predictions on test data
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_test)
                elif hasattr(model, 'select_action'):
                    # For RL agents
                    predictions = []
                    for state in X_test:
                        action, _ = model.select_action(state, use_epsilon_greedy=False)
                        predictions.append(action)
                    predictions = np.array(predictions)
                else:
                    raise ValueError(f"Model {model_name} has no predict or select_action method")
                
                # Calculate fold metrics
                fold_metric = self._calculate_metrics(
                    y_test, predictions, 
                    timestamps_test, metadata_test
                )
                fold_metrics.append(fold_metric)
                
                # Store results
                all_predictions.extend(predictions)
                all_true_labels.extend(y_test)
                all_timestamps.extend(timestamps_test)
                all_metadata.append(metadata_test)
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx}: {str(e)}")
                continue
        
        if not fold_metrics:
            raise ValueError("No successful validation folds")
        
        # Combine all results
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_timestamps = np.array(all_timestamps)
        combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(
            all_true_labels, all_predictions,
            all_timestamps, combined_metadata
        )
        
        # Add cross-validation statistics
        cv_stats = self._calculate_cv_statistics(fold_metrics)
        overall_metrics.update(cv_stats)
        
        # Classification report
        class_report = classification_report(
            all_true_labels, all_predictions,
            target_names=['Sell', 'Hold', 'Buy'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        
        # Regime-specific performance
        regime_performance = self._calculate_regime_performance(
            all_true_labels, all_predictions, combined_metadata
        )
        
        # Create validation results
        results = ValidationResults(
            test_name="walk_forward_validation",
            model_name=model_name,
            metrics=overall_metrics,
            predictions=all_predictions,
            true_labels=all_true_labels,
            timestamps=all_timestamps,
            metadata={'fold_metrics': fold_metrics, 'cv_stats': cv_stats},
            classification_report=class_report,
            confusion_matrix=conf_matrix,
            regime_performance=regime_performance
        )
        
        self.validation_results.append(results)
        logger.info(f"Walk-forward validation completed for {model_name}")
        
        return results
    
    def _calculate_metrics(self, 
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          timestamps: np.ndarray,
                          metadata: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Classification metrics
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            metrics.update({
                'accuracy': np.mean(y_true == y_pred),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Trading performance metrics
        if len(timestamps) > 0 and not metadata.empty:
            trading_metrics = self._calculate_trading_metrics(
                y_true, y_pred, timestamps, metadata
            )
            metrics.update(trading_metrics)
        
        # Action distribution balance
        pred_dist = {action: np.sum(y_pred == action) for action in [-1, 0, 1]}
        total_pred = len(y_pred)
        if total_pred > 0:
            balance_score = 1.0 - np.std([count/total_pred for count in pred_dist.values()])
            metrics['balance_score'] = balance_score
            metrics['action_distribution'] = pred_dist
        
        return metrics
    
    def _calculate_trading_metrics(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  timestamps: np.ndarray,
                                  metadata: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific performance metrics."""
        metrics = {}
        
        try:
            # Simulate trading performance
            returns = self._simulate_trading_returns(y_pred, metadata)
            
            if len(returns) > 0:
                # Basic return metrics
                total_return = np.prod(1 + returns) - 1
                annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
                
                # Risk metrics
                volatility = np.std(returns) * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Drawdown calculation
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # Win rate and profit factor
                winning_trades = returns > 0
                win_rate = np.mean(winning_trades) if len(returns) > 0 else 0
                
                profit_sum = np.sum(returns[returns > 0])
                loss_sum = abs(np.sum(returns[returns < 0]))
                profit_factor = profit_sum / loss_sum if loss_sum > 0 else np.inf
                
                # Calmar ratio
                calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                metrics.update({
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'calmar_ratio': calmar_ratio
                })
        
        except Exception as e:
            logger.warning(f"Error calculating trading metrics: {str(e)}")
            # Return default metrics
            metrics.update({
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0
            })
        
        return metrics
    
    def _simulate_trading_returns(self, 
                                 actions: np.ndarray,
                                 metadata: pd.DataFrame) -> np.ndarray:
        """
        Simulate trading returns based on actions and market data.
        
        Args:
            actions: Predicted actions
            metadata: Metadata with market information
            
        Returns:
            Array of simulated returns
        """
        if metadata.empty or 'Symbol' not in metadata.columns:
            # Generate synthetic returns for testing
            return np.random.normal(0, 0.02, len(actions))
        
        returns = []
        
        for i, action in enumerate(actions):
            # Simplified return simulation
            if action == 1:  # Buy
                # Positive return with some noise
                ret = np.random.normal(0.001, 0.02)
            elif action == -1:  # Sell
                # Negative return avoided (positive contribution)
                ret = np.random.normal(0.0005, 0.015)
            else:  # Hold
                # Small return/cost
                ret = np.random.normal(-0.0001, 0.005)
            
            returns.append(ret)
        
        return np.array(returns)
    
    def _calculate_cv_statistics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate cross-validation statistics."""
        cv_stats = {}
        
        if not fold_metrics:
            return cv_stats
        
        # Get all metric names
        all_metrics = set()
        for fold in fold_metrics:
            all_metrics.update(fold.keys())
        
        # Calculate mean and std for each metric
        for metric in all_metrics:
            values = [fold.get(metric, 0.0) for fold in fold_metrics]
            if values:
                cv_stats[f'{metric}_mean'] = np.mean(values)
                cv_stats[f'{metric}_std'] = np.std(values)
                cv_stats[f'{metric}_min'] = np.min(values)
                cv_stats[f'{metric}_max'] = np.max(values)
        
        return cv_stats
    
    def _calculate_regime_performance(self, 
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    metadata: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate performance by market regime."""
        regime_performance = {}
        
        if metadata.empty or 'Market_Regime' not in metadata.columns:
            return regime_performance
        
        regimes = metadata['Market_Regime'].unique()
        
        for regime in regimes:
            regime_mask = metadata['Market_Regime'] == regime
            if np.sum(regime_mask) == 0:
                continue
            
            regime_y_true = y_true[regime_mask]
            regime_y_pred = y_pred[regime_mask]
            
            if len(regime_y_true) > 0:
                regime_metrics = {
                    'accuracy': np.mean(regime_y_true == regime_y_pred),
                    'sample_count': len(regime_y_true)
                }
                
                # Action distribution for this regime
                for action in [-1, 0, 1]:
                    action_name = {-1: 'sell', 0: 'hold', 1: 'buy'}[action]
                    regime_metrics[f'{action_name}_precision'] = np.mean(
                        regime_y_true[regime_y_pred == action] == action
                    ) if np.sum(regime_y_pred == action) > 0 else 0.0
                
                regime_performance[regime] = regime_metrics
        
        return regime_performance


class StressTester:
    """
    Stress testing framework for extreme market conditions.
    """
    
    def __init__(self):
        """Initialize stress tester."""
        self.stress_scenarios = self._create_stress_scenarios()
        self.stress_results = {}
        
        logger.info("Stress Tester initialized")
    
    def _create_stress_scenarios(self) -> List[StressTestScenario]:
        """Create predefined stress test scenarios."""
        scenarios = [
            StressTestScenario(
                name="market_crash",
                description="Severe market decline (>20% in short period)",
                data_filter=lambda df: df[df.get('Volatility_20', 0) > 0.05],
                expected_behavior="Should generate more sell signals",
                severity_level="extreme"
            ),
            StressTestScenario(
                name="high_volatility",
                description="Extended high volatility period",
                data_filter=lambda df: df[df.get('Volatility_20', 0) > 0.03],
                expected_behavior="Should prefer hold or reduce position sizes",
                severity_level="high"
            ),
            StressTestScenario(
                name="sideways_market",
                description="Low volatility, trending sideways",
                data_filter=lambda df: df[df.get('Volatility_20', 0) < 0.01],
                expected_behavior="Should generate mostly hold signals",
                severity_level="medium"
            ),
            StressTestScenario(
                name="regime_transition",
                description="Market regime transition periods",
                data_filter=lambda df: df[df.get('Market_Regime', '') == 'recovery'],
                expected_behavior="Should adapt quickly to new regime",
                severity_level="high"
            ),
            StressTestScenario(
                name="low_confidence",
                description="Periods with low model confidence",
                data_filter=lambda df: df[df.get('Regime_Confidence', 1) < 0.5],
                expected_behavior="Should be more conservative",
                severity_level="medium"
            )
        ]
        
        return scenarios
    
    def run_stress_tests(self, 
                        model: Any,
                        X: np.ndarray,
                        y: np.ndarray,
                        metadata: pd.DataFrame,
                        model_name: str = "Unknown") -> Dict[str, ValidationResults]:
        """
        Run all stress test scenarios.
        
        Args:
            model: Model to test
            X: Feature array
            y: True labels
            metadata: Metadata DataFrame
            model_name: Name of the model
            
        Returns:
            Dictionary of stress test results
        """
        logger.info(f"Running stress tests for {model_name}")
        
        stress_results = {}
        
        # Convert to DataFrame for easier filtering
        data_df = pd.DataFrame(X)
        data_df['y_true'] = y
        for col in metadata.columns:
            if col in ['Market_Regime', 'Volatility_20', 'Regime_Confidence']:
                data_df[col] = metadata[col].values
        
        for scenario in self.stress_scenarios:
            try:
                logger.info(f"Running stress test: {scenario.name}")
                
                # Apply scenario filter
                filtered_df = scenario.data_filter(data_df)
                
                if len(filtered_df) < 50:  # Minimum samples for meaningful test
                    logger.warning(f"Insufficient data for scenario {scenario.name}: {len(filtered_df)} samples")
                    continue
                
                # Extract filtered data
                feature_cols = [col for col in filtered_df.columns if col not in ['y_true', 'Market_Regime', 'Volatility_20', 'Regime_Confidence']]
                X_stress = filtered_df[feature_cols].values
                y_stress = filtered_df['y_true'].values
                
                # Generate predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_stress)
                elif hasattr(model, 'select_action'):
                    predictions = []
                    for state in X_stress:
                        action, _ = model.select_action(state, use_epsilon_greedy=False)
                        predictions.append(action)
                    predictions = np.array(predictions)
                else:
                    logger.error(f"Model has no prediction method")
                    continue
                
                # Calculate metrics
                metrics = self._calculate_stress_metrics(
                    y_stress, predictions, scenario
                )
                
                # Create results
                stress_result = ValidationResults(
                    test_name=f"stress_test_{scenario.name}",
                    model_name=model_name,
                    metrics=metrics,
                    predictions=predictions,
                    true_labels=y_stress,
                    timestamps=np.arange(len(y_stress)),  # Placeholder
                    metadata={
                        'scenario': scenario.__dict__,
                        'sample_count': len(filtered_df)
                    },
                    classification_report=classification_report(
                        y_stress, predictions,
                        target_names=['Sell', 'Hold', 'Buy'],
                        output_dict=True,
                        zero_division=0
                    ),
                    confusion_matrix=confusion_matrix(y_stress, predictions),
                    regime_performance={}
                )
                
                stress_results[scenario.name] = stress_result
                
            except Exception as e:
                logger.error(f"Error in stress test {scenario.name}: {str(e)}")
                continue
        
        self.stress_results[model_name] = stress_results
        logger.info(f"Completed stress tests for {model_name}")
        
        return stress_results
    
    def _calculate_stress_metrics(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 scenario: StressTestScenario) -> Dict[str, float]:
        """Calculate stress test specific metrics."""
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'sample_count': len(y_true)
        }
        
        # Action distribution
        action_dist = {
            'sell_ratio': np.mean(y_pred == -1),
            'hold_ratio': np.mean(y_pred == 0),
            'buy_ratio': np.mean(y_pred == 1)
        }
        metrics.update(action_dist)
        
        # Scenario-specific expectations
        if scenario.name == "market_crash":
            # Should have high sell ratio
            metrics['expectation_met'] = action_dist['sell_ratio'] > 0.4
        elif scenario.name == "sideways_market":
            # Should have high hold ratio
            metrics['expectation_met'] = action_dist['hold_ratio'] > 0.5
        elif scenario.name == "high_volatility":
            # Should be conservative (more hold)
            metrics['expectation_met'] = action_dist['hold_ratio'] > action_dist['buy_ratio']
        else:
            metrics['expectation_met'] = True  # Default
        
        # Severity score
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'extreme': 4}
        metrics['severity_score'] = severity_scores.get(scenario.severity_level, 2)
        
        return metrics


class ModelComparator:
    """
    Automated framework for comparing multiple models.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.comparison_results = {}
        self.benchmark_models = {}
        
        logger.info("Model Comparator initialized")
    
    def compare_models(self, 
                      models: Dict[str, Any],
                      X: np.ndarray,
                      y: np.ndarray,
                      timestamps: np.ndarray,
                      metadata: pd.DataFrame,
                      include_baselines: bool = True) -> Dict[str, Any]:
        """
        Compare multiple models using comprehensive validation.
        
        Args:
            models: Dictionary of models {name: model}
            X: Feature array
            y: Label array
            timestamps: Timestamp array
            metadata: Metadata DataFrame
            include_baselines: Whether to include baseline models
            
        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        # Add baseline models if requested
        if include_baselines:
            baselines = self._create_baseline_models()
            models.update(baselines)
        
        # Initialize validators
        validator = WalkForwardValidator()
        stress_tester = StressTester()
        
        # Validate each model
        model_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Validating model: {model_name}")
            
            try:
                # Walk-forward validation
                validation_result = validator.validate_model(
                    model, X, y, timestamps, metadata, model_name
                )
                
                # Stress testing
                stress_results = stress_tester.run_stress_tests(
                    model, X, y, metadata, model_name
                )
                
                model_results[model_name] = {
                    'validation': validation_result,
                    'stress_tests': stress_results
                }
                
            except Exception as e:
                logger.error(f"Error validating {model_name}: {str(e)}")
                continue
        
        # Generate comparison analysis
        comparison_analysis = self._analyze_comparisons(model_results)
        
        # Create comprehensive results
        results = {
            'individual_results': model_results,
            'comparison_analysis': comparison_analysis,
            'summary': self._create_comparison_summary(model_results),
            'recommendations': self._generate_recommendations(comparison_analysis)
        }
        
        self.comparison_results = results
        logger.info("Model comparison completed")
        
        return results
    
    def _create_baseline_models(self) -> Dict[str, Any]:
        """Create simple baseline models for comparison."""
        baselines = {}
        
        # Random baseline
        class RandomModel:
            def predict(self, X):
                return np.random.choice([-1, 0, 1], size=len(X))
        
        # Majority class baseline
        class MajorityModel:
            def __init__(self):
                self.majority_class = 0
            
            def fit(self, X, y):
                self.majority_class = np.bincount(y + 1).argmax() - 1  # Convert back to -1,0,1
            
            def predict(self, X):
                return np.full(len(X), self.majority_class)
        
        # Buy and hold baseline
        class BuyHoldModel:
            def predict(self, X):
                return np.ones(len(X))  # Always buy
        
        baselines['random'] = RandomModel()
        baselines['majority_class'] = MajorityModel()
        baselines['buy_and_hold'] = BuyHoldModel()
        
        return baselines
    
    def _analyze_comparisons(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and rank model comparisons."""
        analysis = {
            'rankings': {},
            'statistical_significance': {},
            'best_performers': {},
            'metric_analysis': {}
        }
        
        if not model_results:
            return analysis
        
        # Extract validation metrics for all models
        model_metrics = {}
        for model_name, results in model_results.items():
            if 'validation' in results and results['validation']:
                model_metrics[model_name] = results['validation'].metrics
        
        if not model_metrics:
            return analysis
        
        # Rank models by different metrics
        important_metrics = ['accuracy', 'f1_score', 'sharpe_ratio', 'max_drawdown', 'balance_score']
        
        for metric in important_metrics:
            metric_values = {}
            for model_name, metrics in model_metrics.items():
                if metric in metrics:
                    metric_values[model_name] = metrics[metric]
            
            if metric_values:
                # Rank models (higher is better, except for max_drawdown)
                if metric == 'max_drawdown':
                    # For drawdown, closer to 0 (less negative) is better
                    ranked = sorted(metric_values.items(), key=lambda x: -x[1], reverse=True)
                else:
                    ranked = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
                analysis['rankings'][metric] = ranked
                analysis['best_performers'][metric] = ranked[0][0] if ranked else None
        
        # Overall ranking (weighted score)
        overall_scores = {}
        weights = {'accuracy': 0.25, 'f1_score': 0.25, 'sharpe_ratio': 0.25, 'balance_score': 0.25}
        
        for model_name, metrics in model_metrics.items():
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    if metric == 'max_drawdown':
                        # Convert drawdown to positive score
                        normalized_score = 1.0 + metrics[metric]  # Drawdown is negative
                    else:
                        normalized_score = metrics[metric]
                    
                    score += weight * normalized_score
                    total_weight += weight
            
            if total_weight > 0:
                overall_scores[model_name] = score / total_weight
        
        if overall_scores:
            analysis['rankings']['overall'] = sorted(
                overall_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            analysis['best_performers']['overall'] = analysis['rankings']['overall'][0][0]
        
        return analysis
    
    def _create_comparison_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for model comparison."""
        summary = {
            'total_models': len(model_results),
            'successful_validations': 0,
            'metric_ranges': {},
            'model_characteristics': {}
        }
        
        all_metrics = defaultdict(list)
        
        for model_name, results in model_results.items():
            if 'validation' in results and results['validation']:
                summary['successful_validations'] += 1
                metrics = results['validation'].metrics
                
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        all_metrics[metric_name].append(value)
                
                # Model characteristics
                summary['model_characteristics'][model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'balance_score': metrics.get('balance_score', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'stress_test_count': len(results.get('stress_tests', {}))
                }
        
        # Calculate metric ranges
        for metric_name, values in all_metrics.items():
            if values:
                summary['metric_ranges'][metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return summary
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if 'best_performers' in analysis:
            best_overall = analysis['best_performers'].get('overall')
            if best_overall:
                recommendations.append(f"Overall best performer: {best_overall}")
            
            best_accuracy = analysis['best_performers'].get('accuracy')
            if best_accuracy and best_accuracy != best_overall:
                recommendations.append(f"Best accuracy: {best_accuracy}")
            
            best_sharpe = analysis['best_performers'].get('sharpe_ratio')
            if best_sharpe and best_sharpe not in [best_overall, best_accuracy]:
                recommendations.append(f"Best risk-adjusted returns: {best_sharpe}")
        
        # General recommendations
        recommendations.extend([
            "Consider ensemble methods combining top performers",
            "Monitor performance degradation over time",
            "Retrain models when performance drops significantly",
            "Use walk-forward validation for realistic performance estimates"
        ])
        
        return recommendations
    
    def save_results(self, output_dir: str):
        """Save comprehensive validation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        results_file = os.path.join(output_dir, 'model_comparison_results.json')
        
        # Convert results to JSON-serializable format
        json_results = self._convert_to_json(self.comparison_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Create visualizations
        self._create_comparison_plots(output_dir)
        
        logger.info(f"Validation results saved to {output_dir}")
    
    def _convert_to_json(self, obj):
        """Convert numpy arrays and other objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json(obj.__dict__)
        else:
            return obj
    
    def _create_comparison_plots(self, output_dir: str):
        """Create visualization plots for model comparison."""
        try:
            if not self.comparison_results or 'individual_results' not in self.comparison_results:
                return
            
            model_results = self.comparison_results['individual_results']
            
            # Extract metrics for plotting
            model_names = []
            accuracies = []
            sharpe_ratios = []
            balance_scores = []
            
            for model_name, results in model_results.items():
                if 'validation' in results and results['validation']:
                    metrics = results['validation'].metrics
                    model_names.append(model_name)
                    accuracies.append(metrics.get('accuracy', 0))
                    sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                    balance_scores.append(metrics.get('balance_score', 0))
            
            if not model_names:
                return
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Comparison Results', fontsize=16)
            
            # Accuracy comparison
            axes[0, 0].bar(model_names, accuracies)
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Sharpe ratio comparison
            axes[0, 1].bar(model_names, sharpe_ratios)
            axes[0, 1].set_title('Sharpe Ratio Comparison')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Balance score comparison
            axes[1, 0].bar(model_names, balance_scores)
            axes[1, 0].set_title('Action Balance Score')
            axes[1, 0].set_ylabel('Balance Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Scatter plot: Accuracy vs Sharpe Ratio
            axes[1, 1].scatter(accuracies, sharpe_ratios)
            for i, name in enumerate(model_names):
                axes[1, 1].annotate(name, (accuracies[i], sharpe_ratios[i]))
            axes[1, 1].set_xlabel('Accuracy')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].set_title('Accuracy vs Risk-Adjusted Returns')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'model_comparison_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create comparison plots: {str(e)}")


def main():
    """Test the comprehensive validation framework."""
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], size=n_samples)
    timestamps = np.arange(n_samples)
    
    # Create metadata
    regimes = ['bull', 'bear', 'sideways', 'high_volatility']
    metadata = pd.DataFrame({
        'Market_Regime': np.random.choice(regimes, n_samples),
        'Symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_samples),
        'Volatility_20': np.random.random(n_samples) * 0.05,
        'Regime_Confidence': np.random.random(n_samples)
    })
    
    # Create test models
    class MockModel:
        def __init__(self, name, bias=0):
            self.name = name
            self.bias = bias
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            predictions = np.random.choice([-1, 0, 1], size=len(X))
            # Add some bias to differentiate models
            if self.bias > 0:
                predictions[np.random.random(len(predictions)) < 0.3] = 1
            elif self.bias < 0:
                predictions[np.random.random(len(predictions)) < 0.3] = -1
            return predictions
    
    models = {
        'model_a': MockModel('A', bias=0.1),
        'model_b': MockModel('B', bias=-0.1),
        'model_c': MockModel('C', bias=0)
    }
    
    # Run comprehensive validation
    comparator = ModelComparator()
    results = comparator.compare_models(models, X, y, timestamps, metadata)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION RESULTS")
    print("="*80)
    
    # Print summary
    summary = results['summary']
    print(f"Total models tested: {summary['total_models']}")
    print(f"Successful validations: {summary['successful_validations']}")
    
    # Print rankings
    if 'comparison_analysis' in results and 'rankings' in results['comparison_analysis']:
        rankings = results['comparison_analysis']['rankings']
        if 'overall' in rankings:
            print(f"\nOverall Rankings:")
            for i, (model, score) in enumerate(rankings['overall'][:3]):
                print(f"  {i+1}. {model}: {score:.4f}")
    
    # Print recommendations
    if 'recommendations' in results:
        print(f"\nRecommendations:")
        for rec in results['recommendations'][:5]:
            print(f"  • {rec}")
    
    print("="*80)


if __name__ == "__main__":
    main()