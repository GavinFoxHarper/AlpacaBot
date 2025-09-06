"""
Training Metrics Utilities for AlpacaBot
Comprehensive performance tracking, visualization, and reporting
Author: AlpacaBot Training System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement at a point in time."""
    timestamp: datetime
    epoch: int
    phase: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSession:
    """Complete training session information."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    model_name: str
    config: Dict[str, Any]
    metrics: List[MetricSnapshot] = field(default_factory=list)
    final_results: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed


class MetricsTracker:
    """
    Comprehensive metrics tracking system for training monitoring.
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize metrics tracker.
        
        Args:
            session_id: Unique identifier for training session
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics_history = defaultdict(list)
        self.current_session = TrainingSession(
            session_id=self.session_id,
            start_time=datetime.now(),
            end_time=None,
            model_name="unknown",
            config={}
        )
        
        # Performance tracking
        self.best_metrics = {}
        self.metric_trends = defaultdict(deque)
        self.alert_thresholds = {
            'accuracy_decline': -0.05,
            'loss_plateau': 0.001,
            'overfitting_gap': 0.1
        }
        
        # Real-time statistics
        self.rolling_averages = defaultdict(lambda: deque(maxlen=10))
        self.phase_summaries = {}
        
        logger.info(f"Metrics Tracker initialized: {self.session_id}")
    
    def start_session(self, model_name: str, config: Dict[str, Any]):
        """
        Start a new training session.
        
        Args:
            model_name: Name of the model being trained
            config: Training configuration
        """
        self.current_session.model_name = model_name
        self.current_session.config = config
        self.current_session.start_time = datetime.now()
        self.current_session.status = "running"
        
        logger.info(f"Training session started: {model_name}")
    
    def log_metric(self, 
                  metric_name: str, 
                  value: float,
                  epoch: int,
                  phase: str = "training",
                  metadata: Dict[str, Any] = None):
        """
        Log a single metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            epoch: Current epoch
            phase: Training phase
            metadata: Additional metadata
        """
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            epoch=epoch,
            phase=phase,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )
        
        # Store in session
        self.current_session.metrics.append(snapshot)
        
        # Store in history
        self.metrics_history[metric_name].append((epoch, value, phase))
        
        # Update rolling averages
        self.rolling_averages[metric_name].append(value)
        
        # Track best metrics
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = {'value': value, 'epoch': epoch, 'phase': phase}
        else:
            # Assume higher is better for most metrics (except loss)
            is_better = value > self.best_metrics[metric_name]['value']
            if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
                is_better = value < self.best_metrics[metric_name]['value']
            
            if is_better:
                self.best_metrics[metric_name] = {'value': value, 'epoch': epoch, 'phase': phase}
        
        # Check for alerts
        self._check_metric_alerts(metric_name, value, epoch, phase)
    
    def log_batch_metrics(self, 
                         metrics: Dict[str, float],
                         epoch: int,
                         phase: str = "training",
                         metadata: Dict[str, Any] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Current epoch
            phase: Training phase
            metadata: Additional metadata
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, epoch, phase, metadata)
    
    def _check_metric_alerts(self, 
                           metric_name: str, 
                           value: float, 
                           epoch: int, 
                           phase: str):
        """Check for metric alerts and warnings."""
        # Check for accuracy decline
        if 'accuracy' in metric_name and len(self.metrics_history[metric_name]) > 5:
            recent_values = [v[1] for v in self.metrics_history[metric_name][-5:]]
            if len(recent_values) >= 2:
                recent_decline = recent_values[-1] - recent_values[0]
                if recent_decline < self.alert_thresholds['accuracy_decline']:
                    logger.warning(f"Alert: {metric_name} declining - recent change: {recent_decline:.4f}")
        
        # Check for loss plateau
        if 'loss' in metric_name and len(self.metrics_history[metric_name]) > 10:
            recent_values = [v[1] for v in self.metrics_history[metric_name][-10:]]
            if np.std(recent_values) < self.alert_thresholds['loss_plateau']:
                logger.warning(f"Alert: {metric_name} plateaued - std: {np.std(recent_values):.6f}")
        
        # Check for overfitting
        if metric_name == 'val_accuracy' and 'train_accuracy' in self.metrics_history:
            train_history = self.metrics_history['train_accuracy']
            if train_history:
                latest_train = train_history[-1][1]
                gap = latest_train - value
                if gap > self.alert_thresholds['overfitting_gap']:
                    logger.warning(f"Alert: Potential overfitting - train/val gap: {gap:.4f}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current (latest) values for all metrics."""
        current_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                current_metrics[metric_name] = history[-1][1]
        return current_metrics
    
    def get_metric_history(self, metric_name: str) -> List[Tuple[int, float, str]]:
        """
        Get complete history for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of (epoch, value, phase) tuples
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_rolling_average(self, metric_name: str, window: int = 10) -> float:
        """
        Get rolling average for a metric.
        
        Args:
            metric_name: Name of the metric
            window: Window size for rolling average
            
        Returns:
            Rolling average value
        """
        if metric_name not in self.metrics_history:
            return 0.0
        
        recent_values = [v[1] for v in self.metrics_history[metric_name][-window:]]
        return np.mean(recent_values) if recent_values else 0.0
    
    def complete_phase(self, phase_name: str) -> Dict[str, Any]:
        """
        Mark a training phase as complete and generate summary.
        
        Args:
            phase_name: Name of the completed phase
            
        Returns:
            Phase summary statistics
        """
        phase_metrics = defaultdict(list)
        
        # Collect metrics for this phase
        for snapshot in self.current_session.metrics:
            if snapshot.phase == phase_name:
                phase_metrics[snapshot.metric_name].append(snapshot.value)
        
        # Calculate phase statistics
        phase_summary = {}
        for metric_name, values in phase_metrics.items():
            if values:
                phase_summary[f"{metric_name}_mean"] = np.mean(values)
                phase_summary[f"{metric_name}_std"] = np.std(values)
                phase_summary[f"{metric_name}_min"] = np.min(values)
                phase_summary[f"{metric_name}_max"] = np.max(values)
                phase_summary[f"{metric_name}_final"] = values[-1]
        
        phase_summary['phase_name'] = phase_name
        phase_summary['sample_count'] = len([s for s in self.current_session.metrics if s.phase == phase_name])
        phase_summary['duration_epochs'] = len(set(
            s.epoch for s in self.current_session.metrics if s.phase == phase_name
        ))
        
        self.phase_summaries[phase_name] = phase_summary
        
        logger.info(f"Phase '{phase_name}' completed - {phase_summary['sample_count']} metrics recorded")
        
        return phase_summary
    
    def complete_session(self, final_results: Dict[str, Any] = None):
        """
        Complete the current training session.
        
        Args:
            final_results: Final evaluation results
        """
        self.current_session.end_time = datetime.now()
        self.current_session.status = "completed"
        self.current_session.final_results = final_results or {}
        
        # Calculate session duration
        duration = self.current_session.end_time - self.current_session.start_time
        
        logger.info(f"Training session completed: {self.session_id} (Duration: {duration})")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        current_metrics = self.get_current_metrics()
        
        summary = {
            'session_info': {
                'session_id': self.session_id,
                'model_name': self.current_session.model_name,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                'status': self.current_session.status,
                'total_metrics': len(self.current_session.metrics)
            },
            'current_metrics': current_metrics,
            'best_metrics': self.best_metrics,
            'phase_summaries': self.phase_summaries,
            'metric_trends': self._calculate_trends(),
            'alerts': self._get_active_alerts()
        }
        
        return summary
    
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate trends for all metrics."""
        trends = {}
        
        for metric_name, history in self.metrics_history.items():
            if len(history) >= 5:
                recent_values = [v[1] for v in history[-5:]]
                epochs = list(range(len(recent_values)))
                
                # Simple linear regression
                slope = np.corrcoef(epochs, recent_values)[0, 1] if len(recent_values) > 1 else 0
                
                if slope > 0.1:
                    trends[metric_name] = "improving"
                elif slope < -0.1:
                    trends[metric_name] = "declining"
                else:
                    trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"
        
        return trends
    
    def _get_active_alerts(self) -> List[str]:
        """Get list of active alerts."""
        # This is a simplified implementation
        # In practice, you'd maintain a more sophisticated alert system
        alerts = []
        
        trends = self._calculate_trends()
        for metric_name, trend in trends.items():
            if 'accuracy' in metric_name and trend == "declining":
                alerts.append(f"{metric_name} is declining")
            elif 'loss' in metric_name and trend == "stable":
                alerts.append(f"{metric_name} has plateaued")
        
        return alerts


class PerformanceAnalyzer:
    """
    Advanced performance analysis and reporting system.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.analysis_cache = {}
        logger.info("Performance Analyzer initialized")
    
    def analyze_training_progression(self, 
                                   metrics_tracker: MetricsTracker) -> Dict[str, Any]:
        """
        Analyze training progression and identify patterns.
        
        Args:
            metrics_tracker: Metrics tracker instance
            
        Returns:
            Analysis results
        """
        analysis = {
            'convergence_analysis': {},
            'learning_efficiency': {},
            'stability_metrics': {},
            'phase_performance': {},
            'recommendations': []
        }
        
        # Convergence analysis
        for metric_name, history in metrics_tracker.metrics_history.items():
            if len(history) >= 10:
                values = [v[1] for v in history]
                epochs = [v[0] for v in history]
                
                # Calculate convergence metrics
                convergence_info = self._analyze_convergence(epochs, values)
                analysis['convergence_analysis'][metric_name] = convergence_info
        
        # Learning efficiency
        analysis['learning_efficiency'] = self._calculate_learning_efficiency(metrics_tracker)
        
        # Stability metrics
        analysis['stability_metrics'] = self._calculate_stability_metrics(metrics_tracker)
        
        # Phase performance comparison
        analysis['phase_performance'] = self._compare_phase_performance(metrics_tracker)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_convergence(self, epochs: List[int], values: List[float]) -> Dict[str, Any]:
        """Analyze convergence patterns for a metric."""
        convergence_info = {
            'converged': False,
            'convergence_epoch': None,
            'convergence_value': None,
            'convergence_rate': 0.0,
            'final_stability': 0.0
        }
        
        if len(values) < 10:
            return convergence_info
        
        # Simple convergence detection based on moving average stability
        window_size = min(20, len(values) // 4)
        moving_avg = pd.Series(values).rolling(window_size).mean()
        
        # Check for convergence (low variance in final portion)
        final_portion = moving_avg.dropna()[-window_size:]
        if len(final_portion) > 0:
            stability = np.std(final_portion) / (np.mean(final_portion) + 1e-8)
            
            if stability < 0.05:  # 5% relative stability threshold
                convergence_info['converged'] = True
                convergence_info['convergence_value'] = final_portion.mean()
                convergence_info['final_stability'] = stability
                
                # Find approximate convergence epoch
                for i, avg in enumerate(moving_avg.dropna()):
                    if abs(avg - convergence_info['convergence_value']) < 0.02:
                        convergence_info['convergence_epoch'] = epochs[i + window_size - 1]
                        break
        
        # Calculate convergence rate
        if len(values) > 1:
            total_change = abs(values[-1] - values[0])
            total_epochs = epochs[-1] - epochs[0]
            convergence_info['convergence_rate'] = total_change / max(total_epochs, 1)
        
        return convergence_info
    
    def _calculate_learning_efficiency(self, metrics_tracker: MetricsTracker) -> Dict[str, Any]:
        """Calculate learning efficiency metrics."""
        efficiency = {
            'epochs_to_best': {},
            'improvement_rate': {},
            'learning_curve_smoothness': {}
        }
        
        for metric_name, best_info in metrics_tracker.best_metrics.items():
            # Epochs to reach best performance
            efficiency['epochs_to_best'][metric_name] = best_info['epoch']
            
            # Improvement rate
            history = metrics_tracker.get_metric_history(metric_name)
            if len(history) > 1:
                initial_value = history[0][1]
                best_value = best_info['value']
                total_epochs = best_info['epoch'] - history[0][0]
                
                if total_epochs > 0:
                    improvement = abs(best_value - initial_value)
                    efficiency['improvement_rate'][metric_name] = improvement / total_epochs
            
            # Learning curve smoothness (inverse of volatility)
            if len(history) > 5:
                values = [v[1] for v in history]
                volatility = np.std(np.diff(values))
                efficiency['learning_curve_smoothness'][metric_name] = 1.0 / (1.0 + volatility)
        
        return efficiency
    
    def _calculate_stability_metrics(self, metrics_tracker: MetricsTracker) -> Dict[str, Any]:
        """Calculate stability and robustness metrics."""
        stability = {
            'metric_volatility': {},
            'trend_consistency': {},
            'plateau_detection': {}
        }
        
        for metric_name, history in metrics_tracker.metrics_history.items():
            if len(history) < 10:
                continue
            
            values = [v[1] for v in history]
            
            # Metric volatility
            changes = np.diff(values)
            stability['metric_volatility'][metric_name] = np.std(changes)
            
            # Trend consistency
            # Divide into segments and check if trends are consistent
            segment_size = max(5, len(values) // 5)
            segments = [values[i:i+segment_size] for i in range(0, len(values)-segment_size, segment_size)]
            
            segment_trends = []
            for segment in segments:
                if len(segment) > 1:
                    trend = np.corrcoef(range(len(segment)), segment)[0, 1]
                    segment_trends.append(1 if trend > 0 else -1 if trend < 0 else 0)
            
            if segment_trends:
                consistency = np.std(segment_trends)
                stability['trend_consistency'][metric_name] = 1.0 / (1.0 + consistency)
            
            # Plateau detection
            recent_values = values[-min(20, len(values) // 2):]
            plateau_score = 1.0 - (np.std(recent_values) / (np.mean(recent_values) + 1e-8))
            stability['plateau_detection'][metric_name] = max(0, min(1, plateau_score))
        
        return stability
    
    def _compare_phase_performance(self, metrics_tracker: MetricsTracker) -> Dict[str, Any]:
        """Compare performance across different training phases."""
        phase_comparison = {
            'best_phase_per_metric': {},
            'phase_improvement_rates': {},
            'phase_stability_ranking': {}
        }
        
        # Group metrics by phase
        phase_metrics = defaultdict(lambda: defaultdict(list))
        
        for snapshot in metrics_tracker.current_session.metrics:
            phase_metrics[snapshot.phase][snapshot.metric_name].append(snapshot.value)
        
        # Find best phase for each metric
        for metric_name in metrics_tracker.metrics_history.keys():
            phase_averages = {}
            for phase, metrics in phase_metrics.items():
                if metric_name in metrics and metrics[metric_name]:
                    phase_averages[phase] = np.mean(metrics[metric_name])
            
            if phase_averages:
                best_phase = max(phase_averages, key=phase_averages.get)
                if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
                    best_phase = min(phase_averages, key=phase_averages.get)
                
                phase_comparison['best_phase_per_metric'][metric_name] = {
                    'phase': best_phase,
                    'value': phase_averages[best_phase]
                }
        
        return phase_comparison
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Check convergence
        convergence_analysis = analysis.get('convergence_analysis', {})
        non_converged = [m for m, info in convergence_analysis.items() if not info.get('converged', False)]
        
        if non_converged:
            recommendations.append(f"Metrics not converged: {', '.join(non_converged[:3])}. Consider increasing training duration.")
        
        # Check learning efficiency
        efficiency = analysis.get('learning_efficiency', {})
        epochs_to_best = efficiency.get('epochs_to_best', {})
        
        slow_learners = [m for m, epochs in epochs_to_best.items() if epochs > 100]
        if slow_learners:
            recommendations.append(f"Slow convergence detected for: {', '.join(slow_learners[:3])}. Consider increasing learning rate.")
        
        # Check stability
        stability = analysis.get('stability_metrics', {})
        volatility = stability.get('metric_volatility', {})
        
        unstable_metrics = [m for m, vol in volatility.items() if vol > 0.1]
        if unstable_metrics:
            recommendations.append(f"High volatility in: {', '.join(unstable_metrics[:3])}. Consider regularization or learning rate reduction.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Training appears stable. Monitor for overfitting and consider early stopping.")
        
        return recommendations


class ReportGenerator:
    """
    Comprehensive report generation for training results.
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.template_dir = "templates"
        self.output_dir = "reports"
        
    def generate_training_report(self, 
                               metrics_tracker: MetricsTracker,
                               analyzer: PerformanceAnalyzer,
                               output_path: str = None) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            metrics_tracker: Metrics tracker instance
            analyzer: Performance analyzer instance
            output_path: Output file path
            
        Returns:
            Generated report content
        """
        # Get analysis results
        analysis = analyzer.analyze_training_progression(metrics_tracker)
        session_summary = metrics_tracker.get_session_summary()
        
        # Generate report content
        report_content = self._create_report_content(session_summary, analysis)
        
        # Save to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Training report saved to: {output_path}")
        
        return report_content
    
    def _create_report_content(self, 
                             session_summary: Dict[str, Any],
                             analysis: Dict[str, Any]) -> str:
        """Create formatted report content."""
        
        session_info = session_summary['session_info']
        current_metrics = session_summary['current_metrics']
        best_metrics = session_summary['best_metrics']
        
        report = f"""
# AlpacaBot Training Report

## Session Information
- **Session ID**: {session_info['session_id']}
- **Model**: {session_info['model_name']}
- **Start Time**: {session_info['start_time']}
- **End Time**: {session_info.get('end_time', 'Running')}
- **Status**: {session_info['status']}
- **Total Metrics Recorded**: {session_info['total_metrics']:,}

## Current Performance
"""
        
        # Add current metrics
        for metric_name, value in current_metrics.items():
            report += f"- **{metric_name}**: {value:.4f}\n"
        
        report += "\n## Best Performance Achieved\n"
        
        # Add best metrics
        for metric_name, info in best_metrics.items():
            report += f"- **{metric_name}**: {info['value']:.4f} (Epoch {info['epoch']}, Phase: {info['phase']})\n"
        
        # Add convergence analysis
        convergence = analysis.get('convergence_analysis', {})
        if convergence:
            report += "\n## Convergence Analysis\n"
            for metric_name, info in convergence.items():
                status = "✓ Converged" if info['converged'] else "⚠ Not Converged"
                report += f"- **{metric_name}**: {status}\n"
                if info['converged']:
                    report += f"  - Convergence Epoch: {info['convergence_epoch']}\n"
                    report += f"  - Convergence Value: {info['convergence_value']:.4f}\n"
        
        # Add recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report += "\n## Recommendations\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        
        # Add phase summaries
        phase_summaries = session_summary.get('phase_summaries', {})
        if phase_summaries:
            report += "\n## Phase Performance Summary\n"
            for phase_name, summary in phase_summaries.items():
                report += f"\n### {phase_name}\n"
                report += f"- Duration: {summary['duration_epochs']} epochs\n"
                report += f"- Metrics Recorded: {summary['sample_count']}\n"
                
                # Show key final metrics
                key_metrics = ['accuracy_final', 'loss_final', 'f1_score_final']
                for key_metric in key_metrics:
                    if key_metric in summary:
                        report += f"- {key_metric.replace('_final', '').title()}: {summary[key_metric]:.4f}\n"
        
        report += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
    
    def create_visualization_dashboard(self, 
                                     metrics_tracker: MetricsTracker,
                                     output_dir: str):
        """
        Create comprehensive visualization dashboard.
        
        Args:
            metrics_tracker: Metrics tracker instance
            output_dir: Output directory for visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create training progress plots
        self._plot_training_progress(metrics_tracker, output_dir)
        
        # Create phase comparison plots
        self._plot_phase_comparison(metrics_tracker, output_dir)
        
        # Create convergence analysis plots
        self._plot_convergence_analysis(metrics_tracker, output_dir)
        
        logger.info(f"Visualization dashboard created in: {output_dir}")
    
    def _plot_training_progress(self, metrics_tracker: MetricsTracker, output_dir: str):
        """Create training progress plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - {metrics_tracker.session_id}', fontsize=16)
            
            # Plot key metrics
            key_metrics = ['accuracy', 'loss', 'f1_score', 'sharpe_ratio']
            
            for i, metric_name in enumerate(key_metrics):
                if metric_name in metrics_tracker.metrics_history:
                    row, col = i // 2, i % 2
                    history = metrics_tracker.get_metric_history(metric_name)
                    
                    epochs = [h[0] for h in history]
                    values = [h[1] for h in history]
                    phases = [h[2] for h in history]
                    
                    # Plot main line
                    axes[row, col].plot(epochs, values, 'b-', alpha=0.7, linewidth=2)
                    
                    # Color by phase
                    unique_phases = list(set(phases))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_phases)))
                    
                    for phase, color in zip(unique_phases, colors):
                        phase_mask = [p == phase for p in phases]
                        phase_epochs = [e for e, m in zip(epochs, phase_mask) if m]
                        phase_values = [v for v, m in zip(values, phase_mask) if m]
                        
                        if phase_epochs:
                            axes[row, col].scatter(phase_epochs, phase_values, 
                                                 c=[color], label=phase, alpha=0.6, s=20)
                    
                    axes[row, col].set_title(f'{metric_name.replace("_", " ").title()}')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel(metric_name)
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create training progress plots: {str(e)}")
    
    def _plot_phase_comparison(self, metrics_tracker: MetricsTracker, output_dir: str):
        """Create phase comparison plots."""
        try:
            phase_summaries = metrics_tracker.phase_summaries
            if not phase_summaries:
                return
            
            # Extract phase data
            phases = list(phase_summaries.keys())
            metrics_to_plot = ['accuracy_mean', 'loss_mean', 'f1_score_mean']
            
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            fig.suptitle('Phase Performance Comparison', fontsize=16)
            
            for i, metric in enumerate(metrics_to_plot):
                values = []
                phase_names = []
                
                for phase_name, summary in phase_summaries.items():
                    if metric in summary:
                        values.append(summary[metric])
                        phase_names.append(phase_name)
                
                if values:
                    axes[i].bar(range(len(phase_names)), values)
                    axes[i].set_title(metric.replace('_mean', '').replace('_', ' ').title())
                    axes[i].set_xlabel('Training Phase')
                    axes[i].set_ylabel(metric.split('_')[0])
                    axes[i].set_xticks(range(len(phase_names)))
                    axes[i].set_xticklabels(phase_names, rotation=45, ha='right')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'phase_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create phase comparison plots: {str(e)}")
    
    def _plot_convergence_analysis(self, metrics_tracker: MetricsTracker, output_dir: str):
        """Create convergence analysis plots."""
        try:
            analyzer = PerformanceAnalyzer()
            analysis = analyzer.analyze_training_progression(metrics_tracker)
            convergence_analysis = analysis.get('convergence_analysis', {})
            
            if not convergence_analysis:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Convergence Analysis', fontsize=16)
            
            metrics_to_plot = list(convergence_analysis.keys())[:4]
            
            for i, metric_name in enumerate(metrics_to_plot):
                row, col = i // 2, i % 2
                info = convergence_analysis[metric_name]
                history = metrics_tracker.get_metric_history(metric_name)
                
                epochs = [h[0] for h in history]
                values = [h[1] for h in history]
                
                # Plot metric values
                axes[row, col].plot(epochs, values, 'b-', alpha=0.7, linewidth=2, label='Actual')
                
                # Add moving average
                if len(values) > 10:
                    window = min(20, len(values) // 4)
                    moving_avg = pd.Series(values).rolling(window).mean()
                    axes[row, col].plot(epochs, moving_avg, 'r--', alpha=0.8, linewidth=2, label='Moving Average')
                
                # Mark convergence point if converged
                if info['converged'] and info['convergence_epoch']:
                    axes[row, col].axvline(x=info['convergence_epoch'], color='g', 
                                         linestyle=':', alpha=0.8, label='Convergence')
                    axes[row, col].axhline(y=info['convergence_value'], color='g', 
                                         linestyle=':', alpha=0.8)
                
                axes[row, col].set_title(f"{metric_name} ({'Converged' if info['converged'] else 'Not Converged'})")
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(metric_name)
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create convergence analysis plots: {str(e)}")


def main():
    """Test the training metrics utilities."""
    # Create metrics tracker
    tracker = MetricsTracker("test_session")
    tracker.start_session("test_model", {"learning_rate": 0.001})
    
    # Simulate training metrics
    phases = ["foundation", "specialization", "integration"]
    
    for epoch in range(100):
        phase = phases[epoch // 35] if epoch // 35 < len(phases) else phases[-1]
        
        # Simulate improving metrics with some noise
        base_accuracy = 0.5 + 0.4 * (1 - np.exp(-epoch / 30))
        accuracy = base_accuracy + np.random.normal(0, 0.02)
        
        base_loss = 1.0 * np.exp(-epoch / 25) + 0.1
        loss = base_loss + np.random.normal(0, 0.05)
        
        f1_score = accuracy * 0.9 + np.random.normal(0, 0.01)
        sharpe = np.random.normal(0.8, 0.2)
        
        # Log metrics
        tracker.log_batch_metrics({
            'accuracy': accuracy,
            'loss': loss,
            'f1_score': f1_score,
            'sharpe_ratio': sharpe
        }, epoch, phase)
        
        # Complete phases
        if epoch in [34, 69]:
            tracker.complete_phase(phase)
    
    # Complete session
    tracker.complete_session({'final_accuracy': 0.85, 'final_loss': 0.15})
    
    # Generate analysis
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze_training_progression(tracker)
    
    # Generate report
    report_gen = ReportGenerator()
    report = report_gen.generate_training_report(tracker, analyzer)
    
    # Create visualizations
    os.makedirs("test_output", exist_ok=True)
    report_gen.create_visualization_dashboard(tracker, "test_output")
    
    print("="*80)
    print("TRAINING METRICS UTILITIES TEST")
    print("="*80)
    print(report)
    print("="*80)
    print("Visualizations created in: test_output/")
    print("="*80)


if __name__ == "__main__":
    main()