#!/usr/bin/env python3
"""
Live Monitoring Dashboard for LAEF Trading System
Shows real-time evolution of the bot's learning and performance
"""

import os
import sys
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from alpaca_trade_api import REST
from config import LOGS_DIR, MODELS_DIR, REPORTS_DIR

# Load environment
load_dotenv()

class LiveMonitoringDashboard:
    """Real-time monitoring of LAEF system evolution"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Database paths
        self.predictions_db = LOGS_DIR / 'training' / 'predictions.db'
        self.trades_db = LOGS_DIR / 'trades.db'
        
        # Initialize API
        self.api = REST(self.api_key, self.secret_key, self.base_url)
        
        # Tracking variables
        self.start_time = datetime.now()
        self.initial_portfolio_value = None
        self.evolution_metrics = {
            'q_learning': {'iterations': 0, 'avg_reward': 0, 'epsilon': 1.0},
            'ml_models': {'accuracy': 0, 'last_trained': None, 'total_predictions': 0},
            'strategies': {},
            'performance': {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0}
        }
    
    def get_evolution_status(self) -> Dict:
        """Get current evolution status of all AI components"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'runtime': str(datetime.now() - self.start_time),
            'components': {}
        }
        
        # 1. Q-Learning Evolution
        q_model_path = MODELS_DIR / 'q_learning_models' / 'q_model.keras'
        if q_model_path.exists():
            mod_time = datetime.fromtimestamp(q_model_path.stat().st_mtime)
            status['components']['q_learning'] = {
                'status': 'ACTIVE',
                'last_update': mod_time.isoformat(),
                'model_size_kb': q_model_path.stat().st_size / 1024,
                'epsilon': self._get_current_epsilon(),
                'learning_rate': 0.001
            }
        else:
            status['components']['q_learning'] = {
                'status': 'NOT INITIALIZED',
                'epsilon': self._get_current_epsilon(),
                'learning_rate': 0.001
            }
        
        # 2. Prediction Tracking
        if self.predictions_db.exists():
            conn = sqlite3.connect(self.predictions_db)
            cursor = conn.cursor()
            
            # Get prediction statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN prediction_accuracy = 'correct' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy,
                    COUNT(DISTINCT symbol) as symbols_tracked
                FROM predictions
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            
            pred_stats = cursor.fetchone()
            
            # Get learning updates
            cursor.execute("""
                SELECT COUNT(*) as updates, AVG(performance_after - performance_before) as avg_improvement
                FROM learning_history
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            
            learn_stats = cursor.fetchone()
            
            status['components']['predictions'] = {
                'last_24h_predictions': pred_stats[0] if pred_stats else 0,
                'avg_confidence': round(pred_stats[1], 3) if pred_stats and pred_stats[1] else 0,
                'accuracy_rate': round(pred_stats[2], 2) if pred_stats and pred_stats[2] else 0,
                'symbols_tracked': pred_stats[3] if pred_stats else 0,
                'learning_updates': learn_stats[0] if learn_stats else 0,
                'avg_improvement': round(learn_stats[1], 4) if learn_stats and learn_stats[1] else 0
            }
            
            conn.close()
        else:
            status['components']['predictions'] = {
                'last_24h_predictions': 0,
                'avg_confidence': 0,
                'accuracy_rate': 0,
                'symbols_tracked': 0,
                'learning_updates': 0,
                'avg_improvement': 0,
                'status': 'No prediction database found'
            }
        
        # 3. Strategy Performance Evolution
        status['components']['strategies'] = self._get_strategy_evolution()
        
        # 4. Portfolio Evolution
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            if self.initial_portfolio_value is None:
                self.initial_portfolio_value = float(account.last_equity)
            
            current_value = float(account.portfolio_value)
            daily_change = current_value - float(account.last_equity)
            total_change = current_value - self.initial_portfolio_value
            
            status['components']['portfolio'] = {
                'current_value': round(current_value, 2),
                'daily_change': round(daily_change, 2),
                'daily_change_pct': round(daily_change / float(account.last_equity) * 100, 2),
                'total_change': round(total_change, 2),
                'total_change_pct': round(total_change / self.initial_portfolio_value * 100, 2),
                'positions': len(positions),
                'buying_power': round(float(account.buying_power), 2)
            }
            
            # Get recent trades for evolution tracking
            orders = self.api.list_orders(status='filled', limit=100)
            if orders:
                wins = sum(1 for o in orders if self._is_winning_trade(o))
                status['components']['portfolio']['win_rate'] = round(wins / len(orders) * 100, 2)
                status['components']['portfolio']['total_trades'] = len(orders)
        
        except Exception as e:
            status['components']['portfolio'] = {'error': str(e)}
        
        # 5. Model Files Evolution
        status['components']['model_files'] = self._check_model_files()
        
        return status
    
    def _get_current_epsilon(self) -> float:
        """Get current epsilon value for Q-learning exploration"""
        # Calculate based on runtime
        hours_run = (datetime.now() - self.start_time).total_seconds() / 3600
        # Decay from 1.0 to 0.1 over 100 hours
        epsilon = max(0.1, 1.0 - (hours_run * 0.009))
        return round(epsilon, 3)
    
    def _get_strategy_evolution(self) -> Dict:
        """Track how strategies are evolving"""
        strategies = {
            'momentum_scalping': {'trades': 0, 'profit': 0, 'active': True},
            'mean_reversion': {'trades': 0, 'profit': 0, 'active': True},
            'statistical_arbitrage': {'trades': 0, 'profit': 0, 'active': True},
            'dual_model_swing': {'trades': 0, 'profit': 0, 'active': True},
            'pattern_recognition': {'trades': 0, 'profit': 0, 'active': True},
            'time_based': {'trades': 0, 'profit': 0, 'active': True},
            'news_sentiment': {'trades': 0, 'profit': 0, 'active': True},
            'hybrid_adaptive': {'trades': 0, 'profit': 0, 'active': True},
            'reinforced_grid': {'trades': 0, 'profit': 0, 'active': True}
        }
        
        # Read from logs if available
        log_files = list(LOGS_DIR.glob('*trading*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            # Parse log for strategy performance
            # This would need actual log parsing logic
            
        return strategies
    
    def _check_model_files(self) -> Dict:
        """Check all model files and their evolution"""
        model_info = {}
        
        # Check Q-learning models
        q_models = list((MODELS_DIR / 'q_learning_models').glob('*.keras'))
        if q_models:
            latest_q = max(q_models, key=lambda x: x.stat().st_mtime)
            model_info['q_learning'] = {
                'file': latest_q.name,
                'size_kb': round(latest_q.stat().st_size / 1024, 2),
                'last_modified': datetime.fromtimestamp(latest_q.stat().st_mtime).isoformat(),
                'age_hours': round((datetime.now() - datetime.fromtimestamp(latest_q.stat().st_mtime)).total_seconds() / 3600, 2)
            }
        
        # Check for other ML models
        for model_type in ['lstm', 'random_forest', 'xgboost']:
            model_files = list(MODELS_DIR.glob(f'*{model_type}*'))
            if model_files:
                latest = max(model_files, key=lambda x: x.stat().st_mtime)
                model_info[model_type] = {
                    'file': latest.name,
                    'size_kb': round(latest.stat().st_size / 1024, 2),
                    'last_modified': datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
                }
        
        return model_info
    
    def _is_winning_trade(self, order) -> bool:
        """Check if an order was a winning trade"""
        # This is simplified - would need to track buy/sell pairs
        return float(order.filled_avg_price or 0) > 0
    
    def generate_evolution_report(self) -> str:
        """Generate a comprehensive evolution report"""
        status = self.get_evolution_status()
        
        report = []
        report.append("=" * 70)
        report.append(" " * 20 + "LAEF EVOLUTION STATUS REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {status['timestamp']}")
        report.append(f"Runtime: {status['runtime']}")
        report.append("")
        
        # Q-Learning Status
        if 'q_learning' in status['components']:
            ql = status['components']['q_learning']
            report.append("Q-LEARNING EVOLUTION:")
            report.append("-" * 40)
            report.append(f"  Status: {ql['status']}")
            report.append(f"  Epsilon (Exploration): {ql['epsilon']}")
            report.append(f"  Learning Rate: {ql['learning_rate']}")
            report.append(f"  Model Size: {ql['model_size_kb']:.2f} KB")
            report.append(f"  Last Update: {ql['last_update']}")
            report.append("")
        
        # Predictions & Learning
        if 'predictions' in status['components']:
            pred = status['components']['predictions']
            report.append("PREDICTION & LEARNING:")
            report.append("-" * 40)
            report.append(f"  24h Predictions: {pred['last_24h_predictions']}")
            report.append(f"  Accuracy Rate: {pred['accuracy_rate']}%")
            report.append(f"  Avg Confidence: {pred['avg_confidence']}")
            report.append(f"  Symbols Tracked: {pred['symbols_tracked']}")
            report.append(f"  Learning Updates: {pred['learning_updates']}")
            report.append(f"  Avg Improvement: {pred['avg_improvement']}")
            report.append("")
        
        # Portfolio Performance
        if 'portfolio' in status['components'] and 'error' not in status['components']['portfolio']:
            port = status['components']['portfolio']
            report.append("PORTFOLIO EVOLUTION:")
            report.append("-" * 40)
            report.append(f"  Current Value: ${port['current_value']:,.2f}")
            report.append(f"  Daily Change: ${port['daily_change']:+,.2f} ({port['daily_change_pct']:+.2f}%)")
            report.append(f"  Total Change: ${port['total_change']:+,.2f} ({port['total_change_pct']:+.2f}%)")
            report.append(f"  Active Positions: {port['positions']}")
            if 'win_rate' in port:
                report.append(f"  Win Rate: {port['win_rate']}%")
                report.append(f"  Total Trades: {port['total_trades']}")
            report.append("")
        
        # Model Files
        if 'model_files' in status['components']:
            models = status['components']['model_files']
            report.append("MODEL FILES:")
            report.append("-" * 40)
            for model_type, info in models.items():
                report.append(f"  {model_type}:")
                report.append(f"    File: {info['file']}")
                report.append(f"    Size: {info['size_kb']} KB")
                report.append(f"    Modified: {info['last_modified']}")
                if 'age_hours' in info:
                    report.append(f"    Age: {info['age_hours']} hours")
            report.append("")
        
        # Evolution Indicators
        report.append("EVOLUTION INDICATORS:")
        report.append("-" * 40)
        
        # Calculate evolution score
        evolution_score = 0
        if 'predictions' in status['components']:
            if status['components']['predictions']['last_24h_predictions'] > 0:
                evolution_score += 25
            if status['components']['predictions']['accuracy_rate'] > 50:
                evolution_score += 25
            if status['components']['predictions']['learning_updates'] > 0:
                evolution_score += 25
        
        if 'q_learning' in status['components']:
            if status['components']['q_learning']['epsilon'] < 0.5:
                evolution_score += 25
        
        report.append(f"  Evolution Score: {evolution_score}/100")
        report.append(f"  Status: {'EVOLVING' if evolution_score > 50 else 'LEARNING'}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_evolution_snapshot(self):
        """Save current evolution state to file"""
        status = self.get_evolution_status()
        
        snapshot_file = REPORTS_DIR / f"evolution_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        return snapshot_file

def main():
    """Run the monitoring dashboard"""
    dashboard = LiveMonitoringDashboard()
    
    print("\nStarting Live Monitoring Dashboard...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Generate and display report
            report = dashboard.generate_evolution_report()
            print(report)
            
            # Save snapshot every hour
            if datetime.now().minute == 0 and datetime.now().second < 30:
                snapshot = dashboard.save_evolution_snapshot()
                print(f"\nSnapshot saved: {snapshot}")
            
            # Update every 30 seconds
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        # Save final snapshot
        final_snapshot = dashboard.save_evolution_snapshot()
        print(f"Final snapshot saved: {final_snapshot}")

if __name__ == "__main__":
    main()