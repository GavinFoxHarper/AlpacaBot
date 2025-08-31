#!/usr/bin/env python3
"""
Test the Live Monitoring Dashboard functionality
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_live_monitoring():
    """Test the live monitoring system components"""
    
    print("="*80)
    print("TESTING LIVE MONITORING DASHBOARD")
    print("="*80)
    
    # Test 1: Check if monitoring dashboard can be imported
    print("\nTEST 1: Import Live Monitoring Dashboard")
    try:
        from live_monitoring_dashboard import LiveMonitoringDashboard
        print("[SUCCESS] Live monitoring dashboard imported successfully")
    except Exception as e:
        print(f"[ERROR] Failed to import live monitoring dashboard: {e}")
        return
    
    # Test 2: Initialize the monitoring system
    print("\nTEST 2: Initialize Monitoring System")
    try:
        dashboard = LiveMonitoringDashboard()
        print("[SUCCESS] Monitoring dashboard initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize monitoring system: {e}")
        print("This might be due to missing Alpaca API credentials")
        
        # Try to initialize without API
        try:
            print("Attempting to run in offline mode...")
            dashboard = MockLiveMonitoringDashboard()
            print("[SUCCESS] Monitoring dashboard initialized in offline mode")
        except:
            return
    
    # Test 3: Check evolution status
    print("\nTEST 3: Get Evolution Status")
    try:
        status = dashboard.get_evolution_status()
        print("[SUCCESS] Evolution status retrieved")
        
        print("\nEvolution Status Summary:")
        print(f"  Timestamp: {status['timestamp']}")
        print(f"  Runtime: {status['runtime']}")
        print(f"  Components: {len(status['components'])} active")
        
        # Show component details
        for component, details in status['components'].items():
            print(f"\n  {component.upper()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"    {key}: {value}")
        
    except Exception as e:
        print(f"[ERROR] Failed to get evolution status: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Generate evolution report
    print("\nTEST 4: Generate Evolution Report")
    try:
        report = dashboard.generate_evolution_report()
        print("[SUCCESS] Evolution report generated")
        
        print("\nEvolution Report Preview (first 20 lines):")
        print("-" * 60)
        report_lines = report.split('\n')[:20]
        for line in report_lines:
            print(line)
        if len(report.split('\n')) > 20:
            print("... (truncated)")
        print("-" * 60)
        
    except Exception as e:
        print(f"[ERROR] Failed to generate evolution report: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Check required directories and files
    print("\nTEST 5: Check Required Files and Directories")
    required_paths = [
        'logs',
        'models',
        'reports',
        'logs/training',
        'models/q_learning_models'
    ]
    
    for path_name in required_paths:
        path_obj = Path(path_name)
        if path_obj.exists():
            print(f"[SUCCESS] {path_name}: exists")
        else:
            print(f"[WARNING] {path_name}: missing (will be created when needed)")
    
    # Test 6: Check for model files
    print("\nTEST 6: Check for AI Model Files")
    model_files = [
        'models/q_learning_models/q_model.keras',
        'logs/training/predictions.db',
        'logs/trades.db'
    ]
    
    for model_file in model_files:
        model_path = Path(model_file)
        if model_path.exists():
            size_kb = model_path.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            print(f"[SUCCESS] {model_file}: exists ({size_kb:.1f} KB, modified {mod_time.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"[INFO] {model_file}: not found (normal for first run)")
    
    print("\n" + "="*80)
    print("LIVE MONITORING TEST COMPLETE")
    print("="*80)


class MockLiveMonitoringDashboard:
    """Mock version for testing without API credentials"""
    
    def __init__(self):
        self.start_time = datetime.now()
        
    def get_evolution_status(self):
        """Mock evolution status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'runtime': str(datetime.now() - self.start_time),
            'components': {
                'q_learning': {
                    'status': 'SIMULATED',
                    'last_update': datetime.now().isoformat(),
                    'model_size_kb': 75.96,
                    'epsilon': 0.1,
                    'learning_rate': 0.001
                },
                'strategies': {
                    'momentum_scalping': {'trades': 5, 'profit': 123.45, 'active': True},
                    'mean_reversion': {'trades': 3, 'profit': 67.89, 'active': True}
                },
                'portfolio': {
                    'current_value': 51234.56,
                    'daily_change': 234.56,
                    'daily_change_pct': 0.46,
                    'positions': 3,
                    'win_rate': 60.0,
                    'total_trades': 10
                }
            }
        }
    
    def generate_evolution_report(self):
        """Mock evolution report"""
        return f"""
LAEF LIVE MONITORING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

SYSTEM STATUS: ACTIVE (MOCK MODE)
Runtime: {datetime.now() - self.start_time}

Q-LEARNING EVOLUTION:
  Status: SIMULATED
  Epsilon: 0.1
  Learning Rate: 0.001
  
PORTFOLIO EVOLUTION:
  Current Value: $51,234.56
  Daily Change: +$234.56 (+0.46%)
  Active Positions: 3
  Win Rate: 60%
  Total Trades: 10

STRATEGIES:
  Momentum Scalping: 5 trades, $123.45 profit
  Mean Reversion: 3 trades, $67.89 profit

EVOLUTION INDICATORS:
  Evolution Score: 75/100 (GOOD)
  System Health: HEALTHY
  
This is a mock report for testing purposes.
"""


if __name__ == "__main__":
    test_live_monitoring()