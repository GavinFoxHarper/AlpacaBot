#!/usr/bin/env python3
"""
Comprehensive backtest runner - tests all menu options
Goes through the front-end menu system
"""

import sys
import os
import json
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laef_unified_system import LAEFUnifiedSystem

def run_all_backtest_options():
    """Run all backtest menu options comprehensively"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST TESTING - ALL MENU OPTIONS")
    print("="*80)
    print("\nThis will test all backtest menu options through the front-end system")
    print("Testing sequence: Quick -> Advanced -> Strategy Comparison -> View Results\n")
    
    # Initialize system
    system = LAEFUnifiedSystem(debug_mode=False)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: Quick Backtest
    print("\n" + "="*80)
    print("TEST 1: QUICK BACKTEST (Menu Option 1)")
    print("="*80)
    try:
        print("\nExecuting Quick Backtest with default LAEF settings...")
        system._run_quick_backtest()
        results['quick_backtest'] = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'description': 'LAEF Default AI/ML System - Last 3 months'
        }
        print("[✓] Quick Backtest completed")
    except Exception as e:
        results['quick_backtest'] = {'status': 'failed', 'error': str(e)}
        print(f"[✗] Quick Backtest failed: {e}")
    
    time.sleep(2)
    
    # Test 2: Advanced Backtest with different configurations
    print("\n" + "="*80)
    print("TEST 2: ADVANCED BACKTEST (Menu Option 2)")
    print("="*80)
    
    # Test multiple advanced configurations
    advanced_configs = [
        {
            'name': 'Conservative',
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_cash': 25000,
            'risk_per_trade': 0.01,
            'max_position_size': 0.05
        },
        {
            'name': 'Aggressive',
            'start_date': '2024-04-01',
            'end_date': '2024-06-30',
            'initial_cash': 100000,
            'risk_per_trade': 0.05,
            'max_position_size': 0.20
        },
        {
            'name': 'Balanced',
            'start_date': '2024-07-01',
            'end_date': '2024-09-30',
            'initial_cash': 50000,
            'risk_per_trade': 0.02,
            'max_position_size': 0.10
        }
    ]
    
    results['advanced_backtest'] = []
    
    for config in advanced_configs:
        print(f"\nTesting Advanced Config: {config['name']}")
        print(f"  Period: {config['start_date']} to {config['end_date']}")
        print(f"  Initial Cash: ${config['initial_cash']:,}")
        print(f"  Risk Level: {config['risk_per_trade']*100}%")
        
        try:
            from trading.hybrid_trading_engine import LAEFBacktester
            
            backtester = LAEFBacktester(
                initial_cash=config['initial_cash'],
                custom_config=config
            )
            
            # Test with different stock selections
            test_results = backtester.run_backtest(use_smart_selection=True)
            
            results['advanced_backtest'].append({
                'config_name': config['name'],
                'config': config,
                'results': test_results if test_results else 'No trades executed',
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            })
            print(f"  [✓] {config['name']} config completed")
            
        except Exception as e:
            results['advanced_backtest'].append({
                'config_name': config['name'],
                'status': 'failed',
                'error': str(e)
            })
            print(f"  [✗] {config['name']} config failed: {e}")
        
        time.sleep(1)
    
    # Test 3: Strategy Comparison
    print("\n" + "="*80)
    print("TEST 3: STRATEGY COMPARISON BACKTEST (Menu Option 3)")
    print("="*80)
    
    try:
        print("\nComparing all available strategies...")
        system._run_strategy_comparison()
        results['strategy_comparison'] = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'description': 'Compared all 9 LAEF strategies'
        }
        print("[✓] Strategy Comparison completed")
    except Exception as e:
        results['strategy_comparison'] = {'status': 'failed', 'error': str(e)}
        print(f"[✗] Strategy Comparison failed: {e}")
    
    time.sleep(2)
    
    # Test 4: View Previous Results
    print("\n" + "="*80)
    print("TEST 4: VIEW PREVIOUS RESULTS & ANALYSIS (Menu Option 4)")
    print("="*80)
    
    try:
        print("\nAccessing previous backtest results...")
        system._view_backtest_analysis()
        results['view_previous'] = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'description': 'Viewed and analyzed previous results'
        }
        print("[✓] Previous Results viewing completed")
    except Exception as e:
        results['view_previous'] = {'status': 'failed', 'error': str(e)}
        print(f"[✗] Previous Results viewing failed: {e}")
    
    # Save comprehensive results
    results_file = f"reports/comprehensive_backtest_{timestamp}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[✓] Comprehensive results saved to: {results_file}")
    except Exception as e:
        print(f"\n[✗] Failed to save results: {e}")
    
    return results

if __name__ == "__main__":
    print("Starting comprehensive backtest testing...")
    results = run_all_backtest_options()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING COMPLETE")
    print("="*80)
    print("\nSummary:")
    for test_name, test_result in results.items():
        if isinstance(test_result, dict) and 'status' in test_result:
            status = "✓" if test_result['status'] == 'completed' else "✗"
            print(f"  [{status}] {test_name}")
        elif isinstance(test_result, list):
            completed = sum(1 for r in test_result if r.get('status') == 'completed')
            print(f"  [{completed}/{len(test_result)}] {test_name}")