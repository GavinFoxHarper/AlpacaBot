#!/usr/bin/env python3
"""
Automated comprehensive backtest - bypasses input prompts
"""

import sys
import os
import json
import warnings
from datetime import datetime
import time
from unittest.mock import patch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_comprehensive_backtests():
    """Run all backtest options directly"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST TESTING - ALL OPTIONS")
    print("="*80)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Import after path setup
    from trading.hybrid_trading_engine import LAEFBacktester
    
    # Test 1: Quick Backtest (Default Settings)
    print("\n" + "="*80)
    print("TEST 1: QUICK BACKTEST - DEFAULT LAEF SETTINGS")
    print("="*80)
    
    try:
        print("\nConfiguration:")
        print("  - Strategy: LAEF Multi-Strategy System")
        print("  - Period: Last 3 months")
        print("  - Initial Cash: $50,000")
        print("  - Stock Selection: AI-driven smart selection")
        
        backtester = LAEFBacktester(
            initial_cash=50000,
            custom_config={
                'start_date': '2024-10-01',
                'end_date': '2024-12-31'
            }
        )
        
        print("\nRunning Quick Backtest...")
        quick_results = backtester.run_backtest(use_smart_selection=True)
        
        results['quick_backtest'] = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'initial_cash': 50000,
                'period': '2024-10-01 to 2024-12-31',
                'smart_selection': True
            },
            'results': quick_results
        }
        print("[SUCCESS] Quick Backtest completed")
        
    except Exception as e:
        results['quick_backtest'] = {'status': 'failed', 'error': str(e)}
        print(f"[ERROR] Quick Backtest failed: {e}")
    
    time.sleep(1)
    
    # Test 2: Advanced Backtests - Multiple Configurations
    print("\n" + "="*80)
    print("TEST 2: ADVANCED BACKTESTS - MULTIPLE CONFIGURATIONS")
    print("="*80)
    
    advanced_configs = [
        {
            'name': 'Conservative Strategy',
            'initial_cash': 25000,
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'risk_per_trade': 0.01,
            'max_position_size': 0.05,
            'stop_loss': 0.01,
            'profit_target': 0.02
        },
        {
            'name': 'Aggressive Strategy',
            'initial_cash': 100000,
            'start_date': '2024-04-01',
            'end_date': '2024-06-30',
            'risk_per_trade': 0.05,
            'max_position_size': 0.20,
            'stop_loss': 0.03,
            'profit_target': 0.05
        },
        {
            'name': 'Balanced Strategy',
            'initial_cash': 50000,
            'start_date': '2024-07-01',
            'end_date': '2024-09-30',
            'risk_per_trade': 0.02,
            'max_position_size': 0.10,
            'stop_loss': 0.02,
            'profit_target': 0.03
        },
        {
            'name': 'Long-term Growth',
            'initial_cash': 75000,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'risk_per_trade': 0.015,
            'max_position_size': 0.08,
            'stop_loss': 0.015,
            'profit_target': 0.04
        }
    ]
    
    results['advanced_backtests'] = []
    
    for config in advanced_configs:
        print(f"\n[{config['name']}]")
        print(f"  Period: {config['start_date']} to {config['end_date']}")
        print(f"  Capital: ${config['initial_cash']:,}")
        print(f"  Risk: {config['risk_per_trade']*100}% per trade")
        print(f"  Max Position: {config['max_position_size']*100}%")
        
        try:
            backtester = LAEFBacktester(
                initial_cash=config['initial_cash'],
                custom_config=config
            )
            
            print(f"  Running backtest...")
            test_results = backtester.run_backtest(use_smart_selection=True)
            
            results['advanced_backtests'].append({
                'name': config['name'],
                'status': 'completed',
                'config': config,
                'results': test_results,
                'timestamp': datetime.now().isoformat()
            })
            print(f"  [SUCCESS] {config['name']} completed")
            
        except Exception as e:
            results['advanced_backtests'].append({
                'name': config['name'],
                'status': 'failed',
                'error': str(e)
            })
            print(f"  [ERROR] {config['name']} failed: {e}")
        
        time.sleep(1)
    
    # Test 3: Strategy Comparison
    print("\n" + "="*80)
    print("TEST 3: STRATEGY COMPARISON - ALL LAEF STRATEGIES")
    print("="*80)
    
    strategies_to_test = [
        'momentum_scalping',
        'mean_reversion',
        'statistical_arbitrage',
        'dual_model_swing',
        'pattern_recognition',
        'time_based',
        'news_sentiment',
        'hybrid_adaptive',
        'reinforced_grid'
    ]
    
    results['strategy_comparison'] = []
    
    print("\nTesting individual strategies:")
    for strategy in strategies_to_test:
        print(f"\n[{strategy.upper()}]")
        
        try:
            backtester = LAEFBacktester(
                initial_cash=50000,
                custom_config={
                    'start_date': '2024-07-01',
                    'end_date': '2024-12-31',
                    'active_strategy': strategy
                }
            )
            
            print(f"  Running {strategy} strategy...")
            strategy_results = backtester.run_backtest(
                symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                use_smart_selection=False
            )
            
            results['strategy_comparison'].append({
                'strategy': strategy,
                'status': 'completed',
                'results': strategy_results,
                'timestamp': datetime.now().isoformat()
            })
            print(f"  [SUCCESS] {strategy} completed")
            
        except Exception as e:
            results['strategy_comparison'].append({
                'strategy': strategy,
                'status': 'failed',
                'error': str(e)
            })
            print(f"  [ERROR] {strategy} failed: {e}")
        
        time.sleep(0.5)
    
    # Test 4: Different Asset Groups
    print("\n" + "="*80)
    print("TEST 4: ASSET GROUP TESTING")
    print("="*80)
    
    asset_groups = [
        {
            'name': 'Tech Giants',
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
        },
        {
            'name': 'EV & Clean Energy',
            'symbols': ['TSLA', 'RIVN', 'NIO', 'PLUG', 'ENPH']
        },
        {
            'name': 'Financial Sector',
            'symbols': ['JPM', 'BAC', 'GS', 'MS', 'WFC']
        },
        {
            'name': 'High Volatility',
            'symbols': ['GME', 'AMC', 'BBBY', 'SPCE', 'WISH']
        }
    ]
    
    results['asset_groups'] = []
    
    for group in asset_groups:
        print(f"\n[{group['name']}]")
        print(f"  Symbols: {', '.join(group['symbols'])}")
        
        try:
            backtester = LAEFBacktester(
                initial_cash=50000,
                custom_config={
                    'start_date': '2024-07-01',
                    'end_date': '2024-12-31'
                }
            )
            
            print(f"  Running backtest...")
            group_results = backtester.run_backtest(
                symbols=group['symbols'],
                use_smart_selection=False
            )
            
            results['asset_groups'].append({
                'name': group['name'],
                'symbols': group['symbols'],
                'status': 'completed',
                'results': group_results,
                'timestamp': datetime.now().isoformat()
            })
            print(f"  [SUCCESS] {group['name']} completed")
            
        except Exception as e:
            results['asset_groups'].append({
                'name': group['name'],
                'status': 'failed',
                'error': str(e)
            })
            print(f"  [ERROR] {group['name']} failed: {e}")
        
        time.sleep(1)
    
    # Save comprehensive results
    results_file = f"reports/comprehensive_backtest_{timestamp}.json"
    try:
        os.makedirs('reports', exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[SUCCESS] Results saved to: {results_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save results: {e}")
    
    return results, results_file

if __name__ == "__main__":
    print("Starting comprehensive backtest testing...")
    print("This will test all menu options and configurations")
    
    results, results_file = run_comprehensive_backtests()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING COMPLETE")
    print("="*80)
    
    # Summary
    print("\nTest Summary:")
    
    # Quick Backtest
    if 'quick_backtest' in results:
        status = "SUCCESS" if results['quick_backtest'].get('status') == 'completed' else "FAILED"
        print(f"  1. Quick Backtest: {status}")
    
    # Advanced Backtests
    if 'advanced_backtests' in results:
        completed = sum(1 for r in results['advanced_backtests'] if r.get('status') == 'completed')
        total = len(results['advanced_backtests'])
        print(f"  2. Advanced Backtests: {completed}/{total} completed")
    
    # Strategy Comparison
    if 'strategy_comparison' in results:
        completed = sum(1 for r in results['strategy_comparison'] if r.get('status') == 'completed')
        total = len(results['strategy_comparison'])
        print(f"  3. Strategy Comparison: {completed}/{total} strategies tested")
    
    # Asset Groups
    if 'asset_groups' in results:
        completed = sum(1 for r in results['asset_groups'] if r.get('status') == 'completed')
        total = len(results['asset_groups'])
        print(f"  4. Asset Groups: {completed}/{total} groups tested")
    
    print(f"\nFull results saved to: {results_file}")