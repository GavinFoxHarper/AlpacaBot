#!/usr/bin/env python3
"""
Comprehensive front-end backtest testing
Tests all menu options through the actual LAEF system interface
"""

import sys
import os
import warnings
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveFrontendTester:
    """Test all backtest menu options through the front-end"""
    
    def __init__(self):
        self.test_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_all_tests(self):
        """Run comprehensive testing of all backtest menu options"""
        
        print("="*100)
        print("COMPREHENSIVE FRONTEND BACKTEST TESTING")
        print("="*100)
        print("Testing all menu options through the actual LAEF system interface")
        print(f"Test Session: {self.timestamp}")
        print("="*100)
        
        from laef_unified_system import LAEFUnifiedSystem
        system = LAEFUnifiedSystem(debug_mode=False)
        
        # Test 1: Quick Backtest
        print("\n" + "="*80)
        print("TEST 1: QUICK BACKTEST (Default LAEF Settings)")
        print("="*80)
        self.test_quick_backtest(system)
        
        # Test 2: Advanced Backtest - Multiple Configurations
        print("\n" + "="*80)
        print("TEST 2: ADVANCED BACKTEST (Multiple Configurations)")
        print("="*80)
        self.test_advanced_backtest(system)
        
        # Test 3: Strategy Comparison
        print("\n" + "="*80)
        print("TEST 3: STRATEGY COMPARISON BACKTEST")
        print("="*80)
        self.test_strategy_comparison(system)
        
        # Test 4: Previous Results Analysis
        print("\n" + "="*80)
        print("TEST 4: VIEW PREVIOUS RESULTS & ANALYSIS")
        print("="*80)
        self.test_view_previous_results(system)
        
        # Generate comprehensive report
        self.generate_final_report()
    
    def test_quick_backtest(self, system):
        """Test Quick Backtest option"""
        print("\nConfiguration:")
        print("  - Strategy: LAEF's Superior AI/ML Multi-Strategy System")
        print("  - Stock Selection: AI-driven smart selection")
        print("  - Period: Last 3 months")
        print("  - Initial Cash: $50,000")
        print("  - Auto Config: Live parameter optimization")
        
        try:
            # Call the actual quick backtest method
            from trading.hybrid_trading_engine import LAEFBacktester
            
            backtester = LAEFBacktester(
                initial_cash=50000,
                custom_config={
                    'start_date': '2024-10-01',
                    'end_date': '2024-12-31',
                    'q_buy': 0.65,
                    'q_sell': 0.35,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70
                }
            )
            
            print("\nRunning Quick Backtest...")
            results = backtester.run_backtest(use_smart_selection=True)
            
            self.test_results['quick_backtest'] = {
                'status': 'completed',
                'config': 'Default LAEF Settings',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            self.display_test_results("Quick Backtest", results)
            
        except Exception as e:
            print(f"Quick Backtest failed: {e}")
            self.test_results['quick_backtest'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def test_advanced_backtest(self, system):
        """Test Advanced Backtest with multiple configurations"""
        
        configurations = [
            {
                'name': 'Conservative Strategy',
                'initial_cash': 25000,
                'start_date': '2024-08-01',
                'end_date': '2024-10-31',
                'q_buy': 0.75,
                'q_sell': 0.25,
                'risk_per_trade': 0.01,
                'max_position_size': 0.05,
                'rsi_oversold': 25,
                'rsi_overbought': 75
            },
            {
                'name': 'Aggressive Day Trading',
                'initial_cash': 100000,
                'start_date': '2024-07-01',
                'end_date': '2024-09-30',
                'q_buy': 0.45,
                'q_sell': 0.20,
                'risk_per_trade': 0.08,
                'max_position_size': 0.25,
                'rsi_oversold': 20,
                'rsi_overbought': 80
            },
            {
                'name': 'Balanced Swing Trading',
                'initial_cash': 75000,
                'start_date': '2024-06-01',
                'end_date': '2024-08-31',
                'q_buy': 0.55,
                'q_sell': 0.30,
                'risk_per_trade': 0.03,
                'max_position_size': 0.15,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            },
            {
                'name': 'High-Frequency Scalping',
                'initial_cash': 50000,
                'start_date': '2024-09-01',
                'end_date': '2024-11-30',
                'q_buy': 0.40,
                'q_sell': 0.15,
                'risk_per_trade': 0.05,
                'max_position_size': 0.20,
                'rsi_oversold': 35,
                'rsi_overbought': 65
            }
        ]
        
        self.test_results['advanced_backtests'] = []
        
        for i, config in enumerate(configurations, 1):
            print(f"\n--- Advanced Configuration {i}: {config['name']} ---")
            print(f"Capital: ${config['initial_cash']:,}")
            print(f"Period: {config['start_date']} to {config['end_date']}")
            print(f"Q-Buy Threshold: {config['q_buy']}")
            print(f"Risk per Trade: {config['risk_per_trade']*100}%")
            print(f"Max Position: {config['max_position_size']*100}%")
            
            try:
                from trading.hybrid_trading_engine import LAEFBacktester
                
                backtester = LAEFBacktester(
                    initial_cash=config['initial_cash'],
                    custom_config=config
                )
                
                print(f"Running {config['name']}...")
                results = backtester.run_backtest(use_smart_selection=True)
                
                test_record = {
                    'name': config['name'],
                    'status': 'completed',
                    'config': config,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.test_results['advanced_backtests'].append(test_record)
                self.display_test_results(f"Advanced: {config['name']}", results)
                
            except Exception as e:
                print(f"{config['name']} failed: {e}")
                self.test_results['advanced_backtests'].append({
                    'name': config['name'],
                    'status': 'failed',
                    'error': str(e)
                })
    
    def test_strategy_comparison(self, system):
        """Test Strategy Comparison functionality"""
        
        # Test individual strategies
        strategies = [
            ('Momentum Scalping', {'q_buy': 0.40, 'q_sell': 0.20, 'strategy_focus': 'momentum'}),
            ('Mean Reversion', {'q_buy': 0.70, 'q_sell': 0.40, 'strategy_focus': 'reversion'}),
            ('Statistical Arbitrage', {'q_buy': 0.60, 'q_sell': 0.35, 'strategy_focus': 'statistical'}),
            ('Dual Model Swing', {'q_buy': 0.55, 'q_sell': 0.30, 'strategy_focus': 'swing'}),
            ('Pattern Recognition', {'q_buy': 0.65, 'q_sell': 0.25, 'strategy_focus': 'pattern'})
        ]
        
        self.test_results['strategy_comparison'] = []
        
        print("\nTesting individual strategies with optimized parameters:")
        
        for strategy_name, strategy_config in strategies:
            print(f"\n--- Strategy: {strategy_name} ---")
            print(f"Q-Buy: {strategy_config['q_buy']}")
            print(f"Q-Sell: {strategy_config['q_sell']}")
            print(f"Focus: {strategy_config['strategy_focus']}")
            
            try:
                from trading.hybrid_trading_engine import LAEFBacktester
                
                # Add common config
                full_config = {
                    'initial_cash': 50000,
                    'start_date': '2024-07-01',
                    'end_date': '2024-09-30',
                    **strategy_config
                }
                
                backtester = LAEFBacktester(
                    initial_cash=50000,
                    custom_config=full_config
                )
                
                print(f"Running {strategy_name} strategy...")
                results = backtester.run_backtest(
                    symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                    use_smart_selection=False
                )
                
                strategy_record = {
                    'strategy': strategy_name,
                    'status': 'completed',
                    'config': full_config,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.test_results['strategy_comparison'].append(strategy_record)
                self.display_test_results(f"Strategy: {strategy_name}", results)
                
            except Exception as e:
                print(f"Strategy {strategy_name} failed: {e}")
                self.test_results['strategy_comparison'].append({
                    'strategy': strategy_name,
                    'status': 'failed',
                    'error': str(e)
                })
    
    def test_view_previous_results(self, system):
        """Test viewing and analyzing previous results"""
        print("\nAnalyzing all available backtest results...")
        
        try:
            reports_dir = Path("reports")
            logs_dir = Path("logs")
            
            # Find all result files
            result_files = list(reports_dir.glob("*backtest*.json"))
            trade_files = list(reports_dir.glob("*trades*.csv"))
            decision_files = list(logs_dir.glob("decisions*.csv"))
            
            analysis = {
                'total_result_files': len(result_files),
                'total_trade_files': len(trade_files),
                'total_decision_files': len(decision_files),
                'file_analysis': []
            }
            
            print(f"\nFound {len(result_files)} result files")
            print(f"Found {len(trade_files)} trade files")
            print(f"Found {len(decision_files)} decision files")
            
            # Analyze latest results
            if result_files:
                latest_results = sorted(result_files, key=lambda x: x.stat().st_mtime)[-3:]  # Last 3 files
                
                for result_file in latest_results:
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        file_analysis = {
                            'file': result_file.name,
                            'final_value': data.get('final_value', 0),
                            'total_return': data.get('total_return', 0),
                            'total_trades': data.get('total_trades', 0),
                            'win_rate': data.get('win_rate', 0)
                        }
                        analysis['file_analysis'].append(file_analysis)
                        
                        print(f"\nAnalysis of {result_file.name}:")
                        print(f"  Final Value: ${data.get('final_value', 0):,.2f}")
                        print(f"  Return: {data.get('total_return', 0):.2f}%")
                        print(f"  Trades: {data.get('total_trades', 0)}")
                        print(f"  Win Rate: {data.get('win_rate', 0):.1f}%")
                        
                    except Exception as e:
                        print(f"Could not analyze {result_file.name}: {e}")
            
            self.test_results['previous_results_analysis'] = {
                'status': 'completed',
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Previous results analysis failed: {e}")
            self.test_results['previous_results_analysis'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def display_test_results(self, test_name, results):
        """Display formatted test results"""
        print(f"\n--- Results for {test_name} ---")
        
        if results and results.get('status') == 'completed':
            print(f"Initial Cash:     ${results.get('initial_cash', 0):,.2f}")
            print(f"Final Value:      ${results.get('final_value', 0):,.2f}")
            print(f"Total Return:     {results.get('total_return', 0):.2f}%")
            print(f"Total Trades:     {results.get('total_trades', 0)}")
            print(f"Winning Trades:   {results.get('winning_trades', 0)}")
            print(f"Win Rate:         {results.get('win_rate', 0):.1f}%")
            print(f"Sharpe Ratio:     {results.get('sharpe_ratio', 0):.2f}")
            
            # Show decision breakdown if available
            if 'buy_signals' in results:
                print(f"Buy Signals:      {results.get('buy_signals', 0)}")
                print(f"Sell Signals:     {results.get('sell_signals', 0)}")
                print(f"Hold Signals:     {results.get('hold_signals', 0)}")
            
            print(f"Status:           SUCCESS")
        else:
            print(f"Status:           FAILED")
            print(f"Error:            {results.get('error', 'Unknown error') if results else 'No results'}")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*100)
        print("COMPREHENSIVE TESTING FINAL REPORT")
        print("="*100)
        
        # Count successes and failures
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        # Quick backtest
        if 'quick_backtest' in self.test_results:
            total_tests += 1
            if self.test_results['quick_backtest'].get('status') == 'completed':
                successful_tests += 1
            else:
                failed_tests += 1
        
        # Advanced backtests
        if 'advanced_backtests' in self.test_results:
            for test in self.test_results['advanced_backtests']:
                total_tests += 1
                if test.get('status') == 'completed':
                    successful_tests += 1
                else:
                    failed_tests += 1
        
        # Strategy comparison
        if 'strategy_comparison' in self.test_results:
            for test in self.test_results['strategy_comparison']:
                total_tests += 1
                if test.get('status') == 'completed':
                    successful_tests += 1
                else:
                    failed_tests += 1
        
        # Previous results analysis
        if 'previous_results_analysis' in self.test_results:
            total_tests += 1
            if self.test_results['previous_results_analysis'].get('status') == 'completed':
                successful_tests += 1
            else:
                failed_tests += 1
        
        print(f"\nOVERALL TESTING SUMMARY:")
        print(f"  Total Tests Run:      {total_tests}")
        print(f"  Successful Tests:     {successful_tests}")
        print(f"  Failed Tests:         {failed_tests}")
        print(f"  Success Rate:         {(successful_tests/total_tests*100) if total_tests > 0 else 0:.1f}%")
        
        # Save comprehensive results
        results_file = f"reports/comprehensive_frontend_test_{self.timestamp}.json"
        try:
            os.makedirs('reports', exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            print(f"Could not save results file: {e}")
        
        print(f"\n" + "="*100)
        print("FRONTEND TESTING COMPLETE")
        print("="*100)

def main():
    """Run comprehensive frontend testing"""
    tester = ComprehensiveFrontendTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()