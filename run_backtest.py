#!/usr/bin/env python3
"""
Run a comprehensive backtest with the LAEF system
"""

import sys
from datetime import datetime, timedelta
from trading.enhanced_backtest_engine import EnhancedBacktestEngine

def run_comprehensive_backtest():
    """Run a comprehensive backtest with multiple configurations"""
    
    print("="*70)
    print("LAEF COMPREHENSIVE BACKTEST")
    print("="*70)
    
    # Test different time periods
    test_configs = [
        {
            'name': 'Recent 3 Months',
            'start': '2024-10-01',
            'end': '2024-12-31',
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        },
        {
            'name': 'Last Month',
            'start': '2024-12-01',
            'end': '2024-12-31',
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
        }
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"Period: {config['start']} to {config['end']}")
        print(f"Symbols: {', '.join(config['symbols'])}")
        print("="*50)
        
        try:
            # Create engine with optimized settings
            engine = EnhancedBacktestEngine(
                initial_cash=50000,
                custom_config={
                    'q_buy': 0.35,
                    'q_sell': 0.30,
                    'ml_profit_peak': 0.35,
                    'day_trade_profit': 0.008,
                    'swing_trade_profit': 0.02,
                    'stop_loss_pct': 0.98,
                    'max_risk_per_trade': 0.03,
                    'max_position_size': 0.10
                }
            )
            
            # Run backtest
            results = engine.run_backtest(
                symbols=config['symbols'],
                start_date=config['start'],
                end_date=config['end'],
                use_smart_selection=False
            )
            
            if results:
                # Calculate actual return percentage
                actual_return = ((results['final_value'] - results['initial_cash']) / results['initial_cash']) * 100
                
                print(f"\n✓ BACKTEST RESULTS for {config['name']}:")
                print(f"  Initial Cash: ${results['initial_cash']:,.2f}")
                print(f"  Final Value: ${results['final_value']:,.2f}")
                print(f"  Total Return: {actual_return:.2f}%")
                print(f"  Total P&L: ${results['final_value'] - results['initial_cash']:,.2f}")
                print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
                print(f"  Total Trades: {results.get('total_trades', 0)}")
                print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
                
                # Show symbol breakdown
                print(f"\n  Symbol Performance:")
                for symbol, pnl_data in results.get('symbol_pnl', {}).items():
                    win_rate = (pnl_data['wins'] / pnl_data['trades'] * 100) if pnl_data['trades'] > 0 else 0
                    print(f"    {symbol:5} | Trades: {pnl_data['trades']:3} | Wins: {pnl_data['wins']:3} | P&L: ${pnl_data['total_pnl']:8.2f} | Win Rate: {win_rate:.1f}%")
                
                all_results.append({
                    'config': config,
                    'results': results,
                    'actual_return': actual_return
                })
            else:
                print(f"✗ Backtest failed for {config['name']}")
                
        except Exception as e:
            print(f"✗ Error running backtest for {config['name']}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    
    if all_results:
        avg_return = sum(r['actual_return'] for r in all_results) / len(all_results)
        best_result = max(all_results, key=lambda x: x['actual_return'])
        worst_result = min(all_results, key=lambda x: x['actual_return'])
        
        print(f"Tests Completed: {len(all_results)}/{len(test_configs)}")
        print(f"Average Return: {avg_return:.2f}%")
        print(f"Best Period: {best_result['config']['name']} ({best_result['actual_return']:.2f}%)")
        print(f"Worst Period: {worst_result['config']['name']} ({worst_result['actual_return']:.2f}%)")
    else:
        print("No successful backtests completed")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_backtest()
    
    if results:
        print("\n✓ Backtest completed successfully!")
        print("Check the reports/ and logs/ folders for detailed results.")
    else:
        print("\n✗ All backtests failed. Please check the configuration.")
        sys.exit(1)