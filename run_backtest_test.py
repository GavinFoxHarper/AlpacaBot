#!/usr/bin/env python3
"""
Quick backtest test run
"""

import sys
import os
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_backtest():
    """Run a simple backtest"""
    try:
        print("="*70)
        print("RUNNING BACKTEST TEST")
        print("="*70)
        
        # Import the backtester
        from trading.hybrid_trading_engine import LAEFBacktester
        
        # Configure backtest
        print("\nBacktest Configuration:")
        print("  - Period: 2024-10-01 to 2024-12-31")
        print("  - Initial Cash: $50,000")
        print("  - Strategy: LAEF AI/ML Multi-Strategy System")
        print("  - Stock Selection: Smart Selection (AI-driven)")
        
        # Initialize backtester
        backtester = LAEFBacktester(
            initial_cash=50000,
            custom_config={
                'start_date': '2024-10-01',
                'end_date': '2024-12-31',
                'risk_per_trade': 0.02,
                'max_position_size': 0.10
            }
        )
        
        print("\nRunning backtest...")
        print("-" * 50)
        
        # Run the backtest
        results = backtester.run_backtest(use_smart_selection=True)
        
        if results:
            print("\n" + "="*70)
            print("BACKTEST RESULTS")
            print("="*70)
            
            # Display results
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if 'return' in key.lower() or 'ratio' in key.lower():
                        print(f"{key}: {value:.2%}")
                    elif 'cash' in key.lower() or 'value' in key.lower() or 'pnl' in key.lower():
                        print(f"{key}: ${value:,.2f}")
                    else:
                        print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
            
            print("\n" + "="*70)
            print("Backtest completed successfully!")
        else:
            print("\n[ERROR] Backtest failed - no results returned")
            
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("Make sure all required modules are installed")
    except Exception as e:
        print(f"\n[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_backtest()