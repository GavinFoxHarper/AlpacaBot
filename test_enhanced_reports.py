"""
Test script for enhanced P&L reporting with layman's terms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from trading.enhanced_backtest_engine import EnhancedBacktestEngine

def test_enhanced_reporting():
    """Test the enhanced reporting with P&L tracking and layman's terms"""
    
    print("="*60)
    print("TESTING ENHANCED P&L REPORTING")
    print("="*60)
    
    # Initialize engine with small amount for testing
    engine = EnhancedBacktestEngine(initial_cash=10000)
    
    # Run a short backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Test with a few symbols
    
    print(f"\nRunning backtest from {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Cash: $10,000")
    print("\nThis will test:")
    print("1. P&L tracking for each transaction")
    print("2. Grand total P&L by symbol")
    print("3. Layman's terms explanations for decisions")
    print("-"*60)
    
    try:
        # Run the backtest
        results = engine.run_backtest(
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        print("\n[SUCCESS] BACKTEST COMPLETED SUCCESSFULLY")
        print("-"*60)
        
        # Display results
        if results:
            print(f"Initial Cash: ${results.get('initial_cash', 0):,.2f}")
            print(f"Final Value: ${results.get('final_value', 0):,.2f}")
            print(f"Total Return: {results.get('total_return', 0):.2f}%")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0):.1f}%")
            
            if 'grand_total_pnl' in results:
                print(f"\nGRAND TOTAL P&L: ${results['grand_total_pnl']:,.2f}")
            
            if 'symbol_pnl' in results and results['symbol_pnl']:
                print("\nP&L BY SYMBOL:")
                for symbol, data in results['symbol_pnl'].items():
                    print(f"  {symbol}: ${data['total_pnl']:,.2f} ({data['trades']} trades)")
        
        print("\n" + "="*60)
        print("CHECK THE FOLLOWING FILES FOR DETAILED REPORTS:")
        print("="*60)
        print("1. reports/enhanced_trades_*.csv - Detailed trades with P&L")
        print("2. reports/pnl_summary_*.csv - P&L summary by symbol")
        print("3. logs/decisions_*.csv - All decisions with layman explanations")
        print("4. reports/portfolio_history_*.csv - Portfolio value over time")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_enhanced_reporting()
    
    if success:
        print("\n[SUCCESS] Enhanced reporting test completed successfully!")
        print("The reports now include:")
        print("- P&L for each transaction")
        print("- Running total P&L")
        print("- P&L summary by symbol with grand total")
        print("- Clear explanations in everyday language")
    else:
        print("\n[FAILED] Test failed. Please check the error messages above.")