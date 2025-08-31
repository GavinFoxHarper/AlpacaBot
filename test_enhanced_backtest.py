#!/usr/bin/env python3
"""
Test the enhanced backtest implementation
"""

import sys
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_backtest():
    """Test the new enhanced backtest engine"""
    
    print("="*80)
    print("TESTING ENHANCED BACKTEST ENGINE")
    print("="*80)
    
    try:
        from trading.enhanced_backtest_engine import EnhancedBacktestEngine
        
        # Test configuration
        config = {
            'q_buy': 0.45,              # Moderate entry threshold
            'q_sell': 0.25,             # Moderate exit threshold
            'rsi_oversold': 30,         # Standard oversold
            'rsi_overbought': 70,       # Standard overbought
            'max_risk_per_trade': 0.05, # 5% risk
            'max_position_size': 0.20,  # 20% max position
        }
        
        print("\nBacktest Configuration:")
        print("  - Initial Cash: $50,000")
        print("  - Period: Last 60 days")
        print("  - Symbols: AAPL, MSFT, GOOGL, TSLA, NVDA")
        print("  - Strategy: Hybrid Trading (Day + Swing)")
        print("  - Risk per trade: 5%")
        print("  - Max position: 20%")
        
        # Initialize engine
        engine = EnhancedBacktestEngine(
            initial_cash=50000,
            custom_config=config
        )
        
        # Calculate date range (last 60 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        print(f"\nDate Range: {start_date} to {end_date}")
        print("\nRunning backtest with actual market data...")
        print("-" * 80)
        
        # Run backtest
        results = engine.run_backtest(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            start_date=start_date,
            end_date=end_date,
            use_smart_selection=False
        )
        
        if results and results.get('status') == 'completed':
            print("\n" + "="*80)
            print("BACKTEST RESULTS")
            print("="*80)
            
            print(f"\nPerformance Metrics:")
            print(f"  Initial Capital:    ${results['initial_cash']:,.2f}")
            print(f"  Final Value:        ${results['final_value']:,.2f}")
            print(f"  Total Return:       {results['total_return']:.2f}%")
            print(f"  Sharpe Ratio:       {results.get('sharpe_ratio', 0):.2f}")
            
            print(f"\nTrading Statistics:")
            print(f"  Total Trades:       {results['total_trades']}")
            print(f"  Winning Trades:     {results.get('winning_trades', 0)}")
            print(f"  Losing Trades:      {results.get('losing_trades', 0)}")
            print(f"  Win Rate:           {results['win_rate']:.1f}%")
            print(f"  Avg Profit:         ${results.get('avg_profit', 0):.2f}")
            print(f"  Avg Loss:           ${results.get('avg_loss', 0):.2f}")
            
            print(f"\nSignal Analysis:")
            print(f"  Total Decisions:    {results.get('total_decisions', 0)}")
            print(f"  Buy Signals:        {results.get('buy_signals', 0)}")
            print(f"  Sell Signals:       {results.get('sell_signals', 0)}")
            print(f"  Hold Signals:       {results.get('hold_signals', 0)}")
            
            # Check if trades were executed
            if results['total_trades'] > 0:
                print("\n✓ SUCCESS: Trades were executed!")
                print("✓ The enhanced backtest engine is working properly")
                
                # Show last few trades if available
                if engine.trades:
                    print("\nLast 5 Trades:")
                    for trade in engine.trades[-5:]:
                        if trade['action'] == 'BUY':
                            print(f"  {trade['date'].strftime('%Y-%m-%d')}: BUY {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")
                        else:
                            pnl_str = f"+${trade['pnl']:.2f}" if trade['pnl'] > 0 else f"-${abs(trade['pnl']):.2f}"
                            print(f"  {trade['date'].strftime('%Y-%m-%d')}: SELL {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f} (PnL: {pnl_str})")
            else:
                print("\nWARNING: No trades were executed")
                print("This could be due to:")
                print("  - Market conditions not triggering signals")
                print("  - Overly conservative thresholds")
                print("  - Limited date range")
                
                # Show some decision samples
                if engine.decision_log:
                    print(f"\nSample decisions (showing 5 of {len(engine.decision_log)}):")
                    for decision in engine.decision_log[:5]:
                        print(f"  {decision['symbol']}: {decision['decision']} - Q:{decision['q_value']:.3f} ML:{decision['ml_confidence']:.3f} RSI:{decision['rsi']:.1f}")
        
        elif results and results.get('status') == 'no_data':
            print("\nERROR: No market data could be fetched")
            print("Check your internet connection and try again")
        else:
            print("\nERROR: Backtest failed")
            print("Check the logs for details")
            
    except ImportError as e:
        print(f"\nERROR: Import failed - {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install yfinance pandas numpy")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_backtest()