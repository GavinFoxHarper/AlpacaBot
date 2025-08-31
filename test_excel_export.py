"""Test Excel export functionality"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.enhanced_backtest_engine import EnhancedBacktestEngine

def test_excel_export():
    """Test the Excel export functionality with a short backtest"""
    print("Testing Excel export functionality...")
    
    # Initialize engine with small initial cash for quick test
    engine = EnhancedBacktestEngine(initial_cash=10000)
    
    # Run a very short backtest (just 5 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    symbols = ['AAPL']  # Just one symbol for quick test
    
    print(f"Running backtest from {start_date.date()} to {end_date.date()}")
    results = engine.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Check if Excel file was created
    reports_dir = Path("reports")
    excel_files = list(reports_dir.glob("backtest_results_*.xlsx"))
    
    if excel_files:
        latest_excel = max(excel_files, key=lambda p: p.stat().st_mtime)
        print(f"\n[SUCCESS] Excel file created successfully: {latest_excel}")
        
        # Try to read it back to verify it's valid
        try:
            import pandas as pd
            with pd.ExcelFile(latest_excel) as xls:
                sheet_names = xls.sheet_names
                print(f"[SUCCESS] Excel file contains {len(sheet_names)} sheets: {sheet_names}")
                
                # Read summary sheet
                summary_df = pd.read_excel(xls, 'Summary')
                print("\n[SUCCESS] Summary sheet preview:")
                print(summary_df.to_string())
        except Exception as e:
            print(f"[ERROR] Error reading Excel file: {e}")
            return False
    else:
        print("[ERROR] No Excel file was created")
        return False
    
    print("\n[SUCCESS] Excel export test completed successfully!")
    return True

if __name__ == "__main__":
    test_excel_export()