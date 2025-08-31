#!/usr/bin/env python3
"""
Directly run the backtest function to show the experience
"""

import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "="*70)
    print("NEW USER BACKTEST EXPERIENCE")
    print("="*70)
    
    # Import the main system
    from laef_unified_system import LAEFUnifiedSystem
    
    # Initialize system
    print("\nInitializing LAEF Trading System...")
    system = LAEFUnifiedSystem(debug_mode=False)
    
    # Show what a user would see
    print("\n" + "="*70)
    print("MAIN MENU - User selects option 3 (Backtesting)")
    print("="*70)
    
    # Directly call the quick backtest function
    print("\nNavigating to: Backtesting -> Quick Backtest")
    print("-" * 70)
    
    try:
        # This is what happens when user selects Quick Backtest
        system._run_quick_backtest()
    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()