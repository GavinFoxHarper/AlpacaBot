#!/usr/bin/env python3
"""
Interactive script to run backtest through LAEF menu system
Simulates user interaction with the menu
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main system
from laef_unified_system import LAEFUnifiedSystem

def main():
    print("\n" + "="*70)
    print("Starting LAEF Trading System - New User Experience")
    print("="*70)
    
    # Initialize the system
    system = LAEFUnifiedSystem(debug_mode=False)
    
    # Show main menu and select backtest option
    print("\n" + "="*70)
    print("LAEF AI TRADING SYSTEM - MAIN MENU")
    print("="*70)
    print("\n1. Live Trading (Real Money)")
    print("2. Paper Trading (Simulated)")  
    print("3. Backtesting & Analysis")
    print("4. Performance Reports")
    print("5. System Configuration")
    print("6. Exit")
    
    print("\n[User selecting option 3 - Backtesting & Analysis]")
    print("-" * 50)
    
    # Call the backtest function directly
    system.run_backtesting()

if __name__ == "__main__":
    main()