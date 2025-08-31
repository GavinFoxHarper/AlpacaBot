#!/usr/bin/env python3
"""
Simulates a new user running backtest through the menu system
This script automates the menu selections a user would make
"""

import sys
import os
from io import StringIO
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simulate_user_input():
    """Simulate user selecting backtest options"""
    # Sequence of inputs:
    # 3 - Select Backtesting from main menu
    # 1 - Select Quick Backtest
    # Enter - Continue after results
    # 5 - Back to backtest menu
    # 5 - Back to main menu
    # 6 - Exit
    inputs = ['3', '1', '', '5', '5', '6']
    return iter(inputs)

def main():
    print("\n" + "="*70)
    print("SIMULATING NEW USER EXPERIENCE - BACKTEST FROM MENU")
    print("="*70)
    print("\nSimulating a new user starting LAEF and running a backtest...")
    print("User actions will be: Main Menu -> Backtesting -> Quick Backtest\n")
    
    # Import after path setup
    from laef_unified_system import LAEFUnifiedSystem
    
    # Mock the input function to simulate user selections
    user_inputs = simulate_user_input()
    
    with patch('builtins.input', side_effect=lambda prompt: next(user_inputs, '')):
        try:
            # Create system instance
            system = LAEFUnifiedSystem(debug_mode=False)
            
            # Run the main menu (this will process our simulated inputs)
            system.show_main_menu()
            
        except StopIteration:
            print("\n[Simulation Complete]")
        except Exception as e:
            print(f"\n[Error during simulation]: {e}")

if __name__ == "__main__":
    main()