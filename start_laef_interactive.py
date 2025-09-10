#!/usr/bin/env python3
"""
LAEF Interactive System Launcher
Provides interactive menu for AlpacaBot trading operations
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main system
from laef_unified_system import LAEFUnifiedSystem

def main():
    """Main entry point for interactive LAEF system"""
    print("\n" + "="*70)
    print("Starting LAEF Trading System - Interactive Mode")
    print("="*70 + "\n")
    
    try:
        # Initialize the LAEF system
        system = LAEFUnifiedSystem()
        
        # Run the interactive menu
        while True:
            print("\n" + "="*70)
            print("LAEF AI TRADING SYSTEM - MAIN MENU")
            print("="*70 + "\n")
            
            print("1. Live Trading (Real Money)")
            print("2. Paper Trading (Simulated)")
            print("3. Backtesting & Analysis")
            print("4. Performance Reports")
            print("5. System Configuration")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                print("\n[Live Trading Selected]")
                print("WARNING: This will use real money. Are you sure? (yes/no)")
                confirm = input().strip().lower()
                if confirm == 'yes':
                    system.run_live_trading()
                else:
                    print("Live trading cancelled.")
                    
            elif choice == '2':
                print("\n[Paper Trading Selected]")
                system.run_paper_trading()
                
            elif choice == '3':
                print("\n[Backtesting & Analysis Selected]")
                system.run_backtesting()
                
            elif choice == '4':
                print("\n[Performance Reports Selected]")
                system.show_performance_reports()
                
            elif choice == '5':
                print("\n[System Configuration Selected]")
                system.configure_system()
                
            elif choice == '6':
                print("\n[Exiting LAEF System]")
                print("Thank you for using LAEF Trading System!")
                break
                
            else:
                print("\nInvalid option. Please select 1-6.")
                
    except KeyboardInterrupt:
        print("\n\n[System interrupted by user]")
        print("Shutting down LAEF Trading System...")
        
    except Exception as e:
        print(f"\n[Error] System error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()