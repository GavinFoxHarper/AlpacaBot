"""
Test script for LAEF trading system
"""

import logging
import threading
import time
from typing import List, Dict
import builtins

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Mock input function
original_input = builtins.input

def mock_input(prompt):
    test_system = TestLAEFSystem.instance
    return test_system.get_input(prompt)

builtins.input = mock_input

class TestLAEFSystem:
    """Test harness for LAEF system"""
    
    instance = None
    
    def __init__(self):
        TestLAEFSystem.instance = self
        self.responses: List[str] = []
        self.stop_event = threading.Event()
        
    def test_menu_navigation(self):
        """Test menu system navigation"""
        print("\nTesting Paper Trading...")
        from laef_unified_system import LAEFUnifiedSystem
        
        # Start system
        system = LAEFUnifiedSystem(debug_mode=True)
        
        # Paper trading test
        self.responses = ["2", "1", "7"]  # Paper trading -> Standard -> Exit
        system.run()
        
        print("\nTesting Backtesting...")
        # Backtesting test
        self.responses = ["3", "1", "7"]  # Backtesting -> Quick -> Exit
        system.run()
        
        print("\nTesting Settings...")
        # Settings test
        self.responses = ["1", "7"]  # Live trading -> Exit
        system.run()
        
    def get_input(self, prompt: str = "") -> str:
        """Mock user input"""
        if self.responses:
            response = self.responses.pop(0)
            if prompt:
                print(f"{prompt}{response}")
            return response
        return "7"  # Default to exit
        
    def run_tests(self):
        """Run all system tests"""
        print("=" * 70)
        print("LAEF SYSTEM TEST SUITE")
        print("=" * 70)
        
        try:
            self.test_menu_navigation()
            print("\nAll tests completed!")
            
        except Exception as e:
            print(f"\nTest failed: {e}")
            logging.error("Test error", exc_info=True)
            
        finally:
            self.stop_event.set()
            builtins.input = original_input  # Restore original input

def main():
    test_system = TestLAEFSystem()
    test_system.run_tests()

if __name__ == "__main__":
    main()