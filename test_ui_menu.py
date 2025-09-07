#!/usr/bin/env python3
"""
UI Testing Script for AlpacaBot Menu Interface
Tests all menu options systematically
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class AlpacaBotUITester:
    """Comprehensive UI tester for AlpacaBot menu system"""
    
    def __init__(self):
        self.test_results = []
        self.base_path = Path(__file__).parent
        
    def log_result(self, test_name, status, details=""):
        """Log test result"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {details}")
    
    def test_menu_display(self):
        """Test if the main menu displays correctly"""
        print("\n=== TESTING MENU DISPLAY ===")
        
        # Check if the batch file exists
        batch_file = self.base_path / "start_alpacabot.bat"
        if not batch_file.exists():
            self.log_result("Menu File Existence", "FAIL", "start_alpacabot.bat not found")
            return False
        
        self.log_result("Menu File Existence", "PASS", "start_alpacabot.bat found")
        
        # Read and validate menu structure
        try:
            with open(batch_file, 'r') as f:
                content = f.read()
                
            # Check for menu options
            menu_options = [
                "Live Trading Dashboard",
                "Backtest Mode", 
                "Training Mode",
                "Configuration Manager",
                "Performance Analysis",
                "Exit"
            ]
            
            for option in menu_options:
                if option in content:
                    self.log_result(f"Menu Option: {option}", "PASS", "Found in menu")
                else:
                    self.log_result(f"Menu Option: {option}", "FAIL", "Missing from menu")
                    
        except Exception as e:
            self.log_result("Menu Content Reading", "FAIL", str(e))
            return False
            
        return True
    
    def test_live_trading_option(self):
        """Test Live Trading Dashboard option"""
        print("\n=== TESTING LIVE TRADING DASHBOARD ===")
        
        # Check if live_monitoring_dashboard.py exists
        dashboard_file = self.base_path / "live_monitoring_dashboard.py"
        if not dashboard_file.exists():
            self.log_result("Live Dashboard File", "FAIL", "live_monitoring_dashboard.py not found")
            return False
        
        self.log_result("Live Dashboard File", "PASS", "live_monitoring_dashboard.py exists")
        
        # Test basic syntax validity
        try:
            with open(dashboard_file, 'r') as f:
                content = f.read()
                compile(content, dashboard_file, 'exec')
            self.log_result("Live Dashboard Syntax", "PASS", "Python syntax valid")
        except SyntaxError as e:
            self.log_result("Live Dashboard Syntax", "FAIL", f"Syntax error: {e}")
            return False
        except Exception as e:
            self.log_result("Live Dashboard Syntax", "FAIL", f"Error: {e}")
            return False
        
        # Test if main function exists
        if "def main():" in content:
            self.log_result("Live Dashboard Main Function", "PASS", "main() function found")
        else:
            self.log_result("Live Dashboard Main Function", "FAIL", "main() function missing")
        
        # Test key components
        key_components = [
            "LiveMonitoringDashboard",
            "get_evolution_status", 
            "generate_evolution_report",
            "save_evolution_snapshot"
        ]
        
        for component in key_components:
            if component in content:
                self.log_result(f"Dashboard Component: {component}", "PASS", "Found")
            else:
                self.log_result(f"Dashboard Component: {component}", "FAIL", "Missing")
        
        return True
    
    def test_backtest_option(self):
        """Test Backtest Mode functionality"""
        print("\n=== TESTING BACKTEST MODE ===")
        
        # Check for enhanced_backtest_engine.py
        backtest_file = self.base_path / "trading" / "enhanced_backtest_engine.py"
        if not backtest_file.exists():
            self.log_result("Backtest Engine File", "FAIL", "enhanced_backtest_engine.py not found")
            return False
        
        self.log_result("Backtest Engine File", "PASS", "enhanced_backtest_engine.py exists")
        
        # Test syntax
        try:
            with open(backtest_file, 'r') as f:
                content = f.read()
                compile(content, backtest_file, 'exec')
            self.log_result("Backtest Engine Syntax", "PASS", "Python syntax valid")
        except Exception as e:
            self.log_result("Backtest Engine Syntax", "FAIL", str(e))
            return False
        
        # Check for EnhancedBacktestEngine class
        if "class EnhancedBacktestEngine" in content:
            self.log_result("BacktestEngine Class", "PASS", "EnhancedBacktestEngine found")
        else:
            self.log_result("BacktestEngine Class", "FAIL", "EnhancedBacktestEngine missing")
        
        # Check for run_backtest method
        if "def run_backtest" in content:
            self.log_result("Backtest Run Method", "PASS", "run_backtest method found")
        else:
            self.log_result("Backtest Run Method", "FAIL", "run_backtest method missing")
        
        return True
    
    def test_training_mode(self):
        """Test Training Mode submenu"""
        print("\n=== TESTING TRAINING MODE ===")
        
        # Check training directory
        training_dir = self.base_path / "training"
        if not training_dir.exists():
            self.log_result("Training Directory", "FAIL", "training/ directory not found")
            return False
        
        self.log_result("Training Directory", "PASS", "training/ directory exists")
        
        # Check for training files
        training_files = {
            "comprehensive_trainer.py": "ComprehensiveTrainer",
            "ml_trainer.py": "train_all_models"
        }
        
        for filename, expected_component in training_files.items():
            file_path = training_dir / filename
            if file_path.exists():
                self.log_result(f"Training File: {filename}", "PASS", "File exists")
                
                # Check content
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if expected_component in content:
                        self.log_result(f"Training Component: {expected_component}", "PASS", f"Found in {filename}")
                    else:
                        self.log_result(f"Training Component: {expected_component}", "FAIL", f"Missing from {filename}")
                except Exception as e:
                    self.log_result(f"Training File Reading: {filename}", "FAIL", str(e))
            else:
                self.log_result(f"Training File: {filename}", "FAIL", "File not found")
        
        return True
    
    def test_config_manager(self):
        """Test Configuration Manager functionality"""
        print("\n=== TESTING CONFIGURATION MANAGER ===")
        
        # Check config.py
        config_file = self.base_path / "config.py"
        if not config_file.exists():
            self.log_result("Config File", "FAIL", "config.py not found")
            return False
        
        self.log_result("Config File", "PASS", "config.py exists")
        
        # Test config file content
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for key configuration variables
            config_vars = [
                "PAPER_TRADING",
                "INITIAL_CASH", 
                "MAX_POSITIONS",
                "STOP_LOSS_PERCENT",
                "TAKE_PROFIT_PERCENT"
            ]
            
            for var in config_vars:
                if var in content:
                    self.log_result(f"Config Variable: {var}", "PASS", "Found")
                else:
                    self.log_result(f"Config Variable: {var}", "FAIL", "Missing")
            
            # Check for profile functions
            profile_functions = ["load_profile", "save_profile", "validate_config"]
            for func in profile_functions:
                if func in content:
                    self.log_result(f"Config Function: {func}", "PASS", "Found")
                else:
                    self.log_result(f"Config Function: {func}", "FAIL", "Missing")
                    
        except Exception as e:
            self.log_result("Config File Reading", "FAIL", str(e))
            return False
        
        # Check config_profiles directory
        profiles_dir = self.base_path / "config_profiles"
        if profiles_dir.exists():
            self.log_result("Config Profiles Directory", "PASS", "Directory exists")
        else:
            self.log_result("Config Profiles Directory", "FAIL", "Directory missing")
        
        return True
    
    def test_performance_analysis(self):
        """Test Performance Analysis functionality"""
        print("\n=== TESTING PERFORMANCE ANALYSIS ===")
        
        # Check analysis files
        analysis_files = {
            "show_backtest_results.py": "Show results functionality",
            "detailed_results_analyzer.py": "Detailed analysis functionality"
        }
        
        for filename, description in analysis_files.items():
            file_path = self.base_path / filename
            if file_path.exists():
                self.log_result(f"Analysis File: {filename}", "PASS", "File exists")
                
                # Test syntax
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        compile(content, file_path, 'exec')
                    self.log_result(f"Analysis Syntax: {filename}", "PASS", "Valid Python syntax")
                except Exception as e:
                    self.log_result(f"Analysis Syntax: {filename}", "FAIL", str(e))
            else:
                self.log_result(f"Analysis File: {filename}", "FAIL", "File not found")
        
        # Check utils directory for report generator
        utils_dir = self.base_path / "utils"
        if utils_dir.exists():
            self.log_result("Utils Directory", "PASS", "Directory exists")
            
            # Check for report_generator.py
            report_gen = utils_dir / "report_generator.py"
            if report_gen.exists():
                self.log_result("Report Generator", "PASS", "report_generator.py exists")
            else:
                self.log_result("Report Generator", "FAIL", "report_generator.py missing")
        else:
            self.log_result("Utils Directory", "FAIL", "utils/ directory missing")
        
        # Check logs directory
        logs_dir = self.base_path / "logs"
        if logs_dir.exists():
            self.log_result("Logs Directory", "PASS", "logs/ directory exists")
        else:
            self.log_result("Logs Directory", "FAIL", "logs/ directory missing")
        
        return True
    
    def test_directory_structure(self):
        """Test overall directory structure and required files"""
        print("\n=== TESTING DIRECTORY STRUCTURE ===")
        
        required_dirs = ["core", "trading", "training", "utils", "models", "reports", "logs"]
        for dir_name in required_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                self.log_result(f"Directory: {dir_name}", "PASS", "Directory exists")
            else:
                self.log_result(f"Directory: {dir_name}", "FAIL", "Directory missing")
        
        # Check .env file
        env_file = self.base_path / ".env"
        if env_file.exists():
            self.log_result("Environment File", "PASS", ".env file exists")
        else:
            self.log_result("Environment File", "FAIL", ".env file missing")
        
        # Check requirements.txt
        req_file = self.base_path / "requirements.txt"
        if req_file.exists():
            self.log_result("Requirements File", "PASS", "requirements.txt exists")
        else:
            self.log_result("Requirements File", "FAIL", "requirements.txt missing")
        
        return True
    
    def test_input_validation(self):
        """Test input validation and error handling"""
        print("\n=== TESTING INPUT VALIDATION ===")
        
        # This would test the batch file's input validation
        # Since we can't easily run the batch file interactively, 
        # we'll examine the validation logic in the batch file
        
        batch_file = self.base_path / "start_alpacabot.bat"
        try:
            with open(batch_file, 'r') as f:
                content = f.read()
            
            # Check for input validation
            if 'if "%choice%"==' in content:
                self.log_result("Menu Choice Validation", "PASS", "Input validation found")
            else:
                self.log_result("Menu Choice Validation", "FAIL", "No input validation")
            
            # Check for error handling
            if "Invalid choice" in content:
                self.log_result("Invalid Input Handling", "PASS", "Error handling found")
            else:
                self.log_result("Invalid Input Handling", "FAIL", "No error handling")
            
            # Check for timeout functionality
            if "timeout" in content:
                self.log_result("Timeout Functionality", "PASS", "Timeout features found")
            else:
                self.log_result("Timeout Functionality", "FAIL", "No timeout features")
                
        except Exception as e:
            self.log_result("Input Validation Test", "FAIL", str(e))
        
        return True
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print(" "*20 + "ALPACABOT UI TEST REPORT")
        print("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print(f"\nTEST SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        print(f"\nDETAILED RESULTS:")
        print("-"*70)
        
        for result in self.test_results:
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_icon} [{result['status']:>4}] {result['test']:<40} | {result['details']}")
        
        # Group results by category
        categories = {}
        for result in self.test_results:
            category = result['test'].split(':')[0] if ':' in result['test'] else 'General'
            if category not in categories:
                categories[category] = {'pass': 0, 'fail': 0}
            
            if result['status'] == 'PASS':
                categories[category]['pass'] += 1
            else:
                categories[category]['fail'] += 1
        
        print(f"\nCATEGORY BREAKDOWN:")
        print("-"*50)
        for category, counts in categories.items():
            total = counts['pass'] + counts['fail']
            success_rate = (counts['pass'] / total * 100) if total > 0 else 0
            print(f"{category:<30} | {counts['pass']}/{total} ({success_rate:.1f}%)")
        
        print("\n" + "="*70)
        
        return self.test_results
    
    def run_all_tests(self):
        """Run all UI tests"""
        print("Starting comprehensive UI testing for AlpacaBot...")
        print("="*70)
        
        # Run all tests
        self.test_directory_structure()
        self.test_menu_display()
        self.test_live_trading_option()
        self.test_backtest_option()
        self.test_training_mode()
        self.test_config_manager()
        self.test_performance_analysis()
        self.test_input_validation()
        
        # Generate and return report
        return self.generate_report()

def main():
    """Main testing function"""
    tester = AlpacaBotUITester()
    results = tester.run_all_tests()
    
    # Save results to file
    report_file = Path(__file__).parent / "reports" / f"ui_test_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(f"AlpacaBot UI Test Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for result in results:
            f.write(f"[{result['status']}] {result['test']}: {result['details']}\n")
    
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    main()