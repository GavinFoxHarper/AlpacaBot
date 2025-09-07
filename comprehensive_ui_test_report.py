#!/usr/bin/env python3
"""
Comprehensive UI Test Report for AlpacaBot
Tests all menu options and interactive elements without dependencies
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

class AlpacaBotUITester:
    """Comprehensive UI tester that analyzes AlpacaBot's interface structure"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.test_results = []
        self.ui_coverage = {}
        
    def log_test(self, component: str, test_name: str, status: str, details: str = ""):
        """Log a test result"""
        result = {
            'component': component,
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.test_results.append(result)
        status_icon = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[WARN]")
        print(f"  {status_icon} {test_name}: {details}")
        
    def test_main_menu_interface(self):
        """Test the main command-line menu interface"""
        print("\n=== TESTING MAIN MENU INTERFACE ===")
        
        batch_file = self.base_path / "start_alpacabot.bat"
        
        if not batch_file.exists():
            self.log_test("Main Menu", "Batch File Existence", "FAIL", "start_alpacabot.bat not found")
            return
            
        self.log_test("Main Menu", "Entry Point File", "PASS", "start_alpacabot.bat exists")
        
        # Read and analyze menu structure
        try:
            with open(batch_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Test menu options
            expected_menu_options = [
                ("Live Trading Dashboard", "1."),
                ("Backtest Mode", "2."),
                ("Training Mode", "3."),
                ("Configuration Manager", "4."),
                ("Performance Analysis", "5."),
                ("Exit", "6.")
            ]
            
            menu_coverage = {}
            
            for option_name, option_number in expected_menu_options:
                if option_name in content and option_number in content:
                    self.log_test("Main Menu", f"Option {option_number[0]}: {option_name}", "PASS", "Menu option found")
                    menu_coverage[option_name] = "PRESENT"
                else:
                    self.log_test("Main Menu", f"Option {option_number[0]}: {option_name}", "FAIL", "Menu option missing")
                    menu_coverage[option_name] = "MISSING"
            
            # Test input validation
            if 'set /p choice="Enter your choice' in content:
                self.log_test("Main Menu", "Input Prompt", "PASS", "User input prompt found")
            else:
                self.log_test("Main Menu", "Input Prompt", "FAIL", "No input prompt")
                
            # Test input validation logic
            if 'if "%choice%"==' in content:
                self.log_test("Main Menu", "Input Validation", "PASS", "Input validation logic present")
            else:
                self.log_test("Main Menu", "Input Validation", "FAIL", "No input validation")
                
            # Test error handling
            if "Invalid choice" in content or "ERROR" in content:
                self.log_test("Main Menu", "Error Handling", "PASS", "Error handling present")
            else:
                self.log_test("Main Menu", "Error Handling", "WARN", "Limited error handling")
            
            self.ui_coverage["main_menu"] = menu_coverage
            
        except Exception as e:
            self.log_test("Main Menu", "File Analysis", "FAIL", f"Error reading file: {e}")
    
    def test_submenu_structures(self):
        """Test all submenu structures"""
        print("\n=== TESTING SUBMENU STRUCTURES ===")
        
        batch_file = self.base_path / "start_alpacabot.bat"
        
        try:
            with open(batch_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Test Training Mode submenu (Option 3)
            training_submenu_options = [
                "Q-Learning Agent",
                "ML Models (Random Forest, XGBoost)", 
                "Comprehensive Training (All models)",
                "Back to main menu"
            ]
            
            training_coverage = {}
            for option in training_submenu_options:
                if option in content:
                    self.log_test("Training Submenu", f"Option: {option}", "PASS", "Submenu option found")
                    training_coverage[option] = "PRESENT"
                else:
                    self.log_test("Training Submenu", f"Option: {option}", "FAIL", "Submenu option missing")
                    training_coverage[option] = "MISSING"
            
            # Test Configuration Manager submenu (Option 4)
            config_submenu_options = [
                "View current configuration",
                "Edit configuration",
                "Load configuration profile",
                "Save configuration profile",
                "Validate configuration",
                "Back to main menu"
            ]
            
            config_coverage = {}
            for option in config_submenu_options:
                if option in content:
                    self.log_test("Config Submenu", f"Option: {option}", "PASS", "Submenu option found")
                    config_coverage[option] = "PRESENT"
                else:
                    self.log_test("Config Submenu", f"Option: {option}", "FAIL", "Submenu option missing")
                    config_coverage[option] = "MISSING"
            
            # Test Performance Analysis submenu (Option 5)
            analysis_submenu_options = [
                "Show latest backtest results",
                "Analyze trading performance",
                "Generate detailed report",
                "View trade logs",
                "Back to main menu"
            ]
            
            analysis_coverage = {}
            for option in analysis_submenu_options:
                if option in content:
                    self.log_test("Analysis Submenu", f"Option: {option}", "PASS", "Submenu option found")
                    analysis_coverage[option] = "PRESENT"
                else:
                    self.log_test("Analysis Submenu", f"Option: {option}", "FAIL", "Submenu option missing")
                    analysis_coverage[option] = "MISSING"
            
            self.ui_coverage["training_submenu"] = training_coverage
            self.ui_coverage["config_submenu"] = config_coverage
            self.ui_coverage["analysis_submenu"] = analysis_coverage
            
        except Exception as e:
            self.log_test("Submenu Analysis", "File Reading", "FAIL", f"Error: {e}")
    
    def test_file_dependencies(self):
        """Test that all referenced files exist and are accessible"""
        print("\n=== TESTING FILE DEPENDENCIES ===")
        
        # Test core Python files
        core_files = {
            "live_monitoring_dashboard.py": "Live Trading Dashboard",
            "config.py": "Configuration System",
            "show_backtest_results.py": "Backtest Results Display",
            "detailed_results_analyzer.py": "Performance Analysis"
        }
        
        for filename, description in core_files.items():
            file_path = self.base_path / filename
            if file_path.exists():
                self.log_test("File Dependencies", description, "PASS", f"{filename} exists")
                
                # Test file readability
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # Read first 1000 chars
                    self.log_test("File Dependencies", f"{description} - Readability", "PASS", "File is readable")
                except Exception as e:
                    self.log_test("File Dependencies", f"{description} - Readability", "FAIL", f"Cannot read: {e}")
            else:
                self.log_test("File Dependencies", description, "FAIL", f"{filename} missing")
        
        # Test directory structure
        required_dirs = {
            "core": "Core system modules",
            "trading": "Trading engines", 
            "training": "ML/AI training modules",
            "utils": "Utility functions",
            "models": "Trained models storage",
            "reports": "Generated reports",
            "logs": "System logs",
            "data": "Market data cache"
        }
        
        for dirname, description in required_dirs.items():
            dir_path = self.base_path / dirname
            if dir_path.exists() and dir_path.is_dir():
                self.log_test("Directory Structure", description, "PASS", f"{dirname}/ exists")
            else:
                self.log_test("Directory Structure", description, "FAIL", f"{dirname}/ missing")
    
    def test_input_validation_logic(self):
        """Test input validation and error handling logic"""
        print("\n=== TESTING INPUT VALIDATION LOGIC ===")
        
        batch_file = self.base_path / "start_alpacabot.bat"
        
        try:
            with open(batch_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Test main menu input validation
            main_menu_checks = [
                ('if "%choice%"=="1"', "Option 1 validation"),
                ('if "%choice%"=="2"', "Option 2 validation"),
                ('if "%choice%"=="3"', "Option 3 validation"),
                ('if "%choice%"=="4"', "Option 4 validation"),
                ('if "%choice%"=="5"', "Option 5 validation"),
                ('if "%choice%"=="6"', "Option 6 validation")
            ]
            
            for check_pattern, description in main_menu_checks:
                if check_pattern in content:
                    self.log_test("Input Validation", description, "PASS", "Validation logic found")
                else:
                    self.log_test("Input Validation", description, "FAIL", "Validation logic missing")
            
            # Test error handling for invalid input
            if "Invalid choice" in content:
                self.log_test("Error Handling", "Invalid Input Message", "PASS", "Error message present")
            else:
                self.log_test("Error Handling", "Invalid Input Message", "FAIL", "No error message")
            
            # Test loop-back logic
            if ":menu" in content and "goto menu" in content:
                self.log_test("Navigation", "Menu Loop Logic", "PASS", "Loop-back navigation present")
            else:
                self.log_test("Navigation", "Menu Loop Logic", "FAIL", "No loop-back navigation")
            
            # Test submenu input validation  
            submenu_validation_patterns = [
                ('train_choice', "Training submenu input"),
                ('config_choice', "Config submenu input"),
                ('analysis_choice', "Analysis submenu input")
            ]
            
            for pattern, description in submenu_validation_patterns:
                if pattern in content:
                    self.log_test("Submenu Validation", description, "PASS", "Submenu input handling found")
                else:
                    self.log_test("Submenu Validation", description, "FAIL", "Submenu input handling missing")
                    
        except Exception as e:
            self.log_test("Input Validation", "Analysis", "FAIL", f"Error: {e}")
    
    def test_interactive_components(self):
        """Test interactive components like backtest input prompts"""
        print("\n=== TESTING INTERACTIVE COMPONENTS ===")
        
        batch_file = self.base_path / "start_alpacabot.bat"
        
        try:
            with open(batch_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Test backtest mode user inputs
            backtest_inputs = [
                ('set /p symbols="Enter symbols', "Symbol input prompt"),
                ('set /p start_date="Enter start date', "Start date input prompt"),
                ('set /p end_date="Enter end date', "End date input prompt")
            ]
            
            for pattern, description in backtest_inputs:
                if pattern in content:
                    self.log_test("Backtest Inputs", description, "PASS", "Input prompt found")
                else:
                    self.log_test("Backtest Inputs", description, "FAIL", "Input prompt missing")
            
            # Test timeout functionality
            if "timeout" in content.lower():
                self.log_test("User Experience", "Timeout Features", "PASS", "Timeout functionality present")
            else:
                self.log_test("User Experience", "Timeout Features", "FAIL", "No timeout functionality")
                
            # Test pause functionality
            if "pause" in content.lower():
                self.log_test("User Experience", "Pause/Continue", "PASS", "Pause functionality present")
            else:
                self.log_test("User Experience", "Pause/Continue", "FAIL", "No pause functionality")
            
            # Test warning messages
            warning_patterns = [
                ("WARNING", "Warning messages"),
                ("CAREFUL", "Safety warnings"),
                ("Paper trading", "Paper trading notices")
            ]
            
            for pattern, description in warning_patterns:
                if pattern.lower() in content.lower():
                    self.log_test("Safety Features", description, "PASS", "Safety message present")
                else:
                    self.log_test("Safety Features", description, "FAIL", "Safety message missing")
                    
        except Exception as e:
            self.log_test("Interactive Components", "Analysis", "FAIL", f"Error: {e}")
    
    def test_python_component_structure(self):
        """Test Python component structure and imports"""
        print("\n=== TESTING PYTHON COMPONENT STRUCTURE ===")
        
        python_components = {
            "live_monitoring_dashboard.py": {
                "expected_classes": ["LiveMonitoringDashboard"],
                "expected_functions": ["main", "get_evolution_status"],
                "critical_imports": ["pandas", "numpy"]
            },
            "config.py": {
                "expected_variables": ["PAPER_TRADING", "INITIAL_CASH", "MAX_POSITIONS"],
                "expected_functions": ["load_profile", "save_profile", "validate_config"],
                "critical_imports": ["dotenv"]
            }
        }
        
        for filename, expectations in python_components.items():
            file_path = self.base_path / filename
            
            if not file_path.exists():
                self.log_test("Python Structure", f"{filename} - File", "FAIL", "File missing")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Test for expected classes
                if "expected_classes" in expectations:
                    for class_name in expectations["expected_classes"]:
                        if f"class {class_name}" in content:
                            self.log_test("Python Structure", f"{filename} - Class {class_name}", "PASS", "Class found")
                        else:
                            self.log_test("Python Structure", f"{filename} - Class {class_name}", "FAIL", "Class missing")
                
                # Test for expected functions
                if "expected_functions" in expectations:
                    for func_name in expectations["expected_functions"]:
                        if f"def {func_name}" in content:
                            self.log_test("Python Structure", f"{filename} - Function {func_name}", "PASS", "Function found")
                        else:
                            self.log_test("Python Structure", f"{filename} - Function {func_name}", "FAIL", "Function missing")
                
                # Test for expected variables
                if "expected_variables" in expectations:
                    for var_name in expectations["expected_variables"]:
                        if var_name in content:
                            self.log_test("Python Structure", f"{filename} - Variable {var_name}", "PASS", "Variable found")
                        else:
                            self.log_test("Python Structure", f"{filename} - Variable {var_name}", "FAIL", "Variable missing")
                
                # Test for critical imports (these would fail at runtime if missing)
                if "critical_imports" in expectations:
                    missing_imports = []
                    for import_name in expectations["critical_imports"]:
                        if f"import {import_name}" not in content and f"from {import_name}" not in content:
                            missing_imports.append(import_name)
                    
                    if missing_imports:
                        self.log_test("Python Structure", f"{filename} - Critical Imports", "WARN", f"Missing: {', '.join(missing_imports)}")
                    else:
                        self.log_test("Python Structure", f"{filename} - Critical Imports", "PASS", "All critical imports present")
                        
            except Exception as e:
                self.log_test("Python Structure", f"{filename} - Analysis", "FAIL", f"Error: {e}")
    
    def analyze_ui_completeness(self):
        """Analyze overall UI completeness and coverage"""
        print("\n=== UI COMPLETENESS ANALYSIS ===")
        
        # Calculate coverage metrics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        warned_tests = sum(1 for result in self.test_results if result['status'] == 'WARN')
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOVERALL UI COVERAGE:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ({pass_rate:.1f}%)")
        print(f"  Failed: {failed_tests}")
        print(f"  Warnings: {warned_tests}")
        
        # Critical UI elements analysis
        critical_elements = {
            "Menu Structure": 0,
            "Input Validation": 0, 
            "File Dependencies": 0,
            "Error Handling": 0,
            "Navigation Logic": 0
        }
        
        # Count critical element coverage
        for result in self.test_results:
            component = result['component']
            status = result['status']
            
            if 'Menu' in component and status == 'PASS':
                critical_elements["Menu Structure"] += 1
            elif 'Input' in component and status == 'PASS':
                critical_elements["Input Validation"] += 1
            elif 'Dependencies' in component and status == 'PASS':
                critical_elements["File Dependencies"] += 1
            elif 'Error' in component and status == 'PASS':
                critical_elements["Error Handling"] += 1
            elif 'Navigation' in component and status == 'PASS':
                critical_elements["Navigation Logic"] += 1
        
        print(f"\nCRITICAL ELEMENT COVERAGE:")
        for element, count in critical_elements.items():
            print(f"  {element}: {count} tests passed")
        
        return {
            "total_tests": total_tests,
            "pass_rate": pass_rate,
            "critical_coverage": critical_elements,
            "ui_coverage": self.ui_coverage
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive UI test report"""
        print("\n" + "="*80)
        print(" "*25 + "ALPACABOT COMPREHENSIVE UI TEST REPORT")
        print("="*80)
        
        # Run all tests
        self.test_main_menu_interface()
        self.test_submenu_structures()
        self.test_file_dependencies()
        self.test_input_validation_logic()
        self.test_interactive_components()
        self.test_python_component_structure()
        
        # Analyze completeness
        completeness = self.analyze_ui_completeness()
        
        # Issues and recommendations
        print(f"\n=== CRITICAL ISSUES FOUND ===")
        
        critical_issues = []
        recommendations = []
        
        for result in self.test_results:
            if result['status'] == 'FAIL' and any(keyword in result['test'].lower() 
                                                for keyword in ['menu', 'input', 'navigation', 'error']):
                critical_issues.append(f"• {result['component']}: {result['test']} - {result['details']}")
        
        if critical_issues:
            for issue in critical_issues[:10]:  # Show top 10 critical issues
                print(issue)
        else:
            print("No critical UI issues found!")
        
        # Generate recommendations
        if completeness['pass_rate'] < 80:
            recommendations.append("• UI completion rate is below 80% - focus on missing components")
        
        if any('missing' in str(coverage).lower() for coverage in self.ui_coverage.values()):
            recommendations.append("• Some menu options are missing - verify batch file completeness")
        
        print(f"\n=== RECOMMENDATIONS ===")
        if recommendations:
            for rec in recommendations:
                print(rec)
        else:
            print("• UI structure appears complete and well-implemented")
            print("• Consider adding more interactive validation")
            print("• Test with actual user interactions for full validation")
        
        # Save detailed report
        report_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "summary": {
                "total_tests": completeness["total_tests"],
                "pass_rate": completeness["pass_rate"],
                "critical_issues": len(critical_issues),
                "recommendations": len(recommendations)
            },
            "detailed_results": self.test_results,
            "ui_coverage": self.ui_coverage,
            "critical_coverage": completeness["critical_coverage"]
        }
        
        # Save to file
        report_file = self.base_path / "reports" / f"comprehensive_ui_test_report_{int(time.time())}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n=== REPORT SAVED ===")
        print(f"Detailed JSON report: {report_file}")
        print(f"Summary: {completeness['pass_rate']:.1f}% UI coverage with {len(critical_issues)} critical issues")
        
        return report_data

def main():
    """Run comprehensive UI testing"""
    print("Starting Comprehensive UI Testing for AlpacaBot...")
    
    tester = AlpacaBotUITester()
    report = tester.generate_comprehensive_report()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE!")
    print(f"Check the reports/ folder for detailed JSON report")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()