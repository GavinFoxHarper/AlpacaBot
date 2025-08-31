#!/usr/bin/env python3
"""
Comprehensive UI Test Framework for AlpacaBot Trading Application
Systematically tests all frontend user interface elements
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

class UITestFramework:
    """Comprehensive UI testing framework for AlpacaBot"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'ui_components_tested': {},
            'menu_structures': {},
            'issues_found': [],
            'coverage_summary': {},
            'recommendations': []
        }
        self.total_components_tested = 0
        self.total_issues_found = 0
    
    def test_component(self, component_name: str, file_path: str, test_type: str = 'interactive'):
        """Test a UI component and record results"""
        print(f"\n{'='*70}")
        print(f"TESTING: {component_name}")
        print(f"File: {file_path}")
        print(f"{'='*70}")
        
        component_result = {
            'file_path': file_path,
            'test_type': test_type,
            'status': 'unknown',
            'menu_structure': [],
            'issues': [],
            'functionality_tested': [],
            'test_timestamp': datetime.now().isoformat()
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            component_result['status'] = 'file_not_found'
            component_result['issues'].append(f"File not found: {file_path}")
            self.total_issues_found += 1
            self.test_results['ui_components_tested'][component_name] = component_result
            return component_result
        
        try:
            # Analyze file for UI structure
            menu_structure = self._analyze_file_structure(file_path)
            component_result['menu_structure'] = menu_structure
            
            # Attempt to run the component (non-interactive test)
            execution_result = self._test_component_execution(file_path, component_name)
            component_result.update(execution_result)
            
            # Record success
            component_result['status'] = 'tested'
            self.total_components_tested += 1
            
        except Exception as e:
            component_result['status'] = 'error'
            component_result['issues'].append(f"Test error: {str(e)}")
            self.total_issues_found += 1
        
        self.test_results['ui_components_tested'][component_name] = component_result
        return component_result
    
    def _analyze_file_structure(self, file_path: str) -> List[Dict]:
        """Analyze file to extract menu structure and UI elements"""
        menu_items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Look for menu patterns
                menu_patterns = [
                    r'print\("(\d+)\.\s*([^"]+)"\)',  # Numbered menu items
                    r'print\("([^"]*menu[^"]*)"\)',  # Menu-related prints
                    r'input\("([^"]+)"\)',  # Input prompts
                    r'choice\s*==\s*["\']([^"\']+)["\']',  # Menu choices
                    r'if\s+choice\s*==\s*["\'](\d+)["\']'  # Numeric choices
                ]
                
                import re
                for pattern in menu_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            if len(match) == 2:  # Number and description
                                menu_items.append({
                                    'option': match[0],
                                    'description': match[1],
                                    'type': 'numbered_option'
                                })
                            else:
                                menu_items.append({
                                    'text': match[0],
                                    'type': 'menu_text'
                                })
                        else:
                            menu_items.append({
                                'text': match,
                                'type': 'interactive_element'
                            })
                
                # Look for function definitions that might be menu handlers
                function_patterns = [
                    r'def\s+([^(]+)\([^)]*\):',  # Function definitions
                    r'class\s+([^(:\s]+)'  # Class definitions
                ]
                
                for pattern in function_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if any(keyword in match.lower() for keyword in 
                              ['menu', 'interactive', 'main', 'show', 'display', 'analyze']):
                            menu_items.append({
                                'function': match,
                                'type': 'handler_function'
                            })
                
        except Exception as e:
            menu_items.append({
                'error': f"Failed to analyze structure: {e}",
                'type': 'analysis_error'
            })
        
        return menu_items
    
    def _test_component_execution(self, file_path: str, component_name: str) -> Dict:
        """Test component execution with timeout"""
        result = {
            'execution_status': 'unknown',
            'output_captured': '',
            'errors_found': [],
            'functionality_tested': []
        }
        
        try:
            # Run with short timeout to capture initial output
            cmd = [sys.executable, file_path]
            
            # Special handling for specific components
            if 'profile_manager' in file_path:
                cmd.append('list')  # Test list command
            elif 'laef_control_interface' in file_path:
                cmd.append('help')  # Test help command
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                input='\n' * 10  # Provide some inputs to handle interactive prompts
            )
            
            result['execution_status'] = 'completed'
            result['output_captured'] = process.stdout
            
            if process.stderr:
                result['errors_found'].append(process.stderr)
            
            # Analyze output for functionality
            output = process.stdout
            if 'menu' in output.lower():
                result['functionality_tested'].append('menu_display')
            if any(keyword in output.lower() for keyword in ['error', 'failed', 'exception']):
                result['errors_found'].append('Runtime errors detected in output')
            if any(keyword in output.lower() for keyword in ['1.', '2.', '3.']):
                result['functionality_tested'].append('numbered_menu_options')
            if 'select' in output.lower() or 'choose' in output.lower():
                result['functionality_tested'].append('user_selection_prompt')
                
        except subprocess.TimeoutExpired:
            result['execution_status'] = 'timeout'
            result['functionality_tested'].append('interactive_interface_detected')
        except subprocess.CalledProcessError as e:
            result['execution_status'] = 'error'
            result['errors_found'].append(f"Process error: {e}")
        except Exception as e:
            result['execution_status'] = 'exception'
            result['errors_found'].append(f"Test exception: {e}")
        
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all UI components"""
        print("\n" + "="*80)
        print("COMPREHENSIVE UI TEST FRAMEWORK - AlpacaBot Trading Application")
        print("="*80)
        print(f"Test started at: {datetime.now()}")
        
        # Define all UI components to test
        ui_components = {
            "Live Monitoring Dashboard": "live_monitoring_dashboard.py",
            "Interactive Result Explorer": "utils/interactive_explorer.py",
            "Profile Manager CLI": "utils/profile_manager.py",
            "LAEF Control Interface": "utils/laef_control_interface.py",
            "Log Management Interface": "utils/logging_utils.py",
            "LAEF Unified System Main Menu": "laef_unified_system.py",
            "Run Interactive Backtest": "run_interactive_backtest.py",
            "Comprehensive Backtest Runner": "comprehensive_backtest_runner.py",
            "Show Backtest Results": "show_backtest_results.py"
        }
        
        # Test each component
        for component_name, file_path in ui_components.items():
            full_path = os.path.join(os.getcwd(), file_path)
            self.test_component(component_name, full_path)
            time.sleep(1)  # Brief pause between tests
        
        # Generate summary
        self._generate_test_summary()
        
        # Save results
        self._save_test_results()
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE UI TEST COMPLETED")
        print(f"{'='*80}")
        print(f"Total Components Tested: {self.total_components_tested}")
        print(f"Total Issues Found: {self.total_issues_found}")
        print(f"Test Results Saved: ui_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return self.test_results
    
    def _generate_test_summary(self):
        """Generate comprehensive test summary"""
        summary = {
            'total_components': len(self.test_results['ui_components_tested']),
            'successful_tests': 0,
            'failed_tests': 0,
            'components_with_menus': 0,
            'interactive_components': 0,
            'error_prone_components': [],
            'well_functioning_components': []
        }
        
        for component_name, result in self.test_results['ui_components_tested'].items():
            if result['status'] == 'tested':
                summary['successful_tests'] += 1
                if len(result['issues']) == 0:
                    summary['well_functioning_components'].append(component_name)
            else:
                summary['failed_tests'] += 1
                summary['error_prone_components'].append(component_name)
            
            if result['menu_structure']:
                summary['components_with_menus'] += 1
            
            if 'interactive_interface_detected' in result.get('functionality_tested', []):
                summary['interactive_components'] += 1
        
        self.test_results['coverage_summary'] = summary
        
        # Generate recommendations
        recommendations = []
        
        if summary['failed_tests'] > 0:
            recommendations.append("Fix components that failed to run properly")
        
        if summary['error_prone_components']:
            recommendations.append(f"Address issues in: {', '.join(summary['error_prone_components'])}")
        
        if summary['interactive_components'] < summary['total_components'] * 0.5:
            recommendations.append("Consider adding more interactive elements to improve user experience")
        
        recommendations.extend([
            "Implement consistent error handling across all UI components",
            "Add proper input validation and user-friendly error messages",
            "Consider creating a unified menu framework for consistency",
            "Add help text and usage examples for complex interfaces",
            "Implement graceful handling of interrupted operations"
        ])
        
        self.test_results['recommendations'] = recommendations
    
    def _save_test_results(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ui_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        # Also create a readable report
        report_filename = f"ui_test_report_{timestamp}.md"
        self._generate_markdown_report(report_filename)
    
    def _generate_markdown_report(self, filename: str):
        """Generate a markdown report"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# AlpacaBot UI Test Report\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = self.test_results['coverage_summary']
            f.write(f"- **Total Components Tested**: {summary['total_components']}\n")
            f.write(f"- **Successful Tests**: {summary['successful_tests']}\n")
            f.write(f"- **Failed Tests**: {summary['failed_tests']}\n")
            f.write(f"- **Components with Menus**: {summary['components_with_menus']}\n")
            f.write(f"- **Interactive Components**: {summary['interactive_components']}\n\n")
            
            # Detailed Results
            f.write("## Detailed Test Results\n\n")
            for component_name, result in self.test_results['ui_components_tested'].items():
                f.write(f"### {component_name}\n\n")
                f.write(f"**File**: `{result['file_path']}`\n")
                f.write(f"**Status**: {result['status']}\n")
                f.write(f"**Execution Status**: {result.get('execution_status', 'N/A')}\n\n")
                
                if result['menu_structure']:
                    f.write("**Menu Structure Found**:\n")
                    for item in result['menu_structure'][:10]:  # Limit to first 10 items
                        if item.get('option') and item.get('description'):
                            f.write(f"- {item['option']}. {item['description']}\n")
                        elif item.get('function'):
                            f.write(f"- Function: {item['function']}\n")
                        elif item.get('text'):
                            f.write(f"- {item['text']}\n")
                    f.write("\n")
                
                if result.get('functionality_tested'):
                    f.write("**Functionality Tested**:\n")
                    for func in result['functionality_tested']:
                        f.write(f"- {func}\n")
                    f.write("\n")
                
                if result.get('issues'):
                    f.write("**Issues Found**:\n")
                    for issue in result['issues']:
                        f.write(f"- {issue}\n")
                    f.write("\n")
                
                f.write("---\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in self.test_results['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")

def main():
    """Run comprehensive UI testing"""
    framework = UITestFramework()
    results = framework.run_comprehensive_test()
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    well_functioning = results['coverage_summary'].get('well_functioning_components', [])
    if well_functioning:
        print("\nWell-Functioning Components:")
        for component in well_functioning:
            print(f"  ✓ {component}")
    
    error_prone = results['coverage_summary'].get('error_prone_components', [])
    if error_prone:
        print("\nComponents Requiring Attention:")
        for component in error_prone:
            print(f"  ⚠ {component}")
    
    print(f"\nOverall Test Coverage: {len(results['ui_components_tested'])} components tested")

if __name__ == "__main__":
    main()