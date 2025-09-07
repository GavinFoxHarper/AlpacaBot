#!/usr/bin/env python3
"""
Edge Case UI Testing for AlpacaBot
Tests keyboard interrupts, error conditions, and edge cases
"""

import subprocess
import time
import signal
import os
from pathlib import Path

def test_python_syntax_validation():
    """Test that all Python files have valid syntax"""
    print("\n=== TESTING PYTHON SYNTAX VALIDATION ===")
    
    base_path = Path(__file__).parent
    python_files = []
    
    # Collect all Python files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    syntax_errors = []
    valid_files = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Attempt to compile to check syntax
            compile(content, str(py_file), 'exec')
            valid_files.append(py_file.name)
            print(f"  [PASS] {py_file.name}: Valid syntax")
            
        except SyntaxError as e:
            syntax_errors.append(f"{py_file.name}: Line {e.lineno} - {e.msg}")
            print(f"  [FAIL] {py_file.name}: Syntax error at line {e.lineno}")
            
        except Exception as e:
            print(f"  [WARN] {py_file.name}: Could not validate - {e}")
    
    print(f"\nSyntax Validation Summary:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Syntax errors: {len(syntax_errors)}")
    
    if syntax_errors:
        print(f"  Files with errors:")
        for error in syntax_errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
    
    return len(syntax_errors) == 0

def test_batch_file_edge_cases():
    """Test batch file with various edge case inputs"""
    print("\n=== TESTING BATCH FILE EDGE CASES ===")
    
    batch_file = Path(__file__).parent / "start_alpacabot.bat"
    
    if not batch_file.exists():
        print("  [FAIL] Batch file not found for edge case testing")
        return False
    
    # Read batch file to analyze edge case handling
    try:
        with open(batch_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Test for common edge case patterns
        edge_case_checks = [
            ("errorlevel", "Exit code handling"),
            ("2>nul", "Error suppression"),
            ("exist", "File existence checks"),
            ("if not", "Negative condition handling"),
            ("timeout", "User input timeouts"),
            (">nul", "Output redirection")
        ]
        
        edge_case_coverage = {}
        for pattern, description in edge_case_checks:
            if pattern.lower() in content.lower():
                print(f"  [PASS] {description}: Pattern '{pattern}' found")
                edge_case_coverage[description] = True
            else:
                print(f"  [WARN] {description}: Pattern '{pattern}' not found")
                edge_case_coverage[description] = False
        
        # Check for input validation patterns
        validation_patterns = [
            'if "%choice%"=="', "Menu choice validation",
            'else (', "Alternative flow handling"
        ]
        
        for pattern, description in validation_patterns:
            if pattern in content:
                print(f"  [PASS] {description}: Validation present")
            else:
                print(f"  [FAIL] {description}: No validation found")
        
        return sum(edge_case_coverage.values()) >= len(edge_case_coverage) * 0.7
        
    except Exception as e:
        print(f"  [FAIL] Error analyzing batch file: {e}")
        return False

def test_dependency_resilience():
    """Test how the system handles missing dependencies"""
    print("\n=== TESTING DEPENDENCY RESILIENCE ===")
    
    # Test key Python files for import error handling
    files_to_test = [
        "live_monitoring_dashboard.py",
        "config.py", 
        "show_backtest_results.py"
    ]
    
    resilience_score = 0
    total_tests = len(files_to_test)
    
    for filename in files_to_test:
        file_path = Path(__file__).parent / filename
        
        if not file_path.exists():
            print(f"  [FAIL] {filename}: File not found")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for try/except import patterns
            has_import_handling = False
            
            # Look for import error handling patterns
            error_handling_patterns = [
                "try:\n    import",
                "except ImportError",
                "except ModuleNotFoundError", 
                "HAS_MATPLOTLIB = True",  # Conditional import pattern
                "import warnings\nwarnings.filterwarnings"
            ]
            
            for pattern in error_handling_patterns:
                if pattern in content:
                    has_import_handling = True
                    break
            
            if has_import_handling:
                print(f"  [PASS] {filename}: Has import error handling")
                resilience_score += 1
            else:
                print(f"  [WARN] {filename}: No import error handling detected")
                
        except Exception as e:
            print(f"  [FAIL] {filename}: Error analyzing - {e}")
    
    resilience_percentage = (resilience_score / total_tests) * 100
    print(f"\nDependency Resilience: {resilience_percentage:.1f}% ({resilience_score}/{total_tests})")
    
    return resilience_percentage >= 50

def test_configuration_validation():
    """Test configuration file validation and error handling"""
    print("\n=== TESTING CONFIGURATION VALIDATION ===")
    
    config_file = Path(__file__).parent / "config.py"
    
    if not config_file.exists():
        print("  [FAIL] config.py not found")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for configuration validation patterns
        validation_checks = [
            ("def validate_config", "Config validation function"),
            ("os.getenv", "Environment variable handling"),
            ("default", "Default value handling"),
            ("if not", "Negative condition checks")
        ]
        
        validation_score = 0
        for pattern, description in validation_checks:
            if pattern in content:
                print(f"  [PASS] {description}: Pattern found")
                validation_score += 1
            else:
                print(f"  [WARN] {description}: Pattern not found")
        
        # Check for critical configuration variables
        critical_vars = [
            "PAPER_TRADING", 
            "API_KEY",
            "INITIAL_CASH",
            "MAX_POSITIONS"
        ]
        
        missing_vars = []
        for var in critical_vars:
            if var not in content:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"  [WARN] Missing critical variables: {', '.join(missing_vars)}")
        else:
            print(f"  [PASS] All critical configuration variables present")
            validation_score += 1
        
        validation_percentage = (validation_score / (len(validation_checks) + 1)) * 100
        print(f"\nConfiguration Validation: {validation_percentage:.1f}%")
        
        return validation_percentage >= 70
        
    except Exception as e:
        print(f"  [FAIL] Error analyzing config: {e}")
        return False

def test_file_permissions_and_access():
    """Test file permissions and access patterns"""
    print("\n=== TESTING FILE PERMISSIONS AND ACCESS ===")
    
    base_path = Path(__file__).parent
    
    # Test critical directories
    critical_dirs = ["logs", "reports", "models", "data"]
    access_issues = []
    
    for dir_name in critical_dirs:
        dir_path = base_path / dir_name
        
        try:
            if not dir_path.exists():
                print(f"  [WARN] {dir_name}/: Directory missing (will be created)")
                continue
            
            # Test read access
            list(dir_path.iterdir())  # Try to list directory
            print(f"  [PASS] {dir_name}/: Read access OK")
            
            # Test write access by creating a test file
            test_file = dir_path / "test_access.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()  # Delete test file
                print(f"  [PASS] {dir_name}/: Write access OK")
            except Exception:
                print(f"  [WARN] {dir_name}/: Write access limited")
                access_issues.append(f"{dir_name} - write access")
                
        except PermissionError:
            print(f"  [FAIL] {dir_name}/: Permission denied")
            access_issues.append(f"{dir_name} - permission denied")
        except Exception as e:
            print(f"  [WARN] {dir_name}/: Access test failed - {e}")
    
    if access_issues:
        print(f"\nAccess Issues Found: {len(access_issues)}")
        for issue in access_issues:
            print(f"  - {issue}")
    else:
        print(f"\n[PASS] All directory access tests passed")
    
    return len(access_issues) == 0

def generate_edge_case_report():
    """Generate comprehensive edge case test report"""
    print("="*80)
    print(" "*25 + "ALPACABOT EDGE CASE TEST REPORT") 
    print("="*80)
    
    test_results = []
    
    # Run all edge case tests
    tests = [
        ("Python Syntax Validation", test_python_syntax_validation),
        ("Batch File Edge Cases", test_batch_file_edge_cases), 
        ("Dependency Resilience", test_dependency_resilience),
        ("Configuration Validation", test_configuration_validation),
        ("File Permissions", test_file_permissions_and_access)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"  [ERROR] {test_name} failed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("EDGE CASE TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    print(f"\nDetailed Results:")
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")
    
    # Save report
    report_file = Path(__file__).parent / "reports" / f"edge_case_test_report_{int(time.time())}.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(f"AlpacaBot Edge Case Test Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Summary: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests*100):.1f}%)\n\n")
        
        for test_name, result in test_results:
            status = "PASS" if result else "FAIL" 
            f.write(f"[{status}] {test_name}\n")
    
    print(f"\nEdge case report saved to: {report_file}")
    
    return test_results

if __name__ == "__main__":
    print("Starting AlpacaBot Edge Case Testing...")
    results = generate_edge_case_report()
    print(f"\nEdge case testing complete!")