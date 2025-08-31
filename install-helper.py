#!/usr/bin/env python3
"""
AlpacaBot Installation Helper
Handles Python 3.13 compatibility issues and installs requirements
"""

import subprocess
import sys
import platform
from pathlib import Path

def check_python_version():
    """Check Python version and warn about compatibility"""
    version_info = sys.version_info
    print(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    if version_info.major == 3 and version_info.minor == 13:
        print("⚠️  Warning: Python 3.13 detected")
        print("Some packages may not have stable releases for Python 3.13 yet.")
        print("Using compatible alternatives where possible.\n")
        return "3.13"
    elif version_info.major == 3 and version_info.minor >= 8:
        print("✅ Python version is compatible\n")
        return "compatible"
    else:
        print("❌ Python 3.8+ is required")
        sys.exit(1)

def install_requirements(python_version):
    """Install requirements based on Python version"""
    
    # Core packages that work with Python 3.13
    core_packages = [
        "alpaca-trade-api",
        "alpaca-py",
        "python-dotenv",
        "pandas",
        "numpy",
        "yfinance",
        "ta",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "requests",
        "beautifulsoup4",
        "lxml",
        "aiohttp",
        "vaderSentiment",
        "textblob",
        "tqdm",
        "colorama",
        "schedule",
        "pytz",
        "sqlalchemy",
        "loguru",
        "xgboost",
        "lightgbm",
        "statsmodels"
    ]
    
    # Packages that might have issues with Python 3.13
    optional_packages = {
        "tensorflow": "2.20.0rc0",  # Release candidate for Python 3.13
        "torch": None,  # May not be available for 3.13 yet
        "ta-lib": None,  # Requires special installation on Windows
    }
    
    print("Installing core packages...")
    failed_packages = []
    
    for package in core_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} failed to install")
            failed_packages.append(package)
    
    # Try optional packages
    print("\nAttempting optional packages...")
    
    if python_version == "3.13":
        # Try TensorFlow RC for Python 3.13
        print("Installing TensorFlow 2.20.0rc0 (release candidate)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.20.0rc0"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            print("  ✅ TensorFlow RC installed")
        except:
            print("  ℹ️  TensorFlow RC not installed (not critical)")
            print("     The system will use XGBoost and LightGBM instead")
    
    # Platform-specific instructions for ta-lib
    if platform.system() == "Windows":
        print("\n📌 Note for Windows users:")
        print("   TA-Lib requires a special installation on Windows.")
        print("   You can either:")
        print("   1. Download the wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("   2. Or skip it - the system will work without it")
    
    return failed_packages

def create_minimal_requirements():
    """Create a minimal requirements file that definitely works"""
    minimal_reqs = """# Minimal requirements for AlpacaBot - Python 3.13 compatible
alpaca-trade-api
python-dotenv
pandas
numpy
yfinance
requests
scikit-learn
matplotlib

# If these fail, the system can still run with reduced functionality
# tensorflow==2.20.0rc0  # Uncomment if you want to try the RC
# xgboost
# lightgbm
"""
    
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_reqs)
    
    print("\n✅ Created requirements_minimal.txt with essential packages only")

def verify_installation():
    """Verify critical packages are installed"""
    critical_packages = {
        "alpaca_trade_api": "Alpaca Trading API",
        "dotenv": "Environment variables",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "sklearn": "Machine learning"
    }
    
    print("\nVerifying critical packages...")
    all_good = True
    
    for module_name, description in critical_packages.items():
        try:
            __import__(module_name)
            print(f"  ✅ {description} ({module_name})")
        except ImportError:
            print(f"  ❌ {description} ({module_name}) - REQUIRED")
            all_good = False
    
    return all_good

def main():
    print("="*60)
    print("AlpacaBot Installation Helper")
    print("="*60)
    print()
    
    # Check Python version
    py_version = check_python_version()
    
    # Offer options
    print("Select installation option:")
    print("1. Install all compatible packages (recommended)")
    print("2. Install minimal requirements only")
    print("3. Create requirements files and exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nInstalling all compatible packages...")
        failed = install_requirements(py_version)
        
        if failed:
            print(f"\n⚠️  {len(failed)} packages failed to install:")
            for pkg in failed:
                print(f"   - {pkg}")
            print("\nThese are mostly optional. The system should still work.")
        
        if verify_installation():
            print("\n✅ All critical packages installed successfully!")
            print("The AlpacaBot system is ready to run.")
        else:
            print("\n⚠️  Some critical packages are missing.")
            print("Try option 2 for minimal installation.")
            
    elif choice == "2":
        create_minimal_requirements()
        print("\nInstalling minimal requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"])
        
        if verify_installation():
            print("\n✅ Minimal installation complete!")
            print("The system will run with basic functionality.")
        
    elif choice == "3":
        create_minimal_requirements()
        print("\nCreated requirements_minimal.txt")
        print("Run: pip install -r requirements_minimal.txt")
        
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Ensure your .env file has Alpaca API credentials")
    print("2. Run: python laef_unified_system.py")
    print("="*60)

if __name__ == "__main__":
    main()