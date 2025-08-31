#!/usr/bin/env python3
"""
Robust wrapper for running the Live Monitoring Dashboard
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    try:
        from live_monitoring_dashboard import LiveMonitoringDashboard
        
        dashboard = LiveMonitoringDashboard()
        
        print("\nStarting Live Monitoring Dashboard...")
        print("Press Ctrl+C to stop\n")
        
        # For testing, just generate one report
        report = dashboard.generate_evolution_report()
        print(report)
        
        print("\n[Dashboard is working correctly]")
        print("To run continuously, use: python live_monitoring_dashboard.py")
        
    except ImportError as e:
        print(f"Error importing dashboard: {e}")
        print("Please ensure all dependencies are installed")
        return 1
        
    except Exception as e:
        print(f"Error running dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())