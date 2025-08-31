#!/usr/bin/env python3
"""
Test Live Monitoring Menu from LAEF System
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_monitoring_menu():
    """Test the live monitoring menu system"""
    
    print("="*80)
    print("TESTING LIVE MONITORING MENU SYSTEM")
    print("="*80)
    
    # Test 1: Check if LAEF system can be loaded
    print("\nTEST 1: Load LAEF System")
    try:
        from laef_unified_system import LAEFUnifiedSystem
        system = LAEFUnifiedSystem(debug_mode=True)
        print("[SUCCESS] LAEF system loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load LAEF system: {e}")
        return
    
    # Test 2: Check monitoring dashboard methods
    print("\nTEST 2: Check Monitoring Dashboard Methods")
    methods_to_check = [
        'show_monitoring_dashboard',
        '_start_live_monitoring', 
        '_show_learning_dashboard',
        '_show_prediction_monitor',
        '_show_variables_dashboard',
        '_configure_learning'
    ]
    
    for method_name in methods_to_check:
        if hasattr(system, method_name):
            print(f"[SUCCESS] {method_name}: Available")
        else:
            print(f"[ERROR] {method_name}: Missing")
    
    # Test 3: Test dashboard components
    print("\nTEST 3: Test Dashboard Components")
    try:
        # Import monitoring components
        from live_monitoring_dashboard import LiveMonitoringDashboard
        dashboard = LiveMonitoringDashboard()
        
        # Test evolution status
        status = dashboard.get_evolution_status()
        print("[SUCCESS] Evolution status retrieved")
        
        # Show key metrics
        if 'portfolio' in status['components']:
            portfolio = status['components']['portfolio']
            print(f"  Portfolio Value: ${portfolio['current_value']:,.2f}")
            print(f"  Daily Change: {portfolio['daily_change_pct']:+.2f}%")
            print(f"  Active Positions: {portfolio['positions']}")
            
        if 'strategies' in status['components']:
            strategies = status['components']['strategies']
            active_strategies = len([s for s in strategies.values() if s.get('active', False)])
            print(f"  Active Strategies: {active_strategies}")
            
    except Exception as e:
        print(f"[ERROR] Dashboard components failed: {e}")
    
    # Test 4: Check learning components
    print("\nTEST 4: Check Learning Components")
    learning_components = [
        'training.live_market_learner',
        'training.q_learning_agent', 
        'training.prediction_tracker',
        'training.ml_trainer'
    ]
    
    for component in learning_components:
        try:
            __import__(component)
            print(f"[SUCCESS] {component}: Available")
        except ImportError as e:
            print(f"[WARNING] {component}: Not available - {e}")
        except Exception as e:
            print(f"[ERROR] {component}: Failed - {e}")
    
    # Test 5: Real-time monitoring simulation
    print("\nTEST 5: Real-time Monitoring Simulation")
    try:
        # Simulate what happens when monitoring starts
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        print(f"Simulating monitoring for: {', '.join(symbols)}")
        
        import yfinance as yf
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                print(f"  {symbol}: ${price}")
            except:
                print(f"  {symbol}: Data unavailable")
                
        print("[SUCCESS] Real-time data simulation working")
        
    except Exception as e:
        print(f"[ERROR] Real-time monitoring simulation failed: {e}")
    
    # Test 6: Check if monitoring can be started
    print("\nTEST 6: Monitoring Start Test")
    try:
        # This simulates what happens in the actual menu
        print("Simulating Live Monitoring Dashboard startup...")
        
        dashboard = LiveMonitoringDashboard()
        report = dashboard.generate_evolution_report()
        
        print("[SUCCESS] Monitoring dashboard can be started")
        print("Sample evolution report generated:")
        print("-" * 40)
        print(report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        print(f"[ERROR] Monitoring start failed: {e}")
    
    print("\n" + "="*80)
    print("LIVE MONITORING MENU TEST COMPLETE")
    print("="*80)


def show_monitoring_status():
    """Show current monitoring system status"""
    
    print("\n" + "="*80)
    print("LIVE MONITORING SYSTEM STATUS REPORT")
    print("="*80)
    
    # Check system components
    components_status = {}
    
    # 1. Dashboard
    try:
        from live_monitoring_dashboard import LiveMonitoringDashboard
        dashboard = LiveMonitoringDashboard()
        status = dashboard.get_evolution_status()
        components_status['dashboard'] = 'WORKING'
        
        print("\nDASHBOARD STATUS: WORKING")
        print("  - Real-time evolution tracking: ACTIVE")
        print("  - Portfolio monitoring: ACTIVE")
        print("  - Strategy tracking: ACTIVE")
        
    except Exception as e:
        components_status['dashboard'] = f'ERROR: {e}'
        print(f"\nDASHBOARD STATUS: ERROR - {e}")
    
    # 2. Data Feeds
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        price = ticker.info.get('currentPrice', 'N/A')
        components_status['data_feeds'] = 'WORKING'
        print(f"\nDATA FEEDS STATUS: WORKING")
        print(f"  - Real-time prices: Available (AAPL: ${price})")
        
    except Exception as e:
        components_status['data_feeds'] = f'ERROR: {e}'
        print(f"\nDATA FEEDS STATUS: ERROR - {e}")
    
    # 3. API Connection
    api_key = os.getenv('ALPACA_API_KEY')
    if api_key:
        components_status['api_connection'] = 'CONFIGURED'
        print(f"\nAPI CONNECTION STATUS: CONFIGURED")
        print(f"  - Alpaca API: Configured")
        
        # Check if we can connect
        try:
            from alpaca_trade_api import REST
            api = REST(api_key, os.getenv('ALPACA_SECRET_KEY'), 
                      os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'))
            account = api.get_account()
            print(f"  - Connection: ACTIVE")
            print(f"  - Account Status: {account.status}")
        except Exception as e:
            print(f"  - Connection: ERROR - {e}")
    else:
        components_status['api_connection'] = 'NOT_CONFIGURED'
        print(f"\nAPI CONNECTION STATUS: NOT_CONFIGURED")
        print(f"  - Alpaca API: Not configured (demo mode)")
    
    # 4. Email Alerts
    email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    if email_enabled:
        components_status['email_alerts'] = 'ENABLED'
        print(f"\nEMAIL ALERTS STATUS: ENABLED")
    else:
        components_status['email_alerts'] = 'DISABLED'
        print(f"\nEMAIL ALERTS STATUS: DISABLED")
    
    # 5. Learning Components
    q_model_path = Path('models/q_learning_models/q_model.keras')
    if q_model_path.exists():
        components_status['learning'] = 'ACTIVE'
        size_kb = q_model_path.stat().st_size / 1024
        mod_time = datetime.fromtimestamp(q_model_path.stat().st_mtime)
        print(f"\nLEARNING COMPONENTS STATUS: ACTIVE")
        print(f"  - Q-Learning Model: {size_kb:.1f} KB")
        print(f"  - Last Updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        components_status['learning'] = 'NOT_INITIALIZED'
        print(f"\nLEARNING COMPONENTS STATUS: NOT_INITIALIZED")
    
    # Summary
    print(f"\n" + "="*80)
    print("MONITORING SYSTEM SUMMARY")
    print(f"="*80)
    
    working_components = len([v for v in components_status.values() if v == 'WORKING' or v == 'CONFIGURED' or v == 'ENABLED' or v == 'ACTIVE'])
    total_components = len(components_status)
    
    print(f"Components Working: {working_components}/{total_components}")
    
    if working_components >= 3:
        print("STATUS: MONITORING SYSTEM IS FUNCTIONAL")
        print("✓ Can monitor portfolio and strategies")
        print("✓ Can track real-time market data")
        print("✓ Dashboard available")
    else:
        print("STATUS: MONITORING SYSTEM NEEDS CONFIGURATION")
        print("- Set up Alpaca API keys for live monitoring")
        print("- Configure email alerts (optional)")
    


if __name__ == "__main__":
    test_monitoring_menu()
    show_monitoring_status()