"""
LAEF Unified Trading System - Main Interface
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from news_integration_system import NewsIntegrationEngine
from multi_strategy_orchestrator import MultiStrategyOrchestrator, MarketRegime, MarketConditions
from risk_adjusted_reward_system import RiskAdjustedRewardSystem
from unified_q_value_handler import UnifiedQValueHandler

class LAEFUnifiedSystem:
    """Main LAEF trading system interface"""
    
    def __init__(self, config_path: str = None, debug_mode: bool = False):
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.initialize_components()
        
        self.logger.info("LAEF System Initialized - Corrected Architecture")
        
    def initialize_components(self):
        """Initialize system components"""
        try:
            # Initialize trading components
            self.q_handler = UnifiedQValueHandler(self.config)
            self.reward_system = RiskAdjustedRewardSystem(self.config)
            
            # Initialize news integration
            self.news_engine = NewsIntegrationEngine(self.config)
            
            # Initialize strategy orchestration
            self.strategy_orchestrator = MultiStrategyOrchestrator(
                self.config,
                self._load_strategies(),
                self.news_engine
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def run(self):
        """Main menu loop"""
        while True:
            self.show_main_menu()
            try:
                choice = input("\nSelect an option (1-7): ").strip()
                
                if choice == "1":
                    self.start_live_trading()
                elif choice == "2":
                    self.start_paper_trading()
                elif choice == "3":
                    self.run_backtesting()
                elif choice == "4":
                    self.show_monitoring_dashboard()
                elif choice == "5":
                    self.run_optimization()
                elif choice == "6":
                    self.manage_settings()
                elif choice == "7":
                    print("\nExiting LAEF system...")
                    break
                else:
                    print("\nInvalid option. Please try again.")
                    
            except Exception as e:
                self.logger.error(f"Menu error: {e}")
                input("Press Enter to continue...")
                
    def show_main_menu(self):
        """Display main menu"""
        print("\n" + "=" * 70)
        print("LAEF UNIFIED TRADING SYSTEM")
        print("Learning-Augmented Equity Framework")
        print("=" * 70)
        print("\n1. Live Trading (Real Money - Alpaca Live)")
        print("2. Paper Trading (Virtual Money - Alpaca Paper)")
        print("3. Backtesting (Historical Analysis)")
        print("4. Live Monitoring & Learning Dashboard")
        print("5. Optimization & Analysis")
        print("6. Settings")
        print("7. Exit")
        
    def start_live_trading(self):
        """Start live trading mode"""
        print("\n" + "=" * 70)
        print("LIVE TRADING - REAL MONEY")
        print("Connecting to Alpaca Live Trading API")
        print("=" * 70)
        
        try:
            from trading.unified_trading_engine import LAEFLiveTrader
            trader = LAEFLiveTrader(paper_trading=False)
            
            # Get user confirmation
            confirm = input("\nWARNING: This will trade with REAL money. Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("Live trading cancelled.")
                return
                
            # Start trading system
            print("\nInitializing live trading system...")
            trader.start()
            
        except Exception as e:
            self.logger.error(f"Live trading error: {e}")
            print(f"[ERROR] Live trading failed: {e}")
            input("\nPress Enter to return to main menu...")
            
    def start_paper_trading(self):
        """Start paper trading mode"""
        print("\n" + "=" * 70)
        print("PAPER TRADING - VIRTUAL MONEY")
        print("Connecting to Alpaca Paper Trading API")
        print("=" * 70)
        
        print("\n1. Standard Paper Trading")
        print("2. Aggressive Paper Trading (More Opportunities)")
        print("3. Conservative Paper Trading (Lower Risk)")
        print("4. Back to Main Menu")
        
        try:
            from trading.unified_trading_engine import LAEFLiveTrader
            
            choice = input("\nSelect trading mode (1-4): ").strip()
            
            if choice == "1":
                trader = LAEFLiveTrader(paper_trading=True)
            elif choice == "2":
                config = self.config.copy()
                config.update({'risk_level': 'aggressive'})
                trader = LAEFLiveTrader(paper_trading=True, config=config)
            elif choice == "3":
                config = self.config.copy()
                config.update({'risk_level': 'conservative'})
                trader = LAEFLiveTrader(paper_trading=True, config=config)
            elif choice == "4":
                return
            else:
                print("Invalid option")
                return
                
            print("\nStarting paper trading system...")
            trader.start()
            
        except Exception as e:
            self.logger.error(f"Paper trading error: {e}")
            print(f"[ERROR] Paper trading failed: {e}")
            input("\nPress Enter to return to main menu...")
            
    def run_backtesting(self):
        """Run backtesting analysis"""
        print("\n" + "=" * 70)
        print("LAEF BACKTESTING SYSTEM")
        print("Historical Trading Analysis - No Real Money")
        print("=" * 70)
        
        print("\n1. Quick Backtest (LAEF Default Settings)")
        print("2. Advanced Backtest (Full Configuration)")
        print("3. Strategy Comparison Backtest")
        print("4. View Previous Results & Analysis")
        print("5. Back to Main Menu")
        
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                self._run_quick_backtest()
            elif choice == "2":
                self._run_advanced_backtest()
            elif choice == "3":
                self._run_strategy_comparison()
            elif choice == "4":
                self._view_backtest_results()
            elif choice == "5":
                return
            else:
                print("Invalid option")
                return
                
        except Exception as e:
            self.logger.error(f"Backtesting error: {e}")
            print(f"\n[ERROR] Backtesting failed: {e}")
            input("\nPress Enter to continue...")
            
    def _run_quick_backtest(self):
        """Run quick backtest with default settings"""
        print("\n" + "=" * 50)
        print("QUICK BACKTEST - LAEF DEFAULT AI/ML SYSTEM")
        print("=" * 50)
        
        print("\nBacktest Configuration:")
        print("  - Strategy: LAEF's Superior AI/ML Multi-Strategy System")
        print("  - Stock Selection: LAEF Smart Selection (AI-driven)")
        print("  - Period: Last 3 months")
        print("  - Initial Cash: $50,000")
        print("  - Auto Config: Live parameter optimization")
        
        try:
            from trading.unified_trading_engine import LAEFBacktester
            
            backtest = LAEFBacktester()
            results = backtest.run_quick_backtest()
            
            self._display_backtest_results(results)
            
        except Exception as e:
            self.logger.error(f"Quick backtest error: {e}")
            print(f"[ERROR] Quick backtest failed: {e}")
            input("\nPress Enter to continue...")
            
    def show_monitoring_dashboard(self):
        """Show live monitoring dashboard"""
        pass  # Implementation would go here
        
    def run_optimization(self):
        """Run system optimization"""
        pass  # Implementation would go here
        
    def manage_settings(self):
        """Manage system settings"""
        pass  # Implementation would go here
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load system configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
                
        # Default configuration
        return {
            'paper_trading': True,
            'logging_level': 'INFO',
            'reward_type': 'sharpe_adjusted',
            'risk_level': 'standard'
        }
        
    def _load_strategies(self) -> Dict:
        """Load trading strategies"""
        # This would load actual strategy implementations
        return {}
        
    def _display_backtest_results(self, results: Dict):
        """Display backtest results"""
        # Implementation would go here
        pass