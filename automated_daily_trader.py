#!/usr/bin/env python3
"""
Automated Daily Trading System for LAEF
Starts at 9:30 AM, monitors market, sends daily reports
"""

import os
import sys
import time
import json
import smtplib
import sqlite3
import logging
import threading
import pandas as pd
from datetime import datetime, timedelta, time as datetime_time
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()  # Load environment first

from alpaca_trade_api import REST
from config import *

class AutomatedDailyTrader:
    """Automated daily trading with learning and reporting"""
    
    def __init__(self):
        # API Setup
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.api = REST(self.api_key, self.secret_key, self.base_url)
        
        # Email configuration
        self.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_from = os.getenv('EMAIL_FROM', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        self.email_to = os.getenv('EMAIL_TO', '').split(',')
        
        # Trading configuration
        self.trading_active = False
        self.start_time = None
        self.end_time = None
        
        # Daily tracking
        self.daily_metrics = {
            'start_portfolio_value': 0,
            'trades_executed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'total_profit': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'predictions_made': 0,
            'learning_updates': 0,
            'errors': []
        }
        
        # Learning tracking
        self.learning_data = {
            'new_patterns': [],
            'strategy_adjustments': [],
            'model_improvements': [],
            'market_insights': []
        }
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup daily logging"""
        log_dir = LOGS_DIR / 'daily_trading'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        log_file = log_dir / f"daily_trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def wait_for_market_open(self):
        """Wait until market opens at 9:30 AM EST"""
        self.logger.info("Waiting for market to open...")
        
        while True:
            clock = self.api.get_clock()
            
            if clock.is_open:
                self.logger.info("Market is open! Starting trading...")
                return True
            
            # Calculate time until market opens
            now = datetime.now(clock.next_open.tzinfo)
            time_to_open = (clock.next_open - now).total_seconds()
            
            if time_to_open > 0:
                self.logger.info(f"Market opens in {time_to_open/60:.1f} minutes")
                
                # Wait efficiently
                if time_to_open > 3600:  # More than 1 hour
                    time.sleep(1800)  # Check every 30 minutes
                elif time_to_open > 300:  # More than 5 minutes
                    time.sleep(240)  # Check every 4 minutes
                else:
                    time.sleep(30)  # Check every 30 seconds
            else:
                time.sleep(10)  # Market should be open soon
    
    def initialize_daily_session(self):
        """Initialize the daily trading session"""
        self.start_time = datetime.now()
        
        # Get initial portfolio value
        account = self.api.get_account()
        self.daily_metrics['start_portfolio_value'] = float(account.portfolio_value)
        
        self.logger.info(f"Daily session initialized")
        self.logger.info(f"Starting portfolio value: ${self.daily_metrics['start_portfolio_value']:,.2f}")
        
        # Check existing positions
        positions = self.api.list_positions()
        self.logger.info(f"Starting with {len(positions)} open positions")
    
    def run_trading_strategies(self):
        """Run the main LAEF trading strategies"""
        try:
            # Import and run LAEF engine
            from laef.laef_ai_trading_engine import LAEFAITradingEngine
            from trading.alpaca_broker import AlpacaBroker
            from core.portfolio_manager import PortfolioManager
            
            # Initialize components
            broker = AlpacaBroker(
                api_key=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                paper_trading=True
            )
            
            portfolio_manager = PortfolioManager(
                broker=broker,
                max_positions=MAX_POSITIONS,
                max_position_size=MAX_POSITION_SIZE
            )
            
            engine = LAEFAITradingEngine(
                broker=broker,
                portfolio_manager=portfolio_manager,
                config=STRATEGIES
            )
            
            self.logger.info("LAEF Trading Engine started")
            
            # Run trading loop
            self.trading_active = True
            
            while self.trading_active:
                clock = self.api.get_clock()
                
                if not clock.is_open:
                    self.logger.info("Market closed. Ending trading session.")
                    break
                
                # Execute trading logic
                try:
                    engine.execute_trading_cycle()
                    self.daily_metrics['trades_executed'] += 1
                except Exception as e:
                    self.logger.error(f"Trading cycle error: {e}")
                    self.daily_metrics['errors'].append(str(e))
                
                # Check for stop conditions
                if self.check_stop_conditions():
                    break
                
                # Wait before next cycle
                time.sleep(60)  # Run every minute
            
        except Exception as e:
            self.logger.error(f"Fatal trading error: {e}")
            self.daily_metrics['errors'].append(f"Fatal: {str(e)}")
    
    def check_stop_conditions(self) -> bool:
        """Check if trading should stop"""
        # Check daily loss limit
        account = self.api.get_account()
        current_value = float(account.portfolio_value)
        daily_loss = current_value - self.daily_metrics['start_portfolio_value']
        
        if daily_loss < -MAX_DAILY_LOSS:
            self.logger.warning(f"Daily loss limit reached: ${daily_loss:.2f}")
            return True
        
        # Check time (stop 30 minutes before close)
        clock = self.api.get_clock()
        if clock.next_close:
            time_to_close = (clock.next_close - datetime.now(clock.next_close.tzinfo)).total_seconds()
            if time_to_close < 1800:  # 30 minutes
                self.logger.info("Approaching market close. Stopping trading.")
                return True
        
        return False
    
    def track_learning_progress(self):
        """Track what the bot has learned today"""
        # Check prediction database
        predictions_db = LOGS_DIR / 'training' / 'predictions.db'
        
        if predictions_db.exists():
            conn = sqlite3.connect(predictions_db)
            cursor = conn.cursor()
            
            # Get today's predictions
            cursor.execute("""
                SELECT COUNT(*), AVG(confidence), AVG(CASE WHEN prediction_accuracy = 'correct' THEN 1 ELSE 0 END)
                FROM predictions
                WHERE DATE(timestamp) = DATE('now')
            """)
            
            result = cursor.fetchone()
            if result:
                self.daily_metrics['predictions_made'] = result[0]
                if result[0] > 0:
                    avg_confidence = result[1] or 0
                    accuracy = (result[2] or 0) * 100
                    
                    self.learning_data['market_insights'].append(
                        f"Made {result[0]} predictions with {accuracy:.1f}% accuracy and {avg_confidence:.2f} avg confidence"
                    )
            
            # Get learning updates
            cursor.execute("""
                SELECT COUNT(*), AVG(performance_after - performance_before)
                FROM learning_history
                WHERE DATE(timestamp) = DATE('now')
            """)
            
            result = cursor.fetchone()
            if result and result[0]:
                self.daily_metrics['learning_updates'] = result[0]
                improvement = result[1] or 0
                
                self.learning_data['model_improvements'].append(
                    f"Applied {result[0]} learning updates with avg improvement: {improvement:.4f}"
                )
            
            conn.close()
        
        # Check for new patterns learned
        self.check_pattern_learning()
        
        # Check strategy performance
        self.analyze_strategy_performance()
    
    def check_pattern_learning(self):
        """Check for new patterns the bot has learned"""
        # This would analyze logs for pattern recognition
        log_files = list(LOGS_DIR.glob(f"*{datetime.now().strftime('%Y%m%d')}*.log"))
        
        patterns_found = []
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if 'pattern detected' in content.lower():
                        patterns_found.append(log_file.name)
            except:
                pass
        
        if patterns_found:
            self.learning_data['new_patterns'].append(
                f"Detected patterns in {len(patterns_found)} log files"
            )
    
    def analyze_strategy_performance(self):
        """Analyze how each strategy performed today"""
        # Get today's orders
        orders = self.api.list_orders(status='all', after=self.start_time.isoformat())
        
        strategy_performance = {}
        
        # Analyze orders by strategy (would need strategy tagging in real implementation)
        for order in orders:
            # Extract strategy from order (this is simplified)
            strategy = 'unknown'
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'trades': 0, 'profit': 0}
            
            strategy_performance[strategy]['trades'] += 1
            
            if order.status == 'filled' and order.side == 'sell':
                # Calculate profit (simplified)
                strategy_performance[strategy]['profit'] += float(order.filled_qty or 0) * 0.01
        
        # Record insights
        for strategy, perf in strategy_performance.items():
            if perf['trades'] > 0:
                self.learning_data['strategy_adjustments'].append(
                    f"{strategy}: {perf['trades']} trades, ${perf['profit']:.2f} profit"
                )
    
    def generate_daily_report(self) -> str:
        """Generate comprehensive daily report"""
        self.end_time = datetime.now()
        
        # Get final portfolio value
        account = self.api.get_account()
        final_value = float(account.portfolio_value)
        daily_change = final_value - self.daily_metrics['start_portfolio_value']
        daily_change_pct = (daily_change / self.daily_metrics['start_portfolio_value']) * 100
        
        # Get trade statistics
        orders = self.api.list_orders(status='filled', after=self.start_time.isoformat())
        filled_orders = [o for o in orders if o.status == 'filled']
        
        report = []
        report.append("=" * 70)
        report.append(" " * 20 + "LAEF DAILY TRADING REPORT")
        report.append("=" * 70)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        report.append(f"Trading Hours: {self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')}")
        report.append("")
        
        report.append("PORTFOLIO PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Starting Value:     ${self.daily_metrics['start_portfolio_value']:,.2f}")
        report.append(f"Ending Value:       ${final_value:,.2f}")
        report.append(f"Daily Change:       ${daily_change:+,.2f} ({daily_change_pct:+.2f}%)")
        report.append("")
        
        report.append("TRADING ACTIVITY:")
        report.append("-" * 40)
        report.append(f"Total Orders:       {len(filled_orders)}")
        report.append(f"Trades Executed:    {self.daily_metrics['trades_executed']}")
        report.append(f"Predictions Made:   {self.daily_metrics['predictions_made']}")
        report.append("")
        
        report.append("LEARNING & EVOLUTION:")
        report.append("-" * 40)
        report.append(f"Learning Updates:   {self.daily_metrics['learning_updates']}")
        
        if self.learning_data['market_insights']:
            report.append("\nMarket Insights:")
            for insight in self.learning_data['market_insights']:
                report.append(f"  - {insight}")
        
        if self.learning_data['model_improvements']:
            report.append("\nModel Improvements:")
            for improvement in self.learning_data['model_improvements']:
                report.append(f"  - {improvement}")
        
        if self.learning_data['strategy_adjustments']:
            report.append("\nStrategy Performance:")
            for adjustment in self.learning_data['strategy_adjustments']:
                report.append(f"  - {adjustment}")
        
        if self.learning_data['new_patterns']:
            report.append("\nNew Patterns Learned:")
            for pattern in self.learning_data['new_patterns']:
                report.append(f"  - {pattern}")
        
        report.append("")
        
        # Current positions
        positions = self.api.list_positions()
        if positions:
            report.append(f"OPEN POSITIONS ({len(positions)}):")
            report.append("-" * 40)
            for pos in positions[:5]:  # Show first 5
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100
                report.append(f"  {pos.symbol}: {pos.qty} shares, P&L: ${unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.2f}%)")
            if len(positions) > 5:
                report.append(f"  ... and {len(positions) - 5} more")
            report.append("")
        
        # Errors if any
        if self.daily_metrics['errors']:
            report.append("ERRORS & WARNINGS:")
            report.append("-" * 40)
            for error in self.daily_metrics['errors'][:5]:
                report.append(f"  - {error}")
            report.append("")
        
        report.append("EVOLUTION SUMMARY:")
        report.append("-" * 40)
        
        # Calculate evolution score
        evolution_score = 0
        if self.daily_metrics['predictions_made'] > 10:
            evolution_score += 25
        if self.daily_metrics['learning_updates'] > 0:
            evolution_score += 25
        if daily_change > 0:
            evolution_score += 25
        if len(self.learning_data['market_insights']) > 0:
            evolution_score += 25
        
        report.append(f"Daily Evolution Score: {evolution_score}/100")
        report.append(f"Bot Status: {'EVOLVING SUCCESSFULLY' if evolution_score >= 50 else 'LEARNING IN PROGRESS'}")
        
        report.append("")
        report.append("=" * 70)
        report.append("End of Daily Report")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def send_email_report(self, report: str):
        """Send daily report via email"""
        if not self.email_enabled:
            self.logger.info("Email reporting disabled")
            return
        
        if not self.email_from or not self.email_password or not self.email_to:
            self.logger.warning("Email configuration incomplete")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            msg['Subject'] = f"LAEF Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Add report as body
            msg.attach(MIMEText(report, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Email report sent to {', '.join(self.email_to)}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email report: {e}")
    
    def save_report(self, report: str):
        """Save report to file"""
        report_dir = REPORTS_DIR / 'daily_reports'
        report_dir.mkdir(exist_ok=True, parents=True)
        
        report_file = report_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {report_file}")
        return report_file
    
    def run(self):
        """Main execution loop"""
        self.logger.info("=" * 60)
        self.logger.info("LAEF Automated Daily Trader Starting")
        self.logger.info("=" * 60)
        
        try:
            # Wait for market open
            if self.wait_for_market_open():
                # Initialize session
                self.initialize_daily_session()
                
                # Start monitoring thread
                monitor_thread = threading.Thread(target=self.continuous_monitoring)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Run trading strategies
                self.run_trading_strategies()
                
                # Track learning progress
                self.track_learning_progress()
                
                # Generate report
                report = self.generate_daily_report()
                print("\n" + report)
                
                # Save report
                report_file = self.save_report(report)
                
                # Send email report
                self.send_email_report(report)
                
                self.logger.info("Daily trading session completed successfully")
            
        except Exception as e:
            self.logger.error(f"Fatal error in daily trader: {e}")
            # Send error report
            error_report = f"LAEF Daily Trader Error\n\n{str(e)}"
            self.send_email_report(error_report)
        
        finally:
            self.trading_active = False
            self.logger.info("Automated Daily Trader shutdown complete")
    
    def continuous_monitoring(self):
        """Continuous monitoring thread"""
        while self.trading_active:
            try:
                # Monitor portfolio
                account = self.api.get_account()
                current_value = float(account.portfolio_value)
                change = current_value - self.daily_metrics['start_portfolio_value']
                
                self.logger.debug(f"Portfolio: ${current_value:,.2f} ({change:+,.2f})")
                
                # Sleep for monitoring interval
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

if __name__ == "__main__":
    trader = AutomatedDailyTrader()
    trader.run()