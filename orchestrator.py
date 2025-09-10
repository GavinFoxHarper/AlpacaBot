#!/usr/bin/env python3
"""
LAEF Orchestrator - Autonomous Trading Bot Management System
Handles automated daily trading, monitoring, and reporting for AlpacaBot
Compliant with NYSE trading calendar and timezone-aware operations
"""

import os
import sys
import logging
import json
import time
import signal
import argparse
import threading
import traceback
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
import pytz
import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from dotenv import load_dotenv
import psutil
import redis
import filelock

# Trading calendar and market hours
import pandas_market_calendars as mcal

# Internal imports
from laef_unified_system import LAEFUnifiedSystem
from trading.unified_trading_engine import UnifiedTradingEngine as LAEFLiveTrader
# from training.live_market_learner import LiveMarketLearner  # Module not found
from core.performance_tracker import BacktestTracker as PerformanceTracker
from utils.logging_utils import setup_structured_logging
# from utils.email_reporter import EmailReporter  # Module not found
from data.market_data_fetcher import DailyMarketMonitor as MarketDataFetcher


class TradingMode(Enum):
    """Trading execution modes"""
    LIVE = "live"
    PAPER = "paper"
    MONITOR_ONLY = "monitor_only"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"


class SystemState(Enum):
    """System operational states"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration"""
    mode: TradingMode
    timezone: str = "America/New_York"
    market_open: dtime = dtime(9, 0, 0)
    market_close: dtime = dtime(16, 0, 0)
    pre_market: dtime = dtime(8, 30, 0)
    after_market: dtime = dtime(17, 0, 0)
    
    # Process management
    max_retries: int = 3
    retry_delay: int = 60
    health_check_interval: int = 30
    pid_file: str = "orchestrator.pid"
    lock_file: str = "orchestrator.lock"
    
    # Logging and reporting
    log_dir: Path = Path("logs")
    report_dir: Path = Path("reports")
    metrics_dir: Path = Path("metrics")
    
    # Redis config for distributed locking
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Performance thresholds
    max_drawdown: float = 0.10
    min_sharpe: float = 0.5
    error_threshold: int = 5
    
    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['mode'] = self.mode.value
        for key in ['market_open', 'market_close', 'pre_market', 'after_market']:
            result[key] = result[key].strftime("%H:%M:%S")
        for key in ['log_dir', 'report_dir', 'metrics_dir']:
            result[key] = str(result[key])
        return result


class LAEFOrchestrator:
    """Main orchestrator for LAEF trading system"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.state = SystemState.IDLE
        self.logger = self._setup_logging()
        self.scheduler = None
        self.lock = None
        self.redis_client = None
        self.trading_thread = None
        self.monitoring_thread = None
        self.metrics_collector = None
        self.email_reporter = None
        
        # Component instances
        self.trading_system = None
        self.market_learner = None
        self.performance_tracker = None
        
        # Runtime metrics
        self.start_time = None
        self.errors_count = 0
        self.trades_count = 0
        self.last_health_check = None
        
        # Initialize directories
        self._initialize_directories()
        
        # Load environment
        load_dotenv()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup structured JSON logging with trace IDs"""
        log_file = self.config.log_dir / f"orchestrator_{datetime.now():%Y%m%d}.log"
        logger = setup_structured_logging(
            name="LAEFOrchestrator",
            log_file=str(log_file),
            level=logging.INFO,
            json_format=True
        )
        return logger
        
    def _initialize_directories(self):
        """Create required directories"""
        for dir_path in [self.config.log_dir, self.config.report_dir, self.config.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown()
        
    def acquire_lock(self) -> bool:
        """Acquire exclusive lock for single instance"""
        try:
            # File-based lock for local single instance
            self.lock = filelock.FileLock(self.config.lock_file, timeout=1)
            self.lock.acquire(blocking=False)
            
            # Redis-based distributed lock if available
            if self.config.redis_host:
                try:
                    self.redis_client = redis.Redis(
                        host=self.config.redis_host,
                        port=self.config.redis_port,
                        db=self.config.redis_db,
                        decode_responses=True
                    )
                    
                    # Set lock with expiration
                    lock_key = f"orchestrator:{os.getpid()}"
                    if not self.redis_client.set(lock_key, "1", nx=True, ex=300):
                        self.logger.warning("Another orchestrator instance may be running")
                        return False
                except Exception as e:
                    self.logger.warning(f"Redis lock unavailable: {e}")
                    
            # Write PID file
            with open(self.config.pid_file, 'w') as f:
                f.write(str(os.getpid()))
                
            return True
            
        except filelock.Timeout:
            self.logger.error("Failed to acquire lock - another instance may be running")
            return False
            
    def release_lock(self):
        """Release all locks"""
        if self.lock:
            self.lock.release()
            
        if self.redis_client:
            try:
                lock_key = f"orchestrator:{os.getpid()}"
                self.redis_client.delete(lock_key)
            except:
                pass
                
        # Remove PID file
        if Path(self.config.pid_file).exists():
            os.remove(self.config.pid_file)
            
    def is_market_open(self) -> bool:
        """Check if US stock market is open"""
        ny_tz = pytz.timezone(self.config.timezone)
        now = datetime.now(ny_tz)
        
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        
        # Check if today is a trading day
        today = now.date()
        schedule = nyse.schedule(start_date=today, end_date=today)
        
        if schedule.empty:
            return False
            
        # Check if within market hours
        market_open = schedule.iloc[0]['market_open'].tz_convert(ny_tz)
        market_close = schedule.iloc[0]['market_close'].tz_convert(ny_tz)
        
        return market_open <= now <= market_close
        
    def get_next_trading_day(self) -> datetime:
        """Get next trading day"""
        ny_tz = pytz.timezone(self.config.timezone)
        now = datetime.now(ny_tz)
        
        nyse = mcal.get_calendar('NYSE')
        
        # Get next 10 days schedule
        end_date = now.date() + timedelta(days=10)
        schedule = nyse.schedule(start_date=now.date(), end_date=end_date)
        
        # Find next trading day
        for idx, row in schedule.iterrows():
            market_open = row['market_open'].tz_convert(ny_tz)
            if market_open > now:
                return market_open
                
        return None
        
    def initialize_components(self):
        """Initialize trading components"""
        self.logger.info("Initializing trading components")
        
        try:
            # Initialize main trading system
            self.trading_system = LAEFUnifiedSystem(debug_mode=False)
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker()
            
            # Initialize market learner for live monitoring
            if self.config.mode in [TradingMode.LIVE, TradingMode.PAPER, TradingMode.MONITOR_ONLY]:
                # self.market_learner = LiveMarketLearner()  # Module not found
                pass
                
            # Initialize email reporter
            if os.getenv("EMAIL_ENABLED", "false").lower() == "true":
                # self.email_reporter = EmailReporter()  # Module not found
                pass
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            return False
            
    def run_pre_market_checks(self) -> Dict[str, Any]:
        """Run pre-market validation checks"""
        self.logger.info("Running pre-market checks")
        
        checks = {
            "timestamp": datetime.now().isoformat(),
            "market_status": "unknown",
            "api_connection": False,
            "data_feed": False,
            "account_status": False,
            "risk_limits": False,
            "system_resources": False,
            "errors": []
        }
        
        try:
            # Check market status
            checks["market_status"] = "open" if self.is_market_open() else "closed"
            
            # Check API connection
            from trading.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(paper_trading=self.config.mode == TradingMode.PAPER)
            account = broker.get_account()
            checks["api_connection"] = account is not None
            checks["account_status"] = account.status == "ACTIVE" if account else False
            
            # Check data feed
            fetcher = MarketDataFetcher()
            test_data = fetcher.get_latest_price("SPY")
            checks["data_feed"] = test_data is not None
            
            # Check risk limits
            if account:
                checks["risk_limits"] = float(account.cash) > 1000
                
            # Check system resources
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            checks["system_resources"] = (
                memory.percent < 90 and 
                cpu < self.config.max_cpu_percent
            )
            
        except Exception as e:
            checks["errors"].append(str(e))
            self.logger.error(f"Pre-market check failed: {e}")
            
        return checks
        
    def start_paper_trading(self):
        """Start paper trading session"""
        self.logger.info("Starting paper trading session")
        
        try:
            from trading.unified_trading_engine import UnifiedTradingEngine as LAEFLiveTrader
            
            trader = LAEFLiveTrader(paper_trading=True)
            
            # Run trading loop
            while self.state == SystemState.RUNNING:
                if self.is_market_open():
                    trader.execute_trading_cycle()
                    time.sleep(60)  # Check every minute
                else:
                    self.logger.info("Market closed, waiting...")
                    time.sleep(300)  # Check every 5 minutes
                    
        except Exception as e:
            self.logger.error(f"Paper trading error: {e}")
            self.errors_count += 1
            
    def start_live_monitoring(self):
        """Start live market monitoring"""
        self.logger.info("Starting live market monitoring")
        
        try:
            if self.market_learner:
                self.market_learner.start_monitoring()
                
                while self.state == SystemState.RUNNING:
                    if self.is_market_open():
                        metrics = self.market_learner.get_current_metrics()
                        self.log_metrics(metrics)
                        time.sleep(30)  # Update every 30 seconds
                    else:
                        time.sleep(300)  # Check every 5 minutes when closed
                        
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            
    def log_metrics(self, metrics: Dict):
        """Log performance metrics"""
        metrics_file = self.config.metrics_dir / f"metrics_{datetime.now():%Y%m%d}.jsonl"
        
        with open(metrics_file, 'a') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                **metrics
            }, f)
            f.write('\n')
            
    def generate_daily_report(self) -> Dict:
        """Generate end-of-day report"""
        self.logger.info("Generating daily report")
        
        report = {
            "date": datetime.now().date().isoformat(),
            "mode": self.config.mode.value,
            "runtime_hours": 0,
            "trades_executed": self.trades_count,
            "errors_encountered": self.errors_count,
            "performance": {},
            "system_health": {}
        }
        
        try:
            if self.start_time:
                runtime = datetime.now() - self.start_time
                report["runtime_hours"] = runtime.total_seconds() / 3600
                
            if self.performance_tracker:
                report["performance"] = self.performance_tracker.get_summary()
                
            # System health metrics
            report["system_health"] = {
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
            
            # Save report
            report_file = self.config.report_dir / f"daily_report_{datetime.now():%Y%m%d}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            # Send email if configured
            if self.email_reporter:
                self.email_reporter.send_daily_report(report)
                
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            
        return report
        
    def setup_scheduler(self):
        """Setup APScheduler for automated runs"""
        self.logger.info("Setting up scheduler")
        
        executors = {
            'default': ThreadPoolExecutor(10),
            'processpool': ProcessPoolExecutor(3)
        }
        
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 30
        }
        
        self.scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone=self.config.timezone
        )
        
        # Schedule daily trading at market open (9:00 AM ET)
        self.scheduler.add_job(
            func=self.daily_trading_job,
            trigger=CronTrigger(
                hour=9, minute=0,
                day_of_week='mon-fri',
                timezone=self.config.timezone
            ),
            id='daily_trading',
            name='Daily Trading Job'
        )
        
        # Schedule pre-market preparation (8:30 AM ET)
        self.scheduler.add_job(
            func=self.pre_market_job,
            trigger=CronTrigger(
                hour=8, minute=30,
                day_of_week='mon-fri',
                timezone=self.config.timezone
            ),
            id='pre_market',
            name='Pre-Market Preparation'
        )
        
        # Schedule end-of-day report (4:30 PM ET)
        self.scheduler.add_job(
            func=self.end_of_day_job,
            trigger=CronTrigger(
                hour=16, minute=30,
                day_of_week='mon-fri',
                timezone=self.config.timezone
            ),
            id='end_of_day',
            name='End of Day Report'
        )
        
        # Schedule health checks every 30 seconds
        self.scheduler.add_job(
            func=self.health_check,
            trigger='interval',
            seconds=self.config.health_check_interval,
            id='health_check',
            name='System Health Check'
        )
        
        self.scheduler.start()
        self.logger.info("Scheduler started with all jobs configured")
        
    def daily_trading_job(self):
        """Daily trading job executed at market open"""
        self.logger.info("Executing daily trading job")
        
        if not self.is_market_open():
            self.logger.info("Market is closed, skipping trading")
            return
            
        self.state = SystemState.RUNNING
        
        if self.config.mode == TradingMode.PAPER:
            self.trading_thread = threading.Thread(target=self.start_paper_trading)
            self.trading_thread.start()
            
        elif self.config.mode == TradingMode.MONITOR_ONLY:
            self.monitoring_thread = threading.Thread(target=self.start_live_monitoring)
            self.monitoring_thread.start()
            
        elif self.config.mode == TradingMode.LIVE:
            self.logger.warning("Live trading mode - ensure proper authorization")
            # Implement live trading with additional safety checks
            
    def pre_market_job(self):
        """Pre-market preparation job"""
        self.logger.info("Executing pre-market preparation")
        
        # Run system checks
        checks = self.run_pre_market_checks()
        
        # Log results
        self.logger.info(f"Pre-market checks: {json.dumps(checks, indent=2)}")
        
        # Initialize components if needed
        if not self.trading_system:
            self.initialize_components()
            
    def end_of_day_job(self):
        """End of day reporting job"""
        self.logger.info("Executing end-of-day tasks")
        
        # Generate daily report
        report = self.generate_daily_report()
        
        # Clean up resources
        self.state = SystemState.IDLE
        
    def health_check(self):
        """System health check"""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
                
            # Check CPU usage
            cpu = psutil.cpu_percent(interval=1)
            if cpu > self.config.max_cpu_percent:
                self.logger.warning(f"High CPU usage: {cpu}%")
                
            # Check error threshold
            if self.errors_count > self.config.error_threshold:
                self.logger.error("Error threshold exceeded, consider restart")
                
            self.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            
    def run(self):
        """Main orchestrator run loop"""
        self.logger.info("Starting LAEF Orchestrator")
        
        # Acquire lock
        if not self.acquire_lock():
            self.logger.error("Failed to acquire lock, exiting")
            return 1
            
        try:
            self.start_time = datetime.now()
            self.state = SystemState.STARTING
            
            # Initialize components
            if not self.initialize_components():
                return 1
                
            # Setup scheduler
            self.setup_scheduler()
            
            self.state = SystemState.RUNNING
            self.logger.info("Orchestrator running, press Ctrl+C to stop")
            
            # Keep running
            while self.state == SystemState.RUNNING:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
            
        except Exception as e:
            self.logger.error(f"Orchestrator error: {e}")
            self.logger.error(traceback.format_exc())
            return 1
            
        finally:
            self.shutdown()
            
        return 0
        
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating shutdown")
        self.state = SystemState.STOPPING
        
        # Stop scheduler
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            
        # Stop threads
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
            
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            
        # Generate final report
        self.generate_daily_report()
        
        # Release lock
        self.release_lock()
        
        self.logger.info("Shutdown complete")
        

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LAEF Trading Orchestrator")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "monitor", "backtest", "dry-run"],
        default="paper",
        help="Trading mode"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trading)"
    )
    
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode"
    )
    
    parser.add_argument(
        "--live-monitor-only",
        action="store_true",
        help="Only run live market monitoring"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.dry_run:
        mode = TradingMode.DRY_RUN
    elif args.paper:
        mode = TradingMode.PAPER
    elif args.live_monitor_only:
        mode = TradingMode.MONITOR_ONLY
    else:
        mode = TradingMode[args.mode.upper()]
        
    # Create config
    config = OrchestratorConfig(mode=mode)
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
    # Create and run orchestrator
    orchestrator = LAEFOrchestrator(config)
    
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())