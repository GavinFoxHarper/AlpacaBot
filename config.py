"""
AlpacaBot LAEF System Configuration
===================================
Central configuration file for all system components
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Alpaca API Settings
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
ALPACA_DATA_URL = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')

# API Rate Limiting
API_RATE_LIMIT = 200  # requests per minute
API_RETRY_COUNT = 3
API_TIMEOUT = 30  # seconds

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================

# Trading Mode
PAPER_TRADING = True  # Set to False for live trading (BE CAREFUL!)
TRADING_HOURS_ONLY = True  # Only trade during market hours

# Position Management
MAX_POSITIONS = 10  # Maximum number of concurrent positions
MAX_POSITION_SIZE = 1000  # Maximum $ per position
MIN_POSITION_SIZE = 100  # Minimum $ per position
POSITION_SIZING_METHOD = 'equal'  # 'equal', 'kelly', 'risk_parity'

# Risk Management
STOP_LOSS_PERCENT = 2.0  # Default stop loss percentage
TAKE_PROFIT_PERCENT = 5.0  # Default take profit percentage
TRAILING_STOP_PERCENT = 1.5  # Trailing stop percentage
MAX_DAILY_LOSS = 500  # Maximum daily loss in $
MAX_DRAWDOWN_PERCENT = 10.0  # Maximum account drawdown

# Trading Thresholds (for backward compatibility)
TRADING_THRESHOLDS = {
    'q_buy': 0.58,
    'q_sell': 0.42,
    'ml_profit_peak': 0.58,
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'sell_profit_pct': TAKE_PROFIT_PERCENT / 100,
    'stop_loss_pct': 1 - (STOP_LOSS_PERCENT / 100),
    'trailing_stop_pct': TRAILING_STOP_PERCENT / 100
}

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

# Available strategies and their weights
STRATEGIES = {
    'momentum_scalping': {
        'enabled': True,
        'weight': 0.15,
        'params': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1000000
        }
    },
    'mean_reversion': {
        'enabled': True,
        'weight': 0.15,
        'params': {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14
        }
    },
    'statistical_arbitrage': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'lookback_period': 60,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5
        }
    },
    'dual_model_swing': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'ml_threshold': 0.65,
            'holding_period': 5
        }
    },
    'pattern_recognition': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'min_pattern_confidence': 0.7,
            'patterns': ['head_shoulders', 'double_bottom', 'triangle']
        }
    },
    'time_based': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'morning_session': {'start': '09:30', 'end': '11:00'},
            'afternoon_session': {'start': '14:00', 'end': '15:30'}
        }
    },
    'news_sentiment': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'sentiment_threshold': 0.6,
            'news_sources': ['reuters', 'bloomberg', 'wsj']
        }
    },
    'hybrid_adaptive': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'adaptation_period': 20,
            'performance_window': 50
        }
    },
    'reinforced_grid': {
        'enabled': True,
        'weight': 0.10,
        'params': {
            'grid_levels': 5,
            'grid_spacing': 0.5,
            'learning_rate': 0.001
        }
    }
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Data Sources
DATA_PROVIDER = 'alpaca'  # 'alpaca', 'yfinance', 'polygon'
HISTORICAL_DATA_DAYS = 365  # Days of historical data to fetch
CACHE_ENABLED = True  # Enable data caching
CACHE_EXPIRY_HOURS = 24  # Cache expiry time

# Symbols to Trade
SYMBOLS_FILE = 'tickers_cleaned.csv'  # File containing symbols
DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'
]

# Market Data Settings
TIMEFRAME = '1Min'  # '1Min', '5Min', '15Min', '1Hour', '1Day'
BARS_TO_FETCH = 1000  # Number of bars to fetch

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================

# Model Settings
ML_MODELS_DIR = Path('models')
ML_RETRAIN_DAYS = 30  # Retrain models every N days
ML_VALIDATION_SPLIT = 0.2  # Train/validation split
ML_TEST_SPLIT = 0.1  # Test set split

# Feature Engineering
TECHNICAL_INDICATORS = [
    'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'rsi', 'macd', 'bollinger_bands',
    'atr', 'obv', 'adx'
]

# Model Parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    'lstm': {
        'units': 50,
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32
    }
}

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

# Backtesting Settings
BACKTEST_START_DATE = '2022-01-01'
BACKTEST_END_DATE = '2023-12-31'
INITIAL_CAPITAL = 10000  # Starting capital for backtesting
INITIAL_CASH = 10000  # Alias for compatibility
COMMISSION = 0.001  # Trading commission (0.1%)
SLIPPAGE = 0.001  # Slippage (0.1%)
MAX_RISK_PER_TRADE = 0.02  # Maximum risk per trade (2% of portfolio)

# Additional configuration for compatibility
COOLDOWN_MINUTES = 5  # Cooldown between trades
OVERRIDE_FIFO_FOR_STOP_LOSS = True  # Allow stop loss override of FIFO

# Performance Metrics
METRICS_TO_CALCULATE = [
    'total_return', 'annual_return', 'sharpe_ratio',
    'max_drawdown', 'win_rate', 'profit_factor',
    'avg_win', 'avg_loss', 'total_trades'
]

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
CONFIG_PROFILES_DIR = BASE_DIR / 'config_profiles'

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR, CONFIG_PROFILES_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Trade log file (defined after LOGS_DIR)
TRADE_LOG_FILE = LOGS_DIR / 'trades.log'

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / f"alpacabot_{datetime.now().strftime('%Y%m%d')}.log"

# System Settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
DRY_RUN = os.getenv('DRY_RUN', 'False').lower() == 'true'  # Simulate trades without execution
NOTIFICATION_ENABLED = False  # Enable email/SMS notifications
HEARTBEAT_INTERVAL = 60  # System health check interval (seconds)

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config() -> tuple[bool, List[str]]:
    """
    Validate configuration settings
    Returns: (is_valid, list_of_issues)
    """
    issues = []
    
    # Check API keys
    if not ALPACA_API_KEY:
        issues.append("ALPACA_API_KEY not set in environment")
    if not ALPACA_SECRET_KEY:
        issues.append("ALPACA_SECRET_KEY not set in environment")
    
    # Check strategy weights
    total_weight = sum(s['weight'] for s in STRATEGIES.values() if s['enabled'])
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"Strategy weights don't sum to 1.0 (current: {total_weight})")
    
    # Check risk parameters
    if STOP_LOSS_PERCENT <= 0:
        issues.append("STOP_LOSS_PERCENT must be positive")
    if MAX_POSITIONS <= 0:
        issues.append("MAX_POSITIONS must be positive")
    
    # Check directories
    if not BASE_DIR.exists():
        issues.append(f"Base directory doesn't exist: {BASE_DIR}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def load_profile(profile_name: str) -> bool:
    """
    Load a configuration profile
    """
    profile_path = CONFIG_PROFILES_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        print(f"Profile not found: {profile_name}")
        return False
    
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    # Update global variables with profile settings
    for key, value in profile.items():
        if key in globals():
            globals()[key] = value
    
    print(f"Loaded profile: {profile_name}")
    return True

def save_profile(profile_name: str) -> bool:
    """
    Save current configuration as a profile
    """
    profile = {
        'PAPER_TRADING': PAPER_TRADING,
        'MAX_POSITIONS': MAX_POSITIONS,
        'MAX_POSITION_SIZE': MAX_POSITION_SIZE,
        'STOP_LOSS_PERCENT': STOP_LOSS_PERCENT,
        'TAKE_PROFIT_PERCENT': TAKE_PROFIT_PERCENT,
        'STRATEGIES': STRATEGIES,
        'TIMEFRAME': TIMEFRAME
    }
    
    profile_path = CONFIG_PROFILES_DIR / f"{profile_name}.json"
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=4)
    
    print(f"Saved profile: {profile_name}")
    return True

# ============================================================================
# INITIALIZATION
# ============================================================================

# Validate configuration on import
if __name__ != "__main__":
    is_valid, issues = validate_config()
    if not is_valid:
        print("[WARNING] Configuration Issues Detected:")
        for issue in issues:
            print(f"   - {issue}")
        if not DEBUG_MODE:
            print("\n[INFO] Running in SAFE MODE with paper trading enabled")
            PAPER_TRADING = True
    else:
        print("[OK] Configuration validated successfully")

# Export key configuration items
__all__ = [
    'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL',
    'PAPER_TRADING', 'STRATEGIES', 'MAX_POSITIONS',
    'STOP_LOSS_PERCENT', 'TAKE_PROFIT_PERCENT', 'TRADING_THRESHOLDS',
    'INITIAL_CASH', 'MAX_RISK_PER_TRADE', 'COOLDOWN_MINUTES',
    'validate_config', 'load_profile', 'save_profile'
]
