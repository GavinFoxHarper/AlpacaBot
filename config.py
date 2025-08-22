#!/usr/bin/env python3
"""Configuration settings for LAEF trading system"""

import os
from datetime import datetime, timedelta

# State features and settings
STATE_SIZE = 12  # Input features for ML models
STATE_FEATURES = [
    'price', 'volume', 'rsi', 'macd', 'signal',
    'sma_20', 'sma_50', 'bollinger_upper', 'bollinger_lower',
    'atr', 'momentum', 'volatility'
]

# AI Model settings 
LEARNING_RATE = 0.001
MODEL_PATH = 'models/laef_model.keras'

# Trading thresholds
TRADING_THRESHOLDS = {
    'q_buy': 0.65,           # Q-learning buy threshold
    'q_sell': 0.35,          # Q-learning sell threshold
    'ml_profit_peak': 0.58,  # ML confidence for taking profit
    'rsi_oversold': 30,      # RSI oversold level
    'rsi_overbought': 70,    # RSI overbought level
    'stop_loss_pct': 0.97,   # Stop loss at 3% down
    'trailing_stop_pct': 0.06  # 6% trailing stop
}

# Risk management
INITIAL_CASH = 100000  # Starting cash for trading
MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_POSITION_SIZE = 0.10  # Max 10% of portfolio in one position
COOLDOWN_MINUTES = 60  # Minutes to wait after stopped out

# Backtesting settings
BACKTEST_START_DATE = '2024-01-01'
BACKTEST_END_DATE = '2024-12-31'

# Training settings
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# System paths
LOG_DIR = 'logs'
CONFIG_DIR = 'config_profiles'
MODEL_DIR = 'models'

# Ensure directories exist
for directory in [LOG_DIR, CONFIG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)