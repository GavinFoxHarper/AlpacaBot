#!/usr/bin/env python3
"""
Initialize the prediction tracking database and integrate it with LAEF system
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
import json

# Create directories
logs_dir = Path("logs/training")
logs_dir.mkdir(parents=True, exist_ok=True)

# Initialize database
db_path = logs_dir / "predictions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create predictions table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        symbol TEXT,
        action TEXT,
        confidence REAL,
        q_values TEXT,
        price REAL,
        outcome TEXT,
        profit REAL,
        prediction_accuracy TEXT
    )
''')

# Create learning history table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS learning_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        model_type TEXT,
        performance_before REAL,
        performance_after REAL,
        parameters TEXT,
        update_reason TEXT
    )
''')

# Create strategy performance table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS strategy_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        strategy_name TEXT,
        trades_count INTEGER,
        win_rate REAL,
        profit_loss REAL,
        sharpe_ratio REAL
    )
''')

# Insert initial learning history entry
cursor.execute('''
    INSERT INTO learning_history (timestamp, model_type, performance_before, performance_after, parameters, update_reason)
    VALUES (?, ?, ?, ?, ?, ?)
''', (datetime.now(), 'q_learning', 0.0, 0.0, json.dumps({'epsilon': 1.0, 'learning_rate': 0.001}), 'initialization'))

# Insert sample predictions for testing
sample_predictions = [
    (datetime.now(), 'AAPL', 'buy', 0.75, json.dumps([0.8, 0.6, 0.5]), 175.50, 'pending', 0.0, 'pending'),
    (datetime.now(), 'GOOGL', 'hold', 0.65, json.dumps([0.7, 0.7, 0.6]), 140.25, 'pending', 0.0, 'pending'),
    (datetime.now(), 'MSFT', 'sell', 0.80, json.dumps([0.4, 0.5, 0.8]), 380.75, 'pending', 0.0, 'pending'),
]

cursor.executemany('''
    INSERT INTO predictions (timestamp, symbol, action, confidence, q_values, price, outcome, profit, prediction_accuracy)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
''', sample_predictions)

# Insert sample strategy performance
strategies = [
    ('momentum_scalping', 5, 0.60, 150.00, 1.2),
    ('mean_reversion', 3, 0.67, 75.50, 0.9),
    ('hybrid_adaptive', 8, 0.75, 250.00, 1.5),
]

for strategy in strategies:
    cursor.execute('''
        INSERT INTO strategy_performance (timestamp, strategy_name, trades_count, win_rate, profit_loss, sharpe_ratio)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), *strategy))

conn.commit()
conn.close()

print(f"[SUCCESS] Prediction tracking database initialized at: {db_path}")
print("[SUCCESS] Sample data inserted for testing")
print("\nDatabase contains:")
print("  - predictions table (for tracking AI predictions)")
print("  - learning_history table (for tracking model improvements)")
print("  - strategy_performance table (for tracking strategy metrics)")
print("\n[INFO] Now run 'python live_monitoring_dashboard.py' to see the data in the dashboard")