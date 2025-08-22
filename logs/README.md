# LAEF Log Directory

This directory contains system logs and performance data.

## Log Types

1. Trading Logs
- trade_history_YYYYMMDD.csv: All executed trades
- trading_decisions_YYYYMMDD.log: Trading logic and explanations
- portfolio_YYYYMMDD.csv: Portfolio value and positions

2. Learning Logs
- learning_progress_YYYYMMDD.log: ML training progress
- prediction_accuracy_YYYYMMDD.csv: Prediction tracking
- model_updates_YYYYMMDD.log: Model modification details

3. Performance Analysis
- backtest_results_YYYYMMDD.json: Backtest performance data
- optimization_report_YYYYMMDD.txt: Parameter optimization results
- risk_analysis_YYYYMMDD.txt: Risk metrics and analysis

## Log Rotation
- Logs are created daily
- Stored for 30 days by default
- Performance data kept for 1 year