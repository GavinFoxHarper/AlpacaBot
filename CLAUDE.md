# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LAEF (Learning-Augmented Equity Framework) is an advanced algorithmic trading platform that combines machine learning with traditional trading strategies. The system features:

- Adaptive trading strategies with real-time market analysis
- Hybrid approach combining Q-learning with momentum-based scalping
- Multi-timeframe pattern recognition and backtesting
- Risk management and position sizing optimization
- Enhanced news sentiment integration
- Advanced risk-adjusted reward calculation

## Core Components

### Trading Engines
- **Unified Trading Engine**: Main engine supporting multiple strategies and adaptive mode
- **Hybrid Trading Engine**: Combines day and swing trading with momentum analysis 
- **LAEF AI Strategy**: Pure machine learning-based decision making
- **Unified Q-Value Handler**: Advanced Q-value processing with market regime adaptation

### Market Analysis
- Pattern Recognition Analyzer: Detects micro and macro market patterns
- Smart Stock Selector: Intelligent stock screening and selection
- Enhanced Live Learner: Continuous model improvement with advanced performance tracking
- News Integration System: Real-time news sentiment analysis and trading signals

### Risk Management
- Dynamic position sizing based on market conditions
- Multi-level stop loss and profit target system
- Risk-adjusted reward system with multiple reward types
- Advanced drawdown and volatility management

## Development Commands

```bash
# Environment Setup
pip install -r requirements.txt 

# Testing
python -m pytest tests/
pytest tests/test_laef_system.py    # System validation
pytest tests/test_laef_ai_strategy.py  # AI strategy tests

# Run Trading System
python main_trading.py              # Main trading interface
python start_laef_interactive.py    # Interactive mode
```

### Development Notes

1. **Configuration**: Use config_profiles/ directory for custom trading configurations. Key parameters:
   - q_buy/q_sell: AI confidence thresholds
   - momentum_threshold: Price momentum triggers
   - position_size: Risk management limits
   - reward_type: Risk-adjusted reward calculation method
   - market_regime_weights: Strategy weights per market regime

2. **Trading Modes**:
   - paper_trading: Virtual trading for testing
   - live_learning: ML model training with enhanced feedback loop
   - backtesting: Strategy validation
   - regime_specific: Market regime-specific strategy selection

3. **Testing Requirements**: 
   - All new strategies must have unit tests
   - Backtesting required before deployment
   - Performance metrics validation
   - Risk-adjusted performance testing

## Project Architecture

The codebase follows a modular architecture:

```
laef/
├── core/            # Core trading components
├── data/            # Market data management
├── training/        # ML model training
├── trading/         # Trading engines and strategy
├── optimization/    # Parameter optimization
├── news/           # News sentiment analysis
└── utils/          # Helper utilities
```

### Key Files
- laef_unified_system.py: Main trading system interface
- strategy_pattern_recognition.py: Market pattern detection
- parameter_optimizer.py: Trading parameter optimization
- live_market_learner.py: Real-time ML improvements
- q_value_handler.py: Advanced Q-value processing
- risk_adjusted_reward_system.py: Enhanced reward calculation
- news_integration_system.py: News sentiment analysis

## Security and Risk

- API keys managed through environment variables
- Conservative risk limits enforced by default
- Real-time monitoring and alerts
- Trade validation and sanitization
- Multiple risk-adjusted reward types
- Advanced drawdown protection