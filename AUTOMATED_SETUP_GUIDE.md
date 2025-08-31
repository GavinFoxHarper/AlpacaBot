# LAEF Automated Daily Trading Setup Guide

This guide will help you set up the LAEF trading system to run automatically every weekday at 9:30 AM and send you daily reports via email.

## 🚀 Quick Setup

### 1. Email Configuration (5 minutes)

```bash
python setup_email_config.py
```

This will help you configure:
- Email server settings (Gmail, Outlook, etc.)
- Your email credentials
- Recipients for daily reports
- Test email functionality

**For Gmail Users:**
- Use an "App Password" instead of your regular password
- Go to: Google Account > Security > 2-Step Verification > App passwords

### 2. Automated Scheduling (2 minutes)

**Run as Administrator:**
```bash
setup_task_scheduler.bat
```

This creates a Windows scheduled task that:
- Runs Monday-Friday at 9:25 AM
- Waits for market open at 9:30 AM
- Trades throughout the day
- Sends daily report via email
- Automatically shuts down at market close

## 📊 Monitoring & Evolution Tracking

### Live Monitoring Dashboard
```bash
python live_monitoring_dashboard.py
```

Shows real-time:
- Q-Learning evolution (epsilon decay, model updates)
- Prediction accuracy and confidence trends  
- Strategy performance evolution
- Portfolio changes and learning progress
- ML model file modifications

### Key Evolution Variables Tracked:

**Q-Learning System:**
- Epsilon (exploration vs exploitation): Starts at 1.0, decays to 0.1 over 100 hours
- Learning rate: Adaptive based on performance
- Model weights: Updated after each prediction outcome
- Reward accumulation: Tracks long-term learning progress

**Prediction Engine:**
- Accuracy rate: Percentage of correct predictions
- Confidence evolution: How certain the bot becomes
- Pattern recognition: New patterns learned daily
- Market regime detection: Adapts to market conditions

**Strategy Evolution:**
- Performance weighting: Better strategies get more allocation
- Parameter optimization: Continuous hyperparameter tuning
- Risk adjustment: Dynamic stop-loss and take-profit levels
- Correlation analysis: Identifies redundant strategies

## 📧 Daily Report Contents

Each morning after trading, you'll receive an email with:

### Portfolio Performance
- Starting vs ending value
- Daily P&L and percentage change
- Risk metrics and drawdown
- Position summary

### Learning Evolution
- New predictions made and their accuracy
- Model improvements and weight updates
- Strategy performance changes
- Pattern recognition discoveries

### Trading Activity
- Number of trades executed
- Win rate and profit factor
- Strategy breakdown
- Risk management actions

### Evolution Metrics
- Evolution score (0-100)
- Learning velocity
- Adaptation rate
- Market insight generation

## 🔧 Advanced Configuration

### Environment Variables (.env file)
```bash
# API Configuration
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Email Configuration
EMAIL_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_FROM=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=recipient1@email.com,recipient2@email.com

# Trading Configuration
PAPER_TRADING=true
MAX_POSITIONS=10
MAX_POSITION_SIZE=1000
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=5.0
```

### Strategy Configuration
The bot starts with all 9 strategies enabled:
1. **Momentum Scalping** - Quick profits on price momentum
2. **Mean Reversion** - Trades oversold/overbought conditions
3. **Statistical Arbitrage** - Pair trading with correlation analysis
4. **Dual-Model Swing** - Combines Q-learning + ML predictions
5. **Pattern Recognition** - CNN/LSTM for chart patterns
6. **Time-Based Algorithm** - Session-specific behavior
7. **News Sentiment** - NLP analysis of market news
8. **Hybrid Adaptive** - Multi-strategy combination
9. **Reinforced Grid Search** - Parameter optimization

## 🧠 How the Bot Evolves

### Learning Process
1. **Morning**: System starts, loads latest models
2. **Pre-market**: Analyzes overnight news and pre-market data
3. **Market Open**: Begins making predictions and trades
4. **Throughout Day**: 
   - Tracks prediction outcomes
   - Updates Q-learning model
   - Adjusts strategy weights
   - Learns new patterns
5. **Market Close**: Generates learning summary
6. **After Hours**: Sends detailed report via email

### Evolution Indicators
- **Epsilon Decay**: Shows learning progress (1.0 → 0.1)
- **Accuracy Improvement**: Prediction accuracy over time
- **Strategy Adaptation**: Performance-based weight adjustments
- **Pattern Discovery**: New market patterns learned
- **Risk Evolution**: Dynamic risk management improvements

## 📈 Performance Tracking

### Files Generated Daily:
- `logs/daily_trading/daily_trading_YYYYMMDD.log`
- `reports/daily_reports/daily_report_YYYYMMDD.txt`
- `reports/evolution_snapshot_YYYYMMDD_HHMMSS.json`
- Database: `logs/training/predictions.db`

### Key Metrics Tracked:
- Total return vs S&P 500
- Sharpe ratio evolution
- Maximum drawdown
- Win rate trends
- Learning velocity
- Model complexity growth

## 🛡️ Safety Features

### Risk Management
- Daily loss limits ($500 default)
- Position size limits ($1000 default)
- Stop-loss on all positions (2% default)
- Maximum 10 concurrent positions
- Paper trading mode (no real money risk)

### Error Handling
- Automatic retry on API failures
- Graceful shutdown on errors
- Email alerts for critical issues
- Comprehensive error logging
- Safe mode activation on problems

## 🔍 Troubleshooting

### Common Issues:

**Email Not Working:**
1. Check Gmail App Password (not regular password)
2. Verify SMTP settings
3. Run `python setup_email_config.py` again

**Task Scheduler Issues:**
1. Run `setup_task_scheduler.bat` as Administrator
2. Check Python is in system PATH
3. Verify .env file exists and is configured

**API Connection Problems:**
1. Verify Alpaca API keys in .env
2. Check internet connection
3. Confirm paper trading URL is correct

**Bot Not Learning:**
1. Check if predictions.db exists in logs/training/
2. Verify model files in models/ directory
3. Check daily logs for learning updates

### Manual Testing:
```bash
# Test API connection
python test_api_connection.py

# Test monitoring dashboard
python test_monitoring.py

# Test email configuration
python setup_email_config.py

# Run backtest to verify strategies
python run_backtest_enhanced.py

# Manual trading session
python automated_daily_trader.py
```

## 📞 Support

### Log Files:
- Daily trading: `logs/daily_trading/`
- System logs: `logs/`
- Error logs: Check Task Scheduler event logs

### Key Commands:
```bash
# View scheduled task
schtasks /query /tn "LAEF Daily Trading Bot"

# Delete scheduled task
schtasks /delete /tn "LAEF Daily Trading Bot" /f

# Check system status
python check_system_status.py

# View evolution progress
python live_monitoring_dashboard.py
```

---

## 🎯 Expected Evolution Timeline

### Week 1: Learning Phase
- High epsilon (exploration)
- Lower accuracy (~40-60%)
- Strategy weight balancing
- Pattern discovery

### Week 2-4: Adaptation Phase
- Decreasing epsilon
- Improving accuracy (~60-75%)
- Strategy specialization
- Risk parameter optimization

### Month 2+: Evolution Phase
- Low epsilon (exploitation)
- High accuracy (~75%+)
- Automated strategy selection
- Advanced pattern recognition

The bot is designed to continuously evolve and improve its performance over time through reinforcement learning and pattern recognition. Daily reports will show this evolution in real-time.

**Remember: Start with paper trading to verify everything works before considering live trading!**