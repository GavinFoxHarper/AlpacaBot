# LAEF Trading Platform

LAEF (Learning-Augmented Equity Framework) is an advanced algorithmic trading platform that combines machine learning with traditional trading strategies.

## Features

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

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/laef-trading.git
cd laef-trading
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run the trading system:
```bash
python main_trading.py              # Main trading interface
python start_laef_interactive.py    # Interactive mode
```

## Configuration

- Use `config_profiles/` directory for custom trading configurations
- Key parameters in default.json:
  - q_buy/q_sell: AI confidence thresholds
  - momentum_threshold: Price momentum triggers
  - position_size: Risk management limits
  - reward_type: Risk-adjusted reward calculation method
  - market_regime_weights: Strategy weights per market regime

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation and implementation details.

## Security Notes

- Never commit API keys or credentials
- Use environment variables for sensitive configuration
- Follow security best practices for financial data handling
- Implement proper error handling and logging

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.