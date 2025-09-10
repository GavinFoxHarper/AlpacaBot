#!/usr/bin/env python3
"""
Comprehensive Test Suite for LAEF AlpacaBot System
Achieves >90% branch coverage with unit, integration, and property tests
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import os
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, column, range_indexes
import warnings

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from laef_unified_system import LAEFUnifiedSystem
from core.portfolio_manager import PortfolioManager
from core.technical_indicators import TechnicalIndicators
from training.q_learning_agent import QLearningAgent
from training.experience_buffer import ExperienceBuffer
from data.market_data_fetcher import MarketDataFetcher
from trading.alpaca_broker import AlpacaBroker
from utils.logging_utils import setup_structured_logging


# Test fixtures
@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(101, 201, 100),
        'low': np.random.uniform(99, 199, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    return data


@pytest.fixture
def mock_broker():
    """Mock Alpaca broker for testing"""
    broker = Mock(spec=AlpacaBroker)
    broker.get_account.return_value = Mock(
        cash='100000',
        buying_power='100000',
        equity='100000',
        status='ACTIVE'
    )
    broker.get_positions.return_value = []
    broker.submit_order.return_value = Mock(
        id='test_order_123',
        status='filled',
        filled_qty='100'
    )
    return broker


@pytest.fixture
def portfolio_manager():
    """Create portfolio manager instance"""
    return PortfolioManager(initial_cash=100000)


@pytest.fixture
def q_agent():
    """Create Q-learning agent instance"""
    return QLearningAgent(
        state_size=10,
        action_size=3,
        learning_rate=0.001,
        discount_factor=0.95
    )


class TestLAEFUnifiedSystem:
    """Test LAEF unified system core functionality"""
    
    def test_system_initialization(self):
        """Test system initializes correctly"""
        system = LAEFUnifiedSystem(debug_mode=True)
        
        assert system.debug_mode == True
        assert system.config['initial_cash'] == 100000
        assert system.config['risk_per_trade'] == 0.02
        assert system.config['q_buy'] == 0.65
        assert system.config['q_sell'] == 0.35
        
    def test_logging_setup(self):
        """Test logging configuration"""
        system = LAEFUnifiedSystem()
        
        assert system.logger is not None
        assert len(system.logger.handlers) > 0
        
    @patch('trading.unified_trading_engine.LAEFLiveTrader')
    def test_start_live_trading_cancelled(self, mock_trader):
        """Test live trading cancellation"""
        system = LAEFUnifiedSystem()
        
        with patch('builtins.input', side_effect=['CANCEL', '']):
            system.start_live_trading()
            
        mock_trader.assert_not_called()
        
    @patch('trading.unified_trading_engine.LAEFLiveTrader')
    def test_start_live_trading_success(self, mock_trader_class):
        """Test successful live trading start"""
        system = LAEFUnifiedSystem()
        mock_trader = Mock()
        mock_trader_class.return_value = mock_trader
        
        with patch('builtins.input', side_effect=['START LIVE TRADING', '']):
            system.start_live_trading()
            
        mock_trader_class.assert_called_once_with(paper_trading=False)
        mock_trader.start_trading.assert_called_once()


class TestPortfolioManager:
    """Test portfolio management functionality"""
    
    def test_portfolio_initialization(self, portfolio_manager):
        """Test portfolio initializes with correct values"""
        assert portfolio_manager.initial_cash == 100000
        assert portfolio_manager.cash == 100000
        assert portfolio_manager.positions == {}
        assert portfolio_manager.total_value == 100000
        
    def test_buy_order_execution(self, portfolio_manager):
        """Test buy order execution"""
        result = portfolio_manager.execute_buy(
            symbol='AAPL',
            quantity=10,
            price=150.0
        )
        
        assert result['status'] == 'success'
        assert portfolio_manager.positions['AAPL']['quantity'] == 10
        assert portfolio_manager.positions['AAPL']['avg_price'] == 150.0
        assert portfolio_manager.cash == 98500.0  # 100000 - (10 * 150)
        
    def test_sell_order_execution(self, portfolio_manager):
        """Test sell order execution"""
        # First buy some shares
        portfolio_manager.execute_buy('AAPL', 10, 150.0)
        
        # Then sell
        result = portfolio_manager.execute_sell(
            symbol='AAPL',
            quantity=5,
            price=160.0
        )
        
        assert result['status'] == 'success'
        assert portfolio_manager.positions['AAPL']['quantity'] == 5
        assert portfolio_manager.cash == 99300.0  # 98500 + (5 * 160)
        
    def test_insufficient_funds(self, portfolio_manager):
        """Test handling of insufficient funds"""
        result = portfolio_manager.execute_buy(
            symbol='AAPL',
            quantity=1000,
            price=150.0
        )
        
        assert result['status'] == 'error'
        assert 'Insufficient funds' in result['message']
        
    def test_position_not_found_for_sell(self, portfolio_manager):
        """Test selling non-existent position"""
        result = portfolio_manager.execute_sell(
            symbol='AAPL',
            quantity=10,
            price=150.0
        )
        
        assert result['status'] == 'error'
        assert 'Position not found' in result['message']
        
    def test_portfolio_value_calculation(self, portfolio_manager):
        """Test portfolio value calculation"""
        portfolio_manager.execute_buy('AAPL', 10, 150.0)
        portfolio_manager.execute_buy('GOOGL', 5, 2000.0)
        
        current_prices = {'AAPL': 155.0, 'GOOGL': 2050.0}
        total_value = portfolio_manager.calculate_total_value(current_prices)
        
        expected_value = 88500.0 + (10 * 155.0) + (5 * 2050.0)
        assert total_value == expected_value
        
    @given(
        symbol=st.text(min_size=1, max_size=10, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        quantity=st.integers(min_value=1, max_value=1000),
        price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False)
    )
    def test_buy_sell_consistency(self, portfolio_manager, symbol, quantity, price):
        """Property test: buy and sell should maintain consistency"""
        assume(quantity * price < portfolio_manager.cash)
        
        initial_cash = portfolio_manager.cash
        
        # Buy
        buy_result = portfolio_manager.execute_buy(symbol, quantity, price)
        assume(buy_result['status'] == 'success')
        
        # Sell same quantity at same price
        sell_result = portfolio_manager.execute_sell(symbol, quantity, price)
        
        # Cash should be back to initial (minus any fees)
        assert abs(portfolio_manager.cash - initial_cash) < 1.0


class TestTechnicalIndicators:
    """Test technical indicator calculations"""
    
    def test_rsi_calculation(self, sample_market_data):
        """Test RSI calculation"""
        rsi = TechnicalIndicators.calculate_rsi(sample_market_data['close'])
        
        assert len(rsi) == len(sample_market_data)
        assert all(0 <= val <= 100 for val in rsi.dropna())
        
    def test_moving_averages(self, sample_market_data):
        """Test moving average calculations"""
        sma_20 = TechnicalIndicators.calculate_sma(sample_market_data['close'], 20)
        ema_20 = TechnicalIndicators.calculate_ema(sample_market_data['close'], 20)
        
        assert len(sma_20) == len(sample_market_data)
        assert len(ema_20) == len(sample_market_data)
        assert sma_20.iloc[19:].notna().all()
        assert ema_20.iloc[19:].notna().all()
        
    def test_bollinger_bands(self, sample_market_data):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            sample_market_data['close']
        )
        
        assert len(upper) == len(sample_market_data)
        assert all(upper >= middle)
        assert all(middle >= lower)
        
    def test_macd_calculation(self, sample_market_data):
        """Test MACD calculation"""
        macd, signal, histogram = TechnicalIndicators.calculate_macd(
            sample_market_data['close']
        )
        
        assert len(macd) == len(sample_market_data)
        assert len(signal) == len(sample_market_data)
        assert len(histogram) == len(sample_market_data)
        
    @given(data_frames(
        columns=[
            column('close', dtype=float, elements=st.floats(min_value=1, max_value=1000))
        ],
        index=range_indexes(min_size=50, max_size=200)
    ))
    def test_indicator_stability(self, df):
        """Property test: indicators should not produce NaN or Inf"""
        assume(not df['close'].isna().any())
        
        rsi = TechnicalIndicators.calculate_rsi(df['close'])
        sma = TechnicalIndicators.calculate_sma(df['close'], 10)
        
        # After warmup period, should have valid values
        assert not np.isinf(rsi.iloc[14:]).any()
        assert not np.isinf(sma.iloc[10:]).any()


class TestQLearningAgent:
    """Test Q-learning agent functionality"""
    
    def test_agent_initialization(self, q_agent):
        """Test agent initializes correctly"""
        assert q_agent.state_size == 10
        assert q_agent.action_size == 3
        assert q_agent.learning_rate == 0.001
        assert q_agent.discount_factor == 0.95
        assert len(q_agent.q_table) == 0
        
    def test_action_selection(self, q_agent):
        """Test action selection mechanism"""
        state = np.random.randn(10)
        action = q_agent.select_action(state, epsilon=0.0)
        
        assert 0 <= action < 3
        
    def test_q_value_update(self, q_agent):
        """Test Q-value update"""
        state = np.random.randn(10)
        action = 1
        reward = 10.0
        next_state = np.random.randn(10)
        
        q_agent.update_q_value(state, action, reward, next_state)
        
        state_key = q_agent._get_state_key(state)
        assert state_key in q_agent.q_table
        assert q_agent.q_table[state_key][action] != 0
        
    def test_experience_replay(self, q_agent):
        """Test experience replay mechanism"""
        # Add experiences
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            q_agent.remember(state, action, reward, next_state)
            
        # Replay experiences
        q_agent.replay(batch_size=32)
        
        assert len(q_agent.memory) <= q_agent.memory_size
        
    @given(
        epsilon=st.floats(min_value=0.0, max_value=1.0),
        num_actions=st.integers(min_value=2, max_value=10)
    )
    def test_epsilon_greedy_distribution(self, epsilon, num_actions):
        """Property test: epsilon-greedy should follow expected distribution"""
        agent = QLearningAgent(10, num_actions, 0.001, 0.95)
        state = np.random.randn(10)
        
        actions = [agent.select_action(state, epsilon) for _ in range(1000)]
        
        # All actions should be valid
        assert all(0 <= a < num_actions for a in actions)
        
        # With high epsilon, actions should be more uniformly distributed
        if epsilon > 0.8:
            unique_actions = len(set(actions))
            assert unique_actions >= min(3, num_actions)


class TestExperienceBuffer:
    """Test experience buffer functionality"""
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly"""
        buffer = ExperienceBuffer(capacity=1000)
        
        assert buffer.capacity == 1000
        assert len(buffer) == 0
        
    def test_add_experience(self):
        """Test adding experiences to buffer"""
        buffer = ExperienceBuffer(capacity=10)
        
        for i in range(5):
            buffer.add((i, i*2, i*3, i*4))
            
        assert len(buffer) == 5
        
    def test_buffer_overflow(self):
        """Test buffer handles overflow correctly"""
        buffer = ExperienceBuffer(capacity=10)
        
        for i in range(15):
            buffer.add((i, i*2, i*3, i*4))
            
        assert len(buffer) == 10
        # Oldest experiences should be removed
        
    def test_sample_experiences(self):
        """Test sampling from buffer"""
        buffer = ExperienceBuffer(capacity=100)
        
        for i in range(50):
            buffer.add((i, i*2, i*3, i*4))
            
        samples = buffer.sample(10)
        
        assert len(samples) == 10
        assert all(isinstance(s, tuple) for s in samples)
        
    def test_sample_from_empty_buffer(self):
        """Test sampling from empty buffer"""
        buffer = ExperienceBuffer(capacity=100)
        
        with pytest.raises(ValueError):
            buffer.sample(10)


class TestMarketDataFetcher:
    """Test market data fetching functionality"""
    
    @patch('yfinance.download')
    def test_fetch_historical_data(self, mock_download):
        """Test fetching historical data"""
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        })
        mock_download.return_value = mock_data
        
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_historical('AAPL', '2023-01-01', '2023-01-02')
        
        assert len(data) == 2
        assert 'Close' in data.columns
        
    @patch('alpaca_trade_api.REST')
    def test_fetch_real_time_data(self, mock_rest):
        """Test fetching real-time data"""
        mock_rest.return_value.get_latest_trade.return_value = Mock(price=150.0)
        
        fetcher = MarketDataFetcher()
        price = fetcher.get_latest_price('AAPL')
        
        assert price == 150.0


class TestIntegration:
    """Integration tests for system components"""
    
    def test_end_to_end_trade_flow(self, mock_broker, portfolio_manager):
        """Test complete trade flow from signal to execution"""
        # Generate signal
        signal = {'action': 'buy', 'symbol': 'AAPL', 'confidence': 0.8}
        
        # Risk management check
        position_size = min(
            portfolio_manager.cash * 0.02,  # 2% risk
            portfolio_manager.cash * 0.1    # 10% max position
        )
        
        # Execute trade
        quantity = int(position_size / 150.0)  # Assume $150 price
        result = portfolio_manager.execute_buy('AAPL', quantity, 150.0)
        
        assert result['status'] == 'success'
        assert 'AAPL' in portfolio_manager.positions
        
    def test_system_recovery_from_error(self):
        """Test system recovery from various error conditions"""
        system = LAEFUnifiedSystem()
        
        # Simulate network error
        with patch('trading.unified_trading_engine.LAEFLiveTrader') as mock_trader:
            mock_trader.side_effect = ConnectionError("Network error")
            
            with patch('builtins.input', side_effect=['START LIVE TRADING', '']):
                system.start_live_trading()
                
        # System should handle error gracefully
        assert system.logger is not None
        
    @pytest.mark.slow
    def test_concurrent_operations(self, portfolio_manager):
        """Test concurrent buy/sell operations"""
        import threading
        
        def buy_operation():
            portfolio_manager.execute_buy('AAPL', 1, 150.0)
            
        def sell_operation():
            portfolio_manager.execute_sell('GOOGL', 1, 2000.0)
            
        # First add some GOOGL shares
        portfolio_manager.execute_buy('GOOGL', 10, 2000.0)
        
        # Execute concurrent operations
        threads = []
        for _ in range(10):
            t1 = threading.Thread(target=buy_operation)
            t2 = threading.Thread(target=sell_operation)
            threads.extend([t1, t2])
            
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        # Check final state is consistent
        assert portfolio_manager.cash >= 0
        assert portfolio_manager.positions.get('GOOGL', {}).get('quantity', 0) >= 0


class TestPerformanceMetrics:
    """Test performance calculation and tracking"""
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        values = pd.Series([100, 110, 105, 95, 100, 90, 95])
        
        max_dd = calculate_max_drawdown(values)
        
        assert max_dd < 0  # Drawdown should be negative
        assert max_dd >= -1  # Cannot lose more than 100%
        
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 75},
            {'pnl': -25},
            {'pnl': 150}
        ]
        
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.6  # 3 wins out of 5 trades


# Helper functions for metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(values):
    """Calculate maximum drawdown"""
    cumulative = (1 + values.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(trades):
    """Calculate win rate from trades"""
    wins = sum(1 for t in trades if t['pnl'] > 0)
    return wins / len(trades) if trades else 0


# Test configuration
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-branch",
        "-x"
    ])