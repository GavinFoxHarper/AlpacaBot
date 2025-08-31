"""
Enhanced Backtest Engine - Full Implementation
Connects the sophisticated trading logic to actual market data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedBacktestEngine:
    """Full implementation of backtesting with actual trading logic"""
    
    def __init__(self, initial_cash=50000, custom_config=None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.custom_config = custom_config or {}
        
        # Initialize portfolio tracking
        self.positions = {}  # {symbol: {'shares': n, 'avg_price': p}}
        self.trades = []
        self.portfolio_history = []
        self.decision_log = []
        self.symbol_pnl = {}  # Track P&L by symbol
        
        # Trading engine configuration
        from core.portfolio_manager import FIFOPortfolio
        from trading.hybrid_trading_engine import HybridTradingEngine
        
        self.portfolio = FIFOPortfolio(initial_cash)  # Pass initial_cash to constructor
        self.engine = HybridTradingEngine(self.portfolio, custom_config)
        
        # Technical indicators calculator - simplified for now
        # We'll calculate indicators directly in the methods
        
        # Q-learning and ML simulators (simplified for backtest)
        self.q_learning_simulator = QLearningSimulator()
        self.ml_simulator = MLSimulator()
        
        logger.info(f"Enhanced Backtest Engine initialized with ${initial_cash:,}")
    
    def fetch_market_data(self, symbols, start_date, end_date):
        """Fetch actual market data from Yahoo Finance"""
        market_data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Calculate technical indicators
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = self.calculate_rsi(df['Close'])
                df['MACD'] = self.calculate_macd(df['Close'])
                df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
                
                market_data[symbol] = df
                logger.info(f"Loaded {len(df)} days of data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return market_data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd
    
    def run_backtest(self, symbols=None, start_date=None, end_date=None, use_smart_selection=False):
        """Run full backtest with actual trading logic"""
        
        # Setup symbols
        if use_smart_selection:
            try:
                from optimization.smart_stock_selector import SmartStockSelector
                selector = SmartStockSelector()
                symbols = selector.select_stocks(limit=10)
                logger.info(f"Smart selection chose: {symbols}")
            except:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
                logger.info(f"Smart selection failed, using default: {symbols}")
        else:
            symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # Setup dates
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = start_date or (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        logger.info(f"Starting Enhanced Backtest")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Cash: ${self.initial_cash:,}")
        
        # Fetch market data
        market_data = self.fetch_market_data(symbols, start_date, end_date)
        
        if not market_data:
            logger.error("No market data available")
            return self._generate_empty_results(symbols, start_date, end_date)
        
        # Get all trading dates
        all_dates = set()
        for symbol_data in market_data.values():
            all_dates.update(symbol_data.index)
        all_dates = sorted(list(all_dates))
        
        # Run backtest day by day
        for trading_date in all_dates:
            self._process_trading_day(trading_date, symbols, market_data)
        
        # Calculate final results
        results = self._calculate_results(symbols, start_date, end_date)
        
        # Save detailed logs
        self._save_logs(results)
        
        return results
    
    def _process_trading_day(self, trading_date, symbols, market_data):
        """Process a single trading day"""
        daily_portfolio_value = self.cash
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            symbol_data = market_data[symbol]
            
            # Check if we have data for this date
            if trading_date not in symbol_data.index:
                continue
            
            # Get current day's data
            current_bar = symbol_data.loc[trading_date]
            current_price = current_bar['Close']
            
            # Skip if price is invalid
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # Get technical indicators
            indicators = {
                'rsi': current_bar.get('RSI', 50),
                'macd': current_bar.get('MACD', 0),
                'volume': current_bar.get('Volume', 0),
                'sma_20': current_bar.get('SMA_20', current_price),
                'sma_50': current_bar.get('SMA_50', current_price)
            }
            
            # Skip if indicators are not ready (NaN)
            if pd.isna(indicators['rsi']):
                indicators['rsi'] = 50
            if pd.isna(indicators['macd']):
                indicators['macd'] = 0
            
            # Simulate Q-learning value
            q_value = self.q_learning_simulator.get_q_value(
                symbol, current_price, indicators, self.positions
            )
            
            # Simulate ML confidence
            ml_confidence = self.ml_simulator.get_confidence(
                symbol, current_price, indicators
            )
            
            # Get trade decision from the engine
            decision, confidence, reason, position_data = self.engine.evaluate_trade_decision(
                symbol=symbol,
                q_value=q_value,
                ml_confidence=ml_confidence,
                indicators=indicators,
                current_price=current_price,
                current_time=trading_date,
                force_sell=False
            )
            
            # Convert reason to layman's terms for decision log
            layman_reason = self._convert_to_layman_terms(reason, decision)
            
            # Log the decision
            self.decision_log.append({
                'date': trading_date,
                'symbol': symbol,
                'price': current_price,
                'decision': decision,
                'confidence': confidence,
                'reason': layman_reason,
                'technical_reason': reason,  # Keep original for debugging
                'q_value': q_value,
                'ml_confidence': ml_confidence,
                'rsi': indicators['rsi'],
                'macd': indicators['macd']
            })
            
            # Execute trade if signaled
            if decision == 'buy':
                self._execute_buy(symbol, current_price, position_data, trading_date, reason)
            elif decision == 'sell':
                self._execute_sell(symbol, current_price, position_data, trading_date, reason)
            
            # Update portfolio value for this symbol
            if symbol in self.positions and self.positions[symbol]['shares'] > 0:
                daily_portfolio_value += self.positions[symbol]['shares'] * current_price
        
        # Record daily portfolio value
        self.portfolio_history.append({
            'date': trading_date,
            'cash': self.cash,
            'portfolio_value': daily_portfolio_value,
            'positions': len([p for p in self.positions.values() if p['shares'] > 0])
        })
    
    def _execute_buy(self, symbol, price, position_data, date, reason):
        """Execute a buy order"""
        # Calculate shares to buy
        max_investment = min(
            self.cash * 0.25,  # Max 25% of cash per trade
            position_data.get('max_investment', 5000)
        )
        
        if self.cash < max_investment:
            max_investment = self.cash * 0.9  # Use 90% of remaining cash if low
        
        if max_investment < price:
            return  # Can't afford even one share
        
        shares = int(max_investment / price)
        if shares == 0:
            return
        
        cost = shares * price * 1.001  # Include 0.1% slippage
        
        if cost > self.cash:
            shares = int(self.cash / (price * 1.001))
            if shares == 0:
                return
            cost = shares * price * 1.001
        
        # Execute buy
        self.cash -= cost
        
        if symbol in self.positions:
            # Average up
            existing_shares = self.positions[symbol]['shares']
            existing_avg = self.positions[symbol]['avg_price']
            new_shares = existing_shares + shares
            new_avg = ((existing_shares * existing_avg) + (shares * price)) / new_shares
            self.positions[symbol] = {'shares': new_shares, 'avg_price': new_avg}
        else:
            self.positions[symbol] = {'shares': shares, 'avg_price': price}
        
        # Convert reason to layman's terms
        layman_reason = self._convert_to_layman_terms(reason, 'buy')
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'cost': cost,
            'reason': layman_reason,
            'technical_reason': reason,  # Keep original for debugging
            'cash_after': self.cash,
            'total_position': self.positions[symbol]['shares'],
            'avg_price': self.positions[symbol]['avg_price']
        })
        
        logger.info(f"BUY: {shares} shares of {symbol} at ${price:.2f} - {layman_reason}")
    
    def _execute_sell(self, symbol, price, position_data, date, reason):
        """Execute a sell order"""
        if symbol not in self.positions or self.positions[symbol]['shares'] == 0:
            return
        
        shares = self.positions[symbol]['shares']
        avg_price = self.positions[symbol]['avg_price']
        
        # Calculate proceeds with slippage
        proceeds = shares * price * 0.999  # 0.1% slippage
        
        # Calculate profit/loss
        cost_basis = shares * avg_price
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        # Execute sell
        self.cash += proceeds
        
        # Update running P&L for this symbol
        if symbol not in self.symbol_pnl:
            self.symbol_pnl[symbol] = {'total_pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}
        
        self.symbol_pnl[symbol]['total_pnl'] += pnl
        self.symbol_pnl[symbol]['trades'] += 1
        if pnl > 0:
            self.symbol_pnl[symbol]['wins'] += 1
        else:
            self.symbol_pnl[symbol]['losses'] += 1
        
        # Convert reason to layman's terms
        layman_reason = self._convert_to_layman_terms(reason, 'sell', pnl, pnl_pct)
        
        # Clear position
        self.positions[symbol] = {'shares': 0, 'avg_price': 0}
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'buy_price': avg_price,
            'sell_price': price,
            'cost_basis': cost_basis,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': layman_reason,
            'technical_reason': reason,  # Keep original for debugging
            'cash_after': self.cash,
            'cumulative_pnl': self.symbol_pnl[symbol]['total_pnl']
        })
        
        logger.info(f"SELL: {shares} shares of {symbol} at ${price:.2f} - PnL: ${pnl:.2f} ({pnl_pct:.1f}%) - {layman_reason}")
    
    def _convert_to_layman_terms(self, reason, action, pnl=None, pnl_pct=None):
        """Convert technical trading reasons to layman's terms"""
        
        # Parse the reason
        reason_upper = reason.upper()
        
        if action == 'buy':
            if 'MOMENTUM' in reason_upper:
                if 'Q=' in reason:
                    q_val = reason.split('Q=')[1].split()[0]
                    return f"Strong upward momentum detected. Algorithm confidence: {float(q_val)*100:.0f}%. Good entry point for potential gains."
                return "Strong upward momentum detected. Good buying opportunity."
            
            elif 'MACD' in reason_upper:
                return "Technical indicators show bullish crossover pattern. Stock starting upward trend."
            
            elif 'OVERSOLD' in reason_upper:
                return "Stock appears undervalued based on recent selling. Potential rebound opportunity."
            
            elif 'ML' in reason_upper or 'MACHINE LEARNING' in reason_upper:
                return "AI model predicts price increase based on market patterns. High probability setup."
            
            elif 'DIP' in reason_upper:
                return "Buying the dip - stock pulled back to attractive price level."
            
            else:
                return "Technical indicators suggest good entry point for this stock."
        
        elif action == 'sell':
            profit_status = "profit" if pnl and pnl > 0 else "loss"
            pnl_str = f" (${pnl:.2f}, {pnl_pct:.1f}%)" if pnl is not None else ""
            
            if 'STOP LOSS' in reason_upper:
                return f"Stop loss triggered to limit losses. Protecting capital{pnl_str}."
            
            elif 'PROFIT TARGET' in reason_upper or 'TAKE PROFIT' in reason_upper:
                return f"Profit target reached! Locking in gains{pnl_str}."
            
            elif 'TRAILING STOP' in reason_upper:
                return f"Trailing stop hit - securing profits as price pulled back{pnl_str}."
            
            elif 'CONVICTION LOST' in reason_upper:
                return f"Algorithm confidence dropped. Exiting position to preserve capital{pnl_str}."
            
            elif 'ML PROFIT PEAK' in reason_upper:
                return f"AI model detects potential price peak. Taking profits{pnl_str}."
            
            elif 'MAX HOLD TIME' in reason_upper:
                return f"Maximum holding period reached. Risk management exit{pnl_str}."
            
            elif 'OVERBOUGHT' in reason_upper:
                return f"Stock appears overextended. Taking profits before potential pullback{pnl_str}."
            
            elif 'FORCE' in reason_upper:
                return f"End of trading session - closing all positions{pnl_str}."
            
            else:
                return f"Exit signal triggered. Closing position with {profit_status}{pnl_str}."
        
        else:  # hold
            if 'NO ENTRY SIGNAL' in reason_upper:
                return "Waiting for better opportunity. No clear buy signal yet."
            elif 'DAY TRADE' in reason_upper or 'SWING' in reason_upper:
                return "Holding position - indicators suggest more upside potential."
            else:
                return "No action needed - maintaining current position."
    
    def _calculate_results(self, symbols, start_date, end_date):
        """Calculate backtest results"""
        # Calculate final portfolio value
        final_value = self.cash
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                # Get last known price (simplified - should use actual last price)
                final_value += position['shares'] * position['avg_price']
        
        # Calculate metrics
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100
        
        # Analyze trades
        total_trades = len([t for t in self.trades if t['action'] == 'SELL'])
        winning_trades = len([t for t in self.trades if t['action'] == 'SELL' and t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['action'] == 'SELL' and t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average profit/loss
        profits = [t['pnl'] for t in self.trades if t['action'] == 'SELL' and t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['action'] == 'SELL' and t['pnl'] < 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Calculate Sharpe ratio (simplified)
        if self.portfolio_history:
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_val = self.portfolio_history[i-1]['portfolio_value']
                curr_val = self.portfolio_history[i]['portfolio_value']
                if prev_val > 0:
                    daily_return = (curr_val - prev_val) / prev_val
                    returns.append(daily_return)
            
            if returns:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        results = {
            'status': 'completed',
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'total_decisions': len(self.decision_log),
            'buy_signals': len([d for d in self.decision_log if d['decision'] == 'buy']),
            'sell_signals': len([d for d in self.decision_log if d['decision'] == 'sell']),
            'hold_signals': len([d for d in self.decision_log if d['decision'] == 'hold'])
        }
        
        return results
    
    def _generate_empty_results(self, symbols, start_date, end_date):
        """Generate empty results when no data available"""
        return {
            'status': 'no_data',
            'initial_cash': self.initial_cash,
            'final_value': self.initial_cash,
            'total_return': 0,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': 0,
            'win_rate': 0
        }
    
    def _save_logs(self, results):
        """Save detailed logs and reports with enhanced P&L tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Save enhanced trades with P&L
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            
            # Add running total P&L column
            all_sells = trades_df[trades_df['action'] == 'SELL']
            if not all_sells.empty:
                trades_df['running_total_pnl'] = 0
                running_pnl = 0
                for idx, row in trades_df.iterrows():
                    if row['action'] == 'SELL':
                        running_pnl += row['pnl']
                    trades_df.at[idx, 'running_total_pnl'] = running_pnl
            
            trades_file = reports_dir / f"enhanced_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Enhanced trades with P&L saved to {trades_file}")
        
        # Save P&L summary by symbol
        if self.symbol_pnl:
            pnl_summary = []
            grand_total_pnl = 0
            
            for symbol, data in self.symbol_pnl.items():
                win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                pnl_summary.append({
                    'symbol': symbol,
                    'total_trades': data['trades'],
                    'winning_trades': data['wins'],
                    'losing_trades': data['losses'],
                    'win_rate': f"{win_rate:.1f}%",
                    'total_pnl': data['total_pnl'],
                    'avg_pnl_per_trade': data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0
                })
                grand_total_pnl += data['total_pnl']
            
            # Add grand total row
            pnl_summary.append({
                'symbol': 'GRAND TOTAL',
                'total_trades': sum(d['trades'] for d in self.symbol_pnl.values()),
                'winning_trades': sum(d['wins'] for d in self.symbol_pnl.values()),
                'losing_trades': sum(d['losses'] for d in self.symbol_pnl.values()),
                'win_rate': f"{(sum(d['wins'] for d in self.symbol_pnl.values()) / sum(d['trades'] for d in self.symbol_pnl.values()) * 100):.1f}%" if sum(d['trades'] for d in self.symbol_pnl.values()) > 0 else "0%",
                'total_pnl': grand_total_pnl,
                'avg_pnl_per_trade': grand_total_pnl / sum(d['trades'] for d in self.symbol_pnl.values()) if sum(d['trades'] for d in self.symbol_pnl.values()) > 0 else 0
            })
            
            pnl_df = pd.DataFrame(pnl_summary)
            pnl_file = reports_dir / f"pnl_summary_{timestamp}.csv"
            pnl_df.to_csv(pnl_file, index=False)
            logger.info(f"P&L summary by symbol saved to {pnl_file}")
            
            # Print summary to console
            print("\n" + "="*60)
            print("PROFIT & LOSS SUMMARY BY SYMBOL")
            print("="*60)
            for row in pnl_summary:
                if row['symbol'] == 'GRAND TOTAL':
                    print("-"*60)
                print(f"{row['symbol']:<12} | Trades: {row['total_trades']:>3} | Win Rate: {row['win_rate']:>6} | P&L: ${row['total_pnl']:>10.2f}")
            print("="*60 + "\n")
        
        # Save decision log with layman terms
        if self.decision_log:
            decisions_df = pd.DataFrame(self.decision_log)
            decisions_file = logs_dir / f"decisions_{timestamp}.csv"
            decisions_df.to_csv(decisions_file, index=False)
            logger.info(f"Decisions with explanations saved to {decisions_file}")
        
        # Save portfolio history
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_file = reports_dir / f"portfolio_history_{timestamp}.csv"
            portfolio_df.to_csv(portfolio_file, index=False)
            logger.info(f"Portfolio history saved to {portfolio_file}")
        
        # Enhanced results with P&L data
        results['symbol_pnl'] = self.symbol_pnl
        results['grand_total_pnl'] = sum(data['total_pnl'] for data in self.symbol_pnl.values()) if self.symbol_pnl else 0
        
        # Save results summary as JSON
        results_file = reports_dir / f"backtest_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Enhanced results saved to {results_file}")
        
        # Save results as Excel with multiple sheets
        excel_file = reports_dir / f"backtest_results_{timestamp}.xlsx"
        self._save_excel_report(excel_file, results, timestamp)
        logger.info(f"Excel report saved to {excel_file}")
    
    def _save_excel_report(self, filepath, results, timestamp):
        """Save comprehensive Excel report with multiple sheets"""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Summary Metrics
                summary_data = {
                    'Metric': [
                        'Total Return (%)', 
                        'Annual Return (%)',
                        'Volatility (%)',
                        'Sharpe Ratio',
                        'Max Drawdown (%)',
                        'Win Rate (%)',
                        'Total Trades',
                        'Grand Total P&L ($)',
                        'Initial Capital ($)',
                        'Final Value ($)'
                    ],
                    'Value': [
                        results.get('total_return', 0),
                        results.get('annual_return', 0),
                        results.get('volatility', 0),
                        results.get('sharpe_ratio', 0),
                        results.get('max_drawdown', 0),
                        results.get('win_rate', 0),
                        results.get('total_trades', 0),
                        results.get('grand_total_pnl', 0),
                        self.initial_cash,
                        results.get('final_value', 0)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Trades
                if self.trades:
                    trades_df = pd.DataFrame(self.trades)
                    # Remove timezone info from datetime columns for Excel compatibility
                    if 'date' in trades_df.columns:
                        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.tz_localize(None)
                    # Ensure proper column order
                    trade_cols = ['date', 'symbol', 'action', 'shares', 'price', 
                                 'commission', 'reason', 'value']
                    trades_df = trades_df[[col for col in trade_cols if col in trades_df.columns]]
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Sheet 3: Decision Log
                if self.decision_log:
                    decisions_df = pd.DataFrame(self.decision_log)
                    # Remove timezone info from datetime columns
                    if 'date' in decisions_df.columns:
                        decisions_df['date'] = pd.to_datetime(decisions_df['date']).dt.tz_localize(None)
                    decisions_df.to_excel(writer, sheet_name='Decisions', index=False)
                
                # Sheet 4: Portfolio History
                if self.portfolio_history:
                    portfolio_df = pd.DataFrame(self.portfolio_history)
                    # Remove timezone info from datetime columns
                    if 'date' in portfolio_df.columns:
                        portfolio_df['date'] = pd.to_datetime(portfolio_df['date']).dt.tz_localize(None)
                    portfolio_df.to_excel(writer, sheet_name='Portfolio History', index=False)
                
                # Sheet 5: Symbol P&L Analysis
                if results.get('symbol_pnl'):
                    pnl_data = []
                    for symbol, data in results['symbol_pnl'].items():
                        pnl_data.append({
                            'Symbol': symbol,
                            'Realized P&L': data.get('realized_pnl', 0),
                            'Unrealized P&L': data.get('unrealized_pnl', 0),
                            'Total P&L': data.get('total_pnl', 0),
                            'Trade Count': data.get('trade_count', 0)
                        })
                    if pnl_data:
                        pnl_df = pd.DataFrame(pnl_data)
                        pnl_df.to_excel(writer, sheet_name='Symbol PnL', index=False)
                
                # Format the Excel file
                workbook = writer.book
                for worksheet in workbook.worksheets:
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    # Format headers
                    for cell in worksheet[1]:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
                        cell.alignment = Alignment(horizontal="center")
                        
        except Exception as e:
            logger.error(f"Error saving Excel report: {e}")
            # Continue without failing the entire backtest


class QLearningSimulator:
    """Simulates Q-learning values for backtesting"""
    
    def get_q_value(self, symbol, price, indicators, positions):
        """Generate Q-value based on market conditions"""
        # Simplified Q-value calculation
        q_value = 0.5  # Base value
        
        # Adjust based on RSI
        if indicators['rsi'] < 30:
            q_value += 0.2  # Oversold bonus
        elif indicators['rsi'] > 70:
            q_value -= 0.2  # Overbought penalty
        
        # Adjust based on MACD
        if indicators['macd'] > 0:
            q_value += 0.1
        else:
            q_value -= 0.1
        
        # Adjust based on position
        if symbol in positions and positions[symbol]['shares'] > 0:
            q_value -= 0.15  # Already have position
        
        # Add some randomness for exploration
        q_value += np.random.normal(0, 0.05)
        
        return np.clip(q_value, 0, 1)


class MLSimulator:
    """Simulates ML confidence for backtesting"""
    
    def get_confidence(self, symbol, price, indicators):
        """Generate ML confidence score"""
        # Simplified ML confidence
        confidence = 0.5  # Base confidence
        
        # Pattern recognition simulation
        if indicators['rsi'] < 35 and indicators['macd'] > 0:
            confidence += 0.2  # Bullish divergence
        
        if indicators['sma_20'] > indicators['sma_50']:
            confidence += 0.1  # Uptrend
        
        # Volume confirmation
        if indicators['volume'] > 0:
            confidence += 0.05
        
        # Add some noise
        confidence += np.random.normal(0, 0.1)
        
        return np.clip(confidence, 0, 1)