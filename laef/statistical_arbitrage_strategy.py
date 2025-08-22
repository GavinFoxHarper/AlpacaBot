"""
Statistical Arbitrage Strategy for LAEF Trading Platform

Implements various statistical arbitrage methods:
1. Pairs Trading (Cointegration-based)
2. Mean Reversion on Spreads
3. Factor-based Statistical Arbitrage
4. Volatility Arbitrage
5. Cross-sectional Momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression

class ArbType(Enum):
    PAIRS = "pairs_trading"
    SPREAD = "spread_trading"
    FACTOR = "factor_based"
    VOLATILITY = "volatility_arb"
    CROSS_SECTIONAL = "cross_sectional"

@dataclass
class ArbSignal:
    """Statistical arbitrage trading signal"""
    arb_type: ArbType
    long_symbols: Set[str]  # Symbols to long
    short_symbols: Set[str]  # Symbols to short
    hedge_ratio: float      # Hedge ratio for pairs
    zscore: float          # Current z-score
    expected_return: float
    holding_period: int    # Expected holding period in minutes
    confidence: float
    timestamp: datetime

class StatisticalArbitrageStrategy:
    """Statistical arbitrage strategy implementation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 60)  # periods for statistics
        self.zscore_threshold = config.get('zscore_threshold', 2.0)  # entry threshold
        self.zscore_exit = config.get('zscore_exit', 0.5)  # exit threshold
        self.min_correlation = config.get('min_correlation', 0.7)
        self.min_cointegration = config.get('min_cointegration', 0.05)  # p-value threshold
        
        # Position sizing
        self.max_positions = config.get('max_pairs_positions', 5)
        self.position_size = config.get('position_size', 0.1)  # 10% per pair
        
        # Market variables
        self.sectors = self._initialize_sectors()
        self.factor_betas = {}  # symbol -> factor loadings
        self.pair_correlations = {}  # (symbol1, symbol2) -> correlation
        self.cointegration_tests = {}  # (symbol1, symbol2) -> p-value
        
    def analyze_opportunities(self, market_data: Dict[str, pd.DataFrame],
                            timestamp: datetime = None) -> List[ArbSignal]:
        """
        Analyze statistical arbitrage opportunities
        
        Args:
            market_data: Dict of price data for each symbol
            timestamp: Current timestamp
            
        Returns:
            List of arbitrage signals
        """
        try:
            timestamp = timestamp or datetime.now()
            signals = []
            
            # 1. Pairs Trading Signals
            pairs_signals = self._analyze_pairs_trading(market_data)
            signals.extend(pairs_signals)
            
            # 2. Spread Trading Signals
            spread_signals = self._analyze_spread_trading(market_data)
            signals.extend(spread_signals)
            
            # 3. Factor-based Signals
            factor_signals = self._analyze_factor_arbitrage(market_data)
            signals.extend(factor_signals)
            
            # 4. Volatility Arbitrage Signals
            vol_signals = self._analyze_volatility_arbitrage(market_data)
            signals.extend(vol_signals)
            
            # 5. Cross-sectional Momentum
            momentum_signals = self._analyze_cross_sectional(market_data)
            signals.extend(momentum_signals)
            
            # Rank and filter signals
            valid_signals = self._filter_signals(signals)
            ranked_signals = self._rank_signals(valid_signals)
            
            return ranked_signals
            
        except Exception as e:
            self.logger.error(f"Statistical arbitrage analysis failed: {e}")
            return []
            
    def _analyze_pairs_trading(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbSignal]:
        """Analyze pairs trading opportunities"""
        try:
            pairs_signals = []
            
            # Get all symbol combinations
            symbols = list(market_data.keys())
            symbol_pairs = [
                (s1, s2) for i, s1 in enumerate(symbols)
                for s2 in symbols[i+1:]
            ]
            
            for symbol1, symbol2 in symbol_pairs:
                # Get price data
                data1 = market_data[symbol1]['close']
                data2 = market_data[symbol2]['close']
                
                if len(data1) < self.lookback_period or len(data2) < self.lookback_period:
                    continue
                    
                # Calculate correlation
                correlation = data1.corr(data2)
                self.pair_correlations[(symbol1, symbol2)] = correlation
                
                if correlation > self.min_correlation:
                    # Test for cointegration
                    pvalue = self._test_cointegration(data1, data2)
                    self.cointegration_tests[(symbol1, symbol2)] = pvalue
                    
                    if pvalue < self.min_cointegration:
                        # Calculate hedge ratio
                        hedge_ratio = self._calculate_hedge_ratio(data1, data2)
                        
                        # Calculate spread
                        spread = data1 - hedge_ratio * data2
                        zscore = self._calculate_zscore(spread)
                        
                        # Generate signal if spread is significant
                        if abs(zscore) > self.zscore_threshold:
                            signal = ArbSignal(
                                arb_type=ArbType.PAIRS,
                                long_symbols={symbol1} if zscore < 0 else {symbol2},
                                short_symbols={symbol2} if zscore < 0 else {symbol1},
                                hedge_ratio=hedge_ratio,
                                zscore=zscore,
                                expected_return=abs(zscore) * 0.01,  # 1% per zscore
                                holding_period=60,  # 1 hour default
                                confidence=min(0.9, correlation),
                                timestamp=datetime.now()
                            )
                            pairs_signals.append(signal)
                            
            return pairs_signals
            
        except Exception as e:
            self.logger.error(f"Pairs trading analysis failed: {e}")
            return []
            
    def _analyze_spread_trading(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbSignal]:
        """Analyze spread trading opportunities"""
        try:
            spread_signals = []
            
            # Group symbols by sector
            sector_symbols = self._group_by_sector(market_data.keys())
            
            for sector, symbols in sector_symbols.items():
                if len(symbols) < 2:
                    continue
                    
                # Calculate sector index
                sector_prices = pd.DataFrame({
                    sym: market_data[sym]['close']
                    for sym in symbols
                })
                
                sector_index = sector_prices.mean(axis=1)
                
                # Find divergences from sector
                for symbol in symbols:
                    prices = market_data[symbol]['close']
                    
                    # Calculate relative performance
                    rel_perf = prices / sector_index
                    zscore = self._calculate_zscore(rel_perf)
                    
                    if abs(zscore) > self.zscore_threshold:
                        # Create market neutral signal
                        other_symbols = set(symbols) - {symbol}
                        weights = self._calculate_hedge_weights(
                            prices, sector_prices[list(other_symbols)]
                        )
                        
                        signal = ArbSignal(
                            arb_type=ArbType.SPREAD,
                            long_symbols={symbol} if zscore < 0 else other_symbols,
                            short_symbols=other_symbols if zscore < 0 else {symbol},
                            hedge_ratio=1.0,
                            zscore=zscore,
                            expected_return=abs(zscore) * 0.008,  # 0.8% per zscore
                            holding_period=120,  # 2 hours default
                            confidence=0.7,
                            timestamp=datetime.now()
                        )
                        spread_signals.append(signal)
                        
            return spread_signals
            
        except Exception as e:
            self.logger.error(f"Spread trading analysis failed: {e}")
            return []
            
    def _analyze_factor_arbitrage(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbSignal]:
        """Analyze factor-based arbitrage opportunities"""
        try:
            factor_signals = []
            
            # Calculate factor returns (simplified)
            factors = self._calculate_factor_returns(market_data)
            
            # Update factor betas
            self._update_factor_betas(market_data, factors)
            
            # Find factor divergences
            for symbol, data in market_data.items():
                if symbol not in self.factor_betas:
                    continue
                    
                # Calculate expected return from factors
                expected_return = self._calculate_factor_expected_return(
                    symbol, factors
                )
                
                # Compare to actual return
                actual_return = data['close'].pct_change().iloc[-1]
                
                # Calculate divergence
                divergence = actual_return - expected_return
                zscore = divergence / np.std(data['close'].pct_change())
                
                if abs(zscore) > self.zscore_threshold:
                    signal = ArbSignal(
                        arb_type=ArbType.FACTOR,
                        long_symbols={symbol} if zscore < 0 else set(),
                        short_symbols=set() if zscore < 0 else {symbol},
                        hedge_ratio=1.0,
                        zscore=zscore,
                        expected_return=abs(divergence),
                        holding_period=240,  # 4 hours default
                        confidence=0.6,
                        timestamp=datetime.now()
                    )
                    factor_signals.append(signal)
                    
            return factor_signals
            
        except Exception as e:
            self.logger.error(f"Factor arbitrage analysis failed: {e}")
            return []
            
    def _analyze_volatility_arbitrage(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbSignal]:
        """Analyze volatility arbitrage opportunities"""
        try:
            vol_signals = []
            
            for symbol, data in market_data.items():
                # Calculate realized vs. implied vol (if available)
                realized_vol = data['close'].pct_change().std() * np.sqrt(252)
                implied_vol = self._get_implied_volatility(symbol)
                
                if implied_vol is not None:
                    # Check for vol arbitrage opportunities
                    vol_spread = implied_vol - realized_vol
                    zscore = vol_spread / np.std(data['close'].pct_change())
                    
                    if abs(zscore) > self.zscore_threshold:
                        signal = ArbSignal(
                            arb_type=ArbType.VOLATILITY,
                            long_symbols={symbol} if zscore > 0 else set(),
                            short_symbols=set() if zscore > 0 else {symbol},
                            hedge_ratio=1.0,
                            zscore=zscore,
                            expected_return=abs(vol_spread) * 0.1,
                            holding_period=480,  # 8 hours default
                            confidence=0.5,
                            timestamp=datetime.now()
                        )
                        vol_signals.append(signal)
                        
            return vol_signals
            
        except Exception as e:
            self.logger.error(f"Volatility arbitrage analysis failed: {e}")
            return []
            
    def _analyze_cross_sectional(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbSignal]:
        """Analyze cross-sectional momentum opportunities"""
        try:
            momentum_signals = []
            
            # Calculate returns for ranking
            returns = pd.DataFrame({
                symbol: data['close'].pct_change()
                for symbol, data in market_data.items()
            })
            
            # Calculate momentum scores
            momentum = self._calculate_momentum_scores(returns)
            
            # Find top/bottom performers
            top_symbols = set(momentum.nlargest(3).index)
            bottom_symbols = set(momentum.nsmallest(3).index)
            
            if top_symbols and bottom_symbols:
                signal = ArbSignal(
                    arb_type=ArbType.CROSS_SECTIONAL,
                    long_symbols=top_symbols,
                    short_symbols=bottom_symbols,
                    hedge_ratio=1.0,
                    zscore=2.0,  # Default confidence
                    expected_return=0.01,  # 1% expected
                    holding_period=1440,  # 24 hours default
                    confidence=0.6,
                    timestamp=datetime.now()
                )
                momentum_signals.append(signal)
                
            return momentum_signals
            
        except Exception as e:
            self.logger.error(f"Cross-sectional analysis failed: {e}")
            return []
            
    def _filter_signals(self, signals: List[ArbSignal]) -> List[ArbSignal]:
        """Filter arbitrage signals"""
        valid_signals = []
        
        for signal in signals:
            # Check basic validity
            if not signal.long_symbols and not signal.short_symbols:
                continue
                
            # Check expected return threshold
            min_return = self.config.get('min_expected_return', 0.005)  # 0.5%
            if signal.expected_return < min_return:
                continue
                
            valid_signals.append(signal)
            
        return valid_signals
        
    def _rank_signals(self, signals: List[ArbSignal]) -> List[ArbSignal]:
        """Rank arbitrage signals by priority"""
        if not signals:
            return []
            
        # Calculate ranking score
        for signal in signals:
            # Base score on Sharpe-like ratio
            score = signal.expected_return / (abs(signal.zscore) + 1e-6)
            
            # Adjust for strategy type
            type_weights = {
                ArbType.PAIRS: 1.0,
                ArbType.SPREAD: 0.9,
                ArbType.FACTOR: 0.8,
                ArbType.VOLATILITY: 0.7,
                ArbType.CROSS_SECTIONAL: 0.6
            }
            score *= type_weights.get(signal.arb_type, 0.5)
            
            # Adjust confidence
            signal.confidence = min(0.95, score)
            
        # Sort by score
        ranked_signals = sorted(
            signals,
            key=lambda x: x.confidence,
            reverse=True
        )
        
        return ranked_signals
        
    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> float:
        """Test for cointegration between two price series"""
        try:
            # Run ADF test on spread
            spread = series1 - series2
            result = sm.tsa.stattools.adfuller(spread)
            return result[1]  # Return p-value
        except:
            return 1.0  # Return 1.0 if test fails
            
    def _calculate_hedge_ratio(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate hedge ratio between two series"""
        try:
            # Use linear regression
            model = LinearRegression()
            model.fit(series2.values.reshape(-1, 1), series1.values)
            return model.coef_[0]
        except:
            return 1.0
            
    def _calculate_zscore(self, series: pd.Series) -> float:
        """Calculate z-score of latest value"""
        try:
            return (series.iloc[-1] - series.mean()) / series.std()
        except:
            return 0.0
            
    def _initialize_sectors(self) -> Dict[str, List[str]]:
        """Initialize sector classifications"""
        return {
            'TECH': [],
            'FINANCE': [],
            'HEALTHCARE': [],
            'ENERGY': [],
            'CONSUMER': []
        }
        
    # Additional helper methods would be implemented here:
    # - _calculate_factor_returns()
    # - _update_factor_betas()
    # - _calculate_factor_expected_return()
    # - _get_implied_volatility()
    # - _calculate_momentum_scores()
    # - _calculate_hedge_weights()
    # - _group_by_sector()