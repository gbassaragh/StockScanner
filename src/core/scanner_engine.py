"""
Comprehensive Market Scanner Engine

A high-performance scanner engine that extends basic VWAP calculation to concurrent analysis
of 500+ securities using QThreadPool-driven architecture with sophisticated weighted ranking
algorithms, real-time update mechanisms, and integration with risk management for automated
position sizing recommendations.

Features:
- QThreadPool-based concurrent scanning of 500+ securities
- Sophisticated weighted ranking incorporating volume, price change, and ATR
- Real-time update mechanisms for continuous market monitoring
- Custom watchlist management with persistent filter criteria
- Integration with Risk Management Calculator for position sizing
- SQLite persistence for historical analysis and configuration
"""

import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import pandas as pd
import numpy as np
from PyQt6.QtCore import QObject, QThread, QThreadPool, QRunnable, pyqtSignal, QTimer, QMutex
from PyQt6.QtCore import QMutexLocker

# Import interfaces - These will be implemented by other agents
# For now, we'll define minimal interfaces to ensure compatibility
try:
    from ..models.database import DatabaseManager
except ImportError:
    # Define interface stub for development
    class DatabaseManager:
        def save_scanner_results(self, results: List[Dict]) -> bool:
            """Save scanner results to database"""
            pass
        
        def load_watchlist(self, name: str) -> List[str]:
            """Load watchlist from database"""
            return []
        
        def save_watchlist(self, name: str, symbols: List[str]) -> bool:
            """Save watchlist to database"""
            return True
        
        def get_scanner_history(self, symbol: str, days: int = 30) -> List[Dict]:
            """Get historical scanner results for a symbol"""
            return []

try:
    from .data_integration import DataIntegrationManager
except ImportError:
    # Define interface stub for development
    class DataIntegrationManager:
        def get_bulk_market_data(self, symbols: List[str], interval: str = '1min') -> Dict[str, pd.DataFrame]:
            """Get market data for multiple symbols"""
            return {}
        
        def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
            """Get real-time quote for a symbol"""
            return {}
        
        def get_premarket_data(self, symbols: List[str]) -> Dict[str, Dict]:
            """Get premarket data for symbols"""
            return {}

try:
    from .risk_calculator import RiskCalculator
except ImportError:
    # Define interface stub for development
    class RiskCalculator:
        def calculate_position_size(self, symbol: str, price: float, risk_amount: float) -> Dict[str, Any]:
            """Calculate position size for a symbol"""
            return {'shares': 0, 'dollar_amount': 0.0, 'risk_ratio': 0.0}
        
        def get_atr_adjustment(self, symbol: str, atr_value: float) -> float:
            """Get ATR-based risk adjustment"""
            return 1.0


@dataclass
class ScanCriteria:
    """Scanner criteria configuration"""
    min_price: float = 1.0
    max_price: float = 500.0
    min_volume: int = 100000
    min_dollar_volume: float = 1000000.0
    min_float: Optional[float] = None
    max_float: Optional[float] = None
    sectors: List[str] = field(default_factory=list)
    exclude_sectors: List[str] = field(default_factory=list)
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_price_change_percent: float = -100.0
    max_price_change_percent: float = 100.0
    min_volume_ratio: float = 1.0  # Ratio to average volume
    premarket_only: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert criteria to dictionary for storage"""
        return {
            'min_price': self.min_price,
            'max_price': self.max_price,
            'min_volume': self.min_volume,
            'min_dollar_volume': self.min_dollar_volume,
            'min_float': self.min_float,
            'max_float': self.max_float,
            'sectors': self.sectors,
            'exclude_sectors': self.exclude_sectors,
            'min_market_cap': self.min_market_cap,
            'max_market_cap': self.max_market_cap,
            'min_price_change_percent': self.min_price_change_percent,
            'max_price_change_percent': self.max_price_change_percent,
            'min_volume_ratio': self.min_volume_ratio,
            'premarket_only': self.premarket_only
        }


@dataclass
class SecurityMetrics:
    """Comprehensive metrics for a scanned security"""
    symbol: str
    current_price: float
    price_change: float
    price_change_percent: float
    volume: int
    dollar_volume: float
    vwap: float
    atr: float
    float_size: Optional[float]
    market_cap: Optional[float]
    sector: Optional[str]
    premarket_high: Optional[float]
    premarket_low: Optional[float]
    premarket_volume: Optional[int]
    volume_ratio: float  # Current volume / average volume
    volatility_score: float
    momentum_score: float
    composite_score: float
    position_size_suggestion: Optional[Dict[str, Any]]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for storage/export"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'price_change': self.price_change,
            'price_change_percent': self.price_change_percent,
            'volume': self.volume,
            'dollar_volume': self.dollar_volume,
            'vwap': self.vwap,
            'atr': self.atr,
            'float_size': self.float_size,
            'market_cap': self.market_cap,
            'sector': self.sector,
            'premarket_high': self.premarket_high,
            'premarket_low': self.premarket_low,
            'premarket_volume': self.premarket_volume,
            'volume_ratio': self.volume_ratio,
            'volatility_score': self.volatility_score,
            'momentum_score': self.momentum_score,
            'composite_score': self.composite_score,
            'position_size_suggestion': self.position_size_suggestion,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class SecurityScanTask(QRunnable):
    """Individual security scanning task for QThreadPool execution"""
    
    def __init__(self, symbol: str, scanner_engine: 'ScannerEngine'):
        super().__init__()
        self.symbol = symbol
        self.scanner_engine = scanner_engine
        self.setAutoDelete(True)
    
    def run(self):
        """Execute the scanning task for a single security"""
        try:
            metrics = self.scanner_engine._scan_single_security(self.symbol)
            if metrics:
                self.scanner_engine._add_scan_result(metrics)
        except Exception as e:
            logging.error(f"Error scanning {self.symbol}: {str(e)}")
            logging.debug(traceback.format_exc())


class ScannerEngine(QObject):
    """
    Comprehensive Market Scanner Engine
    
    Performs concurrent market analysis across all tradeable securities using QThreadPool
    with sophisticated weighted ranking algorithms and real-time update mechanisms.
    """
    
    # PyQt signals for real-time updates
    scan_completed = pyqtSignal(list)  # List of SecurityMetrics
    scan_progress = pyqtSignal(int, int)  # current, total
    scan_error = pyqtSignal(str)  # error message
    realtime_update = pyqtSignal(str, dict)  # symbol, updated_metrics
    
    def __init__(self, data_manager: DataIntegrationManager, 
                 risk_calculator: RiskCalculator, 
                 db_manager: DatabaseManager):
        super().__init__()
        
        # Core dependencies
        self.data_manager = data_manager
        self.risk_calculator = risk_calculator
        self.db_manager = db_manager
        
        # Threading components
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(20)  # Optimized for concurrent API calls
        self.scan_mutex = QMutex()
        
        # Scanner state
        self.current_results: Dict[str, SecurityMetrics] = {}
        self.scan_criteria = ScanCriteria()
        self.watchlists: Dict[str, List[str]] = {}
        self.is_scanning = False
        self.scan_start_time: Optional[datetime] = None
        
        # Real-time update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._perform_realtime_updates)
        self.update_timer.setInterval(30000)  # 30 seconds
        
        # Performance tracking
        self.scan_metrics = {
            'total_scanned': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'average_scan_time': 0.0,
            'last_scan_duration': 0.0
        }
        
        # Default watchlists
        self._initialize_default_watchlists()
        
        logging.info("Scanner Engine initialized with QThreadPool architecture")
    
    def _initialize_default_watchlists(self):
        """Initialize default watchlists"""
        # Load from database or create defaults
        try:
            self.watchlists['Most Active'] = self.db_manager.load_watchlist('Most Active') or [
                'AAPL', 'TSLA', 'AMD', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META',
                'NFLX', 'BABA', 'PLTR', 'SOFI', 'RIVN', 'LCID', 'NIO', 'XPEV'
            ]
            
            self.watchlists['High Volume'] = self.db_manager.load_watchlist('High Volume') or [
                'SPY', 'QQQ', 'IWM', 'EEM', 'GDX', 'XLF', 'XBI', 'ARKK',
                'TQQQ', 'SQQQ', 'UVXY', 'VXX', 'SPXL', 'SPXS'
            ]
            
            self.watchlists['Premarket Movers'] = []  # Will be populated during scans
            
        except Exception as e:
            logging.error(f"Error loading watchlists: {str(e)}")
            # Set minimal default watchlists
            self.watchlists = {
                'Most Active': ['AAPL', 'TSLA', 'AMD', 'NVDA', 'MSFT'],
                'High Volume': ['SPY', 'QQQ', 'IWM'],
                'Premarket Movers': []
            }
    
    def set_scan_criteria(self, criteria: ScanCriteria):
        """Update scanning criteria"""
        self.scan_criteria = criteria
        logging.info(f"Updated scan criteria: {criteria}")
    
    def get_scan_criteria(self) -> ScanCriteria:
        """Get current scanning criteria"""
        return self.scan_criteria
    
    def add_watchlist(self, name: str, symbols: List[str]):
        """Add or update a watchlist"""
        self.watchlists[name] = symbols
        self.db_manager.save_watchlist(name, symbols)
        logging.info(f"Added/updated watchlist '{name}' with {len(symbols)} symbols")
    
    def remove_watchlist(self, name: str):
        """Remove a watchlist"""
        if name in self.watchlists:
            del self.watchlists[name]
            logging.info(f"Removed watchlist '{name}'")
    
    def get_watchlists(self) -> Dict[str, List[str]]:
        """Get all watchlists"""
        return self.watchlists.copy()
    
    def start_comprehensive_scan(self, symbols: Optional[List[str]] = None):
        """
        Start comprehensive market scan using QThreadPool
        
        Args:
            symbols: Optional list of symbols to scan. If None, uses default universe
        """
        if self.is_scanning:
            logging.warning("Scan already in progress")
            return
        
        self.is_scanning = True
        self.scan_start_time = datetime.now()
        self.current_results.clear()
        
        # Determine symbols to scan
        if symbols is None:
            symbols = self._get_scan_universe()
        
        logging.info(f"Starting comprehensive scan of {len(symbols)} securities")
        
        # Reset performance metrics
        self.scan_metrics['total_scanned'] = len(symbols)
        self.scan_metrics['successful_scans'] = 0
        self.scan_metrics['failed_scans'] = 0
        
        # Submit tasks to thread pool
        for symbol in symbols:
            task = SecurityScanTask(symbol, self)
            self.thread_pool.start(task)
        
        # Monitor completion
        self._monitor_scan_completion()
    
    def _get_scan_universe(self) -> List[str]:
        """Get the universe of symbols to scan"""
        # Combine all watchlists for comprehensive scanning
        symbols = set()
        for watchlist_symbols in self.watchlists.values():
            symbols.update(watchlist_symbols)
        
        # Add additional popular symbols for comprehensive coverage
        additional_symbols = [
            # Major indices and ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'AGG', 'LQD',
            # Major tech stocks
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX',
            # Financial sector
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.A', 'BRK.B',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'CVS',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'NKE', 'MCD', 'SBUX',
            # Popular growth stocks
            'NVDA', 'AMD', 'CRM', 'ADBE', 'PYPL', 'SQ', 'ROKU', 'ZM',
            # Meme stocks and retail favorites
            'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'SOFI', 'WISH', 'CLOV'
        ]
        
        symbols.update(additional_symbols)
        return list(symbols)
    
    def _scan_single_security(self, symbol: str) -> Optional[SecurityMetrics]:
        """
        Scan a single security and calculate comprehensive metrics
        Extended from original calculate_vwap function
        """
        try:
            # Get market data
            market_data = self.data_manager.get_bulk_market_data([symbol], '1min')
            if symbol not in market_data or market_data[symbol].empty:
                return None
            
            df = market_data[symbol]
            
            # Calculate VWAP (extended from original function)
            vwap_df = self._calculate_vwap_extended(df)
            if vwap_df.empty:
                return None
            
            # Get premarket levels (extended from original function)
            premarket_high, premarket_low, premarket_volume = self._get_premarket_levels_extended(vwap_df)
            
            # Get current market state
            current_data = vwap_df.iloc[-1]
            current_price = float(current_data['close'])
            volume = int(current_data['volume'])
            vwap = float(current_data['vwap'])
            
            # Calculate price change
            if len(vwap_df) > 1:
                prev_close = float(vwap_df.iloc[-2]['close'])
                price_change = current_price - prev_close
                price_change_percent = (price_change / prev_close) * 100
            else:
                price_change = 0.0
                price_change_percent = 0.0
            
            # Calculate ATR
            atr = self._calculate_atr(vwap_df)
            
            # Calculate volume metrics
            dollar_volume = current_price * volume
            volume_ratio = self._calculate_volume_ratio(vwap_df)
            
            # Calculate scoring metrics
            volatility_score = self._calculate_volatility_score(atr, current_price)
            momentum_score = self._calculate_momentum_score(price_change_percent, volume_ratio)
            composite_score = self._calculate_composite_score(
                volatility_score, momentum_score, volume_ratio, price_change_percent
            )
            
            # Get additional market data
            realtime_quote = self.data_manager.get_realtime_quote(symbol)
            market_cap = realtime_quote.get('market_cap')
            float_size = realtime_quote.get('float')
            sector = realtime_quote.get('sector')
            
            # Apply filters
            if not self._passes_criteria_filter(
                current_price, volume, dollar_volume, price_change_percent,
                volume_ratio, market_cap, float_size, sector
            ):
                return None
            
            # Get position sizing suggestion
            position_size_suggestion = None
            try:
                position_size_suggestion = self.risk_calculator.calculate_position_size(
                    symbol, current_price, current_price * 0.02  # 2% risk
                )
            except Exception as e:
                logging.debug(f"Position sizing failed for {symbol}: {str(e)}")
            
            # Create comprehensive metrics
            metrics = SecurityMetrics(
                symbol=symbol,
                current_price=current_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                volume=volume,
                dollar_volume=dollar_volume,
                vwap=vwap,
                atr=atr,
                float_size=float_size,
                market_cap=market_cap,
                sector=sector,
                premarket_high=premarket_high,
                premarket_low=premarket_low,
                premarket_volume=premarket_volume,
                volume_ratio=volume_ratio,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                composite_score=composite_score,
                position_size_suggestion=position_size_suggestion,
                last_updated=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error scanning security {symbol}: {str(e)}")
            logging.debug(traceback.format_exc())
            return None
    
    def _calculate_vwap_extended(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extended VWAP calculation from original function
        Supports rolling windows and additional metrics
        """
        if df.empty:
            return df
        
        # Ensure proper column names
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logging.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        df = df.copy()
        
        # Calculate typical price (from original function)
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative volume * typical price (from original function)
        df['cum_vol_tp'] = (df['tp'] * df['volume']).cumsum()
        
        # Calculate cumulative volume (from original function)
        df['cum_vol'] = df['volume'].cumsum()
        
        # Calculate VWAP (from original function)
        df['vwap'] = df['cum_vol_tp'] / df['cum_vol']
        
        # Additional enhancements for desktop application
        # Rolling VWAP for different periods
        df['vwap_20'] = df['tp'].rolling(window=20).apply(
            lambda x: np.average(x, weights=df['volume'].iloc[-len(x):]) if len(x) > 0 else np.nan
        )
        
        # VWAP deviation
        df['vwap_deviation'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
        
        # Volume-weighted price bands
        df['vwap_upper'] = df['vwap'] * 1.02  # 2% above VWAP
        df['vwap_lower'] = df['vwap'] * 0.98  # 2% below VWAP
        
        return df
    
    def _get_premarket_levels_extended(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """
        Extended premarket levels calculation from original function
        Returns high, low, and volume
        """
        if df.empty or 'timestamp' not in df.columns:
            return None, None, None
        
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter premarket data (before 9:30 AM) - from original function
            premarket = df[df['timestamp'].dt.time < time(9, 30)]
            
            if premarket.empty:
                return None, None, None
            
            # Get premarket high and low (from original function)
            premarket_high = float(premarket['high'].max())
            premarket_low = float(premarket['low'].min())
            
            # Additional enhancement: premarket volume
            premarket_volume = int(premarket['volume'].sum())
            
            return premarket_high, premarket_low, premarket_volume
            
        except Exception as e:
            logging.error(f"Error calculating premarket levels: {str(e)}")
            return None, None, None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility assessment"""
        if len(df) < period + 1:
            return 0.0
        
        try:
            # Calculate True Range
            df = df.copy()
            df['prev_close'] = df['close'].shift(1)
            
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR
            atr = df['true_range'].rolling(window=period).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            return 0.0
    
    def _calculate_volume_ratio(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Calculate current volume ratio to average volume"""
        if len(df) < lookback:
            return 1.0
        
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(lookback).mean()
            
            if avg_volume > 0:
                return float(current_volume / avg_volume)
            return 1.0
            
        except Exception as e:
            logging.error(f"Error calculating volume ratio: {str(e)}")
            return 1.0
    
    def _calculate_volatility_score(self, atr: float, price: float) -> float:
        """Calculate volatility score (0-100)"""
        if price <= 0:
            return 0.0
        
        # ATR as percentage of price
        atr_percent = (atr / price) * 100
        
        # Normalize to 0-100 scale (5% ATR = 100 points)
        score = min(atr_percent * 20, 100.0)
        return score
    
    def _calculate_momentum_score(self, price_change_percent: float, volume_ratio: float) -> float:
        """Calculate momentum score based on price change and volume"""
        # Base momentum from price change
        momentum_base = abs(price_change_percent) * 10  # 1% = 10 points
        
        # Volume confirmation multiplier
        volume_multiplier = min(volume_ratio, 3.0)  # Cap at 3x volume
        
        # Combined momentum score
        score = momentum_base * volume_multiplier
        return min(score, 100.0)
    
    def _calculate_composite_score(self, volatility_score: float, momentum_score: float,
                                 volume_ratio: float, price_change_percent: float) -> float:
        """Calculate weighted composite score for ranking"""
        # Weights for different factors
        volatility_weight = 0.25
        momentum_weight = 0.35
        volume_weight = 0.25
        price_change_weight = 0.15
        
        # Normalize volume ratio to 0-100 scale
        volume_score = min(volume_ratio * 25, 100.0)
        
        # Normalize price change to 0-100 scale
        price_score = min(abs(price_change_percent) * 5, 100.0)
        
        # Calculate weighted composite score
        composite = (
            volatility_score * volatility_weight +
            momentum_score * momentum_weight +
            volume_score * volume_weight +
            price_score * price_change_weight
        )
        
        return composite
    
    def _passes_criteria_filter(self, price: float, volume: int, dollar_volume: float,
                              price_change_percent: float, volume_ratio: float,
                              market_cap: Optional[float], float_size: Optional[float],
                              sector: Optional[str]) -> bool:
        """Check if security passes scanning criteria"""
        criteria = self.scan_criteria
        
        # Price filters
        if not (criteria.min_price <= price <= criteria.max_price):
            return False
        
        # Volume filters
        if volume < criteria.min_volume:
            return False
        
        if dollar_volume < criteria.min_dollar_volume:
            return False
        
        if volume_ratio < criteria.min_volume_ratio:
            return False
        
        # Price change filters
        if not (criteria.min_price_change_percent <= price_change_percent <= criteria.max_price_change_percent):
            return False
        
        # Market cap filters
        if market_cap:
            if criteria.min_market_cap and market_cap < criteria.min_market_cap:
                return False
            if criteria.max_market_cap and market_cap > criteria.max_market_cap:
                return False
        
        # Float filters
        if float_size:
            if criteria.min_float and float_size < criteria.min_float:
                return False
            if criteria.max_float and float_size > criteria.max_float:
                return False
        
        # Sector filters
        if sector:
            if criteria.sectors and sector not in criteria.sectors:
                return False
            if criteria.exclude_sectors and sector in criteria.exclude_sectors:
                return False
        
        return True
    
    def _add_scan_result(self, metrics: SecurityMetrics):
        """Thread-safe method to add scan result"""
        with QMutexLocker(self.scan_mutex):
            self.current_results[metrics.symbol] = metrics
            self.scan_metrics['successful_scans'] += 1
            
            # Update progress
            current_progress = len(self.current_results)
            total_progress = self.scan_metrics['total_scanned']
            self.scan_progress.emit(current_progress, total_progress)
    
    def _monitor_scan_completion(self):
        """Monitor scan completion and emit results"""
        def check_completion():
            total_completed = self.scan_metrics['successful_scans'] + self.scan_metrics['failed_scans']
            if total_completed >= self.scan_metrics['total_scanned']:
                self._finalize_scan()
                return
            
            # Check again in 1 second
            QTimer.singleShot(1000, check_completion)
        
        # Start monitoring
        QTimer.singleShot(1000, check_completion)
    
    def _finalize_scan(self):
        """Finalize scan and emit results"""
        self.is_scanning = False
        
        # Calculate scan duration
        if self.scan_start_time:
            duration = (datetime.now() - self.scan_start_time).total_seconds()
            self.scan_metrics['last_scan_duration'] = duration
            self.scan_metrics['average_scan_time'] = (
                self.scan_metrics['average_scan_time'] + duration
            ) / 2
        
        # Sort results by composite score
        sorted_results = sorted(
            self.current_results.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )
        
        # Update premarket movers watchlist
        top_movers = [result.symbol for result in sorted_results[:20]]
        self.watchlists['Premarket Movers'] = top_movers
        
        # Save results to database
        try:
            result_dicts = [result.to_dict() for result in sorted_results]
            self.db_manager.save_scanner_results(result_dicts)
        except Exception as e:
            logging.error(f"Error saving scan results: {str(e)}")
        
        # Emit completion signal
        self.scan_completed.emit(sorted_results)
        
        logging.info(f"Scan completed: {len(sorted_results)} results in {self.scan_metrics['last_scan_duration']:.2f}s")
    
    def get_current_results(self) -> List[SecurityMetrics]:
        """Get current scan results sorted by composite score"""
        return sorted(
            self.current_results.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )
    
    def get_top_results(self, limit: int = 50) -> List[SecurityMetrics]:
        """Get top N scan results"""
        results = self.get_current_results()
        return results[:limit]
    
    def get_result_by_symbol(self, symbol: str) -> Optional[SecurityMetrics]:
        """Get scan result for specific symbol"""
        return self.current_results.get(symbol)
    
    def start_realtime_updates(self):
        """Start real-time updates for current results"""
        if not self.update_timer.isActive():
            self.update_timer.start()
            logging.info("Started real-time updates")
    
    def stop_realtime_updates(self):
        """Stop real-time updates"""
        if self.update_timer.isActive():
            self.update_timer.stop()
            logging.info("Stopped real-time updates")
    
    def _perform_realtime_updates(self):
        """Perform real-time updates for current results"""
        if not self.current_results:
            return
        
        # Update top 20 results in real-time
        top_symbols = [
            result.symbol for result in 
            sorted(self.current_results.values(), key=lambda x: x.composite_score, reverse=True)[:20]
        ]
        
        for symbol in top_symbols:
            try:
                # Get updated quote
                quote = self.data_manager.get_realtime_quote(symbol)
                if quote and 'price' in quote:
                    # Update current result
                    current_result = self.current_results[symbol]
                    old_price = current_result.current_price
                    new_price = float(quote['price'])
                    
                    # Calculate new metrics
                    price_change = new_price - old_price
                    price_change_percent = (price_change / old_price) * 100 if old_price > 0 else 0
                    
                    # Update result
                    current_result.current_price = new_price
                    current_result.price_change += price_change
                    current_result.price_change_percent = price_change_percent
                    current_result.last_updated = datetime.now()
                    
                    # Recalculate scores
                    current_result.momentum_score = self._calculate_momentum_score(
                        price_change_percent, current_result.volume_ratio
                    )
                    current_result.composite_score = self._calculate_composite_score(
                        current_result.volatility_score, current_result.momentum_score,
                        current_result.volume_ratio, price_change_percent
                    )
                    
                    # Emit update signal
                    self.realtime_update.emit(symbol, current_result.to_dict())
                    
            except Exception as e:
                logging.error(f"Error updating {symbol}: {str(e)}")
    
    def export_results(self, format: str = 'dict') -> Union[List[Dict], pd.DataFrame]:
        """Export scan results in specified format"""
        results = self.get_current_results()
        
        if format == 'dict':
            return [result.to_dict() for result in results]
        elif format == 'dataframe':
            return pd.DataFrame([result.to_dict() for result in results])
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scan performance statistics"""
        return {
            'total_results': len(self.current_results),
            'scan_metrics': self.scan_metrics.copy(),
            'thread_pool_active_count': self.thread_pool.activeThreadCount(),
            'thread_pool_max_threads': self.thread_pool.maxThreadCount(),
            'last_scan_time': self.scan_start_time.isoformat() if self.scan_start_time else None,
            'realtime_updates_active': self.update_timer.isActive()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_realtime_updates()
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds
        logging.info("Scanner engine cleanup completed")


# Utility functions for external integration
def create_default_scan_criteria() -> ScanCriteria:
    """Create default scanning criteria for premarket analysis"""
    return ScanCriteria(
        min_price=1.0,
        max_price=500.0,
        min_volume=500000,
        min_dollar_volume=5000000.0,
        min_price_change_percent=2.0,  # At least 2% move
        min_volume_ratio=1.5,  # At least 1.5x average volume
        premarket_only=True
    )


def create_momentum_scan_criteria() -> ScanCriteria:
    """Create scanning criteria focused on momentum plays"""
    return ScanCriteria(
        min_price=5.0,
        max_price=100.0,
        min_volume=1000000,
        min_dollar_volume=10000000.0,
        min_price_change_percent=5.0,  # At least 5% move
        min_volume_ratio=2.0,  # At least 2x average volume
        premarket_only=False
    )


def create_large_cap_scan_criteria() -> ScanCriteria:
    """Create scanning criteria for large cap stocks"""
    return ScanCriteria(
        min_price=50.0,
        max_price=1000.0,
        min_volume=10000000,
        min_dollar_volume=500000000.0,
        min_market_cap=10000000000.0,  # $10B+ market cap
        min_price_change_percent=1.0,
        min_volume_ratio=1.2,
        premarket_only=False
    )