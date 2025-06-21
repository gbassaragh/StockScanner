#!/usr/bin/env python3
"""
Advanced Risk Management Calculator for Professional Trading Platform

This module implements sophisticated position sizing calculations using Kelly Criterion
variant algorithms optimized for trading applications, with ATR-based volatility 
adjustment and comprehensive account parameter persistence.

Key Features:
- Kelly Criterion implementation with win-rate analysis
- ATR-based dynamic risk adjustment for volatility-aware position sizing
- Account parameter persistence via SQLite
- Integration with scanner results for automated recommendations
- Decimal arithmetic precision for all financial calculations
- Real-time risk assessment capabilities for multiple securities

Author: Blitzy Agent
Version: 1.0.0
"""

import logging
import math
import sqlite3
import threading
from decimal import Decimal, getcontext, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import os

# Third-party imports
import pandas as pd
import numpy as np
from scipy import stats
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, QMutex
from PyQt6.QtSql import QSqlDatabase, QSqlQuery, QSqlError

# Configure decimal precision for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskToleranceLevel(Enum):
    """Risk tolerance levels for position sizing calculations."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class AllocationStrategy(Enum):
    """Capital allocation strategies."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_OPTIMAL = "kelly_optimal"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CUSTOM = "custom"


@dataclass
class AccountParameters:
    """Account configuration parameters for risk management."""
    account_id: str
    total_capital: Decimal
    max_position_size: Decimal  # Maximum percentage of capital per position
    max_portfolio_risk: Decimal  # Maximum portfolio risk percentage
    risk_tolerance: RiskToleranceLevel
    allocation_strategy: AllocationStrategy
    max_drawdown_limit: Decimal
    stop_loss_multiplier: Decimal
    target_profit_multiplier: Decimal
    kelly_fraction: Decimal = Decimal('0.25')  # Kelly fraction (partial Kelly)
    max_positions: int = 10
    min_position_size: Decimal = Decimal('100.00')
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityRiskProfile:
    """Risk profile for a specific security."""
    symbol: str
    atr_period: int = 14
    atr_value: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    liquidity_score: Optional[Decimal] = None
    sector_risk: Optional[Decimal] = None
    correlation_factor: Optional[Decimal] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation with detailed analysis."""
    symbol: str
    recommended_shares: int
    recommended_dollar_amount: Decimal
    risk_per_share: Decimal
    total_risk_amount: Decimal
    stop_loss_price: Decimal
    target_price: Optional[Decimal]
    kelly_fraction_used: Decimal
    atr_adjustment_factor: Decimal
    confidence_score: Decimal
    max_loss_percentage: Decimal
    expected_return: Optional[Decimal]
    risk_reward_ratio: Optional[Decimal]
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'recommended_shares': self.recommended_shares,
            'recommended_dollar_amount': float(self.recommended_dollar_amount),
            'risk_per_share': float(self.risk_per_share),
            'total_risk_amount': float(self.total_risk_amount),
            'stop_loss_price': float(self.stop_loss_price),
            'target_price': float(self.target_price) if self.target_price else None,
            'kelly_fraction_used': float(self.kelly_fraction_used),
            'atr_adjustment_factor': float(self.atr_adjustment_factor),
            'confidence_score': float(self.confidence_score),
            'max_loss_percentage': float(self.max_loss_percentage),
            'expected_return': float(self.expected_return) if self.expected_return else None,
            'risk_reward_ratio': float(self.risk_reward_ratio) if self.risk_reward_ratio else None,
            'calculation_timestamp': self.calculation_timestamp.isoformat()
        }


class RiskCalculatorError(Exception):
    """Custom exception for risk calculator errors."""
    pass


class DatabaseManager:
    """Database manager for risk calculator persistence."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager."""
        if db_path is None:
            # Default to application data directory
            app_data_dir = Path(os.environ.get('APPDATA', '')) / 'Blitzy' / 'database'
            app_data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(app_data_dir / 'trading_app.db')
        
        self.db_path = db_path
        self._connection = None
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema for risk management."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create account parameters table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS account_parameters (
                        account_id TEXT PRIMARY KEY,
                        total_capital REAL NOT NULL,
                        max_position_size REAL NOT NULL,
                        max_portfolio_risk REAL NOT NULL,
                        risk_tolerance TEXT NOT NULL,
                        allocation_strategy TEXT NOT NULL,
                        max_drawdown_limit REAL NOT NULL,
                        stop_loss_multiplier REAL NOT NULL,
                        target_profit_multiplier REAL NOT NULL,
                        kelly_fraction REAL NOT NULL,
                        max_positions INTEGER NOT NULL,
                        min_position_size REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # Create security risk profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS security_risk_profiles (
                        symbol TEXT PRIMARY KEY,
                        atr_period INTEGER NOT NULL,
                        atr_value REAL,
                        beta REAL,
                        volatility REAL,
                        liquidity_score REAL,
                        sector_risk REAL,
                        correlation_factor REAL,
                        last_updated TEXT NOT NULL
                    )
                ''')
                
                # Create position size recommendations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS position_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        recommended_shares INTEGER NOT NULL,
                        recommended_dollar_amount REAL NOT NULL,
                        risk_per_share REAL NOT NULL,
                        total_risk_amount REAL NOT NULL,
                        stop_loss_price REAL NOT NULL,
                        target_price REAL,
                        kelly_fraction_used REAL NOT NULL,
                        atr_adjustment_factor REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        max_loss_percentage REAL NOT NULL,
                        expected_return REAL,
                        risk_reward_ratio REAL,
                        calculation_timestamp TEXT NOT NULL
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_symbol ON position_recommendations(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp ON position_recommendations(calculation_timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_profiles_symbol ON security_risk_profiles(symbol)')
                
                conn.commit()
                conn.close()
                logger.info("Risk calculator database initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize risk calculator database: {e}")
                raise RiskCalculatorError(f"Database initialization failed: {e}")
    
    def save_account_parameters(self, params: AccountParameters) -> bool:
        """Save account parameters to database."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO account_parameters 
                    (account_id, total_capital, max_position_size, max_portfolio_risk,
                     risk_tolerance, allocation_strategy, max_drawdown_limit,
                     stop_loss_multiplier, target_profit_multiplier, kelly_fraction,
                     max_positions, min_position_size, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    params.account_id,
                    float(params.total_capital),
                    float(params.max_position_size),
                    float(params.max_portfolio_risk),
                    params.risk_tolerance.value,
                    params.allocation_strategy.value,
                    float(params.max_drawdown_limit),
                    float(params.stop_loss_multiplier),
                    float(params.target_profit_multiplier),
                    float(params.kelly_fraction),
                    params.max_positions,
                    float(params.min_position_size),
                    params.created_at.isoformat(),
                    params.updated_at.isoformat()
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logger.error(f"Failed to save account parameters: {e}")
                return False
    
    def load_account_parameters(self, account_id: str) -> Optional[AccountParameters]:
        """Load account parameters from database."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM account_parameters WHERE account_id = ?', (account_id,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return AccountParameters(
                        account_id=row[0],
                        total_capital=Decimal(str(row[1])),
                        max_position_size=Decimal(str(row[2])),
                        max_portfolio_risk=Decimal(str(row[3])),
                        risk_tolerance=RiskToleranceLevel(row[4]),
                        allocation_strategy=AllocationStrategy(row[5]),
                        max_drawdown_limit=Decimal(str(row[6])),
                        stop_loss_multiplier=Decimal(str(row[7])),
                        target_profit_multiplier=Decimal(str(row[8])),
                        kelly_fraction=Decimal(str(row[9])),
                        max_positions=row[10],
                        min_position_size=Decimal(str(row[11])),
                        created_at=datetime.fromisoformat(row[12]),
                        updated_at=datetime.fromisoformat(row[13])
                    )
                return None
                
            except Exception as e:
                logger.error(f"Failed to load account parameters: {e}")
                return None
    
    def save_security_risk_profile(self, profile: SecurityRiskProfile) -> bool:
        """Save security risk profile to database."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO security_risk_profiles 
                    (symbol, atr_period, atr_value, beta, volatility, liquidity_score,
                     sector_risk, correlation_factor, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile.symbol,
                    profile.atr_period,
                    float(profile.atr_value) if profile.atr_value else None,
                    float(profile.beta) if profile.beta else None,
                    float(profile.volatility) if profile.volatility else None,
                    float(profile.liquidity_score) if profile.liquidity_score else None,
                    float(profile.sector_risk) if profile.sector_risk else None,
                    float(profile.correlation_factor) if profile.correlation_factor else None,
                    profile.last_updated.isoformat()
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logger.error(f"Failed to save security risk profile: {e}")
                return False
    
    def load_security_risk_profile(self, symbol: str) -> Optional[SecurityRiskProfile]:
        """Load security risk profile from database."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM security_risk_profiles WHERE symbol = ?', (symbol,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return SecurityRiskProfile(
                        symbol=row[0],
                        atr_period=row[1],
                        atr_value=Decimal(str(row[2])) if row[2] else None,
                        beta=Decimal(str(row[3])) if row[3] else None,
                        volatility=Decimal(str(row[4])) if row[4] else None,
                        liquidity_score=Decimal(str(row[5])) if row[5] else None,
                        sector_risk=Decimal(str(row[6])) if row[6] else None,
                        correlation_factor=Decimal(str(row[7])) if row[7] else None,
                        last_updated=datetime.fromisoformat(row[8])
                    )
                return None
                
            except Exception as e:
                logger.error(f"Failed to load security risk profile: {e}")
                return None
    
    def save_position_recommendation(self, recommendation: PositionSizeRecommendation) -> bool:
        """Save position size recommendation to database."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO position_recommendations 
                    (symbol, recommended_shares, recommended_dollar_amount, risk_per_share,
                     total_risk_amount, stop_loss_price, target_price, kelly_fraction_used,
                     atr_adjustment_factor, confidence_score, max_loss_percentage,
                     expected_return, risk_reward_ratio, calculation_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    recommendation.symbol,
                    recommendation.recommended_shares,
                    float(recommendation.recommended_dollar_amount),
                    float(recommendation.risk_per_share),
                    float(recommendation.total_risk_amount),
                    float(recommendation.stop_loss_price),
                    float(recommendation.target_price) if recommendation.target_price else None,
                    float(recommendation.kelly_fraction_used),
                    float(recommendation.atr_adjustment_factor),
                    float(recommendation.confidence_score),
                    float(recommendation.max_loss_percentage),
                    float(recommendation.expected_return) if recommendation.expected_return else None,
                    float(recommendation.risk_reward_ratio) if recommendation.risk_reward_ratio else None,
                    recommendation.calculation_timestamp.isoformat()
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logger.error(f"Failed to save position recommendation: {e}")
                return False


class KellyCalculator:
    """Kelly Criterion calculator optimized for trading applications."""
    
    @staticmethod
    def calculate_kelly_fraction(win_rate: Decimal, avg_win: Decimal, avg_loss: Decimal) -> Decimal:
        """
        Calculate Kelly fraction using win rate and average win/loss ratios.
        
        Formula: f = (bp - q) / b
        Where:
        - f = Kelly fraction
        - b = odds (avg_win / avg_loss)
        - p = probability of winning (win_rate)
        - q = probability of losing (1 - win_rate)
        
        Args:
            win_rate: Historical win rate as decimal (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            
        Returns:
            Kelly fraction as decimal
        """
        if win_rate <= 0 or win_rate >= 1:
            raise ValueError("Win rate must be between 0 and 1")
        
        if avg_win <= 0 or avg_loss <= 0:
            raise ValueError("Average win and loss must be positive")
        
        # Calculate odds ratio
        odds_ratio = avg_win / avg_loss
        
        # Calculate Kelly fraction
        kelly_fraction = (odds_ratio * win_rate - (Decimal('1') - win_rate)) / odds_ratio
        
        # Ensure Kelly fraction is between 0 and 1
        return max(Decimal('0'), min(Decimal('1'), kelly_fraction))
    
    @staticmethod
    def calculate_optimal_fraction(historical_returns: List[Decimal], max_fraction: Decimal = Decimal('0.25')) -> Decimal:
        """
        Calculate optimal Kelly fraction from historical returns.
        
        Args:
            historical_returns: List of historical returns as decimals
            max_fraction: Maximum allowed Kelly fraction (default 0.25)
            
        Returns:
            Optimal Kelly fraction
        """
        if not historical_returns:
            return Decimal('0')
        
        # Convert to numpy array for calculations
        returns = np.array([float(r) for r in historical_returns])
        
        # Calculate basic statistics
        mean_return = Decimal(str(np.mean(returns)))
        variance = Decimal(str(np.var(returns)))
        
        # Kelly fraction = mean / variance (for continuous returns)
        if variance > 0:
            kelly_fraction = mean_return / variance
        else:
            kelly_fraction = Decimal('0')
        
        # Apply maximum fraction constraint
        return min(max_fraction, max(Decimal('0'), kelly_fraction))


class ATRCalculator:
    """Average True Range calculator for volatility-based risk adjustment."""
    
    @staticmethod
    def calculate_atr(highs: List[Decimal], lows: List[Decimal], closes: List[Decimal], period: int = 14) -> Optional[Decimal]:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period (default 14)
            
        Returns:
            ATR value or None if insufficient data
        """
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return None
        
        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValueError("Price arrays must have same length")
        
        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(highs)):
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            
            true_range = max(hl, hc, lc)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        # Calculate ATR as simple moving average of True Range
        atr = sum(true_ranges[-period:]) / Decimal(str(period))
        return atr
    
    @staticmethod
    def calculate_volatility_adjustment(atr: Decimal, current_price: Decimal, base_volatility: Decimal = Decimal('0.02')) -> Decimal:
        """
        Calculate volatility adjustment factor based on ATR.
        
        Args:
            atr: Average True Range value
            current_price: Current stock price
            base_volatility: Base volatility assumption (default 2%)
            
        Returns:
            Volatility adjustment factor
        """
        if current_price <= 0:
            raise ValueError("Current price must be positive")
        
        # Calculate ATR as percentage of price
        atr_percentage = atr / current_price
        
        # Adjustment factor = base volatility / ATR percentage
        # Higher volatility = smaller position size
        adjustment_factor = base_volatility / atr_percentage
        
        # Constrain adjustment factor between 0.1 and 2.0
        return max(Decimal('0.1'), min(Decimal('2.0'), adjustment_factor))


class RiskCalculator(QObject):
    """
    Advanced risk management calculator with Kelly Criterion and ATR-based adjustments.
    
    This class provides comprehensive position sizing calculations with:
    - Kelly Criterion implementation for optimal position sizing
    - ATR-based volatility adjustment
    - Account parameter persistence
    - Integration with scanner results
    - Real-time risk assessment capabilities
    """
    
    # Qt signals for real-time updates
    calculation_completed = pyqtSignal(str, dict)  # symbol, recommendation
    risk_parameters_updated = pyqtSignal(str)  # account_id
    error_occurred = pyqtSignal(str, str)  # error_type, error_message
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize risk calculator."""
        super().__init__()
        
        self.db_manager = DatabaseManager(db_path)
        self.kelly_calculator = KellyCalculator()
        self.atr_calculator = ATRCalculator()
        
        # Cache for account parameters and risk profiles
        self._account_cache: Dict[str, AccountParameters] = {}
        self._risk_profile_cache: Dict[str, SecurityRiskProfile] = {}
        
        # Thread safety
        self._cache_lock = QMutex()
        
        # Default account parameters
        self.default_account = AccountParameters(
            account_id="default",
            total_capital=Decimal('100000.00'),
            max_position_size=Decimal('0.10'),  # 10% max per position
            max_portfolio_risk=Decimal('0.02'),  # 2% max portfolio risk
            risk_tolerance=RiskToleranceLevel.MODERATE,
            allocation_strategy=AllocationStrategy.KELLY_OPTIMAL,
            max_drawdown_limit=Decimal('0.20'),  # 20% max drawdown
            stop_loss_multiplier=Decimal('2.0'),  # 2x ATR stop loss
            target_profit_multiplier=Decimal('3.0'),  # 3x ATR target
            kelly_fraction=Decimal('0.25')  # 25% Kelly fraction
        )
        
        # Load or create default account
        self._load_default_account()
        
        logger.info("Risk calculator initialized successfully")
    
    def _load_default_account(self):
        """Load or create default account parameters."""
        existing_params = self.db_manager.load_account_parameters("default")
        if existing_params:
            self.default_account = existing_params
        else:
            # Save default parameters to database
            self.db_manager.save_account_parameters(self.default_account)
        
        # Cache the default account
        self._account_cache["default"] = self.default_account
    
    def get_account_parameters(self, account_id: str = "default") -> Optional[AccountParameters]:
        """Get account parameters with caching."""
        with self._cache_lock:
            if account_id in self._account_cache:
                return self._account_cache[account_id]
            
            # Load from database
            params = self.db_manager.load_account_parameters(account_id)
            if params:
                self._account_cache[account_id] = params
            
            return params
    
    def update_account_parameters(self, params: AccountParameters) -> bool:
        """Update account parameters."""
        try:
            # Update timestamp
            params.updated_at = datetime.now()
            
            # Save to database
            success = self.db_manager.save_account_parameters(params)
            
            if success:
                # Update cache
                with self._cache_lock:
                    self._account_cache[params.account_id] = params
                
                # Emit signal
                self.risk_parameters_updated.emit(params.account_id)
                logger.info(f"Account parameters updated for {params.account_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update account parameters: {e}")
            self.error_occurred.emit("ACCOUNT_UPDATE_ERROR", str(e))
            return False
    
    def get_security_risk_profile(self, symbol: str) -> Optional[SecurityRiskProfile]:
        """Get security risk profile with caching."""
        with self._cache_lock:
            if symbol in self._risk_profile_cache:
                profile = self._risk_profile_cache[symbol]
                # Check if profile is stale (older than 1 hour)
                if datetime.now() - profile.last_updated < timedelta(hours=1):
                    return profile
            
            # Load from database
            profile = self.db_manager.load_security_risk_profile(symbol)
            if profile:
                self._risk_profile_cache[symbol] = profile
            
            return profile
    
    def update_security_risk_profile(self, profile: SecurityRiskProfile) -> bool:
        """Update security risk profile."""
        try:
            # Update timestamp
            profile.last_updated = datetime.now()
            
            # Save to database
            success = self.db_manager.save_security_risk_profile(profile)
            
            if success:
                # Update cache
                with self._cache_lock:
                    self._risk_profile_cache[profile.symbol] = profile
                
                logger.info(f"Risk profile updated for {profile.symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update security risk profile: {e}")
            return False
    
    def calculate_atr_from_data(self, symbol: str, price_data: pd.DataFrame) -> Optional[Decimal]:
        """
        Calculate ATR from price data and update risk profile.
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            ATR value or None if calculation fails
        """
        try:
            if len(price_data) < 14:
                logger.warning(f"Insufficient data for ATR calculation: {symbol}")
                return None
            
            # Extract price arrays
            highs = [Decimal(str(h)) for h in price_data['high'].values]
            lows = [Decimal(str(l)) for l in price_data['low'].values]
            closes = [Decimal(str(c)) for c in price_data['close'].values]
            
            # Calculate ATR
            atr = self.atr_calculator.calculate_atr(highs, lows, closes)
            
            if atr:
                # Update or create risk profile
                profile = self.get_security_risk_profile(symbol)
                if profile is None:
                    profile = SecurityRiskProfile(symbol=symbol)
                
                profile.atr_value = atr
                profile.last_updated = datetime.now()
                
                # Save updated profile
                self.update_security_risk_profile(profile)
                
                logger.info(f"ATR calculated for {symbol}: {atr}")
            
            return atr
            
        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return None
    
    def calculate_position_size(
        self,
        symbol: str,
        current_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        target_price: Optional[Decimal] = None,
        account_id: str = "default",
        scanner_data: Optional[Dict] = None
    ) -> Optional[PositionSizeRecommendation]:
        """
        Calculate optimal position size using Kelly Criterion and ATR adjustment.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            stop_loss_price: Stop loss price (optional, will use ATR if not provided)
            target_price: Target price (optional, will use ATR if not provided)
            account_id: Account identifier
            scanner_data: Additional data from scanner (optional)
            
        Returns:
            Position size recommendation or None if calculation fails
        """
        try:
            # Get account parameters
            account_params = self.get_account_parameters(account_id)
            if not account_params:
                raise RiskCalculatorError(f"Account parameters not found: {account_id}")
            
            # Get security risk profile
            risk_profile = self.get_security_risk_profile(symbol)
            if not risk_profile or not risk_profile.atr_value:
                logger.warning(f"No ATR data available for {symbol}, using default risk calculation")
                risk_profile = SecurityRiskProfile(symbol=symbol, atr_value=current_price * Decimal('0.02'))
            
            # Calculate stop loss if not provided
            if stop_loss_price is None:
                atr_stop_distance = risk_profile.atr_value * account_params.stop_loss_multiplier
                stop_loss_price = current_price - atr_stop_distance
            
            # Calculate target price if not provided
            if target_price is None:
                atr_target_distance = risk_profile.atr_value * account_params.target_profit_multiplier
                target_price = current_price + atr_target_distance
            
            # Calculate risk per share
            risk_per_share = current_price - stop_loss_price
            if risk_per_share <= 0:
                raise RiskCalculatorError("Invalid stop loss price - must be below current price")
            
            # Calculate ATR adjustment factor
            atr_adjustment = self.atr_calculator.calculate_volatility_adjustment(
                risk_profile.atr_value, current_price
            )
            
            # Calculate Kelly fraction
            kelly_fraction = account_params.kelly_fraction
            
            # If we have scanner data with historical performance, use it for Kelly calculation
            if scanner_data and 'historical_returns' in scanner_data:
                historical_returns = [Decimal(str(r)) for r in scanner_data['historical_returns']]
                kelly_fraction = self.kelly_calculator.calculate_optimal_fraction(
                    historical_returns, account_params.kelly_fraction
                )
            
            # Calculate maximum risk amount per position
            max_position_risk = account_params.total_capital * account_params.max_portfolio_risk
            
            # Calculate position size based on risk
            # Position size = risk amount / risk per share
            risk_based_shares = int(max_position_risk / risk_per_share)
            
            # Apply Kelly fraction adjustment
            kelly_adjusted_shares = int(risk_based_shares * kelly_fraction)
            
            # Apply ATR volatility adjustment
            final_shares = int(kelly_adjusted_shares * atr_adjustment)
            
            # Apply position size constraints
            max_position_value = account_params.total_capital * account_params.max_position_size
            max_shares_by_value = int(max_position_value / current_price)
            
            # Take minimum of all constraints
            recommended_shares = min(final_shares, max_shares_by_value)
            
            # Ensure minimum position size
            min_shares = int(account_params.min_position_size / current_price)
            if recommended_shares < min_shares:
                recommended_shares = min_shares
            
            # Calculate final metrics
            recommended_dollar_amount = Decimal(str(recommended_shares)) * current_price
            total_risk_amount = Decimal(str(recommended_shares)) * risk_per_share
            max_loss_percentage = (total_risk_amount / account_params.total_capital) * Decimal('100')
            
            # Calculate expected return and risk-reward ratio
            expected_return = None
            risk_reward_ratio = None
            if target_price:
                potential_profit = (target_price - current_price) * Decimal(str(recommended_shares))
                expected_return = potential_profit / recommended_dollar_amount
                risk_reward_ratio = potential_profit / total_risk_amount
            
            # Calculate confidence score based on data quality
            confidence_score = self._calculate_confidence_score(risk_profile, scanner_data)
            
            # Create recommendation
            recommendation = PositionSizeRecommendation(
                symbol=symbol,
                recommended_shares=recommended_shares,
                recommended_dollar_amount=recommended_dollar_amount,
                risk_per_share=risk_per_share,
                total_risk_amount=total_risk_amount,
                stop_loss_price=stop_loss_price,
                target_price=target_price,
                kelly_fraction_used=kelly_fraction,
                atr_adjustment_factor=atr_adjustment,
                confidence_score=confidence_score,
                max_loss_percentage=max_loss_percentage,
                expected_return=expected_return,
                risk_reward_ratio=risk_reward_ratio
            )
            
            # Save recommendation to database
            self.db_manager.save_position_recommendation(recommendation)
            
            # Emit signal
            self.calculation_completed.emit(symbol, recommendation.to_dict())
            
            logger.info(f"Position size calculated for {symbol}: {recommended_shares} shares")
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            self.error_occurred.emit("POSITION_SIZE_ERROR", str(e))
            return None
    
    def calculate_portfolio_risk(self, positions: List[Dict], account_id: str = "default") -> Dict[str, Decimal]:
        """
        Calculate overall portfolio risk metrics.
        
        Args:
            positions: List of current positions with symbol, shares, current_price
            account_id: Account identifier
            
        Returns:
            Dictionary with portfolio risk metrics
        """
        try:
            account_params = self.get_account_parameters(account_id)
            if not account_params:
                raise RiskCalculatorError(f"Account parameters not found: {account_id}")
            
            total_portfolio_value = Decimal('0')
            total_risk_amount = Decimal('0')
            
            for position in positions:
                symbol = position['symbol']
                shares = Decimal(str(position['shares']))
                current_price = Decimal(str(position['current_price']))
                
                position_value = shares * current_price
                total_portfolio_value += position_value
                
                # Get risk profile for stop loss calculation
                risk_profile = self.get_security_risk_profile(symbol)
                if risk_profile and risk_profile.atr_value:
                    stop_loss_distance = risk_profile.atr_value * account_params.stop_loss_multiplier
                    position_risk = shares * stop_loss_distance
                    total_risk_amount += position_risk
            
            # Calculate risk metrics
            portfolio_risk_percentage = (total_risk_amount / account_params.total_capital) * Decimal('100')
            position_concentration = total_portfolio_value / account_params.total_capital
            
            return {
                'total_portfolio_value': total_portfolio_value,
                'total_risk_amount': total_risk_amount,
                'portfolio_risk_percentage': portfolio_risk_percentage,
                'position_concentration': position_concentration,
                'max_portfolio_risk': account_params.max_portfolio_risk * Decimal('100'),
                'risk_utilization': portfolio_risk_percentage / (account_params.max_portfolio_risk * Decimal('100'))
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio risk: {e}")
            return {}
    
    def _calculate_confidence_score(self, risk_profile: SecurityRiskProfile, scanner_data: Optional[Dict]) -> Decimal:
        """Calculate confidence score for position sizing recommendation."""
        score = Decimal('0.5')  # Base score
        
        # Adjust based on available data
        if risk_profile.atr_value:
            score += Decimal('0.2')
        
        if risk_profile.volatility:
            score += Decimal('0.1')
        
        if risk_profile.beta:
            score += Decimal('0.1')
        
        if scanner_data:
            if 'volume_ratio' in scanner_data:
                score += Decimal('0.05')
            if 'price_trend' in scanner_data:
                score += Decimal('0.05')
        
        return min(Decimal('1.0'), score)
    
    def get_risk_adjusted_allocation(self, candidates: List[Dict], account_id: str = "default") -> List[Dict]:
        """
        Get risk-adjusted allocation for multiple securities.
        
        Args:
            candidates: List of candidate securities from scanner
            account_id: Account identifier
            
        Returns:
            List of allocation recommendations
        """
        try:
            account_params = self.get_account_parameters(account_id)
            if not account_params:
                return []
            
            allocations = []
            
            for candidate in candidates:
                symbol = candidate['symbol']
                current_price = Decimal(str(candidate['current_price']))
                
                # Calculate position size
                recommendation = self.calculate_position_size(
                    symbol=symbol,
                    current_price=current_price,
                    account_id=account_id,
                    scanner_data=candidate
                )
                
                if recommendation:
                    allocations.append({
                        'symbol': symbol,
                        'recommended_shares': recommendation.recommended_shares,
                        'recommended_amount': recommendation.recommended_dollar_amount,
                        'risk_amount': recommendation.total_risk_amount,
                        'confidence_score': recommendation.confidence_score,
                        'stop_loss_price': recommendation.stop_loss_price,
                        'target_price': recommendation.target_price
                    })
            
            # Sort by confidence score and risk-adjusted return
            allocations.sort(key=lambda x: float(x['confidence_score']), reverse=True)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to calculate risk-adjusted allocation: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Initialize risk calculator
    risk_calc = RiskCalculator()
    
    # Example: Calculate position size for a stock
    symbol = "AAPL"
    current_price = Decimal('150.00')
    
    # Create some sample price data for ATR calculation
    price_data = pd.DataFrame({
        'high': [152, 154, 151, 153, 155, 157, 156, 158, 160, 159, 161, 163, 162, 164, 166],
        'low': [148, 150, 147, 149, 151, 153, 152, 154, 156, 155, 157, 159, 158, 160, 162],
        'close': [150, 152, 149, 151, 153, 155, 154, 156, 158, 157, 159, 161, 160, 162, 164]
    })
    
    # Calculate ATR
    atr = risk_calc.calculate_atr_from_data(symbol, price_data)
    print(f"ATR for {symbol}: {atr}")
    
    # Calculate position size
    recommendation = risk_calc.calculate_position_size(
        symbol=symbol,
        current_price=current_price
    )
    
    if recommendation:
        print(f"Position size recommendation for {symbol}:")
        print(f"  Recommended shares: {recommendation.recommended_shares}")
        print(f"  Recommended amount: ${recommendation.recommended_dollar_amount}")
        print(f"  Risk per share: ${recommendation.risk_per_share}")
        print(f"  Total risk: ${recommendation.total_risk_amount}")
        print(f"  Stop loss: ${recommendation.stop_loss_price}")
        print(f"  Target price: ${recommendation.target_price}")
        print(f"  Kelly fraction used: {recommendation.kelly_fraction_used}")
        print(f"  ATR adjustment: {recommendation.atr_adjustment_factor}")
        print(f"  Confidence score: {recommendation.confidence_score}")