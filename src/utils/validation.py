"""
Comprehensive input validation utilities for the professional trading platform.

This module provides pydantic-based data validation, financial data integrity checking,
API response validation, and real-time user input verification with error highlighting
for professional trading applications.

Key Features:
- Trading parameter validation using pydantic models
- Financial data integrity validation for price, volume, and technical indicators
- API response validation for Yahoo Finance, IEX Cloud, and Alpha Vantage
- Real-time validation with error highlighting for invalid configurations
- Trading simulation validation for position sizes and risk parameters
- Configuration validation for JSON schema compliance

Author: Blitzy Development Team
Version: 1.0.0
"""

import re
import json
import logging
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, date, time, timezone
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np
from pydantic import (
    BaseModel, 
    Field, 
    validator, 
    root_validator,
    ValidationError,
    EmailStr,
    constr,
    confloat,
    conint,
    HttpUrl
)


# Configure logging for validation operations
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Enumeration for validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataProvider(Enum):
    """Enumeration for supported data providers."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"


class TradeDirection(Enum):
    """Enumeration for trade directions."""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    """Enumeration for trade status values."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"


class AlertConditionType(Enum):
    """Enumeration for alert condition types."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    VOLUME_SPIKE = "volume_spike"
    PERCENT_CHANGE = "percent_change"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    MOVING_AVERAGE_CROSS = "moving_average_cross"


class ComparisonOperator(Enum):
    """Enumeration for comparison operators in alerts."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL_TO = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="


@dataclass
class ValidationResult:
    """Data class representing validation results with error details."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    severity: ValidationSeverity
    field_errors: Dict[str, List[str]]
    corrected_data: Optional[Dict[str, Any]] = None


class FinancialPrecisionValidator:
    """Validator for financial calculations requiring decimal precision."""
    
    @staticmethod
    def validate_decimal_precision(value: Union[str, float, Decimal], max_decimal_places: int = 4) -> Decimal:
        """
        Validate and convert financial values to Decimal with specified precision.
        
        Args:
            value: Input value to validate and convert
            max_decimal_places: Maximum number of decimal places allowed
            
        Returns:
            Decimal: Validated decimal value
            
        Raises:
            ValidationError: If value cannot be converted or exceeds precision limits
        """
        try:
            if isinstance(value, str):
                # Remove common formatting characters
                clean_value = value.replace(',', '').replace('$', '').strip()
                decimal_value = Decimal(clean_value)
            elif isinstance(value, float):
                # Convert float to string first to avoid floating point precision issues
                decimal_value = Decimal(str(value))
            elif isinstance(value, Decimal):
                decimal_value = value
            else:
                raise ValidationError(f"Invalid type for financial value: {type(value)}")
            
            # Check for excessive decimal places
            if decimal_value.as_tuple().exponent < -max_decimal_places:
                # Round to maximum allowed precision
                quantize_value = Decimal('0.' + '0' * max_decimal_places)
                decimal_value = decimal_value.quantize(quantize_value, rounding=ROUND_HALF_UP)
            
            return decimal_value
            
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid financial value: {value} - {str(e)}")
    
    @staticmethod
    def validate_percentage(value: Union[str, float, Decimal], min_val: float = 0.0, max_val: float = 100.0) -> Decimal:
        """
        Validate percentage values with proper decimal precision.
        
        Args:
            value: Percentage value to validate
            min_val: Minimum allowed percentage
            max_val: Maximum allowed percentage
            
        Returns:
            Decimal: Validated percentage as decimal
        """
        decimal_value = FinancialPrecisionValidator.validate_decimal_precision(value, 2)
        
        if decimal_value < Decimal(str(min_val)) or decimal_value > Decimal(str(max_val)):
            raise ValidationError(f"Percentage {decimal_value} outside valid range [{min_val}, {max_val}]")
        
        return decimal_value


class TradingParameterValidator(BaseModel):
    """Pydantic model for validating trading parameters and risk settings."""
    
    symbol: constr(regex=r'^[A-Z]{1,5}$') = Field(..., description="Stock symbol (1-5 uppercase letters)")
    direction: TradeDirection = Field(..., description="Trade direction (BUY/SELL)")
    quantity: conint(gt=0, le=1000000) = Field(..., description="Share quantity (1-1,000,000)")
    entry_price: confloat(gt=0.0, le=10000.0) = Field(..., description="Entry price per share")
    stop_loss_price: Optional[confloat(gt=0.0, le=10000.0)] = Field(None, description="Stop loss price")
    target_price: Optional[confloat(gt=0.0, le=10000.0)] = Field(None, description="Target price")
    position_size_percent: confloat(ge=0.1, le=100.0) = Field(..., description="Position size as percentage of account")
    risk_amount: confloat(ge=0.01, le=100000.0) = Field(..., description="Maximum risk amount in dollars")
    
    class Config:
        """Pydantic configuration for trading parameters."""
        use_enum_values = True
        validate_assignment = True
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "direction": "BUY",
                "quantity": 100,
                "entry_price": 150.25,
                "stop_loss_price": 145.00,
                "target_price": 160.00,
                "position_size_percent": 2.5,
                "risk_amount": 500.00
            }
        }
    
    @validator('stop_loss_price')
    def validate_stop_loss(cls, v, values):
        """Validate stop loss price relative to entry price and direction."""
        if v is None:
            return v
        
        if 'entry_price' not in values or 'direction' not in values:
            return v
        
        entry_price = values['entry_price']
        direction = values['direction']
        
        if direction == TradeDirection.BUY and v >= entry_price:
            raise ValueError("Stop loss for BUY trade must be below entry price")
        elif direction == TradeDirection.SELL and v <= entry_price:
            raise ValueError("Stop loss for SELL trade must be above entry price")
        
        return v
    
    @validator('target_price')
    def validate_target_price(cls, v, values):
        """Validate target price relative to entry price and direction."""
        if v is None:
            return v
        
        if 'entry_price' not in values or 'direction' not in values:
            return v
        
        entry_price = values['entry_price']
        direction = values['direction']
        
        if direction == TradeDirection.BUY and v <= entry_price:
            raise ValueError("Target price for BUY trade must be above entry price")
        elif direction == TradeDirection.SELL and v >= entry_price:
            raise ValueError("Target price for SELL trade must be below entry price")
        
        return v
    
    @root_validator
    def validate_risk_reward_ratio(cls, values):
        """Validate risk-reward ratio is reasonable."""
        entry_price = values.get('entry_price')
        stop_loss_price = values.get('stop_loss_price')
        target_price = values.get('target_price')
        
        if all([entry_price, stop_loss_price, target_price]):
            risk = abs(entry_price - stop_loss_price)
            reward = abs(target_price - entry_price)
            
            if risk > 0 and reward / risk < 0.5:
                raise ValueError("Risk-reward ratio too unfavorable (reward should be at least 50% of risk)")
        
        return values


class MarketDataValidator(BaseModel):
    """Pydantic model for validating market data integrity."""
    
    symbol: constr(regex=r'^[A-Z]{1,5}$') = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    open_price: confloat(gt=0.0) = Field(..., description="Opening price")
    high_price: confloat(gt=0.0) = Field(..., description="High price")
    low_price: confloat(gt=0.0) = Field(..., description="Low price")
    close_price: confloat(gt=0.0) = Field(..., description="Closing price")
    volume: conint(ge=0) = Field(..., description="Trading volume")
    data_provider: DataProvider = Field(..., description="Data source provider")
    
    class Config:
        """Pydantic configuration for market data."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('high_price')
    def validate_high_price(cls, v, values):
        """Validate high price is the highest of OHLC."""
        if 'open_price' in values and v < values['open_price']:
            raise ValueError("High price cannot be less than open price")
        if 'low_price' in values and v < values['low_price']:
            raise ValueError("High price cannot be less than low price")
        if 'close_price' in values and v < values['close_price']:
            raise ValueError("High price cannot be less than close price")
        return v
    
    @validator('low_price')
    def validate_low_price(cls, v, values):
        """Validate low price is the lowest of OHLC."""
        if 'open_price' in values and v > values['open_price']:
            raise ValueError("Low price cannot be greater than open price")
        if 'close_price' in values and v > values['close_price']:
            raise ValueError("Low price cannot be greater than close price")
        return v
    
    @validator('volume')
    def validate_volume_reasonableness(cls, v):
        """Validate volume is within reasonable bounds."""
        if v > 1_000_000_000:  # 1 billion shares
            raise ValueError("Volume appears unreasonably high")
        return v
    
    @root_validator
    def validate_price_consistency(cls, values):
        """Validate overall price data consistency."""
        open_price = values.get('open_price')
        high_price = values.get('high_price')
        low_price = values.get('low_price')
        close_price = values.get('close_price')
        
        if all([open_price, high_price, low_price, close_price]):
            # Check if high is actually the highest
            prices = [open_price, high_price, low_price, close_price]
            if high_price != max(prices):
                raise ValueError("High price is not the maximum of OHLC prices")
            
            # Check if low is actually the lowest
            if low_price != min(prices):
                raise ValueError("Low price is not the minimum of OHLC prices")
            
            # Check for reasonable price gaps (no more than 50% change)
            max_change = max(abs(high_price - low_price) / open_price, 
                           abs(close_price - open_price) / open_price)
            if max_change > 0.5:
                raise ValueError("Price change exceeds 50% - possible data error")
        
        return values


class APIResponseValidator(BaseModel):
    """Pydantic model for validating API responses from data providers."""
    
    provider: DataProvider = Field(..., description="API data provider")
    request_timestamp: datetime = Field(..., description="Request timestamp")
    response_timestamp: datetime = Field(..., description="Response timestamp")
    status_code: conint(ge=200, le=599) = Field(..., description="HTTP status code")
    response_data: Dict[str, Any] = Field(..., description="Raw response data")
    data_quality_score: confloat(ge=0.0, le=1.0) = Field(default=1.0, description="Data quality score")
    
    class Config:
        """Pydantic configuration for API responses."""
        use_enum_values = True
        validate_assignment = True
    
    @validator('response_timestamp')
    def validate_response_timing(cls, v, values):
        """Validate response timestamp is after request timestamp."""
        if 'request_timestamp' in values and v < values['request_timestamp']:
            raise ValueError("Response timestamp cannot be before request timestamp")
        return v
    
    @validator('status_code')
    def validate_success_status(cls, v):
        """Validate HTTP status code indicates success."""
        if v >= 400:
            raise ValueError(f"HTTP error status code: {v}")
        return v
    
    @root_validator
    def validate_provider_response_format(cls, values):
        """Validate response format matches provider expectations."""
        provider = values.get('provider')
        response_data = values.get('response_data', {})
        
        if provider == DataProvider.YAHOO_FINANCE:
            # Yahoo Finance should have 'chart' or 'quoteResponse' structure
            if not any(key in response_data for key in ['chart', 'quoteResponse', 'quoteSummary']):
                values['data_quality_score'] = 0.3
        
        elif provider == DataProvider.ALPHA_VANTAGE:
            # Alpha Vantage should have time series data or meta data
            if not any(key in response_data for key in ['Time Series (Daily)', 'Meta Data', 'Global Quote']):
                values['data_quality_score'] = 0.3
        
        elif provider == DataProvider.IEX_CLOUD:
            # IEX Cloud should have standard JSON structure
            if not isinstance(response_data, (dict, list)):
                values['data_quality_score'] = 0.3
        
        return values


class PortfolioValidator(BaseModel):
    """Pydantic model for validating portfolio positions and calculations."""
    
    symbol: constr(regex=r'^[A-Z]{1,5}$') = Field(..., description="Stock symbol")
    quantity: conint(ge=0) = Field(..., description="Share quantity")
    average_cost: confloat(gt=0.0) = Field(..., description="Average cost per share")
    current_price: confloat(gt=0.0) = Field(..., description="Current market price")
    allocation_bucket: constr(regex=r'^(Core|Exploratory|Cash)$') = Field(..., description="Allocation bucket")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        """Pydantic configuration for portfolio data."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position."""
        return FinancialPrecisionValidator.validate_decimal_precision(
            self.quantity * self.current_price
        )
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized profit/loss."""
        cost_basis = FinancialPrecisionValidator.validate_decimal_precision(
            self.quantity * self.average_cost
        )
        return self.market_value - cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Calculate unrealized profit/loss percentage."""
        if self.average_cost == 0:
            return Decimal('0')
        
        pnl_pct = (self.current_price - self.average_cost) / self.average_cost * 100
        return FinancialPrecisionValidator.validate_decimal_precision(pnl_pct, 2)
    
    @validator('current_price')
    def validate_reasonable_price_change(cls, v, values):
        """Validate current price is within reasonable bounds of average cost."""
        if 'average_cost' in values:
            avg_cost = values['average_cost']
            price_change_pct = abs(v - avg_cost) / avg_cost
            
            # Warning for extreme price changes (>90%)
            if price_change_pct > 0.9:
                logger.warning(f"Extreme price change detected: {price_change_pct:.1%}")
        
        return v


class RiskManagementValidator(BaseModel):
    """Pydantic model for validating risk management parameters."""
    
    account_value: confloat(gt=0.0) = Field(..., description="Total account value")
    risk_per_trade_percent: confloat(ge=0.1, le=10.0) = Field(..., description="Risk per trade percentage")
    max_position_size_percent: confloat(ge=0.1, le=25.0) = Field(..., description="Maximum position size percentage")
    core_allocation_percent: confloat(ge=40.0, le=80.0) = Field(default=60.0, description="Core allocation percentage")
    exploratory_allocation_percent: confloat(ge=20.0, le=50.0) = Field(default=30.0, description="Exploratory allocation percentage")
    cash_allocation_percent: confloat(ge=5.0, le=20.0) = Field(default=10.0, description="Cash allocation percentage")
    
    class Config:
        """Pydantic configuration for risk management."""
        validate_assignment = True
        schema_extra = {
            "example": {
                "account_value": 100000.0,
                "risk_per_trade_percent": 2.0,
                "max_position_size_percent": 5.0,
                "core_allocation_percent": 60.0,
                "exploratory_allocation_percent": 30.0,
                "cash_allocation_percent": 10.0
            }
        }
    
    @root_validator
    def validate_allocation_totals(cls, values):
        """Validate allocation percentages sum to 100%."""
        core = values.get('core_allocation_percent', 0)
        exploratory = values.get('exploratory_allocation_percent', 0)
        cash = values.get('cash_allocation_percent', 0)
        
        total = core + exploratory + cash
        if abs(total - 100.0) > 0.01:  # Allow for small floating point errors
            raise ValueError(f"Allocation percentages must sum to 100%, got {total}%")
        
        return values
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> Tuple[int, Decimal]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            
        Returns:
            Tuple of (share_quantity, dollar_amount)
        """
        risk_amount = self.account_value * (self.risk_per_trade_percent / 100)
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            raise ValueError("Stop loss price cannot equal entry price")
        
        # Calculate share quantity based on risk amount
        calculated_quantity = int(risk_amount / price_risk)
        
        # Check against maximum position size
        max_position_value = self.account_value * (self.max_position_size_percent / 100)
        max_quantity = int(max_position_value / entry_price)
        
        final_quantity = min(calculated_quantity, max_quantity)
        final_dollar_amount = FinancialPrecisionValidator.validate_decimal_precision(
            final_quantity * entry_price
        )
        
        return final_quantity, final_dollar_amount


class AlertConfigurationValidator(BaseModel):
    """Pydantic model for validating alert configurations."""
    
    symbol: constr(regex=r'^[A-Z]{1,5}$') = Field(..., description="Stock symbol")
    condition_type: AlertConditionType = Field(..., description="Alert condition type")
    trigger_value: confloat(gt=0.0) = Field(..., description="Trigger threshold value")
    comparison_operator: ComparisonOperator = Field(..., description="Comparison operator")
    notification_method: constr(regex=r'^(windows|email|sound|all)$') = Field(..., description="Notification method")
    is_active: bool = Field(default=True, description="Alert active status")
    created_date: datetime = Field(default_factory=datetime.now, description="Alert creation date")
    
    class Config:
        """Pydantic configuration for alert settings."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('trigger_value')
    def validate_trigger_value_range(cls, v, values):
        """Validate trigger value is appropriate for condition type."""
        condition_type = values.get('condition_type')
        
        if condition_type in [AlertConditionType.RSI_OVERSOLD, AlertConditionType.RSI_OVERBOUGHT]:
            if not (0 <= v <= 100):
                raise ValueError("RSI trigger value must be between 0 and 100")
        elif condition_type == AlertConditionType.PERCENT_CHANGE:
            if v > 100:
                raise ValueError("Percentage change trigger should be reasonable (≤100%)")
        
        return v


class ConfigurationValidator(BaseModel):
    """Pydantic model for validating application configuration."""
    
    api_keys: Dict[str, str] = Field(..., description="API key configuration")
    refresh_intervals: Dict[str, conint(ge=1, le=3600)] = Field(..., description="Refresh intervals in seconds")
    display_preferences: Dict[str, Union[str, bool, int]] = Field(..., description="Display preferences")
    export_settings: Dict[str, Union[str, bool]] = Field(..., description="Export configuration")
    performance_settings: Dict[str, Union[int, float, bool]] = Field(..., description="Performance settings")
    
    class Config:
        """Pydantic configuration for application settings."""
        validate_assignment = True
        schema_extra = {
            "example": {
                "api_keys": {
                    "yahoo_finance": "not_required",
                    "alpha_vantage": "demo_key",
                    "iex_cloud": "pk_test_key"
                },
                "refresh_intervals": {
                    "scanner": 30,
                    "portfolio": 60,
                    "alerts": 15
                },
                "display_preferences": {
                    "theme": "dark",
                    "font_size": 12,
                    "show_charts": True
                },
                "export_settings": {
                    "auto_backup": True,
                    "export_format": "excel"
                },
                "performance_settings": {
                    "max_concurrent_requests": 10,
                    "cache_timeout": 300,
                    "enable_logging": True
                }
            }
        }
    
    @validator('api_keys')
    def validate_api_key_formats(cls, v):
        """Validate API key formats for different providers."""
        for provider, key in v.items():
            if provider == "alpha_vantage" and key != "demo_key":
                if not re.match(r'^[A-Z0-9]{16}$', key):
                    raise ValueError(f"Invalid Alpha Vantage API key format: {key}")
            elif provider == "iex_cloud" and key.startswith("pk_"):
                if not re.match(r'^pk_[a-f0-9]{32}$', key):
                    raise ValueError(f"Invalid IEX Cloud API key format: {key}")
        
        return v
    
    @validator('refresh_intervals')
    def validate_refresh_intervals(cls, v):
        """Validate refresh intervals are reasonable."""
        required_intervals = ['scanner', 'portfolio', 'alerts']
        for interval_type in required_intervals:
            if interval_type not in v:
                raise ValueError(f"Missing required refresh interval: {interval_type}")
        
        # Scanner should refresh more frequently than portfolio
        if v.get('scanner', 30) > v.get('portfolio', 60):
            logger.warning("Scanner refresh interval is longer than portfolio refresh interval")
        
        return v


class ValidationEngine:
    """Main validation engine for coordinating all validation operations."""
    
    def __init__(self):
        """Initialize the validation engine."""
        self.logger = logging.getLogger(__name__)
        self._validation_cache = {}
    
    def validate_trading_parameters(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate trading parameters using comprehensive business rules.
        
        Args:
            data: Trading parameter data to validate
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        try:
            validator = TradingParameterValidator(**data)
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                severity=ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "trading_parameters")
    
    def validate_market_data(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate market data integrity and consistency.
        
        Args:
            data: Market data to validate
            
        Returns:
            ValidationResult: Validation results with data quality assessment
        """
        try:
            validator = MarketDataValidator(**data)
            
            # Additional statistical validation
            warnings = []
            if self._detect_price_anomalies(data):
                warnings.append("Potential price anomaly detected")
            
            if self._detect_volume_anomalies(data):
                warnings.append("Unusual volume pattern detected")
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                severity=ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "market_data")
    
    def validate_api_response(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate API response format and data quality.
        
        Args:
            data: API response data to validate
            
        Returns:
            ValidationResult: Validation results with quality score
        """
        try:
            validator = APIResponseValidator(**data)
            
            warnings = []
            if validator.data_quality_score < 0.8:
                warnings.append(f"Low data quality score: {validator.data_quality_score:.2f}")
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                severity=ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "api_response")
    
    def validate_portfolio_data(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate portfolio position data and calculations.
        
        Args:
            data: Portfolio data to validate
            
        Returns:
            ValidationResult: Validation results with P/L accuracy checks
        """
        try:
            validator = PortfolioValidator(**data)
            
            # Additional financial calculation validation
            warnings = []
            if abs(validator.unrealized_pnl_percent) > 50:
                warnings.append(f"Large unrealized P/L: {validator.unrealized_pnl_percent:.1f}%")
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                severity=ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "portfolio_data")
    
    def validate_risk_parameters(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate risk management parameters and allocation strategies.
        
        Args:
            data: Risk management data to validate
            
        Returns:
            ValidationResult: Validation results with risk assessment
        """
        try:
            validator = RiskManagementValidator(**data)
            
            warnings = []
            if validator.risk_per_trade_percent > 5.0:
                warnings.append("High risk per trade percentage (>5%)")
            
            if validator.max_position_size_percent > 15.0:
                warnings.append("High maximum position size (>15%)")
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=warnings,
                severity=ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "risk_parameters")
    
    def validate_alert_configuration(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate alert configuration and notification settings.
        
        Args:
            data: Alert configuration data to validate
            
        Returns:
            ValidationResult: Validation results with alert feasibility checks
        """
        try:
            validator = AlertConfigurationValidator(**data)
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                severity=ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "alert_configuration")
    
    def validate_application_config(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate application configuration and settings.
        
        Args:
            data: Application configuration to validate
            
        Returns:
            ValidationResult: Validation results with configuration compliance
        """
        try:
            validator = ConfigurationValidator(**data)
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                severity=ValidationSeverity.INFO,
                field_errors={}
            )
        except ValidationError as e:
            return self._process_pydantic_error(e, "application_config")
    
    def validate_real_time_input(self, field_name: str, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """
        Perform real-time validation of user input with error highlighting.
        
        Args:
            field_name: Name of the field being validated
            value: Input value to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult: Real-time validation results
        """
        context = context or {}
        errors = []
        warnings = []
        
        try:
            # Symbol validation
            if field_name == "symbol":
                if not re.match(r'^[A-Z]{1,5}$', str(value)):
                    errors.append("Symbol must be 1-5 uppercase letters")
            
            # Price validation
            elif field_name in ["entry_price", "stop_loss_price", "target_price", "current_price"]:
                try:
                    price_val = float(value)
                    if price_val <= 0:
                        errors.append("Price must be greater than zero")
                    elif price_val > 10000:
                        warnings.append("Price is unusually high")
                except (ValueError, TypeError):
                    errors.append("Price must be a valid number")
            
            # Quantity validation
            elif field_name == "quantity":
                try:
                    qty_val = int(value)
                    if qty_val <= 0:
                        errors.append("Quantity must be greater than zero")
                    elif qty_val > 1000000:
                        errors.append("Quantity exceeds maximum limit (1,000,000)")
                except (ValueError, TypeError):
                    errors.append("Quantity must be a valid integer")
            
            # Percentage validation
            elif field_name in ["position_size_percent", "risk_per_trade_percent"]:
                try:
                    pct_val = float(value)
                    if pct_val < 0.1:
                        errors.append("Percentage must be at least 0.1%")
                    elif pct_val > 100:
                        errors.append("Percentage cannot exceed 100%")
                    elif field_name == "risk_per_trade_percent" and pct_val > 5:
                        warnings.append("Risk per trade >5% is considered high")
                except (ValueError, TypeError):
                    errors.append("Percentage must be a valid number")
            
            # Account value validation
            elif field_name == "account_value":
                try:
                    account_val = float(value)
                    if account_val <= 0:
                        errors.append("Account value must be greater than zero")
                    elif account_val < 1000:
                        warnings.append("Account value is very low")
                except (ValueError, TypeError):
                    errors.append("Account value must be a valid number")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                severity=ValidationSeverity.ERROR if errors else (
                    ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO
                ),
                field_errors={field_name: errors} if errors else {}
            )
        
        except Exception as e:
            self.logger.error(f"Real-time validation error for field {field_name}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                severity=ValidationSeverity.ERROR,
                field_errors={field_name: [f"Validation error: {str(e)}"]}
            )
    
    def _process_pydantic_error(self, error: ValidationError, context: str) -> ValidationResult:
        """Process pydantic validation errors into structured format."""
        errors = []
        field_errors = {}
        
        for error_detail in error.errors():
            field_path = '.'.join(str(loc) for loc in error_detail['loc'])
            error_msg = error_detail['msg']
            
            errors.append(f"{field_path}: {error_msg}")
            
            if field_path not in field_errors:
                field_errors[field_path] = []
            field_errors[field_path].append(error_msg)
        
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=[],
            severity=ValidationSeverity.ERROR,
            field_errors=field_errors
        )
    
    def _detect_price_anomalies(self, data: Dict[str, Any]) -> bool:
        """Detect potential price anomalies using statistical analysis."""
        try:
            open_price = data.get('open_price', 0)
            high_price = data.get('high_price', 0)
            low_price = data.get('low_price', 0)
            close_price = data.get('close_price', 0)
            
            if all([open_price, high_price, low_price, close_price]):
                # Check for extreme price gaps
                intraday_range = (high_price - low_price) / open_price
                if intraday_range > 0.2:  # More than 20% intraday range
                    return True
                
                # Check for unusual price patterns
                if low_price == high_price and open_price != close_price:
                    return True  # Flat range with different open/close
                
        except (TypeError, ZeroDivisionError):
            return True  # Data quality issue
        
        return False
    
    def _detect_volume_anomalies(self, data: Dict[str, Any]) -> bool:
        """Detect unusual volume patterns."""
        try:
            volume = data.get('volume', 0)
            
            # Check for suspiciously low volume
            if volume == 0:
                return True
            
            # Check for extremely high volume (>100M shares)
            if volume > 100_000_000:
                return True
            
        except (TypeError, ValueError):
            return True
        
        return False


# Instantiate global validation engine
validation_engine = ValidationEngine()


def validate_trading_parameters(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for trading parameter validation."""
    return validation_engine.validate_trading_parameters(data)


def validate_market_data(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for market data validation."""
    return validation_engine.validate_market_data(data)


def validate_api_response(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for API response validation."""
    return validation_engine.validate_api_response(data)


def validate_portfolio_data(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for portfolio data validation."""
    return validation_engine.validate_portfolio_data(data)


def validate_risk_parameters(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for risk parameter validation."""
    return validation_engine.validate_risk_parameters(data)


def validate_alert_configuration(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for alert configuration validation."""
    return validation_engine.validate_alert_configuration(data)


def validate_application_config(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for application configuration validation."""
    return validation_engine.validate_application_config(data)


def validate_real_time_input(field_name: str, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
    """Convenience function for real-time input validation."""
    return validation_engine.validate_real_time_input(field_name, value, context)


# Export all validation classes and functions
__all__ = [
    'ValidationSeverity',
    'DataProvider', 
    'TradeDirection',
    'TradeStatus',
    'AlertConditionType',
    'ComparisonOperator',
    'ValidationResult',
    'FinancialPrecisionValidator',
    'TradingParameterValidator',
    'MarketDataValidator',
    'APIResponseValidator',
    'PortfolioValidator',
    'RiskManagementValidator',
    'AlertConfigurationValidator',
    'ConfigurationValidator',
    'ValidationEngine',
    'validation_engine',
    'validate_trading_parameters',
    'validate_market_data',
    'validate_api_response',
    'validate_portfolio_data',
    'validate_risk_parameters',
    'validate_alert_configuration',
    'validate_application_config',
    'validate_real_time_input'
]