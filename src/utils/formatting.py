"""
Financial Data Formatting Utilities for Professional Trading Platform

This module provides comprehensive formatting utilities for the desktop trading application,
implementing decimal-precision calculations, currency formatting, percentage calculations,
and data validation with professional-grade accuracy standards required for monetary computations.

Features:
- Decimal-precision financial calculations using Python's decimal module
- Currency formatting with proper precision handling for trading calculations
- Percentage formatting for risk calculations, performance metrics, and scanner displays
- Number formatting for large datasets with thousands separators and scientific notation
- Date/time formatting optimized for financial market data and trading sessions
- Data validation and sanitization for API responses and user input processing

Technical Compliance:
- Follows financial precision standards from technical specification 3.2.2
- Implements DECIMAL(10,4) price formatting and DECIMAL(12,2) P/L calculations
- Supports database schema requirements from section 6.2
- Adheres to professional visual design standards from section 7.7
"""

import re
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, time, date
from typing import Union, Optional, Any, Dict, List
import math

# Set decimal precision context for financial calculations
# Using 28 places for maximum precision in financial computations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

# Financial precision constants aligned with database schema
PRICE_PRECISION = Decimal('0.0001')  # DECIMAL(10,4) for prices
PNL_PRECISION = Decimal('0.01')      # DECIMAL(12,2) for P/L calculations
PERCENTAGE_PRECISION = Decimal('0.0001')  # DECIMAL(5,4) for percentages
SLIPPAGE_PRECISION = Decimal('0.0001')   # DECIMAL(10,4) for slippage amounts

# Market data formatting constants
VOLUME_THOUSANDS_THRESHOLD = 1000
MARKET_CAP_MILLIONS_THRESHOLD = 1000000
FLOAT_MILLIONS_THRESHOLD = 1000000

# Trading session time constants for market data formatting
MARKET_OPEN_TIME = time(9, 30, 0)   # 9:30 AM EST
MARKET_CLOSE_TIME = time(16, 0, 0)  # 4:00 PM EST
PREMARKET_START_TIME = time(4, 0, 0)  # 4:00 AM EST
AFTERHOURS_END_TIME = time(20, 0, 0)  # 8:00 PM EST


class FormattingError(Exception):
    """Custom exception for formatting operations errors."""
    pass


class DecimalValidator:
    """
    Validator class for decimal precision financial data.
    Ensures all monetary values maintain proper precision for professional trading calculations.
    """
    
    @staticmethod
    def validate_price(value: Union[str, int, float, Decimal]) -> Decimal:
        """
        Validate and convert price values to proper decimal precision.
        
        Args:
            value: Price value to validate and convert
            
        Returns:
            Decimal: Validated price with DECIMAL(10,4) precision
            
        Raises:
            FormattingError: If value cannot be converted to valid price
        """
        try:
            if value is None:
                raise FormattingError("Price value cannot be None")
            
            decimal_value = Decimal(str(value))
            
            # Check for reasonable price range (0.0001 to 999999.9999)
            if decimal_value < Decimal('0.0001'):
                raise FormattingError(f"Price {decimal_value} below minimum threshold")
            if decimal_value > Decimal('999999.9999'):
                raise FormattingError(f"Price {decimal_value} exceeds maximum threshold")
                
            return decimal_value.quantize(PRICE_PRECISION)
            
        except (InvalidOperation, ValueError, TypeError) as e:
            raise FormattingError(f"Invalid price value '{value}': {str(e)}")
    
    @staticmethod
    def validate_pnl(value: Union[str, int, float, Decimal]) -> Decimal:
        """
        Validate and convert P/L values to proper decimal precision.
        
        Args:
            value: P/L value to validate and convert
            
        Returns:
            Decimal: Validated P/L with DECIMAL(12,2) precision
            
        Raises:
            FormattingError: If value cannot be converted to valid P/L
        """
        try:
            if value is None:
                return Decimal('0.00')
            
            decimal_value = Decimal(str(value))
            
            # Check for reasonable P/L range (-9999999999.99 to 9999999999.99)
            if abs(decimal_value) > Decimal('9999999999.99'):
                raise FormattingError(f"P/L value {decimal_value} exceeds maximum range")
                
            return decimal_value.quantize(PNL_PRECISION)
            
        except (InvalidOperation, ValueError, TypeError) as e:
            raise FormattingError(f"Invalid P/L value '{value}': {str(e)}")
    
    @staticmethod
    def validate_percentage(value: Union[str, int, float, Decimal]) -> Decimal:
        """
        Validate and convert percentage values to proper decimal precision.
        
        Args:
            value: Percentage value to validate and convert (as decimal, e.g., 0.05 for 5%)
            
        Returns:
            Decimal: Validated percentage with DECIMAL(5,4) precision
            
        Raises:
            FormattingError: If value cannot be converted to valid percentage
        """
        try:
            if value is None:
                return Decimal('0.0000')
            
            decimal_value = Decimal(str(value))
            
            # Check for reasonable percentage range (-99.9999 to 99.9999)
            if abs(decimal_value) > Decimal('99.9999'):
                raise FormattingError(f"Percentage {decimal_value} exceeds maximum range")
                
            return decimal_value.quantize(PERCENTAGE_PRECISION)
            
        except (InvalidOperation, ValueError, TypeError) as e:
            raise FormattingError(f"Invalid percentage value '{value}': {str(e)}")


class CurrencyFormatter:
    """
    Professional currency formatting utilities for trading calculations and portfolio values.
    Implements proper precision handling with support for various display contexts.
    """
    
    @staticmethod
    def format_price(price: Union[str, int, float, Decimal], 
                    symbol: str = '$', 
                    show_cents: bool = True,
                    color_coding: bool = False) -> str:
        """
        Format price values with professional trading precision.
        
        Args:
            price: Price value to format
            symbol: Currency symbol (default: '$')
            show_cents: Whether to show decimal places
            color_coding: Whether to add color coding for gains/losses
            
        Returns:
            str: Formatted price string
            
        Examples:
            >>> format_price(123.4567)
            '$123.46'
            >>> format_price(123.4567, show_cents=False)
            '$123'
            >>> format_price(0.0001)
            '$0.0001'
        """
        try:
            validated_price = DecimalValidator.validate_price(price)
            
            if validated_price == 0:
                return f"{symbol}0.00" if show_cents else f"{symbol}0"
            
            # Handle very small prices (penny stocks) with full precision
            if validated_price < Decimal('1.0000'):
                formatted = f"{symbol}{validated_price:,.4f}"
            elif show_cents:
                formatted = f"{symbol}{validated_price:,.2f}"
            else:
                formatted = f"{symbol}{int(validated_price):,}"
            
            # Add color coding if requested (for UI display)
            if color_coding and validated_price > 0:
                # This would be used with GUI color styling
                formatted = f"<span style='color: #4CAF50'>{formatted}</span>"
            
            return formatted
            
        except FormattingError:
            return f"{symbol}--"
    
    @staticmethod
    def format_pnl(pnl: Union[str, int, float, Decimal],
                   symbol: str = '$',
                   show_sign: bool = True,
                   color_coding: bool = True) -> str:
        """
        Format P/L values with appropriate sign and color indicators.
        
        Args:
            pnl: P/L value to format
            symbol: Currency symbol (default: '$')
            show_sign: Whether to show + sign for positive values
            color_coding: Whether to add color coding for gains/losses
            
        Returns:
            str: Formatted P/L string
            
        Examples:
            >>> format_pnl(123.45)
            '+$123.45'
            >>> format_pnl(-67.89)
            '-$67.89'
            >>> format_pnl(0)
            '$0.00'
        """
        try:
            validated_pnl = DecimalValidator.validate_pnl(pnl)
            
            if validated_pnl == 0:
                return f"{symbol}0.00"
            
            abs_pnl = abs(validated_pnl)
            formatted_amount = f"{symbol}{abs_pnl:,.2f}"
            
            if validated_pnl > 0:
                sign = "+" if show_sign else ""
                formatted = f"{sign}{formatted_amount}"
                if color_coding:
                    formatted = f"<span style='color: #4CAF50'>{formatted}</span>"
            else:
                formatted = f"-{formatted_amount}"
                if color_coding:
                    formatted = f"<span style='color: #F44336'>{formatted}</span>"
            
            return formatted
            
        except FormattingError:
            return f"{symbol}--"
    
    @staticmethod
    def format_portfolio_value(value: Union[str, int, float, Decimal],
                              symbol: str = '$') -> str:
        """
        Format large portfolio values with appropriate scaling.
        
        Args:
            value: Portfolio value to format
            symbol: Currency symbol (default: '$')
            
        Returns:
            str: Formatted portfolio value with scaling
            
        Examples:
            >>> format_portfolio_value(1234567.89)
            '$1.23M'
            >>> format_portfolio_value(12345.67)
            '$12,345.67'
            >>> format_portfolio_value(1234567890)
            '$1.23B'
        """
        try:
            validated_value = DecimalValidator.validate_pnl(value)
            abs_value = abs(validated_value)
            
            if abs_value >= Decimal('1000000000'):  # Billions
                scaled = abs_value / Decimal('1000000000')
                formatted = f"{symbol}{scaled:.2f}B"
            elif abs_value >= Decimal('1000000'):  # Millions
                scaled = abs_value / Decimal('1000000')
                formatted = f"{symbol}{scaled:.2f}M"
            elif abs_value >= Decimal('1000'):  # Thousands
                scaled = abs_value / Decimal('1000')
                formatted = f"{symbol}{scaled:.1f}K"
            else:
                formatted = f"{symbol}{abs_value:,.2f}"
            
            # Add negative sign if needed
            if validated_value < 0:
                formatted = f"-{formatted}"
                
            return formatted
            
        except FormattingError:
            return f"{symbol}--"


class PercentageFormatter:
    """
    Percentage formatting utilities for risk calculations, performance metrics, and scanner displays.
    Handles various percentage contexts with appropriate precision and visual indicators.
    """
    
    @staticmethod
    def format_percentage(value: Union[str, int, float, Decimal],
                         precision: int = 2,
                         show_sign: bool = True,
                         color_coding: bool = True) -> str:
        """
        Format percentage values with proper precision and visual indicators.
        
        Args:
            value: Percentage value as decimal (e.g., 0.05 for 5%)
            precision: Number of decimal places (default: 2)
            show_sign: Whether to show + sign for positive values
            color_coding: Whether to add color coding for gains/losses
            
        Returns:
            str: Formatted percentage string
            
        Examples:
            >>> format_percentage(0.0543)
            '+5.43%'
            >>> format_percentage(-0.0234)
            '-2.34%'
            >>> format_percentage(0.0001, precision=4)
            '+0.0100%'
        """
        try:
            validated_percentage = DecimalValidator.validate_percentage(value)
            
            if validated_percentage == 0:
                return "0.00%"
            
            # Convert to percentage (multiply by 100)
            percentage_value = validated_percentage * 100
            
            # Format with specified precision
            format_str = f"{{:.{precision}f}}%"
            formatted_percent = format_str.format(percentage_value)
            
            if validated_percentage > 0:
                sign = "+" if show_sign else ""
                formatted = f"{sign}{formatted_percent}"
                if color_coding:
                    formatted = f"<span style='color: #4CAF50'>{formatted}</span>"
            else:
                formatted = formatted_percent  # Already has negative sign
                if color_coding:
                    formatted = f"<span style='color: #F44336'>{formatted}</span>"
            
            return formatted
            
        except FormattingError:
            return "--%"
    
    @staticmethod
    def format_risk_percentage(risk_value: Union[str, int, float, Decimal]) -> str:
        """
        Format risk percentage values for position sizing and risk management.
        
        Args:
            risk_value: Risk percentage as decimal (e.g., 0.02 for 2% risk)
            
        Returns:
            str: Formatted risk percentage
            
        Examples:
            >>> format_risk_percentage(0.02)
            '2.00%'
            >>> format_risk_percentage(0.005)
            '0.50%'
        """
        try:
            validated_risk = DecimalValidator.validate_percentage(risk_value)
            
            # Risk percentages are always positive and don't need color coding
            percentage_value = validated_risk * 100
            return f"{percentage_value:.2f}%"
            
        except FormattingError:
            return "--%"
    
    @staticmethod
    def format_win_rate(wins: int, total_trades: int) -> str:
        """
        Calculate and format win rate percentage for trading performance.
        
        Args:
            wins: Number of winning trades
            total_trades: Total number of trades
            
        Returns:
            str: Formatted win rate percentage
            
        Examples:
            >>> format_win_rate(75, 100)
            '75.0%'
            >>> format_win_rate(0, 0)
            'N/A'
        """
        try:
            if total_trades == 0:
                return "N/A"
            
            win_rate = Decimal(wins) / Decimal(total_trades)
            percentage_value = win_rate * 100
            
            return f"{percentage_value:.1f}%"
            
        except (ZeroDivisionError, InvalidOperation):
            return "N/A"


class NumberFormatter:
    """
    Number formatting utilities for large datasets with thousands separators and scientific notation support.
    Handles volume, market cap, float shares, and other large numerical values in trading contexts.
    """
    
    @staticmethod
    def format_volume(volume: Union[str, int, float, Decimal]) -> str:
        """
        Format trading volume with appropriate scaling and separators.
        
        Args:
            volume: Trading volume to format
            
        Returns:
            str: Formatted volume string
            
        Examples:
            >>> format_volume(1234567)
            '1.23M'
            >>> format_volume(12345)
            '12.3K'
            >>> format_volume(123)
            '123'
        """
        try:
            volume_decimal = Decimal(str(volume))
            
            if volume_decimal < 0:
                return "0"
            
            if volume_decimal >= Decimal('1000000'):  # Millions
                scaled = volume_decimal / Decimal('1000000')
                return f"{scaled:.2f}M"
            elif volume_decimal >= Decimal('1000'):  # Thousands
                scaled = volume_decimal / Decimal('1000')
                return f"{scaled:.1f}K"
            else:
                return f"{int(volume_decimal):,}"
            
        except (InvalidOperation, ValueError, TypeError):
            return "0"
    
    @staticmethod
    def format_market_cap(market_cap: Union[str, int, float, Decimal]) -> str:
        """
        Format market capitalization with appropriate scaling.
        
        Args:
            market_cap: Market capitalization value
            
        Returns:
            str: Formatted market cap string
            
        Examples:
            >>> format_market_cap(12345678900)
            '$12.35B'
            >>> format_market_cap(1234567890)
            '$1.23B'
            >>> format_market_cap(123456789)
            '$123.46M'
        """
        try:
            cap_decimal = Decimal(str(market_cap))
            
            if cap_decimal < 0:
                return "$0"
            
            if cap_decimal >= Decimal('1000000000'):  # Billions
                scaled = cap_decimal / Decimal('1000000000')
                return f"${scaled:.2f}B"
            elif cap_decimal >= Decimal('1000000'):  # Millions
                scaled = cap_decimal / Decimal('1000000')
                return f"${scaled:.2f}M"
            elif cap_decimal >= Decimal('1000'):  # Thousands
                scaled = cap_decimal / Decimal('1000')
                return f"${scaled:.1f}K"
            else:
                return f"${int(cap_decimal):,}"
            
        except (InvalidOperation, ValueError, TypeError):
            return "$0"
    
    @staticmethod
    def format_float_shares(float_shares: Union[str, int, float, Decimal]) -> str:
        """
        Format float shares (publicly traded shares) with appropriate scaling.
        
        Args:
            float_shares: Number of float shares
            
        Returns:
            str: Formatted float shares string
            
        Examples:
            >>> format_float_shares(123456789)
            '123.46M'
            >>> format_float_shares(12345678)
            '12.35M'
            >>> format_float_shares(1234567)
            '1.23M'
        """
        try:
            shares_decimal = Decimal(str(float_shares))
            
            if shares_decimal < 0:
                return "0"
            
            if shares_decimal >= Decimal('1000000'):  # Millions
                scaled = shares_decimal / Decimal('1000000')
                return f"{scaled:.2f}M"
            elif shares_decimal >= Decimal('1000'):  # Thousands
                scaled = shares_decimal / Decimal('1000')
                return f"{scaled:.1f}K"
            else:
                return f"{int(shares_decimal):,}"
            
        except (InvalidOperation, ValueError, TypeError):
            return "0"
    
    @staticmethod
    def format_large_number(number: Union[str, int, float, Decimal],
                           precision: int = 2) -> str:
        """
        Format large numbers with scientific notation support for extreme values.
        
        Args:
            number: Number to format
            precision: Decimal places for scaling (default: 2)
            
        Returns:
            str: Formatted number string
            
        Examples:
            >>> format_large_number(1234567890123)
            '1.23T'
            >>> format_large_number(12345678901234567890)
            '1.23e+19'
        """
        try:
            num_decimal = Decimal(str(number))
            abs_num = abs(num_decimal)
            
            if abs_num >= Decimal('1000000000000'):  # Trillions
                scaled = abs_num / Decimal('1000000000000')
                formatted = f"{scaled:.{precision}f}T"
            elif abs_num >= Decimal('1000000000'):  # Billions
                scaled = abs_num / Decimal('1000000000')
                formatted = f"{scaled:.{precision}f}B"
            elif abs_num >= Decimal('1000000'):  # Millions
                scaled = abs_num / Decimal('1000000')
                formatted = f"{scaled:.{precision}f}M"
            elif abs_num >= Decimal('1000'):  # Thousands
                scaled = abs_num / Decimal('1000')
                formatted = f"{scaled:.{precision}f}K"
            else:
                formatted = f"{float(abs_num):,.{precision}f}"
            
            # Handle scientific notation for extremely large numbers
            if abs_num >= Decimal('1000000000000000'):  # Quadrillions and above
                formatted = f"{float(abs_num):.2e}"
            
            # Add negative sign if needed
            if num_decimal < 0:
                formatted = f"-{formatted}"
            
            return formatted
            
        except (InvalidOperation, ValueError, TypeError):
            return "0"


class DateTimeFormatter:
    """
    Date/time formatting utilities optimized for financial market data and trading session management.
    Handles market hours, trading sessions, and timestamp formatting for professional trading contexts.
    """
    
    @staticmethod
    def format_market_timestamp(timestamp: Union[datetime, str, int, float],
                               include_seconds: bool = False,
                               market_context: bool = True) -> str:
        """
        Format timestamps with market context awareness.
        
        Args:
            timestamp: Timestamp to format
            include_seconds: Whether to include seconds in time display
            market_context: Whether to add market session context
            
        Returns:
            str: Formatted timestamp string
            
        Examples:
            >>> format_market_timestamp(datetime(2024, 1, 15, 10, 30, 0))
            '2024-01-15 10:30 (Market Hours)'
            >>> format_market_timestamp(datetime(2024, 1, 15, 7, 0, 0))
            '2024-01-15 07:00 (Premarket)'
        """
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            # Format date and time
            if include_seconds:
                time_format = "%Y-%m-%d %H:%M:%S"
            else:
                time_format = "%Y-%m-%d %H:%M"
            
            formatted_time = dt.strftime(time_format)
            
            if market_context:
                session_info = DateTimeFormatter._get_market_session(dt.time())
                formatted_time += f" ({session_info})"
            
            return formatted_time
            
        except (ValueError, TypeError, AttributeError):
            return "Invalid Time"
    
    @staticmethod
    def format_trade_time(timestamp: Union[datetime, str, int, float]) -> str:
        """
        Format trade execution timestamps for journal and reporting.
        
        Args:
            timestamp: Trade timestamp to format
            
        Returns:
            str: Formatted trade time string
            
        Examples:
            >>> format_trade_time(datetime(2024, 1, 15, 14, 30, 15))
            '01/15/24 2:30:15 PM'
        """
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            return dt.strftime("%m/%d/%y %I:%M:%S %p")
            
        except (ValueError, TypeError, AttributeError):
            return "Invalid Time"
    
    @staticmethod
    def format_date_range(start_date: Union[datetime, date, str],
                         end_date: Union[datetime, date, str]) -> str:
        """
        Format date ranges for reporting and filtering displays.
        
        Args:
            start_date: Start date of range
            end_date: End date of range
            
        Returns:
            str: Formatted date range string
            
        Examples:
            >>> format_date_range(date(2024, 1, 1), date(2024, 1, 31))
            '01/01/2024 - 01/31/2024'
        """
        try:
            if isinstance(start_date, str):
                start_dt = datetime.fromisoformat(start_date).date()
            elif isinstance(start_date, datetime):
                start_dt = start_date.date()
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.fromisoformat(end_date).date()
            elif isinstance(end_date, datetime):
                end_dt = end_date.date()
            else:
                end_dt = end_date
            
            start_str = start_dt.strftime("%m/%d/%Y")
            end_str = end_dt.strftime("%m/%d/%Y")
            
            if start_dt == end_dt:
                return start_str
            else:
                return f"{start_str} - {end_str}"
            
        except (ValueError, TypeError, AttributeError):
            return "Invalid Date Range"
    
    @staticmethod
    def _get_market_session(time_obj: time) -> str:
        """
        Determine market session based on time.
        
        Args:
            time_obj: Time object to evaluate
            
        Returns:
            str: Market session description
        """
        if PREMARKET_START_TIME <= time_obj < MARKET_OPEN_TIME:
            return "Premarket"
        elif MARKET_OPEN_TIME <= time_obj < MARKET_CLOSE_TIME:
            return "Market Hours"
        elif MARKET_CLOSE_TIME <= time_obj < AFTERHOURS_END_TIME:
            return "After Hours"
        else:
            return "Closed"
    
    @staticmethod
    def get_market_session_indicator(timestamp: Union[datetime, str, int, float]) -> Dict[str, Any]:
        """
        Get comprehensive market session information for UI indicators.
        
        Args:
            timestamp: Timestamp to analyze
            
        Returns:
            Dict: Market session information including status, color, and description
            
        Examples:
            >>> get_market_session_indicator(datetime(2024, 1, 15, 10, 30))
            {'session': 'Market Hours', 'color': '#4CAF50', 'active': True, 'description': 'Regular Trading Hours'}
        """
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            time_obj = dt.time()
            
            if PREMARKET_START_TIME <= time_obj < MARKET_OPEN_TIME:
                return {
                    'session': 'Premarket',
                    'color': '#FFC107',  # Amber for premarket
                    'active': True,
                    'description': 'Premarket Trading Session'
                }
            elif MARKET_OPEN_TIME <= time_obj < MARKET_CLOSE_TIME:
                return {
                    'session': 'Market Hours',
                    'color': '#4CAF50',  # Green for active market
                    'active': True,
                    'description': 'Regular Trading Hours'
                }
            elif MARKET_CLOSE_TIME <= time_obj < AFTERHOURS_END_TIME:
                return {
                    'session': 'After Hours',
                    'color': '#FF9800',  # Orange for after hours
                    'active': True,
                    'description': 'After Hours Trading Session'
                }
            else:
                return {
                    'session': 'Closed',
                    'color': '#9E9E9E',  # Grey for closed
                    'active': False,
                    'description': 'Market Closed'
                }
            
        except (ValueError, TypeError, AttributeError):
            return {
                'session': 'Unknown',
                'color': '#F44336',  # Red for error
                'active': False,
                'description': 'Invalid Timestamp'
            }


class DataValidator:
    """
    Data validation and sanitization functions for API responses and user input processing.
    Ensures data integrity and prevents errors in financial calculations and display operations.
    """
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate and sanitize stock symbol input.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            str: Validated and sanitized symbol
            
        Raises:
            FormattingError: If symbol format is invalid
            
        Examples:
            >>> validate_symbol('AAPL')
            'AAPL'
            >>> validate_symbol(' msft ')
            'MSFT'
        """
        if not symbol or not isinstance(symbol, str):
            raise FormattingError("Symbol must be a non-empty string")
        
        # Clean and validate symbol
        cleaned_symbol = symbol.strip().upper()
        
        # Check for valid symbol pattern (1-5 letters, optionally with dots)
        if not re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', cleaned_symbol):
            raise FormattingError(f"Invalid symbol format: {symbol}")
        
        return cleaned_symbol
    
    @staticmethod
    def validate_api_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize API response data for market information.
        
        Args:
            data: Raw API response data
            
        Returns:
            Dict: Validated and sanitized data
            
        Raises:
            FormattingError: If critical data fields are missing or invalid
        """
        if not isinstance(data, dict):
            raise FormattingError("API response data must be a dictionary")
        
        validated_data = {}
        
        # Validate required price fields
        price_fields = ['price', 'open', 'high', 'low', 'close']
        for field in price_fields:
            if field in data:
                try:
                    validated_data[field] = DecimalValidator.validate_price(data[field])
                except FormattingError:
                    validated_data[field] = None
        
        # Validate volume
        if 'volume' in data:
            try:
                volume = int(float(str(data['volume'])))
                validated_data['volume'] = max(0, volume)  # Ensure non-negative
            except (ValueError, TypeError):
                validated_data['volume'] = 0
        
        # Validate percentage changes
        change_fields = ['change_percent', 'premarket_change_percent']
        for field in change_fields:
            if field in data:
                try:
                    validated_data[field] = DecimalValidator.validate_percentage(data[field])
                except FormattingError:
                    validated_data[field] = None
        
        # Validate timestamps
        timestamp_fields = ['timestamp', 'last_updated']
        for field in timestamp_fields:
            if field in data:
                try:
                    if isinstance(data[field], (int, float)):
                        validated_data[field] = datetime.fromtimestamp(data[field])
                    elif isinstance(data[field], str):
                        validated_data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                    else:
                        validated_data[field] = data[field]
                except (ValueError, TypeError):
                    validated_data[field] = datetime.now()
        
        # Preserve other fields as-is (sector, company_name, etc.)
        for key, value in data.items():
            if key not in validated_data:
                validated_data[key] = value
        
        return validated_data
    
    @staticmethod
    def sanitize_user_input(input_value: str, input_type: str = 'text') -> str:
        """
        Sanitize user input for safe processing and storage.
        
        Args:
            input_value: Raw user input
            input_type: Type of input ('text', 'numeric', 'symbol')
            
        Returns:
            str: Sanitized input value
            
        Examples:
            >>> sanitize_user_input('  AAPL  ', 'symbol')
            'AAPL'
            >>> sanitize_user_input('123.45', 'numeric')
            '123.45'
        """
        if not isinstance(input_value, str):
            input_value = str(input_value)
        
        # Basic sanitization
        sanitized = input_value.strip()
        
        if input_type == 'symbol':
            # Remove special characters except dots and hyphens
            sanitized = re.sub(r'[^A-Za-z0-9.\-]', '', sanitized).upper()
        elif input_type == 'numeric':
            # Keep only numbers, decimal points, and negative signs
            sanitized = re.sub(r'[^0-9.\-]', '', sanitized)
        elif input_type == 'text':
            # Remove potentially harmful characters but keep most text
            sanitized = re.sub(r'[<>\'\"&]', '', sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_trade_data(trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete trade data for paper trading operations.
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Dict: Validated trade data
            
        Raises:
            FormattingError: If critical trade data is invalid
        """
        if not isinstance(trade_data, dict):
            raise FormattingError("Trade data must be a dictionary")
        
        validated_trade = {}
        
        # Validate required fields
        required_fields = ['symbol', 'direction', 'quantity', 'entry_price']
        for field in required_fields:
            if field not in trade_data:
                raise FormattingError(f"Missing required field: {field}")
        
        # Validate symbol
        validated_trade['symbol'] = DataValidator.validate_symbol(trade_data['symbol'])
        
        # Validate direction
        if trade_data['direction'].upper() not in ['BUY', 'SELL']:
            raise FormattingError(f"Invalid trade direction: {trade_data['direction']}")
        validated_trade['direction'] = trade_data['direction'].upper()
        
        # Validate quantity
        try:
            quantity = int(trade_data['quantity'])
            if quantity <= 0:
                raise FormattingError("Quantity must be positive")
            validated_trade['quantity'] = quantity
        except (ValueError, TypeError):
            raise FormattingError(f"Invalid quantity: {trade_data['quantity']}")
        
        # Validate prices
        price_fields = ['entry_price', 'stop_loss', 'target_price', 'fill_price']
        for field in price_fields:
            if field in trade_data and trade_data[field] is not None:
                validated_trade[field] = DecimalValidator.validate_price(trade_data[field])
        
        # Validate timestamps
        if 'execution_time' in trade_data:
            try:
                if isinstance(trade_data['execution_time'], str):
                    validated_trade['execution_time'] = datetime.fromisoformat(trade_data['execution_time'])
                else:
                    validated_trade['execution_time'] = trade_data['execution_time']
            except (ValueError, TypeError):
                validated_trade['execution_time'] = datetime.now()
        
        return validated_trade


# Convenience functions for common formatting operations
def format_currency(amount: Union[str, int, float, Decimal], **kwargs) -> str:
    """Convenience function for currency formatting."""
    return CurrencyFormatter.format_price(amount, **kwargs)


def format_percent(value: Union[str, int, float, Decimal], **kwargs) -> str:
    """Convenience function for percentage formatting."""
    return PercentageFormatter.format_percentage(value, **kwargs)


def format_number(number: Union[str, int, float, Decimal], **kwargs) -> str:
    """Convenience function for number formatting."""
    return NumberFormatter.format_large_number(number, **kwargs)


def format_timestamp(timestamp: Union[datetime, str, int, float], **kwargs) -> str:
    """Convenience function for timestamp formatting."""
    return DateTimeFormatter.format_market_timestamp(timestamp, **kwargs)


# Professional formatting presets for common trading contexts
class FormattingPresets:
    """
    Predefined formatting configurations for common trading contexts.
    Provides consistent formatting across the application with professional standards.
    """
    
    @staticmethod
    def scanner_result_format(data: Dict[str, Any]) -> Dict[str, str]:
        """
        Format complete scanner result data for display.
        
        Args:
            data: Raw scanner data
            
        Returns:
            Dict: Formatted scanner result data
        """
        formatted = {}
        
        try:
            # Format price data
            if 'price' in data:
                formatted['price'] = CurrencyFormatter.format_price(data['price'])
            if 'change_percent' in data:
                formatted['change_percent'] = PercentageFormatter.format_percentage(
                    data['change_percent'], color_coding=True)
            
            # Format volume
            if 'volume' in data:
                formatted['volume'] = NumberFormatter.format_volume(data['volume'])
            
            # Format market cap and float
            if 'market_cap' in data:
                formatted['market_cap'] = NumberFormatter.format_market_cap(data['market_cap'])
            if 'float_shares' in data:
                formatted['float_shares'] = NumberFormatter.format_float_shares(data['float_shares'])
            
            # Format timestamp
            if 'timestamp' in data:
                formatted['timestamp'] = DateTimeFormatter.format_market_timestamp(
                    data['timestamp'], market_context=True)
            
            # Copy other fields
            for key, value in data.items():
                if key not in formatted:
                    formatted[key] = str(value) if value is not None else "--"
                    
        except Exception:
            # Fallback formatting on error
            for key, value in data.items():
                formatted[key] = str(value) if value is not None else "--"
        
        return formatted
    
    @staticmethod
    def portfolio_summary_format(data: Dict[str, Any]) -> Dict[str, str]:
        """
        Format portfolio summary data for display.
        
        Args:
            data: Raw portfolio data
            
        Returns:
            Dict: Formatted portfolio summary
        """
        formatted = {}
        
        try:
            # Format portfolio values
            if 'total_value' in data:
                formatted['total_value'] = CurrencyFormatter.format_portfolio_value(data['total_value'])
            if 'total_pnl' in data:
                formatted['total_pnl'] = CurrencyFormatter.format_pnl(data['total_pnl'])
            if 'day_pnl' in data:
                formatted['day_pnl'] = CurrencyFormatter.format_pnl(data['day_pnl'])
            
            # Format performance metrics
            if 'total_return_percent' in data:
                formatted['total_return_percent'] = PercentageFormatter.format_percentage(
                    data['total_return_percent'], color_coding=True)
            
            # Format win rate
            if 'wins' in data and 'total_trades' in data:
                formatted['win_rate'] = PercentageFormatter.format_win_rate(
                    data['wins'], data['total_trades'])
            
            # Format last update
            if 'last_updated' in data:
                formatted['last_updated'] = DateTimeFormatter.format_market_timestamp(
                    data['last_updated'])
            
        except Exception:
            # Fallback formatting on error
            for key, value in data.items():
                formatted[key] = str(value) if value is not None else "--"
        
        return formatted


if __name__ == "__main__":
    # Example usage and testing
    print("Financial Formatting Utilities - Professional Trading Platform")
    print("=" * 60)
    
    # Test currency formatting
    print("\nCurrency Formatting:")
    print(f"Price: {CurrencyFormatter.format_price(123.4567)}")
    print(f"P/L: {CurrencyFormatter.format_pnl(1234.56, color_coding=False)}")
    print(f"Portfolio: {CurrencyFormatter.format_portfolio_value(12345678.90)}")
    
    # Test percentage formatting
    print("\nPercentage Formatting:")
    print(f"Change: {PercentageFormatter.format_percentage(0.0543, color_coding=False)}")
    print(f"Risk: {PercentageFormatter.format_risk_percentage(0.02)}")
    print(f"Win Rate: {PercentageFormatter.format_win_rate(75, 100)}")
    
    # Test number formatting
    print("\nNumber Formatting:")
    print(f"Volume: {NumberFormatter.format_volume(1234567)}")
    print(f"Market Cap: {NumberFormatter.format_market_cap(123456789000)}")
    print(f"Float: {NumberFormatter.format_float_shares(123456789)}")
    
    # Test date/time formatting
    print("\nDate/Time Formatting:")
    test_time = datetime(2024, 1, 15, 10, 30, 0)
    print(f"Market Time: {DateTimeFormatter.format_market_timestamp(test_time)}")
    print(f"Trade Time: {DateTimeFormatter.format_trade_time(test_time)}")
    
    print("\nFormatting utilities initialized successfully!")