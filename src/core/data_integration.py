"""
VWAP Stock Analysis Tool - Multi-Source Data Integration Engine

This module implements sophisticated failover logic across Yahoo Finance (yfinance), 
IEX Cloud (iexfinance), and Alpha Vantage APIs with circuit breaker pattern, 
intelligent rate-limit management, and asyncio-based concurrent processing for 
scalable data retrieval supporting 500+ securities simultaneously.

Key Features:
- Circuit Breaker pattern with pybreaker for robust fault tolerance
- Multi-provider architecture with intelligent weighted failover
- Asyncio + aiohttp concurrent processing with qasync PyQt6 integration
- TTL-based caching with SQLite persistence
- Comprehensive rate limit management and exponential backoff
- Provider performance metrics and dynamic weight calculation
- Advanced error handling with detailed logging

Author: Blitzy Agent
Created: 2024
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# Core async and HTTP libraries
import aiohttp
import qasync
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread

# Data processing libraries
import pandas as pd
import numpy as np

# Circuit breaker and retry logic
import pybreaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# API client libraries
import yfinance as yf
from iexfinance import Stock
from alpha_vantage.timeseries import TimeSeries

# Local imports (interfaces to be defined)
try:
    from ..models.database import DatabaseManager, CacheEntry
except ImportError:
    # Fallback interface definitions
    class DatabaseManager:
        def get_cache_entry(self, cache_key: str) -> Optional[Dict]: pass
        def set_cache_entry(self, cache_key: str, data: Dict, ttl: int = 3600) -> bool: pass
        def cleanup_expired_cache(self) -> int: pass
    
    @dataclass
    class CacheEntry:
        cache_key: str
        data: Dict
        created_at: datetime
        expires_at: datetime


@dataclass
class ProviderConfig:
    """Configuration for market data providers."""
    name: str
    base_weight: int
    max_requests_per_minute: int
    timeout_seconds: int
    circuit_breaker_threshold: int
    circuit_breaker_timeout: int
    enabled: bool = True
    api_key: Optional[str] = None


@dataclass
class ProviderMetrics:
    """Performance metrics for provider monitoring."""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    circuit_breaker_state: str = "CLOSED"
    current_weight: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 100.0 - self.success_rate


@dataclass
class MarketDataRequest:
    """Request structure for market data."""
    symbols: List[str]
    data_type: str  # 'quote', 'historical', 'intraday'
    timeframe: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Optional[Dict] = None


@dataclass
class MarketDataResponse:
    """Response structure for market data."""
    provider: str
    symbols: List[str]
    data: Dict[str, pd.DataFrame]
    success: bool
    error_message: Optional[str] = None
    response_time: float = 0.0
    cached: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RateLimitManager:
    """Manages rate limits for API providers."""
    
    def __init__(self):
        self.request_history: Dict[str, List[datetime]] = {}
        self.rate_limits: Dict[str, int] = {}
        
    def set_rate_limit(self, provider: str, requests_per_minute: int):
        """Set rate limit for a provider."""
        self.rate_limits[provider] = requests_per_minute
        if provider not in self.request_history:
            self.request_history[provider] = []
    
    def can_make_request(self, provider: str) -> bool:
        """Check if a request can be made without exceeding rate limits."""
        if provider not in self.rate_limits:
            return True
            
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.request_history[provider] = [
            req_time for req_time in self.request_history[provider]
            if req_time > minute_ago
        ]
        
        current_requests = len(self.request_history[provider])
        return current_requests < self.rate_limits[provider]
    
    def record_request(self, provider: str):
        """Record a request for rate limiting."""
        if provider not in self.request_history:
            self.request_history[provider] = []
        self.request_history[provider].append(datetime.now())
    
    async def wait_for_rate_limit(self, provider: str):
        """Wait until a request can be made within rate limits."""
        while not self.can_make_request(provider):
            await asyncio.sleep(0.1)


class CircuitBreakerManager:
    """Manages circuit breakers for all providers."""
    
    def __init__(self):
        self.breakers: Dict[str, pybreaker.CircuitBreaker] = {}
        
    def create_breaker(self, provider: str, config: ProviderConfig) -> pybreaker.CircuitBreaker:
        """Create a circuit breaker for a provider."""
        breaker = pybreaker.CircuitBreaker(
            fail_max=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_timeout,
            exclude=[
                # Don't count authentication errors as failures
                aiohttp.ClientResponseError,
            ],
            listeners=[self._create_listener(provider)]
        )
        self.breakers[provider] = breaker
        return breaker
    
    def _create_listener(self, provider: str):
        """Create a listener for circuit breaker state changes."""
        def listener(breaker, event):
            logging.info(f"Circuit breaker for {provider}: {event}")
        return listener
    
    def get_breaker(self, provider: str) -> Optional[pybreaker.CircuitBreaker]:
        """Get circuit breaker for a provider."""
        return self.breakers.get(provider)
    
    def get_state(self, provider: str) -> str:
        """Get current state of provider's circuit breaker."""
        breaker = self.breakers.get(provider)
        if breaker:
            return breaker.current_state
        return "UNKNOWN"


class ProviderPerformanceTracker:
    """Tracks and calculates dynamic weights for providers."""
    
    def __init__(self):
        self.metrics: Dict[str, ProviderMetrics] = {}
        
    def update_metrics(self, provider: str, success: bool, response_time: float, 
                      circuit_state: str):
        """Update performance metrics for a provider."""
        if provider not in self.metrics:
            self.metrics[provider] = ProviderMetrics(provider_name=provider)
            
        metrics = self.metrics[provider]
        metrics.total_requests += 1
        metrics.last_request_time = datetime.now()
        metrics.circuit_breaker_state = circuit_state
        
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
            
        # Update average response time with exponential moving average
        if metrics.average_response_time == 0:
            metrics.average_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            metrics.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * metrics.average_response_time
            )
    
    def calculate_dynamic_weight(self, provider: str, base_weight: int) -> int:
        """Calculate dynamic weight based on performance metrics."""
        if provider not in self.metrics:
            return base_weight
            
        metrics = self.metrics[provider]
        weight_modifier = 1.0
        
        # Success rate modifier
        success_rate = metrics.success_rate
        if success_rate > 95:
            weight_modifier += 0.2
        elif success_rate > 90:
            weight_modifier += 0.1
        elif success_rate < 80:
            weight_modifier -= 0.3
        elif success_rate < 90:
            weight_modifier -= 0.1
            
        # Response time modifier
        if metrics.average_response_time < 500:  # ms
            weight_modifier += 0.15
        elif metrics.average_response_time > 2000:  # ms
            weight_modifier -= 0.15
            
        # Circuit breaker state modifier
        if metrics.circuit_breaker_state == "OPEN":
            weight_modifier -= 0.5
        elif metrics.circuit_breaker_state == "HALF_OPEN":
            weight_modifier -= 0.2
            
        # Calculate final weight
        final_weight = max(1, int(base_weight * weight_modifier))
        metrics.current_weight = final_weight
        
        return final_weight
    
    def get_provider_weights(self, provider_configs: Dict[str, ProviderConfig]) -> Dict[str, int]:
        """Get current weights for all providers."""
        weights = {}
        for provider, config in provider_configs.items():
            if config.enabled:
                weights[provider] = self.calculate_dynamic_weight(provider, config.base_weight)
        return weights


class YahooFinanceProvider:
    """Yahoo Finance data provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = "yahoo_finance"
    
    async def fetch_data(self, request: MarketDataRequest, session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch data from Yahoo Finance."""
        start_time = time.time()
        
        try:
            if request.data_type == 'quote':
                return await self._fetch_quotes(request, session)
            elif request.data_type == 'historical':
                return await self._fetch_historical(request, session)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return MarketDataResponse(
                provider=self.name,
                symbols=request.symbols,
                data={},
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def _fetch_quotes(self, request: MarketDataRequest, 
                           session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch real-time quotes."""
        data = {}
        
        # Process symbols in batches to respect rate limits
        batch_size = 20
        for i in range(0, len(request.symbols), batch_size):
            batch_symbols = request.symbols[i:i + batch_size]
            
            try:
                # Use yfinance synchronously in thread pool
                loop = asyncio.get_event_loop()
                tickers = yf.Tickers(' '.join(batch_symbols))
                
                # Run synchronous yfinance operations in executor
                ticker_data = await loop.run_in_executor(
                    None, lambda: {symbol: tickers.tickers[symbol].info 
                                  for symbol in batch_symbols}
                )
                
                # Convert to DataFrame format
                for symbol, info in ticker_data.items():
                    if info:
                        df = pd.DataFrame([{
                            'symbol': symbol,
                            'price': info.get('regularMarketPrice', 0),
                            'volume': info.get('regularMarketVolume', 0),
                            'market_cap': info.get('marketCap', 0),
                            'timestamp': datetime.now()
                        }])
                        data[symbol] = df
                        
            except Exception as e:
                logging.warning(f"Failed to fetch Yahoo Finance batch {batch_symbols}: {e}")
                continue
        
        response_time = 0  # Will be set by caller
        return MarketDataResponse(
            provider=self.name,
            symbols=request.symbols,
            data=data,
            success=len(data) > 0,
            error_message=None if len(data) > 0 else "No data retrieved",
            response_time=response_time
        )
    
    async def _fetch_historical(self, request: MarketDataRequest, 
                               session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch historical data."""
        data = {}
        
        for symbol in request.symbols:
            try:
                loop = asyncio.get_event_loop()
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist_data = await loop.run_in_executor(
                    None, 
                    lambda: ticker.history(
                        start=request.start_date,
                        end=request.end_date,
                        interval=request.timeframe or "1d"
                    )
                )
                
                if not hist_data.empty:
                    # Add symbol column and clean data
                    hist_data['symbol'] = symbol
                    hist_data.reset_index(inplace=True)
                    data[symbol] = hist_data
                    
            except Exception as e:
                logging.warning(f"Failed to fetch Yahoo Finance historical for {symbol}: {e}")
                continue
        
        return MarketDataResponse(
            provider=self.name,
            symbols=request.symbols,
            data=data,
            success=len(data) > 0,
            error_message=None if len(data) > 0 else "No historical data retrieved"
        )


class IEXCloudProvider:
    """IEX Cloud data provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = "iex_cloud"
    
    async def fetch_data(self, request: MarketDataRequest, session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch data from IEX Cloud."""
        start_time = time.time()
        
        try:
            if request.data_type == 'quote':
                return await self._fetch_quotes(request, session)
            elif request.data_type == 'historical':
                return await self._fetch_historical(request, session)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return MarketDataResponse(
                provider=self.name,
                symbols=request.symbols,
                data={},
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def _fetch_quotes(self, request: MarketDataRequest, 
                           session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch real-time quotes from IEX Cloud."""
        data = {}
        
        for symbol in request.symbols:
            try:
                loop = asyncio.get_event_loop()
                stock = Stock(symbol, token=self.config.api_key)
                
                # Get quote data
                quote_data = await loop.run_in_executor(None, stock.get_quote)
                
                if quote_data:
                    df = pd.DataFrame([{
                        'symbol': symbol,
                        'price': quote_data.get('latestPrice', 0),
                        'volume': quote_data.get('latestVolume', 0),
                        'market_cap': quote_data.get('marketCap', 0),
                        'timestamp': datetime.now()
                    }])
                    data[symbol] = df
                    
            except Exception as e:
                logging.warning(f"Failed to fetch IEX Cloud quote for {symbol}: {e}")
                continue
        
        return MarketDataResponse(
            provider=self.name,
            symbols=request.symbols,
            data=data,
            success=len(data) > 0,
            error_message=None if len(data) > 0 else "No data retrieved"
        )
    
    async def _fetch_historical(self, request: MarketDataRequest, 
                               session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch historical data from IEX Cloud."""
        data = {}
        
        for symbol in request.symbols:
            try:
                loop = asyncio.get_event_loop()
                stock = Stock(symbol, token=self.config.api_key)
                
                # Get historical data
                range_param = "1y"  # Default range
                if request.start_date and request.end_date:
                    days_diff = (request.end_date - request.start_date).days
                    if days_diff <= 30:
                        range_param = "1m"
                    elif days_diff <= 90:
                        range_param = "3m"
                    elif days_diff <= 180:
                        range_param = "6m"
                
                hist_data = await loop.run_in_executor(
                    None,
                    lambda: stock.get_historical_prices(range=range_param)
                )
                
                if hist_data:
                    df = pd.DataFrame(hist_data)
                    df['symbol'] = symbol
                    data[symbol] = df
                    
            except Exception as e:
                logging.warning(f"Failed to fetch IEX Cloud historical for {symbol}: {e}")
                continue
        
        return MarketDataResponse(
            provider=self.name,
            symbols=request.symbols,
            data=data,
            success=len(data) > 0,
            error_message=None if len(data) > 0 else "No historical data retrieved"
        )


class AlphaVantageProvider:
    """Alpha Vantage data provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = "alpha_vantage"
        self.api = TimeSeries(key=config.api_key, output_format='pandas')
    
    async def fetch_data(self, request: MarketDataRequest, session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch data from Alpha Vantage."""
        start_time = time.time()
        
        try:
            if request.data_type == 'quote':
                return await self._fetch_quotes(request, session)
            elif request.data_type == 'historical':
                return await self._fetch_historical(request, session)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return MarketDataResponse(
                provider=self.name,
                symbols=request.symbols,
                data={},
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def _fetch_quotes(self, request: MarketDataRequest, 
                           session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch real-time quotes from Alpha Vantage."""
        data = {}
        
        # Alpha Vantage has strict rate limits, process one symbol at a time
        for symbol in request.symbols[:5]:  # Limit to avoid quota exhaustion
            try:
                loop = asyncio.get_event_loop()
                
                # Get quote data
                quote_data, _ = await loop.run_in_executor(
                    None,
                    self.api.get_quote_endpoint,
                    symbol
                )
                
                if not quote_data.empty:
                    # Convert to standard format
                    df = pd.DataFrame([{
                        'symbol': symbol,
                        'price': float(quote_data.iloc[0]['05. price']),
                        'volume': int(quote_data.iloc[0]['06. volume']),
                        'timestamp': datetime.now()
                    }])
                    data[symbol] = df
                    
                # Wait to respect rate limits
                await asyncio.sleep(12)  # Alpha Vantage free tier: 5 calls/minute
                    
            except Exception as e:
                logging.warning(f"Failed to fetch Alpha Vantage quote for {symbol}: {e}")
                continue
        
        return MarketDataResponse(
            provider=self.name,
            symbols=request.symbols,
            data=data,
            success=len(data) > 0,
            error_message=None if len(data) > 0 else "No data retrieved"
        )
    
    async def _fetch_historical(self, request: MarketDataRequest, 
                               session: aiohttp.ClientSession) -> MarketDataResponse:
        """Fetch historical data from Alpha Vantage."""
        data = {}
        
        # Process limited symbols due to rate limits
        for symbol in request.symbols[:3]:  # Very conservative limit
            try:
                loop = asyncio.get_event_loop()
                
                # Get daily historical data
                hist_data, _ = await loop.run_in_executor(
                    None,
                    self.api.get_daily,
                    symbol,
                    outputsize='compact'
                )
                
                if not hist_data.empty:
                    hist_data['symbol'] = symbol
                    hist_data.reset_index(inplace=True)
                    data[symbol] = hist_data
                
                # Wait to respect rate limits
                await asyncio.sleep(12)
                    
            except Exception as e:
                logging.warning(f"Failed to fetch Alpha Vantage historical for {symbol}: {e}")
                continue
        
        return MarketDataResponse(
            provider=self.name,
            symbols=request.symbols,
            data=data,
            success=len(data) > 0,
            error_message=None if len(data) > 0 else "No historical data retrieved"
        )


class MultiSourceDataIntegration(QObject):
    """
    Main data integration engine with sophisticated failover logic across 
    multiple market data providers.
    """
    
    # Qt signals for integration with PyQt6 application
    data_received = pyqtSignal(dict)  # Emitted when data is successfully retrieved
    error_occurred = pyqtSignal(str, str)  # Emitted on errors (provider, message)
    provider_status_changed = pyqtSignal(str, str)  # provider, status
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Configuration and state
        self.config_path = config_path or "config/api_keys.json"
        self.provider_configs: Dict[str, ProviderConfig] = {}
        self.providers: Dict[str, Union[YahooFinanceProvider, IEXCloudProvider, AlphaVantageProvider]] = {}
        
        # Management components
        self.rate_limit_manager = RateLimitManager()
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.performance_tracker = ProviderPerformanceTracker()
        
        # Database integration
        self.db_manager: Optional[DatabaseManager] = None
        
        # HTTP session for async operations
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize components
        self._load_configuration()
        self._initialize_providers()
        self._setup_circuit_breakers()
        
        # Setup periodic tasks
        self.cache_cleanup_timer = QTimer()
        self.cache_cleanup_timer.timeout.connect(self._cleanup_expired_cache)
        self.cache_cleanup_timer.start(300000)  # 5 minutes
        
    def _setup_logging(self):
        """Setup comprehensive logging for network operations."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logger
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for network errors
        network_handler = logging.FileHandler('logs/network_errors.log')
        network_handler.setLevel(logging.WARNING)
        network_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(network_handler)
        self.logger.setLevel(logging.INFO)
    
    def _load_configuration(self):
        """Load provider configurations from JSON file."""
        try:
            # Default configuration
            default_config = {
                "yahoo_finance": {
                    "name": "yahoo_finance",
                    "base_weight": 40,
                    "max_requests_per_minute": 60,
                    "timeout_seconds": 10,
                    "circuit_breaker_threshold": 5,
                    "circuit_breaker_timeout": 60,
                    "enabled": True
                },
                "iex_cloud": {
                    "name": "iex_cloud",
                    "base_weight": 30,
                    "max_requests_per_minute": 100,
                    "timeout_seconds": 5,
                    "circuit_breaker_threshold": 3,
                    "circuit_breaker_timeout": 30,
                    "enabled": True,
                    "api_key": None
                },
                "alpha_vantage": {
                    "name": "alpha_vantage",
                    "base_weight": 20,
                    "max_requests_per_minute": 5,
                    "timeout_seconds": 15,
                    "circuit_breaker_threshold": 2,
                    "circuit_breaker_timeout": 120,
                    "enabled": True,
                    "api_key": None
                }
            }
            
            # Try to load from file
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    for provider, config in file_config.items():
                        if provider in default_config:
                            default_config[provider].update(config)
            
            # Convert to ProviderConfig objects
            for provider_name, config_dict in default_config.items():
                self.provider_configs[provider_name] = ProviderConfig(**config_dict)
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Use minimal default configuration
            self.provider_configs["yahoo_finance"] = ProviderConfig(
                name="yahoo_finance",
                base_weight=50,
                max_requests_per_minute=60,
                timeout_seconds=10,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=60
            )
    
    def _initialize_providers(self):
        """Initialize all configured data providers."""
        for provider_name, config in self.provider_configs.items():
            if not config.enabled:
                continue
                
            try:
                if provider_name == "yahoo_finance":
                    self.providers[provider_name] = YahooFinanceProvider(config)
                elif provider_name == "iex_cloud":
                    self.providers[provider_name] = IEXCloudProvider(config)
                elif provider_name == "alpha_vantage":
                    self.providers[provider_name] = AlphaVantageProvider(config)
                
                # Set up rate limiting
                self.rate_limit_manager.set_rate_limit(
                    provider_name, config.max_requests_per_minute
                )
                
                self.logger.info(f"Initialized provider: {provider_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {provider_name}: {e}")
                config.enabled = False
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for all providers."""
        for provider_name, config in self.provider_configs.items():
            if config.enabled:
                self.circuit_breaker_manager.create_breaker(provider_name, config)
    
    def set_database_manager(self, db_manager: DatabaseManager):
        """Set the database manager for caching operations."""
        self.db_manager = db_manager
    
    async def initialize_session(self):
        """Initialize aiohttp session for async operations."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _generate_cache_key(self, request: MarketDataRequest) -> str:
        """Generate a unique cache key for the request."""
        key_data = {
            'symbols': sorted(request.symbols),
            'data_type': request.data_type,
            'timeframe': request.timeframe,
            'start_date': request.start_date.isoformat() if request.start_date else None,
            'end_date': request.end_date.isoformat() if request.end_date else None
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[MarketDataResponse]:
        """Check if data exists in cache and is still valid."""
        if not self.db_manager:
            return None
            
        try:
            cache_entry = self.db_manager.get_cache_entry(cache_key)
            if cache_entry and datetime.now() < cache_entry.get('expires_at', datetime.min):
                # Reconstruct response from cache
                data = cache_entry['data']
                return MarketDataResponse(
                    provider="cache",
                    symbols=data.get('symbols', []),
                    data=data.get('data', {}),
                    success=True,
                    cached=True,
                    timestamp=cache_entry.get('created_at', datetime.now())
                )
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
        
        return None
    
    def _store_in_cache(self, cache_key: str, response: MarketDataResponse, ttl: int = 3600):
        """Store response in cache with TTL."""
        if not self.db_manager or not response.success:
            return
            
        try:
            cache_data = {
                'symbols': response.symbols,
                'data': response.data,
                'provider': response.provider,
                'timestamp': response.timestamp.isoformat()
            }
            
            self.db_manager.set_cache_entry(cache_key, cache_data, ttl)
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def _select_provider(self) -> Optional[str]:
        """Select best available provider based on weights and circuit breaker states."""
        weights = self.performance_tracker.get_provider_weights(self.provider_configs)
        available_providers = []
        
        for provider_name, weight in weights.items():
            breaker = self.circuit_breaker_manager.get_breaker(provider_name)
            if not breaker or breaker.current_state == "CLOSED":
                available_providers.extend([provider_name] * weight)
        
        if not available_providers:
            # All circuit breakers are open, try half-open ones
            for provider_name in weights.keys():
                breaker = self.circuit_breaker_manager.get_breaker(provider_name)
                if breaker and breaker.current_state == "HALF_OPEN":
                    available_providers.append(provider_name)
        
        if available_providers:
            import random
            return random.choice(available_providers)
        
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _fetch_with_provider(self, provider_name: str, request: MarketDataRequest) -> MarketDataResponse:
        """Fetch data from a specific provider with retry logic."""
        await self.initialize_session()
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not available")
        
        # Check rate limits
        await self.rate_limit_manager.wait_for_rate_limit(provider_name)
        self.rate_limit_manager.record_request(provider_name)
        
        # Use circuit breaker
        breaker = self.circuit_breaker_manager.get_breaker(provider_name)
        if breaker:
            return await breaker(provider.fetch_data)(request, self.session)
        else:
            return await provider.fetch_data(request, self.session)
    
    async def fetch_market_data(self, request: MarketDataRequest, 
                               use_cache: bool = True) -> MarketDataResponse:
        """
        Fetch market data with intelligent provider selection and failover.
        
        Args:
            request: Market data request specification
            use_cache: Whether to use cached data if available
            
        Returns:
            MarketDataResponse with data from best available provider
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if use_cache:
            cached_response = self._check_cache(cache_key)
            if cached_response:
                self.logger.info(f"Cache hit for request: {cache_key}")
                self.data_received.emit(asdict(cached_response))
                return cached_response
        
        # Try providers in order of preference
        tried_providers = []
        
        while len(tried_providers) < len(self.providers):
            provider_name = self._select_provider()
            if not provider_name or provider_name in tried_providers:
                # Try remaining providers
                remaining = set(self.providers.keys()) - set(tried_providers)
                if remaining:
                    provider_name = next(iter(remaining))
                else:
                    break
            
            tried_providers.append(provider_name)
            
            try:
                self.logger.info(f"Attempting to fetch data from {provider_name}")
                response = await self._fetch_with_provider(provider_name, request)
                
                # Update performance metrics
                circuit_state = self.circuit_breaker_manager.get_state(provider_name)
                self.performance_tracker.update_metrics(
                    provider_name, response.success, response.response_time, circuit_state
                )
                
                if response.success:
                    # Store in cache for future use
                    if use_cache:
                        ttl = 3600 if request.data_type == 'historical' else 300  # 5 min for quotes
                        self._store_in_cache(cache_key, response, ttl)
                    
                    self.logger.info(f"Successfully fetched data from {provider_name}")
                    self.data_received.emit(asdict(response))
                    self.provider_status_changed.emit(provider_name, "SUCCESS")
                    return response
                else:
                    self.logger.warning(f"Provider {provider_name} returned no data: {response.error_message}")
                    self.provider_status_changed.emit(provider_name, "FAILED")
                    
            except Exception as e:
                self.logger.error(f"Provider {provider_name} failed: {e}")
                self.error_occurred.emit(provider_name, str(e))
                self.provider_status_changed.emit(provider_name, "ERROR")
                
                # Update metrics for failure
                circuit_state = self.circuit_breaker_manager.get_state(provider_name)
                self.performance_tracker.update_metrics(
                    provider_name, False, 0, circuit_state
                )
        
        # All providers failed, return empty response
        total_time = (time.time() - start_time) * 1000
        error_response = MarketDataResponse(
            provider="none",
            symbols=request.symbols,
            data={},
            success=False,
            error_message="All providers failed",
            response_time=total_time
        )
        
        self.error_occurred.emit("all_providers", "All data providers failed")
        return error_response
    
    async def fetch_multiple_symbols(self, symbols: List[str], data_type: str = 'quote',
                                   batch_size: int = 50) -> Dict[str, MarketDataResponse]:
        """
        Fetch data for multiple symbols with batching and concurrent processing.
        
        Args:
            symbols: List of stock symbols to fetch
            data_type: Type of data to fetch ('quote', 'historical')
            batch_size: Number of symbols per batch
            
        Returns:
            Dictionary mapping symbols to their responses
        """
        results = {}
        
        # Split symbols into batches
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            request = MarketDataRequest(
                symbols=batch,
                data_type=data_type
            )
            task = asyncio.create_task(self.fetch_market_data(request))
            tasks.append((batch, task))
        
        # Collect results
        for batch, task in tasks:
            try:
                response = await task
                for symbol in batch:
                    results[symbol] = response
            except Exception as e:
                self.logger.error(f"Batch processing failed for {batch}: {e}")
                # Create error responses for failed batch
                for symbol in batch:
                    results[symbol] = MarketDataResponse(
                        provider="error",
                        symbols=[symbol],
                        data={},
                        success=False,
                        error_message=str(e)
                    )
        
        return results
    
    def get_provider_status(self) -> Dict[str, Dict]:
        """Get current status of all providers."""
        status = {}
        
        for provider_name in self.providers.keys():
            metrics = self.performance_tracker.metrics.get(provider_name)
            circuit_state = self.circuit_breaker_manager.get_state(provider_name)
            
            status[provider_name] = {
                'enabled': self.provider_configs[provider_name].enabled,
                'circuit_breaker_state': circuit_state,
                'success_rate': metrics.success_rate if metrics else 0,
                'average_response_time': metrics.average_response_time if metrics else 0,
                'current_weight': metrics.current_weight if metrics else 0,
                'total_requests': metrics.total_requests if metrics else 0
            }
        
        return status
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        if self.db_manager:
            try:
                cleaned_count = self.db_manager.cleanup_expired_cache()
                if cleaned_count > 0:
                    self.logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            except Exception as e:
                self.logger.error(f"Cache cleanup failed: {e}")
    
    def shutdown(self):
        """Shutdown the data integration engine."""
        self.logger.info("Shutting down data integration engine")
        
        # Stop timers
        if self.cache_cleanup_timer.isActive():
            self.cache_cleanup_timer.stop()
        
        # Close session in event loop
        if self.session and not self.session.closed:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.close_session())
            else:
                loop.run_until_complete(self.close_session())


# Convenience function for qasync integration
def create_data_integration_engine(config_path: Optional[str] = None) -> MultiSourceDataIntegration:
    """
    Create and configure a data integration engine instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured MultiSourceDataIntegration instance
    """
    return MultiSourceDataIntegration(config_path)


# Export main classes and functions
__all__ = [
    'MultiSourceDataIntegration',
    'MarketDataRequest',
    'MarketDataResponse',
    'ProviderConfig',
    'ProviderMetrics',
    'create_data_integration_engine'
]