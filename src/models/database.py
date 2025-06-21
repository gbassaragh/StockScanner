"""
VWAP Stock Analysis Tool - Comprehensive SQLite Database Layer

This module implements professional trading application persistence with QSqlDatabase integration,
schema migration management, and comprehensive data access patterns supporting all trading
simulation, portfolio tracking, market data, and alert management requirements.

Key Features:
- SQLite3 database with ACID compliance for critical trading data persistence
- QSqlDatabase integration for Qt-native connectivity and threading support
- Professional schema migration management using database versioning
- Comprehensive data access layer with ORM-like patterns
- Intelligent caching mechanisms with TTL management
- Thread-safe concurrent access patterns for background operations
- Automatic data retention and cleanup policies
- Performance-optimized indexing for high-frequency operations

Technical Implementation:
- Database location: %APPDATA%/Blitzy/database/trading_app.db
- Thread-safe SQLite access using WAL mode
- ACID-compliant transaction management with rollback capabilities
- Connection pooling and singleton pattern for resource management
- Comprehensive error handling and recovery mechanisms
- Automatic backup and corruption detection

Author: Blitzy Agent
Created: 2024
Version: 1.0.0
"""

import logging
import sqlite3
import json
import threading
import uuid
import os
import shutil
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
import pickle
import gzip

# PyQt6 imports for Qt-native database integration
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, QMutex, QMutexLocker
from PyQt6.QtSql import QSqlDatabase, QSqlQuery, QSqlError, QSql

# Configure logging for database operations
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Data class representing a cache entry with TTL management."""
    cache_key: str
    data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    data_provider: str = "unknown"
    data_type: str = "unknown"
    data_quality_score: str = "A"
    
    def __post_init__(self):
        """Ensure datetime objects are properly initialized."""
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.expires_at, str):
            self.expires_at = datetime.fromisoformat(self.expires_at)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() >= self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for storage."""
        return {
            'cache_key': self.cache_key,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'data_provider': self.data_provider,
            'data_type': self.data_type,
            'data_quality_score': self.data_quality_score
        }


class DatabaseConnectionManager:
    """
    Thread-safe SQLite connection manager using QSqlDatabase.
    Implements connection pooling and WAL mode for concurrent access.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_mutex = QMutex()
        self._connections: Dict[int, QSqlDatabase] = {}
        self._initialized = False
        
    def get_connection(self) -> QSqlDatabase:
        """Get thread-specific database connection."""
        thread_id = threading.get_ident()
        
        with QMutexLocker(self.connection_mutex):
            if thread_id not in self._connections:
                # Create new connection for this thread
                connection_name = f"trading_db_{thread_id}"
                db = QSqlDatabase.addDatabase("QSQLITE", connection_name)
                db.setDatabaseName(self.db_path)
                
                if not db.open():
                    error = db.lastError()
                    raise Exception(f"Failed to open database: {error.text()}")
                
                # Configure SQLite for optimal performance and concurrency
                query = QSqlQuery(db)
                query.exec("PRAGMA foreign_keys = ON")
                query.exec("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for concurrency
                query.exec("PRAGMA synchronous = NORMAL")
                query.exec("PRAGMA cache_size = 10000")
                query.exec("PRAGMA temp_store = memory")
                
                self._connections[thread_id] = db
                logger.debug(f"Created database connection for thread {thread_id}")
            
            return self._connections[thread_id]
    
    def close_all_connections(self):
        """Close all database connections."""
        with QMutexLocker(self.connection_mutex):
            for thread_id, db in self._connections.items():
                if db.isOpen():
                    db.close()
                QSqlDatabase.removeDatabase(db.connectionName())
                logger.debug(f"Closed database connection for thread {thread_id}")
            
            self._connections.clear()


class DatabaseSchema:
    """
    Database schema management with versioning and migration support.
    Implements all 7+ primary tables as specified in the technical specification.
    """
    
    CURRENT_VERSION = 1
    
    # Core schema creation SQL statements
    SCHEMA_SQL = {
        "schema_version": """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT NOT NULL
            )
        """,
        
        "trades": """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
                entry_price DECIMAL(10,4) NOT NULL,
                stop_loss DECIMAL(10,4),
                target_price DECIMAL(10,4),
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                fill_price DECIMAL(10,4) NOT NULL,
                execution_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'CLOSED', 'CANCELLED')),
                realized_pnl DECIMAL(12,2) DEFAULT 0.00,
                unrealized_pnl DECIMAL(12,2) DEFAULT 0.00,
                commission DECIMAL(8,2) DEFAULT 0.00,
                slippage DECIMAL(8,4) DEFAULT 0.00,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
        "portfolio_positions": """
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                total_quantity INTEGER NOT NULL,
                average_cost DECIMAL(10,4) NOT NULL,
                current_price DECIMAL(10,4) NOT NULL,
                unrealized_pnl DECIMAL(12,2) DEFAULT 0.00,
                realized_pnl DECIMAL(12,2) DEFAULT 0.00,
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                allocation_bucket TEXT DEFAULT 'general',
                market_value DECIMAL(12,2) GENERATED ALWAYS AS (current_price * ABS(total_quantity)) VIRTUAL,
                cost_basis DECIMAL(12,2) GENERATED ALWAYS AS (average_cost * ABS(total_quantity)) VIRTUAL
            )
        """,
        
        "market_data_cache": """
            CREATE TABLE IF NOT EXISTS market_data_cache (
                cache_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                data_provider TEXT NOT NULL,
                data_type TEXT NOT NULL,
                cached_data BLOB NOT NULL,
                cache_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expiry_time TIMESTAMP NOT NULL,
                data_quality_score TEXT DEFAULT 'A' CHECK(data_quality_score IN ('A', 'B', 'C', 'D', 'F')),
                access_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
        "security_metadata": """
            CREATE TABLE IF NOT EXISTS security_metadata (
                symbol TEXT PRIMARY KEY,
                sector TEXT NOT NULL,
                industry TEXT,
                float_shares REAL NOT NULL,
                market_cap REAL NOT NULL,
                company_name TEXT,
                exchange TEXT,
                currency TEXT DEFAULT 'USD',
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'multiple'
            )
        """,
        
        "user_settings": """
            CREATE TABLE IF NOT EXISTS user_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL,
                setting_category TEXT NOT NULL DEFAULT 'general',
                data_type TEXT NOT NULL DEFAULT 'string' CHECK(data_type IN ('string', 'integer', 'float', 'boolean', 'json')),
                description TEXT,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_encrypted BOOLEAN DEFAULT FALSE
            )
        """,
        
        "trading_journal": """
            CREATE TABLE IF NOT EXISTS trading_journal (
                journal_id TEXT PRIMARY KEY,
                trade_id TEXT,
                entry_title TEXT NOT NULL,
                analysis_notes TEXT,
                market_conditions TEXT,
                lessons_learned TEXT,
                entry_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                entry_type TEXT DEFAULT 'trade_analysis' CHECK(entry_type IN ('trade_analysis', 'market_observation', 'strategy_note')),
                tags TEXT, -- JSON array of tags
                rating INTEGER CHECK(rating BETWEEN 1 AND 5),
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE SET NULL
            )
        """,
        
        "alert_rules": """
            CREATE TABLE IF NOT EXISTS alert_rules (
                alert_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                condition_type TEXT NOT NULL,
                trigger_value DECIMAL(10,4) NOT NULL,
                comparison_operator TEXT NOT NULL CHECK(comparison_operator IN ('>', '<', '>=', '<=', '==', '!=')),
                notification_method TEXT NOT NULL DEFAULT 'desktop',
                is_active BOOLEAN DEFAULT TRUE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_triggered TIMESTAMP,
                trigger_count INTEGER DEFAULT 0,
                expiry_date TIMESTAMP,
                alert_message TEXT,
                severity TEXT DEFAULT 'INFO' CHECK(severity IN ('CRITICAL', 'WARNING', 'INFO'))
            )
        """,
        
        "scanner_results": """
            CREATE TABLE IF NOT EXISTS scanner_results (
                result_id TEXT PRIMARY KEY,
                scan_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                current_price DECIMAL(10,4) NOT NULL,
                price_change DECIMAL(8,4) NOT NULL,
                price_change_percent DECIMAL(6,2) NOT NULL,
                volume INTEGER NOT NULL,
                dollar_volume DECIMAL(15,2) NOT NULL,
                vwap DECIMAL(10,4) NOT NULL,
                atr DECIMAL(8,4) NOT NULL,
                volume_ratio DECIMAL(6,2) NOT NULL,
                volatility_score DECIMAL(5,2) NOT NULL,
                momentum_score DECIMAL(5,2) NOT NULL,
                composite_score DECIMAL(5,2) NOT NULL,
                scan_criteria_hash TEXT NOT NULL -- Hash of scan criteria used
            )
        """,
        
        "watchlists": """
            CREATE TABLE IF NOT EXISTS watchlists (
                watchlist_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                symbols TEXT NOT NULL, -- JSON array of symbols
                description TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_default BOOLEAN DEFAULT FALSE
            )
        """,
        
        "audit_log": """
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                operation_type TEXT NOT NULL CHECK(operation_type IN ('INSERT', 'UPDATE', 'DELETE')),
                record_id TEXT NOT NULL,
                old_values TEXT, -- JSON format
                new_values TEXT, -- JSON format
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_context TEXT DEFAULT 'system'
            )
        """
    }
    
    # Performance-optimized indexes
    INDEXES_SQL = [
        # Primary performance indexes
        "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, execution_time DESC)",
        "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status) WHERE status = 'OPEN'",
        "CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio_positions(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_cache_symbol_provider ON market_data_cache(symbol, data_provider)",
        "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON market_data_cache(expiry_time)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_active ON alert_rules(is_active, symbol) WHERE is_active = 1",
        
        # Security metadata indexes for advanced filtering
        "CREATE INDEX IF NOT EXISTS idx_metadata_sector ON security_metadata(sector)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_float_shares ON security_metadata(float_shares)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_market_cap ON security_metadata(market_cap)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_last_updated ON security_metadata(last_updated)",
        
        # Composite indexes for complex queries
        "CREATE INDEX IF NOT EXISTS idx_trades_performance ON trades(symbol, status, execution_time)",
        "CREATE INDEX IF NOT EXISTS idx_journal_trade_date ON trading_journal(trade_id, entry_date)",
        "CREATE INDEX IF NOT EXISTS idx_settings_category ON user_settings(setting_category, setting_key)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_sector_cap ON security_metadata(sector, market_cap)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_float_cap ON security_metadata(float_shares, market_cap)",
        
        # Scanner and performance indexes
        "CREATE INDEX IF NOT EXISTS idx_scanner_timestamp ON scanner_results(scan_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_scanner_symbol ON scanner_results(symbol, scan_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_scanner_score ON scanner_results(composite_score DESC)",
        
        # Cache performance indexes
        "CREATE INDEX IF NOT EXISTS idx_cache_access ON market_data_cache(last_accessed DESC)",
        "CREATE INDEX IF NOT EXISTS idx_cache_type ON market_data_cache(data_type, symbol)"
    ]
    
    @classmethod
    def create_tables(cls, db: QSqlDatabase) -> bool:
        """Create all database tables with proper error handling."""
        try:
            query = QSqlQuery(db)
            
            # Create tables in dependency order
            for table_name, sql in cls.SCHEMA_SQL.items():
                if not query.exec(sql):
                    error = query.lastError()
                    logger.error(f"Failed to create table {table_name}: {error.text()}")
                    return False
                
                logger.debug(f"Created table: {table_name}")
            
            # Create indexes
            for index_sql in cls.INDEXES_SQL:
                if not query.exec(index_sql):
                    error = query.lastError()
                    logger.warning(f"Failed to create index: {error.text()}")
                    # Continue on index failures - not critical
            
            # Initialize schema version
            if not cls._initialize_schema_version(db):
                return False
            
            logger.info("Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            return False
    
    @classmethod
    def _initialize_schema_version(cls, db: QSqlDatabase) -> bool:
        """Initialize schema version tracking."""
        try:
            query = QSqlQuery(db)
            
            # Check if version already exists
            query.prepare("SELECT version FROM schema_version WHERE version = ?")
            query.addBindValue(cls.CURRENT_VERSION)
            
            if query.exec() and query.next():
                logger.debug(f"Schema version {cls.CURRENT_VERSION} already exists")
                return True
            
            # Insert current version
            query.prepare("""
                INSERT INTO schema_version (version, description) 
                VALUES (?, ?)
            """)
            query.addBindValue(cls.CURRENT_VERSION)
            query.addBindValue("Initial database schema deployment")
            
            if not query.exec():
                error = query.lastError()
                logger.error(f"Failed to initialize schema version: {error.text()}")
                return False
            
            logger.info(f"Initialized schema version {cls.CURRENT_VERSION}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing schema version: {e}")
            return False
    
    @classmethod
    def get_current_version(cls, db: QSqlDatabase) -> int:
        """Get current database schema version."""
        try:
            query = QSqlQuery(db)
            query.exec("SELECT MAX(version) as version FROM schema_version")
            
            if query.next():
                return query.value(0) or 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Error getting schema version: {e}")
            return 0
    
    @classmethod
    def needs_migration(cls, db: QSqlDatabase) -> bool:
        """Check if database needs migration."""
        current_version = cls.get_current_version(db)
        return current_version < cls.CURRENT_VERSION


class DatabaseManager(QObject):
    """
    Comprehensive SQLite database manager with QSqlDatabase integration.
    
    Provides thread-safe access patterns, intelligent caching, automatic cleanup,
    and complete data persistence for all trading application components.
    """
    
    # Signals for database events
    cache_updated = pyqtSignal(str)  # cache_key
    cleanup_completed = pyqtSignal(int)  # cleaned_count
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        
        # Initialize database path
        if db_path is None:
            db_path = self._get_default_db_path()
        
        self.db_path = Path(db_path)
        self.connection_manager = DatabaseConnectionManager(str(self.db_path))
        
        # Thread-safety
        self.mutex = QMutex()
        
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'cleanups': 0
        }
        
        # Initialize database
        self._initialize_database()
        
        # Setup automatic cleanup timer
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._periodic_cleanup)
        self.cleanup_timer.start(300000)  # 5 minutes
        
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    def _get_default_db_path(self) -> str:
        """Get default database path in APPDATA/Blitzy directory."""
        try:
            if os.name == 'nt':  # Windows
                appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
                db_dir = Path(appdata) / 'Blitzy' / 'database'
            else:
                # Fallback for non-Windows (development)
                db_dir = Path.home() / '.blitzy' / 'database'
            
            db_dir.mkdir(parents=True, exist_ok=True)
            return str(db_dir / 'trading_app.db')
            
        except Exception as e:
            logger.error(f"Error creating database directory: {e}")
            # Fallback to current directory
            return 'trading_app.db'
    
    def _initialize_database(self) -> bool:
        """Initialize database with schema creation and migration."""
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get connection and check schema
            db = self.connection_manager.get_connection()
            
            # Check if migration is needed
            if DatabaseSchema.needs_migration(db):
                logger.info("Database migration required")
                if not DatabaseSchema.create_tables(db):
                    raise Exception("Failed to create/migrate database schema")
            
            # Verify database integrity
            if not self._verify_database_integrity(db):
                logger.warning("Database integrity check failed")
            
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.error_occurred.emit(f"Database initialization failed: {str(e)}")
            return False
    
    def _verify_database_integrity(self, db: QSqlDatabase) -> bool:
        """Verify database integrity using SQLite PRAGMA."""
        try:
            query = QSqlQuery(db)
            query.exec("PRAGMA integrity_check")
            
            if query.next():
                result = query.value(0)
                if result == "ok":
                    logger.debug("Database integrity check passed")
                    return True
                else:
                    logger.error(f"Database integrity check failed: {result}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error during integrity check: {e}")
            return False
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions with automatic rollback."""
        db = self.connection_manager.get_connection()
        
        if not db.transaction():
            error = db.lastError()
            raise Exception(f"Failed to start transaction: {error.text()}")
        
        try:
            yield db
            if not db.commit():
                error = db.lastError()
                raise Exception(f"Failed to commit transaction: {error.text()}")
        except Exception as e:
            db.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    # ======================================================================
    # CACHE MANAGEMENT METHODS (for data_integration.py)
    # ======================================================================
    
    def get_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cache entry by key.
        
        Args:
            cache_key: Unique cache identifier
            
        Returns:
            Cache data dictionary or None if not found/expired
        """
        try:
            with QMutexLocker(self.mutex):
                db = self.connection_manager.get_connection()
                query = QSqlQuery(db)
                
                query.prepare("""
                    SELECT cached_data, expiry_time, data_provider, data_type, created_at, cache_time
                    FROM market_data_cache 
                    WHERE cache_key = ? AND expiry_time > CURRENT_TIMESTAMP
                """)
                query.addBindValue(cache_key)
                
                if query.exec() and query.next():
                    # Update access statistics
                    self._update_cache_access(cache_key)
                    
                    # Deserialize cached data
                    cached_data = query.value(0)
                    if isinstance(cached_data, bytes):
                        # Decompress and deserialize
                        try:
                            decompressed = gzip.decompress(cached_data)
                            data = pickle.loads(decompressed)
                        except:
                            # Fallback to JSON
                            data = json.loads(query.value(0).decode() if isinstance(cached_data, bytes) else cached_data)
                    else:
                        data = json.loads(cached_data)
                    
                    self.cache_stats['hits'] += 1
                    
                    return {
                        'data': data,
                        'expires_at': datetime.fromisoformat(query.value(1).replace('Z', '+00:00')),
                        'data_provider': query.value(2),
                        'data_type': query.value(3),
                        'created_at': datetime.fromisoformat(query.value(4).replace('Z', '+00:00')),
                        'cache_time': datetime.fromisoformat(query.value(5).replace('Z', '+00:00'))
                    }
                
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cache entry {cache_key}: {e}")
            return None
    
    def set_cache_entry(self, cache_key: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Store cache entry with TTL.
        
        Args:
            cache_key: Unique cache identifier
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                # Calculate expiry time
                expiry_time = datetime.now() + timedelta(seconds=ttl)
                
                # Serialize and compress data
                serialized_data = pickle.dumps(data)
                compressed_data = gzip.compress(serialized_data)
                
                # Extract metadata from data
                symbol = data.get('symbol', 'unknown')
                data_provider = data.get('provider', 'unknown')
                data_type = data.get('data_type', 'unknown')
                
                # Insert or replace cache entry
                query.prepare("""
                    INSERT OR REPLACE INTO market_data_cache 
                    (cache_key, symbol, data_provider, data_type, cached_data, expiry_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """)
                query.addBindValue(cache_key)
                query.addBindValue(symbol)
                query.addBindValue(data_provider)
                query.addBindValue(data_type)
                query.addBindValue(compressed_data)
                query.addBindValue(expiry_time.isoformat())
                
                if query.exec():
                    self.cache_stats['sets'] += 1
                    self.cache_updated.emit(cache_key)
                    return True
                else:
                    error = query.lastError()
                    logger.error(f"Failed to cache entry {cache_key}: {error.text()}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error caching entry {cache_key}: {e}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                # Count expired entries first
                query.exec("SELECT COUNT(*) FROM market_data_cache WHERE expiry_time <= CURRENT_TIMESTAMP")
                count = 0
                if query.next():
                    count = query.value(0)
                
                # Delete expired entries
                if count > 0:
                    query.exec("DELETE FROM market_data_cache WHERE expiry_time <= CURRENT_TIMESTAMP")
                    self.cache_stats['cleanups'] += 1
                    
                    # Vacuum database to reclaim space
                    if count > 1000:  # Only vacuum for large cleanups
                        query.exec("VACUUM")
                
                self.cleanup_completed.emit(count)
                logger.debug(f"Cleaned up {count} expired cache entries")
                return count
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0
    
    def _update_cache_access(self, cache_key: str):
        """Update cache access statistics."""
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            query.prepare("""
                UPDATE market_data_cache 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP 
                WHERE cache_key = ?
            """)
            query.addBindValue(cache_key)
            query.exec()
            
        except Exception as e:
            logger.debug(f"Error updating cache access for {cache_key}: {e}")
    
    # ======================================================================
    # SCANNER ENGINE METHODS
    # ======================================================================
    
    def save_scanner_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Save scanner results to database.
        
        Args:
            results: List of scanner result dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                # Generate scan criteria hash for this batch
                criteria_hash = hashlib.md5(str(len(results)).encode()).hexdigest()
                
                query.prepare("""
                    INSERT INTO scanner_results 
                    (result_id, symbol, current_price, price_change, price_change_percent,
                     volume, dollar_volume, vwap, atr, volume_ratio, volatility_score,
                     momentum_score, composite_score, scan_criteria_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                for result in results:
                    query.addBindValue(str(uuid.uuid4()))
                    query.addBindValue(result.get('symbol', ''))
                    query.addBindValue(float(result.get('current_price', 0)))
                    query.addBindValue(float(result.get('price_change', 0)))
                    query.addBindValue(float(result.get('price_change_percent', 0)))
                    query.addBindValue(int(result.get('volume', 0)))
                    query.addBindValue(float(result.get('dollar_volume', 0)))
                    query.addBindValue(float(result.get('vwap', 0)))
                    query.addBindValue(float(result.get('atr', 0)))
                    query.addBindValue(float(result.get('volume_ratio', 0)))
                    query.addBindValue(float(result.get('volatility_score', 0)))
                    query.addBindValue(float(result.get('momentum_score', 0)))
                    query.addBindValue(float(result.get('composite_score', 0)))
                    query.addBindValue(criteria_hash)
                    
                    if not query.exec():
                        error = query.lastError()
                        logger.error(f"Failed to save scanner result: {error.text()}")
                        return False
                
                logger.info(f"Saved {len(results)} scanner results to database")
                return True
                
        except Exception as e:
            logger.error(f"Error saving scanner results: {e}")
            return False
    
    def load_watchlist(self, name: str) -> List[str]:
        """
        Load watchlist symbols by name.
        
        Args:
            name: Watchlist name
            
        Returns:
            List of symbols in the watchlist
        """
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            query.prepare("SELECT symbols FROM watchlists WHERE name = ?")
            query.addBindValue(name)
            
            if query.exec() and query.next():
                symbols_json = query.value(0)
                return json.loads(symbols_json) if symbols_json else []
            
            return []
            
        except Exception as e:
            logger.error(f"Error loading watchlist {name}: {e}")
            return []
    
    def save_watchlist(self, name: str, symbols: List[str]) -> bool:
        """
        Save watchlist to database.
        
        Args:
            name: Watchlist name
            symbols: List of symbols
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                # Insert or update watchlist
                query.prepare("""
                    INSERT OR REPLACE INTO watchlists 
                    (watchlist_id, name, symbols, last_modified)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """)
                query.addBindValue(str(uuid.uuid4()))
                query.addBindValue(name)
                query.addBindValue(json.dumps(symbols))
                
                if query.exec():
                    logger.debug(f"Saved watchlist {name} with {len(symbols)} symbols")
                    return True
                else:
                    error = query.lastError()
                    logger.error(f"Failed to save watchlist {name}: {error.text()}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error saving watchlist {name}: {e}")
            return False
    
    def get_scanner_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical scanner results for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of historical scanner results
        """
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            query.prepare("""
                SELECT * FROM scanner_results 
                WHERE symbol = ? AND scan_timestamp >= date('now', '-' || ? || ' days')
                ORDER BY scan_timestamp DESC
            """)
            query.addBindValue(symbol)
            query.addBindValue(days)
            
            results = []
            if query.exec():
                while query.next():
                    results.append({
                        'result_id': query.value('result_id'),
                        'scan_timestamp': query.value('scan_timestamp'),
                        'symbol': query.value('symbol'),
                        'current_price': query.value('current_price'),
                        'price_change': query.value('price_change'),
                        'price_change_percent': query.value('price_change_percent'),
                        'volume': query.value('volume'),
                        'dollar_volume': query.value('dollar_volume'),
                        'vwap': query.value('vwap'),
                        'atr': query.value('atr'),
                        'volume_ratio': query.value('volume_ratio'),
                        'volatility_score': query.value('volatility_score'),
                        'momentum_score': query.value('momentum_score'),
                        'composite_score': query.value('composite_score')
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting scanner history for {symbol}: {e}")
            return []
    
    # ======================================================================
    # TRADING AND PORTFOLIO METHODS (for paper_trading.py)
    # ======================================================================
    
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Save trade to database."""
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                query.prepare("""
                    INSERT OR REPLACE INTO trades 
                    (trade_id, symbol, direction, entry_price, quantity, fill_price, 
                     execution_time, status, realized_pnl, unrealized_pnl, commission, slippage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(trade_data.get('trade_id', str(uuid.uuid4())))
                query.addBindValue(trade_data.get('symbol', ''))
                query.addBindValue(trade_data.get('direction', 'BUY'))
                query.addBindValue(float(trade_data.get('entry_price', 0)))
                query.addBindValue(int(trade_data.get('quantity', 0)))
                query.addBindValue(float(trade_data.get('fill_price', 0)))
                query.addBindValue(trade_data.get('execution_time', datetime.now().isoformat()))
                query.addBindValue(trade_data.get('status', 'OPEN'))
                query.addBindValue(float(trade_data.get('realized_pnl', 0)))
                query.addBindValue(float(trade_data.get('unrealized_pnl', 0)))
                query.addBindValue(float(trade_data.get('commission', 0)))
                query.addBindValue(float(trade_data.get('slippage', 0)))
                
                return query.exec()
                
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    def save_position(self, position_data: Dict[str, Any]) -> bool:
        """Save portfolio position to database."""
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                query.prepare("""
                    INSERT OR REPLACE INTO portfolio_positions 
                    (position_id, symbol, total_quantity, average_cost, current_price, 
                     unrealized_pnl, realized_pnl, allocation_bucket)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(position_data.get('position_id', str(uuid.uuid4())))
                query.addBindValue(position_data.get('symbol', ''))
                query.addBindValue(int(position_data.get('total_quantity', 0)))
                query.addBindValue(float(position_data.get('average_cost', 0)))
                query.addBindValue(float(position_data.get('current_price', 0)))
                query.addBindValue(float(position_data.get('unrealized_pnl', 0)))
                query.addBindValue(float(position_data.get('realized_pnl', 0)))
                query.addBindValue(position_data.get('allocation_bucket', 'general'))
                
                return query.exec()
                
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return False
    
    # ======================================================================
    # SETTINGS AND CONFIGURATION METHODS
    # ======================================================================
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get user setting value."""
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            query.prepare("SELECT setting_value, data_type FROM user_settings WHERE setting_key = ?")
            query.addBindValue(key)
            
            if query.exec() and query.next():
                value = query.value(0)
                data_type = query.value(1)
                
                # Convert based on data type
                if data_type == 'integer':
                    return int(value)
                elif data_type == 'float':
                    return float(value)
                elif data_type == 'boolean':
                    return value.lower() in ('true', '1', 'yes')
                elif data_type == 'json':
                    return json.loads(value)
                else:
                    return value
            
            return default
            
        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return default
    
    def set_setting(self, key: str, value: Any, category: str = 'general') -> bool:
        """Set user setting value."""
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                # Determine data type
                if isinstance(value, bool):
                    data_type = 'boolean'
                    value_str = str(value).lower()
                elif isinstance(value, int):
                    data_type = 'integer'
                    value_str = str(value)
                elif isinstance(value, float):
                    data_type = 'float'
                    value_str = str(value)
                elif isinstance(value, (dict, list)):
                    data_type = 'json'
                    value_str = json.dumps(value)
                else:
                    data_type = 'string'
                    value_str = str(value)
                
                query.prepare("""
                    INSERT OR REPLACE INTO user_settings 
                    (setting_key, setting_value, setting_category, data_type, last_modified)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """)
                query.addBindValue(key)
                query.addBindValue(value_str)
                query.addBindValue(category)
                query.addBindValue(data_type)
                
                return query.exec()
                
        except Exception as e:
            logger.error(f"Error setting {key}: {e}")
            return False
    
    # ======================================================================
    # SECURITY METADATA METHODS
    # ======================================================================
    
    def save_security_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save security metadata to database."""
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                query.prepare("""
                    INSERT OR REPLACE INTO security_metadata 
                    (symbol, sector, industry, float_shares, market_cap, company_name, exchange, currency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metadata.get('symbol', ''))
                query.addBindValue(metadata.get('sector', ''))
                query.addBindValue(metadata.get('industry', ''))
                query.addBindValue(float(metadata.get('float_shares', 0)))
                query.addBindValue(float(metadata.get('market_cap', 0)))
                query.addBindValue(metadata.get('company_name', ''))
                query.addBindValue(metadata.get('exchange', ''))
                query.addBindValue(metadata.get('currency', 'USD'))
                
                return query.exec()
                
        except Exception as e:
            logger.error(f"Error saving security metadata: {e}")
            return False
    
    def get_security_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get security metadata by symbol."""
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            query.prepare("SELECT * FROM security_metadata WHERE symbol = ?")
            query.addBindValue(symbol)
            
            if query.exec() and query.next():
                return {
                    'symbol': query.value('symbol'),
                    'sector': query.value('sector'),
                    'industry': query.value('industry'),
                    'float_shares': query.value('float_shares'),
                    'market_cap': query.value('market_cap'),
                    'company_name': query.value('company_name'),
                    'exchange': query.value('exchange'),
                    'currency': query.value('currency'),
                    'last_updated': query.value('last_updated')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting security metadata for {symbol}: {e}")
            return None
    
    # ======================================================================
    # ALERT MANAGEMENT METHODS
    # ======================================================================
    
    def save_alert_rule(self, alert_data: Dict[str, Any]) -> bool:
        """Save alert rule to database."""
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                query.prepare("""
                    INSERT OR REPLACE INTO alert_rules 
                    (alert_id, symbol, condition_type, trigger_value, comparison_operator,
                     notification_method, is_active, alert_message, severity, expiry_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(alert_data.get('alert_id', str(uuid.uuid4())))
                query.addBindValue(alert_data.get('symbol', ''))
                query.addBindValue(alert_data.get('condition_type', ''))
                query.addBindValue(float(alert_data.get('trigger_value', 0)))
                query.addBindValue(alert_data.get('comparison_operator', '>'))
                query.addBindValue(alert_data.get('notification_method', 'desktop'))
                query.addBindValue(bool(alert_data.get('is_active', True)))
                query.addBindValue(alert_data.get('alert_message', ''))
                query.addBindValue(alert_data.get('severity', 'INFO'))
                query.addBindValue(alert_data.get('expiry_date'))
                
                return query.exec()
                
        except Exception as e:
            logger.error(f"Error saving alert rule: {e}")
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alert rules."""
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            query.exec("""
                SELECT * FROM alert_rules 
                WHERE is_active = 1 AND (expiry_date IS NULL OR expiry_date > CURRENT_TIMESTAMP)
                ORDER BY created_date DESC
            """)
            
            alerts = []
            while query.next():
                alerts.append({
                    'alert_id': query.value('alert_id'),
                    'symbol': query.value('symbol'),
                    'condition_type': query.value('condition_type'),
                    'trigger_value': query.value('trigger_value'),
                    'comparison_operator': query.value('comparison_operator'),
                    'notification_method': query.value('notification_method'),
                    'alert_message': query.value('alert_message'),
                    'severity': query.value('severity'),
                    'created_date': query.value('created_date'),
                    'last_triggered': query.value('last_triggered'),
                    'trigger_count': query.value('trigger_count')
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    # ======================================================================
    # JOURNAL METHODS
    # ======================================================================
    
    def save_journal_entry(self, entry_data: Dict[str, Any]) -> bool:
        """Save trading journal entry."""
        try:
            with self.transaction() as db:
                query = QSqlQuery(db)
                
                query.prepare("""
                    INSERT OR REPLACE INTO trading_journal 
                    (journal_id, trade_id, entry_title, analysis_notes, market_conditions,
                     lessons_learned, entry_type, tags, rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(entry_data.get('journal_id', str(uuid.uuid4())))
                query.addBindValue(entry_data.get('trade_id'))
                query.addBindValue(entry_data.get('entry_title', ''))
                query.addBindValue(entry_data.get('analysis_notes', ''))
                query.addBindValue(entry_data.get('market_conditions', ''))
                query.addBindValue(entry_data.get('lessons_learned', ''))
                query.addBindValue(entry_data.get('entry_type', 'trade_analysis'))
                query.addBindValue(json.dumps(entry_data.get('tags', [])))
                query.addBindValue(entry_data.get('rating'))
                
                return query.exec()
                
        except Exception as e:
            logger.error(f"Error saving journal entry: {e}")
            return False
    
    # ======================================================================
    # MAINTENANCE AND CLEANUP
    # ======================================================================
    
    def _periodic_cleanup(self):
        """Perform periodic database maintenance."""
        try:
            # Clean expired cache
            cleaned_count = self.cleanup_expired_cache()
            
            # Clean old scanner results (keep last 30 days)
            with self.transaction() as db:
                query = QSqlQuery(db)
                query.exec("DELETE FROM scanner_results WHERE scan_timestamp < date('now', '-30 days')")
            
            # Clean old audit logs (keep last 90 days)
            with self.transaction() as db:
                query = QSqlQuery(db)
                query.exec("DELETE FROM audit_log WHERE timestamp < date('now', '-90 days')")
            
            # Update statistics
            if cleaned_count > 0:
                logger.debug(f"Periodic cleanup completed: {cleaned_count} entries removed")
                
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")
    
    def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space."""
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            # VACUUM cannot be done in a transaction
            if query.exec("VACUUM"):
                logger.info("Database vacuum completed successfully")
                return True
            else:
                error = query.lastError()
                logger.error(f"Database vacuum failed: {error.text()}")
                return False
                
        except Exception as e:
            logger.error(f"Error during database vacuum: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            db = self.connection_manager.get_connection()
            query = QSqlQuery(db)
            
            stats = {
                'cache_stats': self.cache_stats.copy(),
                'database_path': str(self.db_path),
                'tables': {}
            }
            
            # Get table row counts
            tables = ['trades', 'portfolio_positions', 'market_data_cache', 'security_metadata',
                     'user_settings', 'trading_journal', 'alert_rules', 'scanner_results', 'watchlists']
            
            for table in tables:
                query.exec(f"SELECT COUNT(*) FROM {table}")
                if query.next():
                    stats['tables'][table] = query.value(0)
            
            # Get database file size
            if self.db_path.exists():
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create database backup."""
        try:
            if backup_path is None:
                backup_path = str(self.db_path.parent / f"trading_app_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    def close(self):
        """Close database connections and cleanup resources."""
        try:
            # Stop cleanup timer
            if self.cleanup_timer.isActive():
                self.cleanup_timer.stop()
            
            # Close all connections
            self.connection_manager.close_all_connections()
            
            logger.info("Database manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database manager: {e}")


# ======================================================================
# CONVENIENCE CLASSES AND FUNCTIONS
# ======================================================================

class TradingDatabase(DatabaseManager):
    """Convenience alias for DatabaseManager to match import in paper_trading.py"""
    pass


def create_database_manager(db_path: Optional[str] = None) -> DatabaseManager:
    """
    Create and initialize a database manager instance.
    
    Args:
        db_path: Optional path to database file
        
    Returns:
        Configured DatabaseManager instance
    """
    return DatabaseManager(db_path)


def get_default_database_path() -> str:
    """Get the default database path for the application."""
    try:
        if os.name == 'nt':  # Windows
            appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
            db_dir = Path(appdata) / 'Blitzy' / 'database'
        else:
            # Fallback for non-Windows
            db_dir = Path.home() / '.blitzy' / 'database'
        
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / 'trading_app.db')
        
    except Exception as e:
        logger.error(f"Error getting default database path: {e}")
        return 'trading_app.db'


# Export main classes and functions
__all__ = [
    'DatabaseManager',
    'TradingDatabase', 
    'CacheEntry',
    'DatabaseSchema',
    'DatabaseConnectionManager',
    'create_database_manager',
    'get_default_database_path'
]