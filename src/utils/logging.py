"""
Professional logging infrastructure for TradingApp Desktop Platform.

This module provides structured JSON logging with correlation ID tracking, automated 
rotation, and comprehensive monitoring integration for all application modules.

Features:
- Structured JSON logging with human-readable text fallback
- Correlation ID tracking with module-specific prefixes
- Daily rotation with compression and configurable retention policies
- Module-specific logging categories for enhanced traceability
- Performance metrics integration with QSqlDatabase persistence
- Thread-safe logging for multi-threaded PyQt6 application
- Automatic log aggregation with TTL management
- Windows %APPDATA% directory integration
- Circuit breaker and failover event tracking
- Comprehensive error context capture

Author: Blitzy Platform
Version: 1.0.0
"""

import os
import sys
import json
import uuid
import time
import gzip
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from contextlib import contextmanager
import logging
import logging.handlers
from logging import LogRecord
import traceback

# Third-party imports for desktop application
import psutil  # For system metrics
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QThread, QTimer
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

# Application-specific imports (will be available at runtime)
try:
    from config.default_settings import get_logging_config
except ImportError:
    # Fallback configuration if config module not available
    def get_logging_config():
        return {
            "log_level": "INFO",
            "rotation_days": 1,
            "retention_operations": 90,
            "retention_errors": 180,
            "max_file_size_mb": 100,
            "enable_json_format": True,
            "enable_compression": True,
            "performance_logging": True
        }


class LogLevel(Enum):
    """Enhanced log levels for trading application context."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PERFORMANCE = "PERFORMANCE"  # Custom level for performance metrics
    BUSINESS = "BUSINESS"        # Custom level for business metrics


class LogCategory(Enum):
    """Module-specific logging categories with enhanced identification."""
    APPLICATION = "application"
    RISK_CALCULATOR = "risk_calculator"
    PAPER_TRADING = "paper_trading"
    TRADING_JOURNAL = "journal"
    DATA_EXPORT = "export"
    DATA_PROVIDER_FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    SCANNER_ENGINE = "scanner"
    ALERT_MANAGER = "alerts"
    DATABASE = "database"
    PERFORMANCE = "performance"
    SECURITY = "security"
    GUI = "gui"
    DATA_INTEGRATION = "data_integration"


class CorrelationIdGenerator:
    """Enhanced correlation ID generator with module-specific prefixes."""
    
    _prefixes = {
        LogCategory.RISK_CALCULATOR: "risk_calc",
        LogCategory.PAPER_TRADING: "paper_trade",
        LogCategory.TRADING_JOURNAL: "journal_op",
        LogCategory.DATA_EXPORT: "export_gen",
        LogCategory.DATA_PROVIDER_FAILOVER: "failover",
        LogCategory.CIRCUIT_BREAKER: "circuit_br",
        LogCategory.SCANNER_ENGINE: "scanner",
        LogCategory.ALERT_MANAGER: "alert_mgr",
        LogCategory.DATABASE: "db_op",
        LogCategory.PERFORMANCE: "perf",
        LogCategory.SECURITY: "security",
        LogCategory.GUI: "gui_op",
        LogCategory.DATA_INTEGRATION: "data_integ",
        LogCategory.APPLICATION: "app"
    }
    
    @classmethod
    def generate(cls, category: LogCategory) -> str:
        """
        Generate correlation ID with module-specific prefix.
        
        Format: {prefix}_YYYYMMDD_HHMMSS_{uuid_suffix}
        """
        prefix = cls._prefixes.get(category, "general")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uuid_suffix = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{uuid_suffix}"


class LogMetrics:
    """Performance metrics collector for logging operations."""
    
    def __init__(self):
        self._metrics = {
            "total_logs": 0,
            "logs_by_level": {level.value: 0 for level in LogLevel},
            "logs_by_category": {cat.value: 0 for cat in LogCategory},
            "error_count": 0,
            "performance_log_count": 0,
            "average_log_time_ms": 0.0,
            "last_log_time": None
        }
        self._lock = threading.Lock()
    
    def record_log(self, level: LogLevel, category: LogCategory, 
                   processing_time_ms: float):
        """Record logging metrics for monitoring."""
        with self._lock:
            self._metrics["total_logs"] += 1
            self._metrics["logs_by_level"][level.value] += 1
            self._metrics["logs_by_category"][category.value] += 1
            self._metrics["last_log_time"] = datetime.now().isoformat()
            
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self._metrics["error_count"] += 1
            
            if level == LogLevel.PERFORMANCE:
                self._metrics["performance_log_count"] += 1
            
            # Update average processing time
            current_avg = self._metrics["average_log_time_ms"]
            total_logs = self._metrics["total_logs"]
            self._metrics["average_log_time_ms"] = (
                (current_avg * (total_logs - 1) + processing_time_ms) / total_logs
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current logging metrics."""
        with self._lock:
            return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics (typically called daily)."""
        with self._lock:
            self._metrics = {
                "total_logs": 0,
                "logs_by_level": {level.value: 0 for level in LogLevel},
                "logs_by_category": {cat.value: 0 for cat in LogCategory},
                "error_count": 0,
                "performance_log_count": 0,
                "average_log_time_ms": 0.0,
                "last_log_time": None
            }


class StructuredFormatter(logging.Formatter):
    """Enhanced structured JSON formatter with fallback to human-readable text."""
    
    def __init__(self, use_json: bool = True, include_system_info: bool = True):
        super().__init__()
        self.use_json = use_json
        self.include_system_info = include_system_info
        self._system_info_cache = None
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get cached system information with TTL."""
        current_time = time.time()
        if (self._system_info_cache is None or 
            self._cache_time is None or 
            current_time - self._cache_time > self._cache_ttl):
            
            try:
                process = psutil.Process()
                self._system_info_cache = {
                    "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                    "cpu_percent": process.cpu_percent(),
                    "thread_count": process.num_threads(),
                    "process_id": os.getpid(),
                    "thread_id": threading.get_ident()
                }
                self._cache_time = current_time
            except Exception:
                self._system_info_cache = {
                    "process_id": os.getpid(),
                    "thread_id": threading.get_ident()
                }
        
        return self._system_info_cache
    
    def format(self, record: LogRecord) -> str:
        """Format log record as structured JSON or human-readable text."""
        # Extract custom attributes
        correlation_id = getattr(record, 'correlation_id', None)
        category = getattr(record, 'category', LogCategory.APPLICATION.value)
        context = getattr(record, 'context', {})
        execution_time_ms = getattr(record, 'execution_time_ms', None)
        
        if self.use_json:
            return self._format_json(record, correlation_id, category, 
                                   context, execution_time_ms)
        else:
            return self._format_text(record, correlation_id, category, 
                                   context, execution_time_ms)
    
    def _format_json(self, record: LogRecord, correlation_id: Optional[str],
                     category: str, context: Dict[str, Any], 
                     execution_time_ms: Optional[float]) -> str:
        """Format as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": category,
            "message": record.getMessage(),
            "logger_name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        if execution_time_ms is not None:
            log_entry["execution_time_ms"] = execution_time_ms
        
        if context:
            log_entry["context"] = context
        
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        if self.include_system_info:
            log_entry["system"] = self._get_system_info()
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def _format_text(self, record: LogRecord, correlation_id: Optional[str],
                     category: str, context: Dict[str, Any], 
                     execution_time_ms: Optional[float]) -> str:
        """Format as human-readable text."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        parts = [
            f"[{timestamp}]",
            f"[{record.levelname:8}]",
            f"[{category}]"
        ]
        
        if correlation_id:
            parts.append(f"[{correlation_id}]")
        
        parts.append(f"{record.getMessage()}")
        
        if execution_time_ms is not None:
            parts.append(f"(took {execution_time_ms:.2f}ms)")
        
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f"({context_str})")
        
        if record.exc_info:
            parts.append(f"\nException: {record.exc_info[1]}")
        
        return " ".join(parts)


class LogFileManager:
    """Enhanced log file management with compression and retention policies."""
    
    def __init__(self, base_directory: Path, retention_policies: Dict[str, int]):
        self.base_directory = Path(base_directory)
        self.retention_policies = retention_policies
        self._ensure_directory_structure()
    
    def _ensure_directory_structure(self):
        """Create required directory structure."""
        directories = [
            self.base_directory,
            self.base_directory / "operations",
            self.base_directory / "errors", 
            self.base_directory / "performance",
            self.base_directory / "business",
            self.base_directory / "archived"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_log_file_path(self, category: LogCategory, level: LogLevel) -> Path:
        """Get appropriate log file path based on category and level."""
        date_str = datetime.now().strftime("%Y%m%d")
        
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            return self.base_directory / "errors" / f"{category.value}_{date_str}.log"
        elif level == LogLevel.PERFORMANCE:
            return self.base_directory / "performance" / f"{category.value}_{date_str}.log"
        elif level == LogLevel.BUSINESS:
            return self.base_directory / "business" / f"{category.value}_{date_str}.log"
        else:
            return self.base_directory / "operations" / f"{category.value}_{date_str}.log"
    
    def compress_old_logs(self):
        """Compress log files older than 1 day."""
        yesterday = datetime.now() - timedelta(days=1)
        date_threshold = yesterday.strftime("%Y%m%d")
        
        for log_dir in ["operations", "errors", "performance", "business"]:
            log_path = self.base_directory / log_dir
            if not log_path.exists():
                continue
                
            for log_file in log_path.glob("*.log"):
                # Extract date from filename
                try:
                    file_date = log_file.stem.split("_")[-1]
                    if file_date < date_threshold:
                        self._compress_file(log_file)
                except (ValueError, IndexError):
                    # Skip files that don't match expected naming pattern
                    continue
    
    def _compress_file(self, file_path: Path):
        """Compress individual log file."""
        compressed_path = file_path.with_suffix(".log.gz")
        
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file after successful compression
            file_path.unlink()
            
        except Exception as e:
            print(f"Failed to compress {file_path}: {e}")
    
    def cleanup_old_logs(self):
        """Remove logs based on retention policies."""
        current_date = datetime.now()
        
        retention_mapping = {
            "operations": self.retention_policies.get("operations", 90),
            "errors": self.retention_policies.get("errors", 180),
            "performance": self.retention_policies.get("performance", 30),
            "business": self.retention_policies.get("business", 90)
        }
        
        for log_type, retention_days in retention_mapping.items():
            cutoff_date = current_date - timedelta(days=retention_days)
            log_dir = self.base_directory / log_type
            
            if not log_dir.exists():
                continue
            
            # Clean up both .log and .log.gz files
            for pattern in ["*.log", "*.log.gz"]:
                for log_file in log_dir.glob(pattern):
                    try:
                        # Extract date from filename
                        base_name = log_file.name.replace(".log.gz", "").replace(".log", "")
                        file_date_str = base_name.split("_")[-1]
                        file_date = datetime.strptime(file_date_str, "%Y%m%d")
                        
                        if file_date < cutoff_date:
                            # Archive before deletion
                            self._archive_log(log_file)
                            log_file.unlink()
                            
                    except (ValueError, IndexError):
                        # Skip files that don't match expected naming pattern
                        continue
    
    def _archive_log(self, log_file: Path):
        """Archive log file before deletion."""
        archive_dir = self.base_directory / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        # Create year/month subdirectory
        try:
            base_name = log_file.name.replace(".log.gz", "").replace(".log", "")
            file_date_str = base_name.split("_")[-1]
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            
            year_month_dir = archive_dir / f"{file_date.year}" / f"{file_date.month:02d}"
            year_month_dir.mkdir(parents=True, exist_ok=True)
            
            archive_path = year_month_dir / log_file.name
            shutil.copy2(log_file, archive_path)
            
        except Exception as e:
            print(f"Failed to archive {log_file}: {e}")


class DatabaseMetricsLogger(QObject):
    """QSqlDatabase integration for metrics persistence."""
    
    metrics_recorded = pyqtSignal(dict)  # Signal for real-time monitoring
    
    def __init__(self, db_connection_name: str = "metrics_db"):
        super().__init__()
        self.db_connection_name = db_connection_name
        self._mutex = QMutex()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize metrics database with required tables."""
        db = QSqlDatabase.addDatabase("QSQLITE", self.db_connection_name)
        
        # Use %APPDATA%/TradingApp/metrics.db as specified in requirements
        app_data_dir = Path(os.getenv("APPDATA", "")) / "TradingApp"
        app_data_dir.mkdir(exist_ok=True)
        db_path = app_data_dir / "metrics.db"
        
        db.setDatabaseName(str(db_path))
        
        if not db.open():
            raise RuntimeError(f"Failed to open metrics database: {db.lastError().text()}")
        
        self._create_tables()
    
    def _create_tables(self):
        """Create metrics tables as specified in technical requirements."""
        db = QSqlDatabase.database(self.db_connection_name)
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS logging_metrics (
                metric_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                processing_time_ms REAL,
                message_length INTEGER,
                correlation_id TEXT,
                thread_id INTEGER,
                memory_mb REAL,
                cpu_percent REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                operation TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS error_context (
                error_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                traceback TEXT,
                correlation_id TEXT,
                context TEXT
            )
            """
        ]
        
        for table_sql in tables:
            query = QSqlQuery(db)
            if not query.exec(table_sql):
                print(f"Failed to create table: {query.lastError().text()}")
    
    def record_log_metrics(self, level: LogLevel, category: LogCategory,
                          processing_time_ms: float, message_length: int,
                          correlation_id: Optional[str] = None,
                          system_info: Optional[Dict[str, Any]] = None):
        """Record log metrics to database."""
        self._mutex.lock()
        try:
            db = QSqlDatabase.database(self.db_connection_name)
            query = QSqlQuery(db)
            
            query.prepare("""
                INSERT INTO logging_metrics 
                (metric_id, level, category, processing_time_ms, message_length,
                 correlation_id, thread_id, memory_mb, cpu_percent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """)
            
            metric_id = str(uuid.uuid4())
            query.addBindValue(metric_id)
            query.addBindValue(level.value)
            query.addBindValue(category.value)
            query.addBindValue(processing_time_ms)
            query.addBindValue(message_length)
            query.addBindValue(correlation_id)
            query.addBindValue(threading.get_ident())
            
            if system_info:
                query.addBindValue(system_info.get("memory_mb", 0.0))
                query.addBindValue(system_info.get("cpu_percent", 0.0))
            else:
                query.addBindValue(0.0)
                query.addBindValue(0.0)
            
            if not query.exec():
                print(f"Failed to record log metrics: {query.lastError().text()}")
            else:
                # Emit signal for real-time monitoring
                metrics_data = {
                    "metric_id": metric_id,
                    "level": level.value,
                    "category": category.value,
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now().isoformat()
                }
                self.metrics_recorded.emit(metrics_data)
                
        finally:
            self._mutex.unlock()
    
    def record_error_context(self, level: LogLevel, category: LogCategory,
                           error_type: str, error_message: str,
                           traceback_str: str, correlation_id: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None):
        """Record detailed error context for debugging."""
        self._mutex.lock()
        try:
            db = QSqlDatabase.database(self.db_connection_name)
            query = QSqlQuery(db)
            
            query.prepare("""
                INSERT INTO error_context 
                (error_id, level, category, error_type, error_message,
                 traceback, correlation_id, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """)
            
            error_id = str(uuid.uuid4())
            query.addBindValue(error_id)
            query.addBindValue(level.value)
            query.addBindValue(category.value)
            query.addBindValue(error_type)
            query.addBindValue(error_message)
            query.addBindValue(traceback_str)
            query.addBindValue(correlation_id)
            query.addBindValue(json.dumps(context) if context else None)
            
            if not query.exec():
                print(f"Failed to record error context: {query.lastError().text()}")
                
        finally:
            self._mutex.unlock()
    
    def cleanup_old_metrics(self, retention_days: int = 30):
        """Clean up old metrics data."""
        self._mutex.lock()
        try:
            db = QSqlDatabase.database(self.db_connection_name)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for table in ["logging_metrics", "performance_metrics"]:
                query = QSqlQuery(db)
                query.prepare(f"DELETE FROM {table} WHERE timestamp < ?")
                query.addBindValue(cutoff_date.isoformat())
                
                if not query.exec():
                    print(f"Failed to cleanup {table}: {query.lastError().text()}")
                    
        finally:
            self._mutex.unlock()


class TradingAppLogger:
    """
    Comprehensive logging infrastructure for the TradingApp Desktop Platform.
    
    Provides structured JSON logging, correlation ID tracking, performance metrics,
    and comprehensive monitoring integration for all application modules.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton implementation for application-wide logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logging infrastructure."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.config = get_logging_config()
        self._setup_log_directory()
        self._setup_metrics()
        self._setup_file_manager()
        self._setup_database_logger()
        self._setup_loggers()
        self._setup_cleanup_timer()
        
        # Thread-local storage for correlation IDs
        self._local = threading.local()
    
    def _setup_log_directory(self):
        """Setup log directory structure in %APPDATA%/TradingApp/logs/."""
        app_data_dir = Path(os.getenv("APPDATA", "")) / "TradingApp"
        self.log_directory = app_data_dir / "logs"
        self.log_directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_metrics(self):
        """Initialize metrics collector."""
        self.metrics = LogMetrics()
    
    def _setup_file_manager(self):
        """Initialize file management with retention policies."""
        retention_policies = {
            "operations": self.config.get("retention_operations", 90),
            "errors": self.config.get("retention_errors", 180),
            "performance": 30,
            "business": 90
        }
        self.file_manager = LogFileManager(self.log_directory, retention_policies)
    
    def _setup_database_logger(self):
        """Initialize database metrics logger."""
        try:
            self.db_logger = DatabaseMetricsLogger()
        except Exception as e:
            print(f"Failed to initialize database logger: {e}")
            self.db_logger = None
    
    def _setup_loggers(self):
        """Setup Python logging infrastructure."""
        # Configure custom log levels
        logging.addLevelName(35, "PERFORMANCE")
        logging.addLevelName(25, "BUSINESS")
        
        # Setup formatters
        self.json_formatter = StructuredFormatter(
            use_json=self.config.get("enable_json_format", True),
            include_system_info=True
        )
        
        self.text_formatter = StructuredFormatter(
            use_json=False,
            include_system_info=False
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.text_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
    
    def _setup_cleanup_timer(self):
        """Setup periodic cleanup timer using QTimer."""
        try:
            from PyQt6.QtCore import QTimer
            self.cleanup_timer = QTimer()
            self.cleanup_timer.timeout.connect(self._periodic_cleanup)
            # Run cleanup every 4 hours
            self.cleanup_timer.start(4 * 60 * 60 * 1000)
        except ImportError:
            # Fallback if Qt not available
            pass
    
    def _periodic_cleanup(self):
        """Perform periodic maintenance tasks."""
        try:
            self.file_manager.compress_old_logs()
            self.file_manager.cleanup_old_logs()
            
            if self.db_logger:
                self.db_logger.cleanup_old_metrics()
                
            # Reset daily metrics at midnight
            current_time = datetime.now()
            if current_time.hour == 0 and current_time.minute < 5:
                self.metrics.reset_metrics()
                
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    @contextmanager
    def correlation_context(self, category: LogCategory, 
                          custom_id: Optional[str] = None):
        """
        Context manager for correlation ID tracking.
        
        Usage:
            with logger.correlation_context(LogCategory.RISK_CALCULATOR):
                logger.info("Risk calculation started", category=LogCategory.RISK_CALCULATOR)
                # ... operations ...
                logger.info("Risk calculation completed", category=LogCategory.RISK_CALCULATOR)
        """
        correlation_id = custom_id or CorrelationIdGenerator.generate(category)
        
        # Store in thread-local storage
        old_correlation_id = getattr(self._local, 'correlation_id', None)
        self._local.correlation_id = correlation_id
        
        try:
            yield correlation_id
        finally:
            # Restore previous correlation ID
            if old_correlation_id:
                self._local.correlation_id = old_correlation_id
            else:
                if hasattr(self._local, 'correlation_id'):
                    delattr(self._local, 'correlation_id')
    
    def _get_current_correlation_id(self) -> Optional[str]:
        """Get current correlation ID from thread-local storage."""
        return getattr(self._local, 'correlation_id', None)
    
    def _log(self, level: LogLevel, message: str, category: LogCategory,
             context: Optional[Dict[str, Any]] = None,
             execution_time_ms: Optional[float] = None,
             correlation_id: Optional[str] = None,
             exc_info: Optional[bool] = None):
        """Internal logging method with comprehensive tracking."""
        start_time = time.time()
        
        # Use provided correlation_id or get from thread-local storage
        final_correlation_id = correlation_id or self._get_current_correlation_id()
        
        # Get appropriate logger
        logger_name = f"tradingapp.{category.value}"
        logger = logging.getLogger(logger_name)
        
        # Ensure logger has appropriate handler
        if not logger.handlers:
            self._add_category_handler(logger, category, level)
        
        # Create log record
        log_level = getattr(logging, level.value)
        if hasattr(logging, level.value):
            logger_method = getattr(logger, level.value.lower(), logger.info)
        else:
            logger_method = logger.info
        
        # Add custom attributes to log record
        extra = {
            'correlation_id': final_correlation_id,
            'category': category.value,
            'context': context or {},
            'execution_time_ms': execution_time_ms
        }
        
        # Log the message
        try:
            if exc_info:
                logger_method(message, exc_info=exc_info, extra=extra)
            else:
                logger_method(message, extra=extra)
        except Exception as e:
            # Fallback logging
            print(f"Logging error: {e} - Original message: {message}")
        
        # Record metrics
        processing_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_log(level, category, processing_time_ms)
        
        # Record to database if available
        if self.db_logger and self.config.get("performance_logging", True):
            try:
                system_info = self.json_formatter._get_system_info()
                self.db_logger.record_log_metrics(
                    level, category, processing_time_ms, len(message),
                    final_correlation_id, system_info
                )
                
                # Record error context for errors
                if level in [LogLevel.ERROR, LogLevel.CRITICAL] and exc_info:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    if exc_type:
                        self.db_logger.record_error_context(
                            level, category, exc_type.__name__, str(exc_value),
                            traceback.format_tb(exc_traceback), final_correlation_id,
                            context
                        )
            except Exception as e:
                print(f"Database logging error: {e}")
    
    def _add_category_handler(self, logger: logging.Logger, 
                            category: LogCategory, level: LogLevel):
        """Add appropriate file handler for category and level."""
        log_file_path = self.file_manager.get_log_file_path(category, level)
        
        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=self.config.get("max_file_size_mb", 100) * 1024 * 1024,
            backupCount=7  # Keep 7 backup files
        )
        
        handler.setFormatter(self.json_formatter)
        handler.setLevel(getattr(logging, level.value))
        
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Let handler filter
    
    # Public logging methods
    def debug(self, message: str, category: LogCategory = LogCategory.APPLICATION,
              context: Optional[Dict[str, Any]] = None,
              execution_time_ms: Optional[float] = None,
              correlation_id: Optional[str] = None):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, category, context, 
                 execution_time_ms, correlation_id)
    
    def info(self, message: str, category: LogCategory = LogCategory.APPLICATION,
             context: Optional[Dict[str, Any]] = None,
             execution_time_ms: Optional[float] = None,
             correlation_id: Optional[str] = None):
        """Log info message."""
        self._log(LogLevel.INFO, message, category, context, 
                 execution_time_ms, correlation_id)
    
    def warning(self, message: str, category: LogCategory = LogCategory.APPLICATION,
                context: Optional[Dict[str, Any]] = None,
                execution_time_ms: Optional[float] = None,
                correlation_id: Optional[str] = None):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, category, context, 
                 execution_time_ms, correlation_id)
    
    def error(self, message: str, category: LogCategory = LogCategory.APPLICATION,
              context: Optional[Dict[str, Any]] = None,
              execution_time_ms: Optional[float] = None,
              correlation_id: Optional[str] = None,
              exc_info: bool = True):
        """Log error message with exception info."""
        self._log(LogLevel.ERROR, message, category, context, 
                 execution_time_ms, correlation_id, exc_info)
    
    def critical(self, message: str, category: LogCategory = LogCategory.APPLICATION,
                 context: Optional[Dict[str, Any]] = None,
                 execution_time_ms: Optional[float] = None,
                 correlation_id: Optional[str] = None,
                 exc_info: bool = True):
        """Log critical message with exception info."""
        self._log(LogLevel.CRITICAL, message, category, context, 
                 execution_time_ms, correlation_id, exc_info)
    
    def performance(self, message: str, category: LogCategory = LogCategory.PERFORMANCE,
                   context: Optional[Dict[str, Any]] = None,
                   execution_time_ms: Optional[float] = None,
                   correlation_id: Optional[str] = None):
        """Log performance metric."""
        self._log(LogLevel.PERFORMANCE, message, category, context, 
                 execution_time_ms, correlation_id)
    
    def business(self, message: str, category: LogCategory = LogCategory.APPLICATION,
                context: Optional[Dict[str, Any]] = None,
                execution_time_ms: Optional[float] = None,
                correlation_id: Optional[str] = None):
        """Log business metric."""
        self._log(LogLevel.BUSINESS, message, category, context, 
                 execution_time_ms, correlation_id)
    
    # Specialized logging methods for module-specific operations
    def log_risk_calculation(self, message: str, calculation_type: str,
                            input_parameters: Dict[str, Any],
                            result: Optional[Dict[str, Any]] = None,
                            execution_time_ms: Optional[float] = None,
                            correlation_id: Optional[str] = None):
        """Log risk calculator specific operations."""
        context = {
            "calculation_type": calculation_type,
            "input_parameters": input_parameters
        }
        if result:
            context["result"] = result
            
        self.info(message, LogCategory.RISK_CALCULATOR, context, 
                 execution_time_ms, correlation_id)
    
    def log_paper_trade(self, message: str, trade_action: str,
                       symbol: str, quantity: int, price: float,
                       portfolio_value: Optional[float] = None,
                       execution_time_ms: Optional[float] = None,
                       correlation_id: Optional[str] = None):
        """Log paper trading operations."""
        context = {
            "trade_action": trade_action,
            "symbol": symbol,
            "quantity": quantity,
            "price": price
        }
        if portfolio_value:
            context["portfolio_value"] = portfolio_value
            
        self.info(message, LogCategory.PAPER_TRADING, context, 
                 execution_time_ms, correlation_id)
    
    def log_journal_operation(self, message: str, operation_type: str,
                             entry_id: Optional[str] = None,
                             entry_size_bytes: Optional[int] = None,
                             execution_time_ms: Optional[float] = None,
                             correlation_id: Optional[str] = None):
        """Log trading journal operations."""
        context = {
            "operation_type": operation_type
        }
        if entry_id:
            context["entry_id"] = entry_id
        if entry_size_bytes:
            context["entry_size_bytes"] = entry_size_bytes
            
        self.info(message, LogCategory.TRADING_JOURNAL, context, 
                 execution_time_ms, correlation_id)
    
    def log_export_operation(self, message: str, export_type: str,
                           record_count: int, file_size_bytes: Optional[int] = None,
                           execution_time_ms: Optional[float] = None,
                           correlation_id: Optional[str] = None):
        """Log data export operations."""
        context = {
            "export_type": export_type,
            "record_count": record_count
        }
        if file_size_bytes:
            context["file_size_bytes"] = file_size_bytes
            
        self.info(message, LogCategory.DATA_EXPORT, context, 
                 execution_time_ms, correlation_id)
    
    def log_provider_failover(self, message: str, failed_provider: str,
                             fallback_provider: str, failure_reason: str,
                             retry_count: int = 0,
                             correlation_id: Optional[str] = None):
        """Log data provider failover events."""
        context = {
            "failed_provider": failed_provider,
            "fallback_provider": fallback_provider,
            "failure_reason": failure_reason,
            "retry_count": retry_count
        }
        
        self.warning(message, LogCategory.DATA_PROVIDER_FAILOVER, context, 
                    correlation_id=correlation_id)
    
    def log_circuit_breaker(self, message: str, circuit_state: str,
                           provider: str, failure_count: int = 0,
                           correlation_id: Optional[str] = None):
        """Log circuit breaker events."""
        context = {
            "circuit_state": circuit_state,
            "provider": provider,
            "failure_count": failure_count
        }
        
        self.warning(message, LogCategory.CIRCUIT_BREAKER, context, 
                    correlation_id=correlation_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current logging metrics."""
        return self.metrics.get_metrics()
    
    def force_cleanup(self):
        """Force immediate cleanup of logs and metrics."""
        self._periodic_cleanup()


# Global logger instance
logger = TradingAppLogger()

# Convenience exports for easy importing
__all__ = [
    'TradingAppLogger',
    'LogLevel', 
    'LogCategory',
    'CorrelationIdGenerator',
    'logger'
]