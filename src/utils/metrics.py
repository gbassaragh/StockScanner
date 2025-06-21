"""
Performance metrics collection utilities for the Blitzy Trading Platform.

This module provides comprehensive metrics persistence using QSqlDatabase integration,
module-specific performance tracking, SLA monitoring, and automated analytics for
application observability and optimization.

Key Features:
- QSqlDatabase integration with metrics persistence to %APPDATA%/TradingApp/metrics.db
- Module-specific tracking for Risk Calculator, Paper Trading, Trading Journal, and Export operations
- SLA monitoring utilities with real-time performance measurement and threshold validation
- Business metrics collection for feature utilization analytics
- Alert threshold monitoring with notification integration
- Automated database schema management with data retention policies

Author: Blitzy Development Team
Version: 1.0.0
"""

import os
import json
import uuid
import time
import logging
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from PyQt6.QtSql import QSqlDatabase, QSqlQuery, QSqlError
    from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
except ImportError:
    # Fallback for development/testing environments
    QSqlDatabase = None
    QSqlQuery = None
    QSqlError = None
    QObject = object
    pyqtSignal = lambda: None
    QTimer = None
    QThread = None


class MetricType(Enum):
    """Enumeration of metric types for categorization."""
    PERFORMANCE = "performance"
    RISK_CALCULATION = "risk_calculation"
    TRADE_SIMULATION = "trade_simulation"
    JOURNAL_OPERATION = "journal_operation"
    EXPORT_OPERATION = "export_operation"
    SYSTEM_HEALTH = "system_health"
    BUSINESS_USAGE = "business_usage"
    SLA_COMPLIANCE = "sla_compliance"


class SeverityLevel(Enum):
    """Alert severity levels for threshold monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Data structure for performance metrics."""
    metric_id: str
    component: str
    operation: str
    duration_ms: int
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class RiskCalculationMetric:
    """Data structure for risk calculator performance metrics."""
    metric_id: str
    calculation_type: str  # 'atr_calculation', 'position_sizing', 'risk_allocation'
    execution_time_ms: int
    input_parameters: Dict[str, Any]
    result_quality_score: float
    timestamp: datetime
    success: bool = True


@dataclass
class SimulationMetric:
    """Data structure for paper trading simulation metrics."""
    metric_id: str
    operation_type: str  # 'portfolio_update', 'pnl_calculation', 'trade_execution'
    execution_time_ms: int
    portfolio_size_records: int
    slippage_calculation_ms: int
    timestamp: datetime
    success: bool = True


@dataclass
class JournalMetric:
    """Data structure for trading journal performance metrics."""
    metric_id: str
    operation_type: str  # 'entry_save', 'search_query', 'bulk_export'
    execution_time_ms: int
    entry_size_bytes: int
    database_transaction_time_ms: int
    timestamp: datetime
    success: bool = True


@dataclass
class ExportMetric:
    """Data structure for data export performance metrics."""
    metric_id: str
    export_type: str  # 'scanner_results', 'trade_history', 'performance_report'
    generation_time_ms: int
    record_count: int
    file_size_bytes: int
    excel_formatting_time_ms: int
    timestamp: datetime
    success: bool = True


@dataclass
class SLAThreshold:
    """SLA threshold configuration."""
    component: str
    operation: str
    threshold_ms: int
    warning_threshold_ms: Optional[int] = None
    severity: SeverityLevel = SeverityLevel.WARNING


@dataclass
class BusinessMetric:
    """Business usage metric data structure."""
    metric_id: str
    module_name: str
    event_type: str
    event_count: int = 1
    success_rate: float = 1.0
    average_execution_time_ms: int = 0
    error_count: int = 0
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MetricsDatabase:
    """
    Database connection and schema management for metrics persistence.
    
    Provides QSqlDatabase integration with automated schema creation,
    data retention policies, and thread-safe operations.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize metrics database connection.
        
        Args:
            database_path: Optional custom database path. Defaults to %APPDATA%/TradingApp/metrics.db
        """
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._connection_name = f"metrics_db_{id(self)}"
        
        # Determine database path
        if database_path is None:
            appdata_path = os.getenv('APPDATA')
            if not appdata_path:
                raise RuntimeError("APPDATA environment variable not found")
            
            trading_app_dir = os.path.join(appdata_path, 'TradingApp')
            os.makedirs(trading_app_dir, exist_ok=True)
            self.database_path = os.path.join(trading_app_dir, 'metrics.db')
        else:
            self.database_path = database_path
            
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        # Initialize database connection
        self._initialize_connection()
        self._create_schema()
        
        self.logger.info(f"Metrics database initialized at: {self.database_path}")

    def _initialize_connection(self) -> bool:
        """Initialize QSqlDatabase connection with error handling."""
        if QSqlDatabase is None:
            self.logger.warning("PyQt6 not available, using fallback mode")
            return False
            
        try:
            # Remove existing connection if it exists
            if QSqlDatabase.contains(self._connection_name):
                QSqlDatabase.removeDatabase(self._connection_name)
            
            # Create new SQLite connection
            db = QSqlDatabase.addDatabase("QSQLITE", self._connection_name)
            db.setDatabaseName(self.database_path)
            
            if not db.open():
                error = db.lastError()
                self.logger.error(f"Failed to open metrics database: {error.text()}")
                return False
                
            self.logger.info("QSqlDatabase connection established successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            return False

    @contextmanager
    def get_connection(self):
        """Thread-safe database connection context manager."""
        with self._lock:
            if QSqlDatabase is None:
                yield None
                return
                
            db = QSqlDatabase.database(self._connection_name)
            if not db.isOpen():
                if not self._initialize_connection():
                    yield None
                    return
                db = QSqlDatabase.database(self._connection_name)
            
            yield db

    def _create_schema(self) -> bool:
        """Create database schema with all required tables and indexes."""
        schema_queries = [
            # Performance metrics table
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                operation TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                success BOOLEAN DEFAULT 1,
                error_message TEXT
            )
            """,
            
            # Risk calculation metrics table
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                metric_id TEXT PRIMARY KEY,
                calculation_type TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                input_parameters TEXT,
                result_quality_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1
            )
            """,
            
            # Paper trading simulation metrics table
            """
            CREATE TABLE IF NOT EXISTS simulation_metrics (
                metric_id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                portfolio_size_records INTEGER,
                slippage_calculation_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1
            )
            """,
            
            # Trading journal metrics table
            """
            CREATE TABLE IF NOT EXISTS journal_metrics (
                metric_id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                entry_size_bytes INTEGER,
                database_transaction_time_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1
            )
            """,
            
            # Data export metrics table
            """
            CREATE TABLE IF NOT EXISTS export_metrics (
                metric_id TEXT PRIMARY KEY,
                export_type TEXT NOT NULL,
                generation_time_ms INTEGER NOT NULL,
                record_count INTEGER,
                file_size_bytes INTEGER,
                excel_formatting_time_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1
            )
            """,
            
            # Business metrics table
            """
            CREATE TABLE IF NOT EXISTS business_metrics (
                metric_id TEXT PRIMARY KEY,
                module_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_count INTEGER DEFAULT 1,
                success_rate REAL,
                average_execution_time_ms INTEGER,
                error_count INTEGER DEFAULT 0,
                last_event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                daily_aggregation_date DATE,
                metadata TEXT
            )
            """,
            
            # SLA violations table
            """
            CREATE TABLE IF NOT EXISTS sla_violations (
                violation_id TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                operation TEXT NOT NULL,
                threshold_ms INTEGER NOT NULL,
                actual_value_ms INTEGER NOT NULL,
                severity TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT 0
            )
            """,
            
            # System health metrics table
            """
            CREATE TABLE IF NOT EXISTS system_health (
                health_id TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        # Performance indexes
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_performance_component_time ON performance_metrics(component, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_performance_operation ON performance_metrics(operation, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_risk_calculation_type ON risk_metrics(calculation_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_simulation_operation ON simulation_metrics(operation_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_journal_operation ON journal_metrics(operation_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_export_type ON export_metrics(export_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_business_module ON business_metrics(module_name, event_type)",
            "CREATE INDEX IF NOT EXISTS idx_sla_component ON sla_violations(component, severity, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_health_component ON system_health(component, status, timestamp DESC)"
        ]
        
        try:
            with self.get_connection() as db:
                if db is None:
                    self.logger.warning("Database connection not available, schema creation skipped")
                    return False
                
                # Execute schema creation queries
                for query_sql in schema_queries:
                    query = QSqlQuery(db)
                    if not query.exec(query_sql):
                        error = query.lastError()
                        self.logger.error(f"Failed to create schema: {error.text()}")
                        return False
                
                # Create indexes
                for index_sql in index_queries:
                    query = QSqlQuery(db)
                    if not query.exec(index_sql):
                        error = query.lastError()
                        self.logger.warning(f"Failed to create index: {error.text()}")
                
                self.logger.info("Database schema created successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Schema creation failed: {e}")
            return False

    def insert_performance_metric(self, metric: PerformanceMetric) -> bool:
        """Insert performance metric into database."""
        try:
            with self.get_connection() as db:
                if db is None:
                    return False
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT INTO performance_metrics 
                    (metric_id, component, operation, duration_ms, timestamp, context, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metric.metric_id)
                query.addBindValue(metric.component)
                query.addBindValue(metric.operation)
                query.addBindValue(metric.duration_ms)
                query.addBindValue(metric.timestamp.isoformat())
                query.addBindValue(json.dumps(metric.context) if metric.context else None)
                query.addBindValue(metric.success)
                query.addBindValue(metric.error_message)
                
                if not query.exec():
                    error = query.lastError()
                    self.logger.error(f"Failed to insert performance metric: {error.text()}")
                    return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert performance metric: {e}")
            return False

    def insert_risk_metric(self, metric: RiskCalculationMetric) -> bool:
        """Insert risk calculation metric into database."""
        try:
            with self.get_connection() as db:
                if db is None:
                    return False
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT INTO risk_metrics 
                    (metric_id, calculation_type, execution_time_ms, input_parameters, result_quality_score, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metric.metric_id)
                query.addBindValue(metric.calculation_type)
                query.addBindValue(metric.execution_time_ms)
                query.addBindValue(json.dumps(metric.input_parameters))
                query.addBindValue(float(metric.result_quality_score))
                query.addBindValue(metric.timestamp.isoformat())
                query.addBindValue(metric.success)
                
                return query.exec()
                
        except Exception as e:
            self.logger.error(f"Failed to insert risk metric: {e}")
            return False

    def insert_simulation_metric(self, metric: SimulationMetric) -> bool:
        """Insert simulation metric into database."""
        try:
            with self.get_connection() as db:
                if db is None:
                    return False
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT INTO simulation_metrics 
                    (metric_id, operation_type, execution_time_ms, portfolio_size_records, slippage_calculation_ms, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metric.metric_id)
                query.addBindValue(metric.operation_type)
                query.addBindValue(metric.execution_time_ms)
                query.addBindValue(metric.portfolio_size_records)
                query.addBindValue(metric.slippage_calculation_ms)
                query.addBindValue(metric.timestamp.isoformat())
                query.addBindValue(metric.success)
                
                return query.exec()
                
        except Exception as e:
            self.logger.error(f"Failed to insert simulation metric: {e}")
            return False

    def insert_journal_metric(self, metric: JournalMetric) -> bool:
        """Insert journal metric into database."""
        try:
            with self.get_connection() as db:
                if db is None:
                    return False
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT INTO journal_metrics 
                    (metric_id, operation_type, execution_time_ms, entry_size_bytes, database_transaction_time_ms, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metric.metric_id)
                query.addBindValue(metric.operation_type)
                query.addBindValue(metric.execution_time_ms)
                query.addBindValue(metric.entry_size_bytes)
                query.addBindValue(metric.database_transaction_time_ms)
                query.addBindValue(metric.timestamp.isoformat())
                query.addBindValue(metric.success)
                
                return query.exec()
                
        except Exception as e:
            self.logger.error(f"Failed to insert journal metric: {e}")
            return False

    def insert_export_metric(self, metric: ExportMetric) -> bool:
        """Insert export metric into database."""
        try:
            with self.get_connection() as db:
                if db is None:
                    return False
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT INTO export_metrics 
                    (metric_id, export_type, generation_time_ms, record_count, file_size_bytes, excel_formatting_time_ms, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metric.metric_id)
                query.addBindValue(metric.export_type)
                query.addBindValue(metric.generation_time_ms)
                query.addBindValue(metric.record_count)
                query.addBindValue(metric.file_size_bytes)
                query.addBindValue(metric.excel_formatting_time_ms)
                query.addBindValue(metric.timestamp.isoformat())
                query.addBindValue(metric.success)
                
                return query.exec()
                
        except Exception as e:
            self.logger.error(f"Failed to insert export metric: {e}")
            return False

    def insert_business_metric(self, metric: BusinessMetric) -> bool:
        """Insert business metric into database."""
        try:
            with self.get_connection() as db:
                if db is None:
                    return False
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT OR REPLACE INTO business_metrics 
                    (metric_id, module_name, event_type, event_count, success_rate, average_execution_time_ms, 
                     error_count, last_event_timestamp, daily_aggregation_date, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(metric.metric_id)
                query.addBindValue(metric.module_name)
                query.addBindValue(metric.event_type)
                query.addBindValue(metric.event_count)
                query.addBindValue(float(metric.success_rate))
                query.addBindValue(metric.average_execution_time_ms)
                query.addBindValue(metric.error_count)
                query.addBindValue(metric.timestamp.isoformat())
                query.addBindValue(metric.timestamp.date().isoformat())
                query.addBindValue(json.dumps(metric.metadata) if metric.metadata else None)
                
                return query.exec()
                
        except Exception as e:
            self.logger.error(f"Failed to insert business metric: {e}")
            return False

    def cleanup_old_metrics(self, retention_days: int = 30) -> bool:
        """Clean up metrics older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with self.get_connection() as db:
                if db is None:
                    return False
                
                # Tables with different retention policies
                cleanup_operations = [
                    ("performance_metrics", 30),
                    ("risk_metrics", 30),
                    ("simulation_metrics", 30),
                    ("journal_metrics", 30),
                    ("export_metrics", 30),
                    ("business_metrics", 90),  # Longer retention for business metrics
                    ("sla_violations", 90),
                    ("system_health", 7)  # Shorter retention for health checks
                ]
                
                for table_name, days in cleanup_operations:
                    table_cutoff = datetime.now() - timedelta(days=days)
                    
                    query = QSqlQuery(db)
                    query.prepare(f"DELETE FROM {table_name} WHERE timestamp < ?")
                    query.addBindValue(table_cutoff.isoformat())
                    
                    if query.exec():
                        rows_affected = query.numRowsAffected()
                        if rows_affected > 0:
                            self.logger.info(f"Cleaned up {rows_affected} old records from {table_name}")
                    else:
                        error = query.lastError()
                        self.logger.warning(f"Failed to cleanup {table_name}: {error.text()}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old metrics: {e}")
            return False


class PerformanceTracker:
    """
    High-level performance tracking utility with context manager support.
    
    Provides decorators and context managers for automatic performance measurement
    with integration to the metrics database.
    """
    
    def __init__(self, metrics_db: MetricsDatabase):
        """
        Initialize performance tracker.
        
        Args:
            metrics_db: MetricsDatabase instance for persistence
        """
        self.metrics_db = metrics_db
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def track_operation(self, component: str, operation: str, context: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking operation performance.
        
        Args:
            component: Component name (e.g., 'risk_calculator', 'paper_trading')
            operation: Operation name (e.g., 'position_sizing', 'trade_execution')
            context: Optional context data
        
        Yields:
            Dictionary containing start_time and other tracking data
        """
        start_time = time.perf_counter()
        tracking_data = {
            'start_time': start_time,
            'component': component,
            'operation': operation,
            'context': context or {}
        }
        
        success = True
        error_message = None
        
        try:
            yield tracking_data
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"Operation {component}.{operation} failed: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)
            
            # Create and store performance metric
            metric = PerformanceMetric(
                metric_id=str(uuid.uuid4()),
                component=component,
                operation=operation,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                context=tracking_data['context'],
                success=success,
                error_message=error_message
            )
            
            self.metrics_db.insert_performance_metric(metric)
            
            if success:
                self.logger.debug(f"{component}.{operation} completed in {duration_ms}ms")

    def track_risk_calculation(self, calculation_type: str, input_parameters: Dict[str, Any], 
                             result_quality_score: float = 1.0):
        """
        Context manager for tracking risk calculation performance.
        
        Args:
            calculation_type: Type of calculation ('atr_calculation', 'position_sizing', 'risk_allocation')
            input_parameters: Input parameters for the calculation
            result_quality_score: Quality score of the calculation result (0.0-1.0)
        """
        return self._track_specialized_metric(
            'risk_calculation',
            calculation_type,
            RiskCalculationMetric,
            {
                'calculation_type': calculation_type,
                'input_parameters': input_parameters,
                'result_quality_score': result_quality_score
            }
        )

    def track_simulation_operation(self, operation_type: str, portfolio_size_records: int = 0):
        """
        Context manager for tracking paper trading simulation performance.
        
        Args:
            operation_type: Type of operation ('portfolio_update', 'pnl_calculation', 'trade_execution')
            portfolio_size_records: Number of portfolio records processed
        """
        return self._track_specialized_metric(
            'simulation',
            operation_type,
            SimulationMetric,
            {
                'operation_type': operation_type,
                'portfolio_size_records': portfolio_size_records,
                'slippage_calculation_ms': 0  # Will be updated during execution
            }
        )

    def track_journal_operation(self, operation_type: str, entry_size_bytes: int = 0):
        """
        Context manager for tracking trading journal operation performance.
        
        Args:
            operation_type: Type of operation ('entry_save', 'search_query', 'bulk_export')
            entry_size_bytes: Size of journal entry in bytes
        """
        return self._track_specialized_metric(
            'journal',
            operation_type,
            JournalMetric,
            {
                'operation_type': operation_type,
                'entry_size_bytes': entry_size_bytes,
                'database_transaction_time_ms': 0  # Will be updated during execution
            }
        )

    def track_export_operation(self, export_type: str, record_count: int = 0):
        """
        Context manager for tracking data export operation performance.
        
        Args:
            export_type: Type of export ('scanner_results', 'trade_history', 'performance_report')
            record_count: Number of records being exported
        """
        return self._track_specialized_metric(
            'export',
            export_type,
            ExportMetric,
            {
                'export_type': export_type,
                'record_count': record_count,
                'file_size_bytes': 0,  # Will be updated during execution
                'excel_formatting_time_ms': 0  # Will be updated during execution
            }
        )

    @contextmanager
    def _track_specialized_metric(self, component: str, operation: str, metric_class, metric_data: Dict[str, Any]):
        """
        Generic context manager for specialized metric tracking.
        
        Args:
            component: Component name
            operation: Operation name
            metric_class: Metric dataclass to create
            metric_data: Initial metric data
        """
        start_time = time.perf_counter()
        tracking_data = {
            'start_time': start_time,
            'metric_data': metric_data
        }
        
        success = True
        
        try:
            yield tracking_data
        except Exception as e:
            success = False
            self.logger.error(f"Specialized metric tracking for {component}.{operation} failed: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)
            
            # Update metric data with final values
            final_metric_data = tracking_data['metric_data'].copy()
            final_metric_data.update({
                'metric_id': str(uuid.uuid4()),
                'execution_time_ms': duration_ms,
                'timestamp': datetime.now(),
                'success': success
            })
            
            # Create metric instance and store
            metric = metric_class(**final_metric_data)
            
            # Insert using appropriate method based on metric type
            if metric_class == RiskCalculationMetric:
                self.metrics_db.insert_risk_metric(metric)
            elif metric_class == SimulationMetric:
                self.metrics_db.insert_simulation_metric(metric)
            elif metric_class == JournalMetric:
                self.metrics_db.insert_journal_metric(metric)
            elif metric_class == ExportMetric:
                self.metrics_db.insert_export_metric(metric)


class SLAMonitor:
    """
    SLA monitoring and threshold validation utility.
    
    Provides real-time performance measurement against configured thresholds
    with automatic alert generation for violations.
    """
    
    def __init__(self, metrics_db: MetricsDatabase):
        """
        Initialize SLA monitor.
        
        Args:
            metrics_db: MetricsDatabase instance for storing violations
        """
        self.metrics_db = metrics_db
        self.logger = logging.getLogger(__name__)
        self.thresholds: Dict[str, SLAThreshold] = {}
        self.violation_callbacks: List[Callable] = []
        
        # Load default SLA thresholds based on technical specification
        self._load_default_thresholds()

    def _load_default_thresholds(self):
        """Load default SLA thresholds from technical specification."""
        default_thresholds = [
            # GUI Response Time Requirements
            SLAThreshold("gui", "button_click", 50, 40, SeverityLevel.WARNING),
            SLAThreshold("gui", "tab_switch", 50, 40, SeverityLevel.WARNING),
            SLAThreshold("gui", "menu_navigation", 50, 40, SeverityLevel.WARNING),
            
            # Chart Rendering Performance
            SLAThreshold("gui", "chart_rendering", 300, 250, SeverityLevel.WARNING),
            SLAThreshold("gui", "data_grid_update", 100, 80, SeverityLevel.WARNING),
            
            # Risk Calculator Performance
            SLAThreshold("risk_calculator", "atr_calculation", 500, 400, SeverityLevel.CRITICAL),
            SLAThreshold("risk_calculator", "position_sizing", 500, 400, SeverityLevel.CRITICAL),
            SLAThreshold("risk_calculator", "risk_allocation", 500, 400, SeverityLevel.CRITICAL),
            
            # Paper Trading Performance
            SLAThreshold("paper_trading", "pnl_update", 200, 150, SeverityLevel.WARNING),
            SLAThreshold("paper_trading", "trade_execution", 200, 150, SeverityLevel.WARNING),
            SLAThreshold("paper_trading", "portfolio_update", 200, 150, SeverityLevel.WARNING),
            
            # Trading Journal Performance
            SLAThreshold("trading_journal", "entry_save", 100, 75, SeverityLevel.WARNING),
            SLAThreshold("trading_journal", "search_query", 100, 75, SeverityLevel.WARNING),
            SLAThreshold("trading_journal", "bulk_export", 500, 400, SeverityLevel.WARNING),
            
            # Data Export Performance
            SLAThreshold("export", "scanner_results", 2000, 1500, SeverityLevel.WARNING),
            SLAThreshold("export", "trade_history", 2000, 1500, SeverityLevel.WARNING),
            SLAThreshold("export", "performance_report", 2000, 1500, SeverityLevel.WARNING),
            
            # Database Operations
            SLAThreshold("database", "query_execution", 100, 75, SeverityLevel.WARNING),
            SLAThreshold("database", "transaction_commit", 250, 200, SeverityLevel.WARNING),
            
            # Market Data Operations
            SLAThreshold("market_data", "single_security_fetch", 2000, 1500, SeverityLevel.WARNING),
            SLAThreshold("market_data", "market_scan", 8000, 6000, SeverityLevel.WARNING),
        ]
        
        for threshold in default_thresholds:
            key = f"{threshold.component}.{threshold.operation}"
            self.thresholds[key] = threshold

    def add_threshold(self, threshold: SLAThreshold):
        """Add or update SLA threshold."""
        key = f"{threshold.component}.{threshold.operation}"
        self.thresholds[key] = threshold
        self.logger.info(f"Added SLA threshold: {key} = {threshold.threshold_ms}ms")

    def check_performance(self, component: str, operation: str, duration_ms: int) -> Optional[Dict[str, Any]]:
        """
        Check performance against SLA thresholds.
        
        Args:
            component: Component name
            operation: Operation name
            duration_ms: Actual duration in milliseconds
        
        Returns:
            Violation details if threshold exceeded, None otherwise
        """
        key = f"{component}.{operation}"
        threshold = self.thresholds.get(key)
        
        if not threshold:
            # No threshold configured for this operation
            return None
        
        violation_data = None
        severity = None
        
        # Check critical threshold
        if duration_ms > threshold.threshold_ms:
            severity = threshold.severity
            violation_data = {
                'component': component,
                'operation': operation,
                'threshold_ms': threshold.threshold_ms,
                'actual_value_ms': duration_ms,
                'severity': severity.value,
                'violation_type': 'threshold_exceeded'
            }
        
        # Check warning threshold if configured
        elif threshold.warning_threshold_ms and duration_ms > threshold.warning_threshold_ms:
            severity = SeverityLevel.WARNING
            violation_data = {
                'component': component,
                'operation': operation,
                'threshold_ms': threshold.warning_threshold_ms,
                'actual_value_ms': duration_ms,
                'severity': severity.value,
                'violation_type': 'warning_threshold_exceeded'
            }
        
        if violation_data:
            # Store violation in database
            self._record_violation(violation_data)
            
            # Notify callbacks
            for callback in self.violation_callbacks:
                try:
                    callback(violation_data)
                except Exception as e:
                    self.logger.error(f"SLA violation callback failed: {e}")
        
        return violation_data

    def _record_violation(self, violation_data: Dict[str, Any]):
        """Record SLA violation in database."""
        try:
            with self.metrics_db.get_connection() as db:
                if db is None:
                    return
                
                query = QSqlQuery(db)
                query.prepare("""
                    INSERT INTO sla_violations 
                    (violation_id, component, operation, threshold_ms, actual_value_ms, severity, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """)
                
                query.addBindValue(str(uuid.uuid4()))
                query.addBindValue(violation_data['component'])
                query.addBindValue(violation_data['operation'])
                query.addBindValue(violation_data['threshold_ms'])
                query.addBindValue(violation_data['actual_value_ms'])
                query.addBindValue(violation_data['severity'])
                query.addBindValue(datetime.now().isoformat())
                
                if not query.exec():
                    error = query.lastError()
                    self.logger.error(f"Failed to record SLA violation: {error.text()}")
                
        except Exception as e:
            self.logger.error(f"Failed to record SLA violation: {e}")

    def add_violation_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for SLA violation notifications."""
        self.violation_callbacks.append(callback)

    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of SLA violations for the specified time period.
        
        Args:
            hours: Time period in hours to analyze
        
        Returns:
            Summary statistics of violations
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.metrics_db.get_connection() as db:
                if db is None:
                    return {}
                
                query = QSqlQuery(db)
                query.prepare("""
                    SELECT component, operation, severity, COUNT(*) as violation_count,
                           AVG(actual_value_ms - threshold_ms) as avg_excess_ms
                    FROM sla_violations 
                    WHERE timestamp > ?
                    GROUP BY component, operation, severity
                    ORDER BY violation_count DESC
                """)
                
                query.addBindValue(cutoff_time.isoformat())
                
                if not query.exec():
                    error = query.lastError()
                    self.logger.error(f"Failed to get violation summary: {error.text()}")
                    return {}
                
                violations = []
                while query.next():
                    violations.append({
                        'component': query.value(0),
                        'operation': query.value(1),
                        'severity': query.value(2),
                        'violation_count': query.value(3),
                        'avg_excess_ms': float(query.value(4)) if query.value(4) else 0.0
                    })
                
                return {
                    'period_hours': hours,
                    'total_violations': len(violations),
                    'violations_by_component': violations,
                    'summary_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get violation summary: {e}")
            return {}


class BusinessMetricsCollector:
    """
    Business metrics collection for feature utilization analytics.
    
    Tracks user interactions, feature adoption, and business KPIs across
    all application modules.
    """
    
    def __init__(self, metrics_db: MetricsDatabase):
        """
        Initialize business metrics collector.
        
        Args:
            metrics_db: MetricsDatabase instance for persistence
        """
        self.metrics_db = metrics_db
        self.logger = logging.getLogger(__name__)
        self._daily_aggregates: Dict[str, BusinessMetric] = {}
        self._lock = threading.RLock()

    def record_feature_usage(self, module_name: str, event_type: str, 
                           execution_time_ms: int = 0, success: bool = True, 
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Record feature usage event.
        
        Args:
            module_name: Module name ('risk_calculator', 'paper_trading', 'journal', 'export')
            event_type: Event type (e.g., 'calculation_request', 'trade_execution', 'entry_created')
            execution_time_ms: Execution time in milliseconds
            success: Whether the operation was successful
            metadata: Optional metadata dictionary
        """
        with self._lock:
            # Create daily aggregate key
            today = datetime.now().date()
            key = f"{module_name}.{event_type}.{today}"
            
            if key in self._daily_aggregates:
                # Update existing aggregate
                metric = self._daily_aggregates[key]
                metric.event_count += 1
                metric.error_count += 0 if success else 1
                
                # Update running averages
                total_events = metric.event_count
                new_success_rate = (metric.success_rate * (total_events - 1) + (1 if success else 0)) / total_events
                new_avg_time = (metric.average_execution_time_ms * (total_events - 1) + execution_time_ms) / total_events
                
                metric.success_rate = new_success_rate
                metric.average_execution_time_ms = int(new_avg_time)
                metric.timestamp = datetime.now()
                
                # Update metadata
                if metadata:
                    if metric.metadata is None:
                        metric.metadata = {}
                    metric.metadata.update(metadata)
            else:
                # Create new aggregate
                metric = BusinessMetric(
                    metric_id=f"{module_name}_{event_type}_{today.isoformat()}",
                    module_name=module_name,
                    event_type=event_type,
                    event_count=1,
                    success_rate=1.0 if success else 0.0,
                    average_execution_time_ms=execution_time_ms,
                    error_count=0 if success else 1,
                    timestamp=datetime.now(),
                    metadata=metadata
                )
                self._daily_aggregates[key] = metric
            
            # Persist to database
            self.metrics_db.insert_business_metric(self._daily_aggregates[key])

    def record_risk_calculator_usage(self, calculation_type: str, execution_time_ms: int, 
                                   success: bool = True, parameters: Optional[Dict[str, Any]] = None):
        """Record risk calculator usage metrics."""
        self.record_feature_usage(
            module_name="risk_calculator",
            event_type=calculation_type,
            execution_time_ms=execution_time_ms,
            success=success,
            metadata={"parameters": parameters} if parameters else None
        )

    def record_trading_simulation_usage(self, operation_type: str, execution_time_ms: int,
                                      success: bool = True, trade_count: int = 0):
        """Record paper trading simulation usage metrics."""
        self.record_feature_usage(
            module_name="paper_trading",
            event_type=operation_type,
            execution_time_ms=execution_time_ms,
            success=success,
            metadata={"trade_count": trade_count}
        )

    def record_journal_usage(self, operation_type: str, execution_time_ms: int,
                           success: bool = True, entry_size: int = 0):
        """Record trading journal usage metrics."""
        self.record_feature_usage(
            module_name="journal",
            event_type=operation_type,
            execution_time_ms=execution_time_ms,
            success=success,
            metadata={"entry_size_bytes": entry_size}
        )

    def record_export_usage(self, export_type: str, execution_time_ms: int,
                          success: bool = True, record_count: int = 0, file_size: int = 0):
        """Record data export usage metrics."""
        self.record_feature_usage(
            module_name="export",
            event_type=export_type,
            execution_time_ms=execution_time_ms,
            success=success,
            metadata={"record_count": record_count, "file_size_bytes": file_size}
        )

    def get_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive usage analytics for the specified period.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Usage analytics summary
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.metrics_db.get_connection() as db:
                if db is None:
                    return {}
                
                # Get module usage summary
                query = QSqlQuery(db)
                query.prepare("""
                    SELECT module_name, event_type, 
                           SUM(event_count) as total_events,
                           AVG(success_rate) as avg_success_rate,
                           AVG(average_execution_time_ms) as avg_execution_time,
                           SUM(error_count) as total_errors
                    FROM business_metrics 
                    WHERE last_event_timestamp > ?
                    GROUP BY module_name, event_type
                    ORDER BY total_events DESC
                """)
                
                query.addBindValue(cutoff_date.isoformat())
                
                if not query.exec():
                    error = query.lastError()
                    self.logger.error(f"Failed to get usage analytics: {error.text()}")
                    return {}
                
                usage_data = []
                while query.next():
                    usage_data.append({
                        'module_name': query.value(0),
                        'event_type': query.value(1),
                        'total_events': query.value(2),
                        'avg_success_rate': float(query.value(3)),
                        'avg_execution_time_ms': float(query.value(4)),
                        'total_errors': query.value(5)
                    })
                
                # Calculate summary statistics
                total_events = sum(item['total_events'] for item in usage_data)
                avg_success_rate = sum(item['avg_success_rate'] * item['total_events'] for item in usage_data) / total_events if total_events > 0 else 0
                
                return {
                    'period_days': days,
                    'total_events': total_events,
                    'overall_success_rate': avg_success_rate,
                    'usage_by_module': usage_data,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get usage analytics: {e}")
            return {}


class MetricsManager:
    """
    Main metrics management class that orchestrates all metrics collection.
    
    Provides a unified interface for performance tracking, SLA monitoring,
    and business metrics collection with automated cleanup and reporting.
    """
    
    def __init__(self, database_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize metrics manager.
        
        Args:
            database_path: Optional custom database path
            config_path: Optional configuration file path
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.database = MetricsDatabase(database_path)
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(self.database)
        self.sla_monitor = SLAMonitor(self.database)
        self.business_metrics = BusinessMetricsCollector(self.database)
        
        # Setup cleanup timer
        self._setup_cleanup_timer()
        
        # Load configuration if provided
        if config_path:
            self._load_configuration(config_path)
        
        self.logger.info("MetricsManager initialized successfully")

    def _setup_cleanup_timer(self):
        """Setup automated cleanup timer."""
        if QTimer is not None:
            self.cleanup_timer = QTimer()
            self.cleanup_timer.timeout.connect(self._perform_cleanup)
            # Run cleanup every 24 hours
            self.cleanup_timer.start(24 * 60 * 60 * 1000)

    def _perform_cleanup(self):
        """Perform automated metrics cleanup."""
        try:
            self.database.cleanup_old_metrics()
            self.logger.info("Automated metrics cleanup completed")
        except Exception as e:
            self.logger.error(f"Automated cleanup failed: {e}")

    def _load_configuration(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load SLA thresholds if configured
            if 'sla_thresholds' in config:
                for threshold_config in config['sla_thresholds']:
                    threshold = SLAThreshold(
                        component=threshold_config['component'],
                        operation=threshold_config['operation'],
                        threshold_ms=threshold_config['threshold_ms'],
                        warning_threshold_ms=threshold_config.get('warning_threshold_ms'),
                        severity=SeverityLevel(threshold_config.get('severity', 'warning'))
                    )
                    self.sla_monitor.add_threshold(threshold)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load configuration from {config_path}: {e}")

    def track_performance(self, component: str, operation: str, context: Optional[Dict[str, Any]] = None):
        """Get performance tracking context manager."""
        return self.performance_tracker.track_operation(component, operation, context)

    def track_risk_calculation(self, calculation_type: str, input_parameters: Dict[str, Any],
                             result_quality_score: float = 1.0):
        """Get risk calculation tracking context manager."""
        return self.performance_tracker.track_risk_calculation(
            calculation_type, input_parameters, result_quality_score
        )

    def track_simulation(self, operation_type: str, portfolio_size_records: int = 0):
        """Get simulation tracking context manager."""
        return self.performance_tracker.track_simulation_operation(operation_type, portfolio_size_records)

    def track_journal(self, operation_type: str, entry_size_bytes: int = 0):
        """Get journal tracking context manager."""
        return self.performance_tracker.track_journal_operation(operation_type, entry_size_bytes)

    def track_export(self, export_type: str, record_count: int = 0):
        """Get export tracking context manager."""
        return self.performance_tracker.track_export_operation(export_type, record_count)

    def check_sla(self, component: str, operation: str, duration_ms: int) -> Optional[Dict[str, Any]]:
        """Check performance against SLA thresholds."""
        return self.sla_monitor.check_performance(component, operation, duration_ms)

    def record_usage(self, module_name: str, event_type: str, execution_time_ms: int = 0,
                    success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Record business metric usage."""
        self.business_metrics.record_feature_usage(
            module_name, event_type, execution_time_ms, success, metadata
        )

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        sla_summary = self.sla_monitor.get_violation_summary(hours)
        usage_analytics = self.business_metrics.get_usage_analytics(hours // 24 if hours >= 24 else 1)
        
        return {
            'period_hours': hours,
            'sla_violations': sla_summary,
            'usage_analytics': usage_analytics,
            'generated_at': datetime.now().isoformat()
        }

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for SLA violation alerts."""
        self.sla_monitor.add_violation_callback(callback)

    def shutdown(self):
        """Shutdown metrics manager and cleanup resources."""
        try:
            # Perform final cleanup
            self.database.cleanup_old_metrics()
            
            # Stop cleanup timer
            if hasattr(self, 'cleanup_timer') and self.cleanup_timer:
                self.cleanup_timer.stop()
            
            self.logger.info("MetricsManager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during metrics manager shutdown: {e}")


# Convenience functions for easy integration
def create_metrics_manager(config_path: Optional[str] = None) -> MetricsManager:
    """
    Create and configure a MetricsManager instance.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Configured MetricsManager instance
    """
    return MetricsManager(config_path=config_path)


def performance_metric(component: str, operation: str):
    """
    Decorator for automatic performance measurement.
    
    Args:
        component: Component name
        operation: Operation name
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to find metrics manager in args/kwargs
            metrics_manager = None
            for arg in args:
                if isinstance(arg, MetricsManager):
                    metrics_manager = arg
                    break
            
            if metrics_manager is None:
                # Create temporary metrics manager
                metrics_manager = create_metrics_manager()
            
            with metrics_manager.track_performance(component, operation):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    manager = create_metrics_manager()
    
    # Example: Track a risk calculation
    with manager.track_risk_calculation('position_sizing', {'symbol': 'AAPL', 'account_balance': 10000}, 0.95) as tracking:
        # Simulate calculation work
        time.sleep(0.1)
        tracking['metric_data']['result_quality_score'] = 0.95
    
    # Example: Track journal operation
    with manager.track_journal('entry_save', 1024) as tracking:
        # Simulate journal save
        time.sleep(0.05)
        tracking['metric_data']['database_transaction_time_ms'] = 25
    
    # Record business usage
    manager.record_usage('risk_calculator', 'position_sizing_request', 85, True, {'symbol': 'AAPL'})
    
    # Get performance summary
    summary = manager.get_performance_summary(1)
    print("Performance Summary:", json.dumps(summary, indent=2))
    
    # Cleanup
    manager.shutdown()