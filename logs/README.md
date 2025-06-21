# Logging System Documentation

## Overview

This document provides comprehensive documentation for the PyQt6 desktop trading application's logging system implementation. The logging architecture is designed to support enterprise-grade observability, performance monitoring, and troubleshooting across all application components including the Risk Management Calculator, Paper Trading Simulator, Trading Journal, and Data Export utilities.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Log Categories and Levels](#log-categories-and-levels)
- [Storage Structure](#storage-structure)
- [Retention Policies](#retention-policies)
- [Correlation ID System](#correlation-id-system)
- [Configuration Management](#configuration-management)
- [Integration Patterns](#integration-patterns)
- [Windows Notification Integration](#windows-notification-integration)
- [Development Guidelines](#development-guidelines)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The logging system implements a **local-first monitoring approach** designed for single-user desktop deployment with structured JSON logging for machine parsing and human-readable fallback options.

### Core Components

```
Application Process
├── Performance Monitor → Metrics Collector → Local Metrics Database
├── Error Handler → Structured Logger → Log File System
├── Health Check Manager → Service Status Tracker → Database
├── Background Service Monitor → Thread Status Monitor → Database
├── Risk Calculator Monitor → Risk Calculation Metrics → Database
├── Paper Trading Monitor → Trading Simulation Metrics → Database
├── Trading Journal Monitor → Journal Performance Metrics → Database
└── Data Export Monitor → Export Generation Metrics → Database
```

### Storage Locations

- **Primary Log Directory**: `%APPDATA%/TradingApp/logs/`
- **Metrics Database**: `%APPDATA%/TradingApp/metrics.db`
- **Configuration**: `logs/logging_config.json`
- **Log Templates**: `logs/log_templates.json`

## Log Categories and Levels

### Standard Categories

| Category | Levels | Retention | Purpose |
|----------|--------|-----------|---------|
| **Application Operations** | INFO, DEBUG | 90 days | Feature usage, workflow tracking |
| **Error Events** | ERROR, CRITICAL | 180 days | Debugging, pattern analysis |
| **Performance Metrics** | DEBUG | 30 days | Performance optimization |
| **Security Events** | WARNING, ERROR | 180 days | Authentication, API access |

### Enhanced Module-Specific Categories

| Category | Levels | Retention | Purpose |
|----------|--------|-----------|---------|
| **Data Provider Failover** | INFO, WARNING | 180 days | Provider switching, failover triggers, recovery events |
| **Circuit Breaker Events** | WARNING, ERROR | 180 days | Circuit state changes, breaker activation, recovery timing |
| **Risk Calculator Operations** | INFO, DEBUG, ERROR | 90 days | ATR calculations, position sizing, risk allocation decisions |
| **Paper Trading Operations** | INFO, DEBUG, ERROR | 90 days | Virtual trade execution, portfolio updates, P/L calculations |
| **Trading Journal Operations** | INFO, DEBUG, ERROR | 90 days | Entry persistence, search operations, data integrity |
| **Data Export Operations** | INFO, DEBUG, ERROR | 90 days | Excel generation, formatting operations, file system writes |

### Log Level Guidelines

- **DEBUG**: Detailed diagnostic information, typically only enabled during development
- **INFO**: General information about application operation and user actions
- **WARNING**: Indicates potential issues that don't prevent operation
- **ERROR**: Error conditions that may impact functionality
- **CRITICAL**: Serious errors that may cause application failure

## Storage Structure

### Directory Layout

```
%APPDATA%/TradingApp/logs/
├── application/
│   ├── trading_app_YYYY-MM-DD.log
│   ├── trading_app_YYYY-MM-DD.log.gz
│   └── ...
├── errors/
│   ├── errors_YYYY-MM-DD.log
│   ├── errors_YYYY-MM-DD.log.gz
│   └── ...
├── performance/
│   ├── performance_YYYY-MM-DD.log
│   ├── performance_YYYY-MM-DD.log.gz
│   └── ...
├── security/
│   ├── security_YYYY-MM-DD.log
│   ├── security_YYYY-MM-DD.log.gz
│   └── ...
├── modules/
│   ├── risk_calculator/
│   │   ├── risk_calc_YYYY-MM-DD.log
│   │   └── risk_calc_YYYY-MM-DD.log.gz
│   ├── paper_trading/
│   │   ├── paper_trade_YYYY-MM-DD.log
│   │   └── paper_trade_YYYY-MM-DD.log.gz
│   ├── trading_journal/
│   │   ├── journal_YYYY-MM-DD.log
│   │   └── journal_YYYY-MM-DD.log.gz
│   ├── data_export/
│   │   ├── export_YYYY-MM-DD.log
│   │   └── export_YYYY-MM-DD.log.gz
│   └── failover/
│       ├── failover_YYYY-MM-DD.log
│       └── failover_YYYY-MM-DD.log.gz
└── archived/
    ├── YYYY/MM/
    └── ...
```

### Log File Format

Logs use structured JSON format for machine parsing with human-readable text fallback:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "correlation_id": "risk_calc_20240115_103045_def456",
  "component": "risk_calculator",
  "level": "INFO",
  "message": "Position sizing calculation completed",
  "context": {
    "module": "risk_calculator",
    "operation": "position_sizing",
    "symbol": "AAPL",
    "account_risk_percent": 2.0,
    "atr_value": 1.23,
    "position_size": 150,
    "execution_time_ms": 45
  },
  "thread_id": "MainThread",
  "user_action": "position_size_request"
}
```

## Retention Policies

### Automated Cleanup Schedule

| Category | Retention Period | Cleanup Frequency | Compression |
|----------|------------------|-------------------|-------------|
| **Operations** | 90 days | Weekly | Daily after 7 days |
| **Errors** | 180 days | Monthly | Daily after 7 days |
| **Performance** | 30 days | Weekly | Daily after 3 days |
| **Security** | 180 days | Monthly | Daily after 7 days |
| **Module Logs** | 90 days | Weekly | Daily after 7 days |
| **Failover Events** | 180 days | Monthly | Daily after 7 days |

### Storage Management

- **Maximum Disk Usage**: 2GB total for logs directory
- **Emergency Cleanup**: Triggered when usage exceeds 1.8GB
- **Archive Storage**: Compressed logs moved to `archived/` directory
- **Metrics Database**: Separate 30-day retention for performance metrics

## Correlation ID System

### Enhanced Correlation ID Patterns

The correlation ID system enables end-to-end request tracing across asynchronous QThread operations:

| Module | Pattern | Example |
|--------|---------|---------|
| **User Actions** | `usr_001_YYYYMMDD_HHMMSS_xxxxx` | `usr_001_20240115_103045_abc123` |
| **Risk Calculator** | `risk_calc_YYYYMMDD_HHMMSS_xxxxx` | `risk_calc_20240115_103045_def456` |
| **Paper Trading** | `paper_trade_YYYYMMDD_HHMMSS_xxxxx` | `paper_trade_20240115_103045_ghi789` |
| **Trading Journal** | `journal_op_YYYYMMDD_HHMMSS_xxxxx` | `journal_op_20240115_103045_jkl012` |
| **Data Export** | `export_gen_YYYYMMDD_HHMMSS_xxxxx` | `export_gen_20240115_103045_mno345` |
| **Provider Failover** | `failover_YYYYMMDD_HHMMSS_xxxxx` | `failover_20240115_103045_pqr678` |
| **Circuit Breaker** | `circuit_br_YYYYMMDD_HHMMSS_xxxxx` | `circuit_br_20240115_103045_stu901` |

### Correlation ID Usage

1. **Generation**: Correlation IDs are generated at the start of each user operation
2. **Propagation**: IDs are passed through all related operations and QThread calls
3. **Logging**: All log entries for a request chain use the same correlation ID
4. **Tracing**: Use correlation IDs to trace complete request flows across modules

### Cross-Thread Tracing

For QThread-based background operations:

```python
# Example correlation ID propagation
def start_risk_calculation(self, symbol, correlation_id=None):
    if not correlation_id:
        correlation_id = f"risk_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generate_random_id()}"
    
    self.logger.info(
        "Starting risk calculation",
        extra={
            "correlation_id": correlation_id,
            "symbol": symbol,
            "component": "risk_calculator"
        }
    )
    
    # Pass correlation_id to worker thread
    worker = RiskCalculationWorker(symbol, correlation_id)
    self.thread_pool.start(worker)
```

## Configuration Management

### Logging Configuration File

The `logs/logging_config.json` file contains comprehensive logging configuration:

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "json": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      "class": "logging_config.JSONFormatter"
    },
    "standard": {
      "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    }
  },
  "handlers": {
    "application_file": {
      "class": "logging.handlers.TimedRotatingFileHandler",
      "filename": "%APPDATA%/TradingApp/logs/application/trading_app.log",
      "when": "midnight",
      "interval": 1,
      "backupCount": 90,
      "formatter": "json"
    },
    "error_file": {
      "class": "logging.handlers.TimedRotatingFileHandler",
      "filename": "%APPDATA%/TradingApp/logs/errors/errors.log",
      "when": "midnight",
      "interval": 1,
      "backupCount": 180,
      "formatter": "json",
      "level": "ERROR"
    },
    "risk_calculator_file": {
      "class": "logging.handlers.TimedRotatingFileHandler",
      "filename": "%APPDATA%/TradingApp/logs/modules/risk_calculator/risk_calc.log",
      "when": "midnight",
      "interval": 1,
      "backupCount": 90,
      "formatter": "json"
    }
  },
  "loggers": {
    "trading_app": {
      "level": "INFO",
      "handlers": ["application_file", "error_file"],
      "propagate": false
    },
    "trading_app.risk_calculator": {
      "level": "DEBUG",
      "handlers": ["risk_calculator_file"],
      "propagate": true
    },
    "trading_app.paper_trading": {
      "level": "DEBUG",
      "handlers": ["paper_trading_file"],
      "propagate": true
    }
  }
}
```

### Dynamic Configuration

- **Runtime Adjustment**: Log levels can be adjusted during runtime through the settings interface
- **Module-Specific Settings**: Each module can have independent logging configuration
- **Performance Mode**: Reduced logging during high-frequency operations
- **Debug Mode**: Enhanced logging for troubleshooting

## Integration Patterns

### PyQt6 Integration

```python
import logging
from PyQt6.QtCore import QObject, pyqtSignal

class LoggingMixin(QObject):
    log_signal = pyqtSignal(str, str, dict)  # level, message, context
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"trading_app.{self.__class__.__name__}")
        
    def log_info(self, message, correlation_id=None, **context):
        extra_context = {"correlation_id": correlation_id, **context}
        self.logger.info(message, extra=extra_context)
        self.log_signal.emit("INFO", message, extra_context)
```

### QThread Logging

```python
class WorkerThread(QThread):
    def __init__(self, correlation_id):
        super().__init__()
        self.correlation_id = correlation_id
        self.logger = logging.getLogger("trading_app.worker")
        
    def run(self):
        self.logger.info(
            "Worker thread started",
            extra={"correlation_id": self.correlation_id, "thread_id": self.currentThreadId()}
        )
        # Work implementation...
```

### Module-Specific Logging Examples

#### Risk Calculator Logging

```python
def calculate_position_size(self, symbol, account_balance, risk_percent, atr):
    correlation_id = f"risk_calc_{int(time.time())}"
    start_time = time.time()
    
    try:
        self.logger.info(
            "Starting position sizing calculation",
            extra={
                "correlation_id": correlation_id,
                "symbol": symbol,
                "account_balance": account_balance,
                "risk_percent": risk_percent,
                "atr": atr
            }
        )
        
        # Calculation logic...
        position_size = self._calculate_size(account_balance, risk_percent, atr)
        
        execution_time = (time.time() - start_time) * 1000
        
        self.logger.info(
            "Position sizing calculation completed",
            extra={
                "correlation_id": correlation_id,
                "symbol": symbol,
                "position_size": position_size,
                "execution_time_ms": execution_time
            }
        )
        
        return position_size
        
    except Exception as e:
        self.logger.error(
            "Position sizing calculation failed",
            extra={
                "correlation_id": correlation_id,
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise
```

#### Paper Trading Logging

```python
def execute_virtual_trade(self, order):
    correlation_id = f"paper_trade_{int(time.time())}"
    
    self.logger.info(
        "Executing virtual trade",
        extra={
            "correlation_id": correlation_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type
        }
    )
    
    # Trade execution with performance tracking...
```

## Windows Notification Integration

### pywin32 Integration

The logging system integrates with Windows desktop notifications through pywin32 for critical alerts and system status updates.

#### Notification Categories

| Log Level | Notification Type | Windows API | Display Duration |
|-----------|------------------|-------------|------------------|
| **CRITICAL** | Modal Dialog + System Tray | `MessageBox` + `Shell_NotifyIcon` | Until acknowledged |
| **ERROR** | Toast Notification | `ToastNotificationManager` | 10 seconds |
| **WARNING** | System Tray | `Shell_NotifyIcon` | 5 seconds |
| **INFO** | Status Bar Update | Internal PyQt6 | 3 seconds |

#### Implementation Example

```python
import win32gui
import win32con
from win32api import GetModuleHandle
from win32gui import Shell_NotifyIcon, NIF_ICON, NIF_MESSAGE, NIF_TIP

class WindowsNotificationHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setup_system_tray()
        
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.show_critical_notification(record)
        elif record.levelno >= logging.WARNING:
            self.show_warning_notification(record)
            
    def show_critical_notification(self, record):
        # System tray notification for critical errors
        Shell_NotifyIcon(
            NIM_MODIFY,
            (self.hwnd, 0, NIF_ICON | NIF_MESSAGE | NIF_TIP,
             win32con.WM_USER + 20, self.hicon, record.getMessage())
        )
        
        # Log the notification event
        notification_logger = logging.getLogger("trading_app.notifications")
        notification_logger.info(
            "Windows notification sent",
            extra={
                "notification_type": "critical",
                "message": record.getMessage(),
                "correlation_id": getattr(record, 'correlation_id', None)
            }
        )
```

#### Notification Logging

All Windows notifications are logged for audit and debugging:

```json
{
  "timestamp": "2024-01-15T10:35:22.456Z",
  "correlation_id": "notification_20240115_103522_xyz789",
  "component": "windows_notifications",
  "level": "INFO",
  "message": "Desktop notification sent",
  "context": {
    "notification_type": "error",
    "original_log_level": "ERROR",
    "notification_method": "toast",
    "display_duration_ms": 10000,
    "user_acknowledged": false
  }
}
```

### Error Handling Integration

Windows notification failures are handled gracefully:

```python
def safe_notify(self, message, level="INFO"):
    try:
        self.send_windows_notification(message, level)
    except Exception as e:
        # Log notification failure without causing application issues
        self.logger.warning(
            "Windows notification failed",
            extra={
                "error": str(e),
                "message": message,
                "level": level,
                "fallback": "status_bar_only"
            }
        )
        # Fallback to status bar notification
        self.update_status_bar(message)
```

## Development Guidelines

### Logging Best Practices

1. **Use Correlation IDs**: Always include correlation IDs for request tracing
2. **Structured Context**: Include relevant context data in log entries
3. **Performance Logging**: Log execution times for operations >50ms
4. **Error Context**: Include full error context and stack traces
5. **Module Identification**: Use module-specific loggers and correlation ID patterns

### Performance Considerations

- **Async Logging**: Use async handlers for high-frequency logging
- **Log Level Management**: Use appropriate log levels to avoid verbose output
- **Context Filtering**: Filter sensitive data from log context
- **Buffer Management**: Configure appropriate buffer sizes for file handlers

### Security Guidelines

- **Sensitive Data**: Never log passwords, API keys, or personal information
- **Data Sanitization**: Sanitize user inputs before logging
- **Access Control**: Ensure log files have appropriate permissions
- **Audit Trail**: Maintain audit logs for security-relevant events

## Troubleshooting

### Common Issues

#### Log Files Not Created

1. Check directory permissions for `%APPDATA%/TradingApp/logs/`
2. Verify logging configuration file syntax
3. Ensure application has write access to log directory

#### Missing Log Entries

1. Verify log level configuration
2. Check correlation ID generation
3. Confirm logger hierarchy configuration

#### Performance Impact

1. Review log level settings (reduce DEBUG logging in production)
2. Check log file rotation configuration
3. Monitor disk space usage

#### Correlation ID Tracing

1. Ensure correlation IDs are properly propagated across threads
2. Verify correlation ID format consistency
3. Check for missing correlation ID in log entries

### Diagnostic Commands

```bash
# Check log directory structure
dir "%APPDATA%\TradingApp\logs" /s

# Monitor log file growth
dir "%APPDATA%\TradingApp\logs\application" /o:d

# Search logs by correlation ID
findstr "risk_calc_20240115" "%APPDATA%\TradingApp\logs\**\*.log"

# Check recent errors
findstr "ERROR\|CRITICAL" "%APPDATA%\TradingApp\logs\errors\*.log" | tail -20
```

### Log Analysis Tools

- **JSON Log Parser**: Use `jq` or Python scripts for JSON log analysis
- **Correlation Tracing**: Custom scripts for end-to-end request tracing
- **Performance Analysis**: Automated scripts for execution time analysis
- **Error Pattern Detection**: Scripts for identifying recurring error patterns

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review log retention and cleanup effectiveness
2. **Monthly**: Analyze error patterns and performance trends
3. **Quarterly**: Review and update logging configuration
4. **As Needed**: Adjust log levels based on operational requirements

### Development Team Resources

- **Log Configuration**: `logs/logging_config.json`
- **Log Templates**: `logs/log_templates.json`
- **Example Implementations**: See module-specific logging examples above
- **Performance Monitoring**: Integration with metrics database for performance analysis

For additional support or questions about the logging system, refer to the technical specification Section 6.5 (Monitoring and Observability) or contact the development team.