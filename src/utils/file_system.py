"""
File System Management Utilities

Provides professional-grade file system management for the TradingApp desktop application,
including directory structure creation, cache management with TTL cleanup, backup operations,
log rotation, and storage monitoring for optimal application data organization.

This module implements the monitoring and observability requirements from section 6.5.2.1,
providing LocalCacheManager TTL-based cleanup operations, storage utilization monitoring,
and comprehensive directory structure management for %APPDATA%/TradingApp.

Key Features:
- Professional application directory structure management
- Cache management with TTL-based cleanup and storage monitoring
- Database backup operations with integrity verification
- Log file rotation with compression and automated cleanup
- Directory permission validation and access verification
- Storage space monitoring with automated cleanup triggers
"""

import os
import json
import shutil
import sqlite3
import gzip
import hashlib
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import time
import stat


class DirectoryType(Enum):
    """Application directory types with specific management policies."""
    DATABASE = "database"
    CACHE = "cache"
    EXPORTS = "exports"
    LOGS = "logs"
    CONFIG = "config"
    ASSETS = "assets"
    METADATA = "metadata"
    MARKET_DATA = "market_data"
    CHARTS = "charts"
    EXCEL_REPORTS = "excel_reports"
    UI_RESOURCES = "ui_resources"
    CHART_TEMPLATES = "chart_templates"
    NOTIFICATION_ICONS = "notification_icons"


class CleanupResult(Enum):
    """Results of cleanup operations for monitoring and alerting."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    NO_ACTION_NEEDED = "no_action_needed"
    PERMISSION_DENIED = "permission_denied"
    DISK_SPACE_CRITICAL = "disk_space_critical"


@dataclass
class CacheEntry:
    """Represents a cache entry with TTL and metadata information."""
    file_path: Path
    created_time: datetime
    last_accessed: datetime
    ttl_hours: int
    file_size: int
    cache_type: str
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has exceeded its TTL."""
        expiry_time = self.created_time + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    @property
    def age_hours(self) -> float:
        """Calculate age of cache entry in hours."""
        return (datetime.now() - self.created_time).total_seconds() / 3600


@dataclass
class StorageMetrics:
    """Storage utilization metrics for monitoring and alerting."""
    total_space: int
    available_space: int
    used_space: int
    cache_size: int
    logs_size: int
    database_size: int
    exports_size: int
    utilization_percent: float
    last_cleanup: Optional[datetime] = None
    
    @property
    def is_space_critical(self) -> bool:
        """Check if available space is critically low (<500MB)."""
        return self.available_space < 500 * 1024 * 1024  # 500MB
    
    @property
    def is_cache_oversized(self) -> bool:
        """Check if cache directory exceeds 500MB threshold."""
        return self.cache_size > 500 * 1024 * 1024  # 500MB


@dataclass
class BackupMetadata:
    """Metadata for backup operations and integrity verification."""
    backup_path: Path
    source_path: Path
    backup_time: datetime
    source_size: int
    backup_size: int
    checksum: str
    backup_type: str
    retention_days: int
    
    @property
    def is_expired(self) -> bool:
        """Check if backup has exceeded its retention period."""
        expiry_date = self.backup_time + timedelta(days=self.retention_days)
        return datetime.now() > expiry_date


class FileSystemManager:
    """
    Comprehensive file system management for TradingApp desktop application.
    
    Implements professional directory structure creation, cache management with TTL-based
    cleanup, backup operations with integrity verification, log rotation, and storage
    monitoring as specified in the technical requirements.
    """
    
    def __init__(self, app_data_root: Optional[Path] = None):
        """
        Initialize file system manager with application data directory.
        
        Args:
            app_data_root: Custom application data root path. Defaults to %APPDATA%/TradingApp
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine application data directory
        if app_data_root:
            self.app_data_root = Path(app_data_root)
        else:
            # Use %APPDATA%/TradingApp on Windows, ~/.tradingapp on other platforms
            if os.name == 'nt':
                self.app_data_root = Path(os.environ['APPDATA']) / 'TradingApp'
            else:
                self.app_data_root = Path.home() / '.tradingapp'
        
        # Directory structure configuration
        self.directories = {
            DirectoryType.DATABASE: self.app_data_root / "database",
            DirectoryType.CACHE: self.app_data_root / "cache",
            DirectoryType.EXPORTS: self.app_data_root / "exports",
            DirectoryType.LOGS: self.app_data_root / "logs",
            DirectoryType.CONFIG: self.app_data_root / "config",
            DirectoryType.ASSETS: self.app_data_root / "assets",
            DirectoryType.METADATA: self.app_data_root / "cache" / "metadata",
            DirectoryType.MARKET_DATA: self.app_data_root / "cache" / "market_data",
            DirectoryType.CHARTS: self.app_data_root / "cache" / "charts",
            DirectoryType.EXCEL_REPORTS: self.app_data_root / "exports" / "excel_reports",
            DirectoryType.UI_RESOURCES: self.app_data_root / "assets" / "ui_resources",
            DirectoryType.CHART_TEMPLATES: self.app_data_root / "assets" / "chart_templates",
            DirectoryType.NOTIFICATION_ICONS: self.app_data_root / "assets" / "notification_icons",
        }
        
        # Cache TTL configuration (hours)
        self.cache_ttl_config = {
            "market_data": 24,      # 24-hour rolling cache
            "metadata": 168,        # 7 days for security metadata
            "charts": 72,           # 3 days for generated charts
            "api_responses": 1,     # 1 hour for API responses
            "quotes": 0.5,          # 30 minutes for real-time quotes
            "fundamentals": 168,    # 7 days for company fundamentals
            "news": 1,              # 1 hour for market news
        }
        
        # Storage thresholds and limits
        self.storage_config = {
            "max_cache_size_mb": 500,
            "max_logs_size_mb": 100,
            "min_free_space_mb": 500,
            "cleanup_threshold_percent": 85,
            "backup_retention_days": 30,
            "log_retention_days": 90,
        }
        
        # File extensions for different content types
        self.file_extensions = {
            "cache": [".json", ".pkl", ".cache", ".tmp"],
            "logs": [".log", ".txt"],
            "database": [".db", ".sqlite", ".sqlite3"],
            "exports": [".xlsx", ".csv", ".pdf"],
            "config": [".json", ".conf", ".ini"],
            "assets": [".png", ".jpg", ".ico", ".qrc"],
        }
    
    def initialize_directory_structure(self) -> bool:
        """
        Create the complete application directory structure with proper permissions.
        
        Creates all required directories for the TradingApp with appropriate
        Windows file system permissions and validates directory accessibility.
        
        Returns:
            bool: True if all directories created successfully, False otherwise
        """
        try:
            self.logger.info(f"Initializing directory structure at {self.app_data_root}")
            
            # Create root application directory
            self.app_data_root.mkdir(parents=True, exist_ok=True)
            
            # Create all required subdirectories
            for dir_type, dir_path in self.directories.items():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
                    # Verify directory is writable
                    test_file = dir_path / ".write_test"
                    test_file.touch()
                    test_file.unlink()
                    
                    self.logger.debug(f"Created and verified directory: {dir_path}")
                    
                except (OSError, PermissionError) as e:
                    self.logger.error(f"Failed to create/verify directory {dir_path}: {e}")
                    return False
            
            # Create initial configuration files if they don't exist
            self._create_initial_config_files()
            
            # Set up initial cache structure
            self._initialize_cache_structure()
            
            self.logger.info("Directory structure initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize directory structure: {e}")
            return False
    
    def _create_initial_config_files(self) -> None:
        """Create initial configuration files with default settings."""
        try:
            config_dir = self.directories[DirectoryType.CONFIG]
            
            # Create default_settings.json if it doesn't exist
            default_settings_path = config_dir / "default_settings.json"
            if not default_settings_path.exists():
                default_settings = {
                    "data_retention": {
                        "market_data_cache": 30,
                        "trade_history": "permanent",
                        "logs": 90,
                        "metadata_cache": 7,
                        "performance_metrics": 30
                    },
                    "cache_settings": {
                        "max_cache_size_mb": 500,
                        "cleanup_threshold_percent": 85,
                        "auto_cleanup_enabled": True,
                        "cleanup_interval_hours": 6
                    },
                    "backup_settings": {
                        "auto_backup_enabled": True,
                        "backup_interval_hours": 24,
                        "retention_days": 30,
                        "compression_enabled": True
                    },
                    "monitoring": {
                        "performance_tracking": True,
                        "storage_monitoring": True,
                        "alert_thresholds": {
                            "disk_space_warning_mb": 1000,
                            "disk_space_critical_mb": 500
                        }
                    }
                }
                
                with open(default_settings_path, 'w') as f:
                    json.dump(default_settings, f, indent=2)
                
                self.logger.debug(f"Created default settings file: {default_settings_path}")
            
            # Create api_keys.json template if it doesn't exist
            api_keys_path = config_dir / "api_keys.json"
            if not api_keys_path.exists():
                api_keys_template = {
                    "yahoo_finance": {
                        "enabled": True,
                        "rate_limit_per_hour": 2000
                    },
                    "alpha_vantage": {
                        "api_key": "",
                        "enabled": False,
                        "rate_limit_per_day": 25
                    },
                    "iex_cloud": {
                        "api_key": "",
                        "enabled": False,
                        "rate_limit_per_month": 500000
                    }
                }
                
                with open(api_keys_path, 'w') as f:
                    json.dump(api_keys_template, f, indent=2)
                
                self.logger.debug(f"Created API keys template: {api_keys_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to create initial config files: {e}")
    
    def _initialize_cache_structure(self) -> None:
        """Initialize cache directory structure with metadata files."""
        try:
            cache_dir = self.directories[DirectoryType.CACHE]
            
            # Create cache metadata file
            cache_metadata_path = cache_dir / ".cache_metadata.json"
            if not cache_metadata_path.exists():
                cache_metadata = {
                    "created": datetime.now().isoformat(),
                    "last_cleanup": None,
                    "total_entries": 0,
                    "cache_types": list(self.cache_ttl_config.keys()),
                    "size_mb": 0
                }
                
                with open(cache_metadata_path, 'w') as f:
                    json.dump(cache_metadata, f, indent=2)
                
                self.logger.debug(f"Created cache metadata file: {cache_metadata_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize cache structure: {e}")
    
    def validate_directory_permissions(self) -> Dict[DirectoryType, bool]:
        """
        Validate read/write permissions for all application directories.
        
        Performs comprehensive permission validation to ensure the application
        can read from and write to all required directories. Critical for
        cache operations, database access, and export functionality.
        
        Returns:
            Dict[DirectoryType, bool]: Permission status for each directory type
        """
        permission_status = {}
        
        for dir_type, dir_path in self.directories.items():
            try:
                # Check if directory exists
                if not dir_path.exists():
                    permission_status[dir_type] = False
                    self.logger.warning(f"Directory does not exist: {dir_path}")
                    continue
                
                # Test read permission
                try:
                    list(dir_path.iterdir())
                except PermissionError:
                    permission_status[dir_type] = False
                    self.logger.error(f"No read permission for directory: {dir_path}")
                    continue
                
                # Test write permission
                try:
                    test_file = dir_path / f".permission_test_{int(time.time())}"
                    test_file.touch()
                    test_file.unlink()
                    permission_status[dir_type] = True
                    self.logger.debug(f"Permissions validated for directory: {dir_path}")
                    
                except PermissionError:
                    permission_status[dir_type] = False
                    self.logger.error(f"No write permission for directory: {dir_path}")
                    
            except Exception as e:
                permission_status[dir_type] = False
                self.logger.error(f"Permission validation failed for {dir_path}: {e}")
        
        return permission_status
    
    def get_storage_metrics(self) -> StorageMetrics:
        """
        Calculate comprehensive storage utilization metrics.
        
        Analyzes disk usage across all application directories to provide
        detailed storage metrics for monitoring and alerting. Supports
        the storage monitoring requirements from section 6.5.2.1.
        
        Returns:
            StorageMetrics: Complete storage utilization information
        """
        try:
            # Get disk usage for the application directory
            total, used, free = shutil.disk_usage(self.app_data_root)
            
            # Calculate directory sizes
            cache_size = self._calculate_directory_size(self.directories[DirectoryType.CACHE])
            logs_size = self._calculate_directory_size(self.directories[DirectoryType.LOGS])
            database_size = self._calculate_directory_size(self.directories[DirectoryType.DATABASE])
            exports_size = self._calculate_directory_size(self.directories[DirectoryType.EXPORTS])
            
            # Calculate utilization percentage
            utilization_percent = (used / total) * 100 if total > 0 else 0
            
            # Get last cleanup time from cache metadata
            last_cleanup = self._get_last_cleanup_time()
            
            metrics = StorageMetrics(
                total_space=total,
                available_space=free,
                used_space=used,
                cache_size=cache_size,
                logs_size=logs_size,
                database_size=database_size,
                exports_size=exports_size,
                utilization_percent=utilization_percent,
                last_cleanup=last_cleanup
            )
            
            self.logger.debug(f"Storage metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate storage metrics: {e}")
            # Return empty metrics in case of error
            return StorageMetrics(
                total_space=0,
                available_space=0,
                used_space=0,
                cache_size=0,
                logs_size=0,
                database_size=0,
                exports_size=0,
                utilization_percent=0
            )
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of all files in directory recursively."""
        total_size = 0
        try:
            if directory.exists():
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Could not calculate size for {directory}: {e}")
        
        return total_size
    
    def _get_last_cleanup_time(self) -> Optional[datetime]:
        """Get the timestamp of the last cache cleanup operation."""
        try:
            cache_metadata_path = self.directories[DirectoryType.CACHE] / ".cache_metadata.json"
            if cache_metadata_path.exists():
                with open(cache_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    last_cleanup_str = metadata.get('last_cleanup')
                    if last_cleanup_str:
                        return datetime.fromisoformat(last_cleanup_str)
        except Exception as e:
            self.logger.warning(f"Could not read last cleanup time: {e}")
        
        return None
    
    def cleanup_cache_by_ttl(self, force_cleanup: bool = False) -> Tuple[CleanupResult, Dict[str, int]]:
        """
        Perform TTL-based cache cleanup operations with comprehensive monitoring.
        
        Implements intelligent cache cleanup based on TTL policies defined in
        cache_ttl_config. Supports automatic and manual cleanup modes with
        detailed result reporting for monitoring integration.
        
        Args:
            force_cleanup: Force cleanup regardless of storage thresholds
            
        Returns:
            Tuple[CleanupResult, Dict[str, int]]: Cleanup result and statistics
        """
        try:
            self.logger.info("Starting TTL-based cache cleanup operation")
            
            # Check if cleanup is needed
            storage_metrics = self.get_storage_metrics()
            if not force_cleanup and not self._should_perform_cleanup(storage_metrics):
                self.logger.debug("Cache cleanup not needed based on current storage metrics")
                return CleanupResult.NO_ACTION_NEEDED, {}
            
            cleanup_stats = {
                "files_processed": 0,
                "files_deleted": 0,
                "space_freed_bytes": 0,
                "errors": 0,
                "cache_types_processed": 0
            }
            
            # Process each cache directory
            cache_entries = self._scan_cache_entries()
            
            for cache_type, entries in cache_entries.items():
                try:
                    cleanup_stats["cache_types_processed"] += 1
                    type_deleted, type_freed = self._cleanup_cache_type(cache_type, entries)
                    cleanup_stats["files_deleted"] += type_deleted
                    cleanup_stats["space_freed_bytes"] += type_freed
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning up cache type {cache_type}: {e}")
                    cleanup_stats["errors"] += 1
            
            # Update cache metadata
            self._update_cache_metadata(cleanup_stats)
            
            # Determine overall cleanup result
            if cleanup_stats["errors"] > 0:
                result = CleanupResult.PARTIAL_SUCCESS if cleanup_stats["files_deleted"] > 0 else CleanupResult.FAILED
            else:
                result = CleanupResult.SUCCESS
            
            self.logger.info(f"Cache cleanup completed: {result.value}, "
                           f"deleted {cleanup_stats['files_deleted']} files, "
                           f"freed {cleanup_stats['space_freed_bytes'] / (1024*1024):.2f} MB")
            
            return result, cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Cache cleanup operation failed: {e}")
            return CleanupResult.FAILED, {"errors": 1}
    
    def _should_perform_cleanup(self, storage_metrics: StorageMetrics) -> bool:
        """Determine if cache cleanup should be performed based on storage metrics."""
        # Check if cache size exceeds threshold
        if storage_metrics.is_cache_oversized:
            return True
        
        # Check if disk space is critical
        if storage_metrics.is_space_critical:
            return True
        
        # Check if overall utilization exceeds threshold
        threshold = self.storage_config["cleanup_threshold_percent"]
        if storage_metrics.utilization_percent > threshold:
            return True
        
        # Check if it's been too long since last cleanup
        if storage_metrics.last_cleanup:
            hours_since_cleanup = (datetime.now() - storage_metrics.last_cleanup).total_seconds() / 3600
            if hours_since_cleanup > 24:  # Force cleanup after 24 hours
                return True
        
        return False
    
    def _scan_cache_entries(self) -> Dict[str, List[CacheEntry]]:
        """Scan all cache directories and categorize entries by type."""
        cache_entries = {}
        
        for cache_type in self.cache_ttl_config.keys():
            cache_entries[cache_type] = []
            
            # Determine cache directory for this type
            if cache_type == "metadata":
                cache_dir = self.directories[DirectoryType.METADATA]
            elif cache_type == "market_data":
                cache_dir = self.directories[DirectoryType.MARKET_DATA]
            elif cache_type == "charts":
                cache_dir = self.directories[DirectoryType.CHARTS]
            else:
                cache_dir = self.directories[DirectoryType.CACHE]
            
            try:
                if cache_dir.exists():
                    for file_path in cache_dir.rglob('*'):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            entry = self._create_cache_entry(file_path, cache_type)
                            if entry:
                                cache_entries[cache_type].append(entry)
                                
            except Exception as e:
                self.logger.warning(f"Error scanning cache directory {cache_dir}: {e}")
        
        return cache_entries
    
    def _create_cache_entry(self, file_path: Path, cache_type: str) -> Optional[CacheEntry]:
        """Create a CacheEntry object from a file path."""
        try:
            stat_info = file_path.stat()
            created_time = datetime.fromtimestamp(stat_info.st_ctime)
            last_accessed = datetime.fromtimestamp(stat_info.st_atime)
            file_size = stat_info.st_size
            ttl_hours = self.cache_ttl_config.get(cache_type, 24)
            
            return CacheEntry(
                file_path=file_path,
                created_time=created_time,
                last_accessed=last_accessed,
                ttl_hours=ttl_hours,
                file_size=file_size,
                cache_type=cache_type
            )
            
        except Exception as e:
            self.logger.warning(f"Could not create cache entry for {file_path}: {e}")
            return None
    
    def _cleanup_cache_type(self, cache_type: str, entries: List[CacheEntry]) -> Tuple[int, int]:
        """Clean up expired entries for a specific cache type."""
        files_deleted = 0
        space_freed = 0
        
        for entry in entries:
            if entry.is_expired:
                try:
                    file_size = entry.file_size
                    entry.file_path.unlink()
                    files_deleted += 1
                    space_freed += file_size
                    self.logger.debug(f"Deleted expired cache entry: {entry.file_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not delete cache entry {entry.file_path}: {e}")
        
        return files_deleted, space_freed
    
    def _update_cache_metadata(self, cleanup_stats: Dict[str, int]) -> None:
        """Update cache metadata file with cleanup statistics."""
        try:
            cache_metadata_path = self.directories[DirectoryType.CACHE] / ".cache_metadata.json"
            
            # Read existing metadata
            metadata = {}
            if cache_metadata_path.exists():
                with open(cache_metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Update with cleanup statistics
            metadata.update({
                "last_cleanup": datetime.now().isoformat(),
                "last_cleanup_stats": cleanup_stats,
                "size_mb": self._calculate_directory_size(self.directories[DirectoryType.CACHE]) / (1024 * 1024)
            })
            
            # Write updated metadata
            with open(cache_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not update cache metadata: {e}")
    
    def create_database_backup(self, database_path: Path, backup_name: Optional[str] = None) -> Optional[BackupMetadata]:
        """
        Create a backup of the SQLite database with integrity verification.
        
        Implements comprehensive database backup operations with SHA-256 checksum
        validation and compression. Supports both manual and automated backup
        creation with configurable retention policies.
        
        Args:
            database_path: Path to the SQLite database file
            backup_name: Optional custom backup name, defaults to timestamp
            
        Returns:
            BackupMetadata: Backup information if successful, None if failed
        """
        try:
            if not database_path.exists():
                self.logger.error(f"Database file does not exist: {database_path}")
                return None
            
            # Generate backup filename
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"trading_app_backup_{timestamp}.db.gz"
            
            backup_dir = self.app_data_root / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / backup_name
            
            self.logger.info(f"Creating database backup: {backup_path}")
            
            # Calculate source file checksum
            source_checksum = self._calculate_file_checksum(database_path)
            source_size = database_path.stat().st_size
            
            # Perform backup with compression
            backup_size = self._compress_file_to_backup(database_path, backup_path)
            
            if backup_size <= 0:
                self.logger.error("Backup creation failed - zero size backup file")
                return None
            
            # Verify backup integrity by decompressing and comparing checksums
            if not self._verify_backup_integrity(backup_path, source_checksum):
                self.logger.error("Backup integrity verification failed")
                backup_path.unlink(missing_ok=True)
                return None
            
            # Create backup metadata
            backup_metadata = BackupMetadata(
                backup_path=backup_path,
                source_path=database_path,
                backup_time=datetime.now(),
                source_size=source_size,
                backup_size=backup_size,
                checksum=source_checksum,
                backup_type="database",
                retention_days=self.storage_config["backup_retention_days"]
            )
            
            # Save backup metadata
            self._save_backup_metadata(backup_metadata)
            
            self.logger.info(f"Database backup created successfully: {backup_path} "
                           f"(compressed {source_size} -> {backup_size} bytes)")
            
            return backup_metadata
            
        except Exception as e:
            self.logger.error(f"Database backup creation failed: {e}")
            return None
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum for file integrity verification."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Checksum calculation failed for {file_path}: {e}")
            return ""
    
    def _compress_file_to_backup(self, source_path: Path, backup_path: Path) -> int:
        """Compress source file to backup location and return compressed size."""
        try:
            with open(source_path, 'rb') as source_file:
                with gzip.open(backup_path, 'wb') as backup_file:
                    shutil.copyfileobj(source_file, backup_file)
            
            return backup_path.stat().st_size
            
        except Exception as e:
            self.logger.error(f"File compression failed: {e}")
            return 0
    
    def _verify_backup_integrity(self, backup_path: Path, expected_checksum: str) -> bool:
        """Verify backup integrity by decompressing and comparing checksums."""
        try:
            # Create temporary file for decompression
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            try:
                # Decompress backup to temporary file
                with gzip.open(backup_path, 'rb') as backup_file:
                    with open(temp_path, 'wb') as temp_file:
                        shutil.copyfileobj(backup_file, temp_file)
                
                # Calculate checksum of decompressed file
                decompressed_checksum = self._calculate_file_checksum(temp_path)
                
                # Compare checksums
                integrity_verified = decompressed_checksum == expected_checksum
                
                if integrity_verified:
                    self.logger.debug("Backup integrity verification successful")
                else:
                    self.logger.error("Backup integrity verification failed - checksum mismatch")
                
                return integrity_verified
                
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            self.logger.error(f"Backup integrity verification failed: {e}")
            return False
    
    def _save_backup_metadata(self, backup_metadata: BackupMetadata) -> None:
        """Save backup metadata for tracking and retention management."""
        try:
            metadata_dir = self.app_data_root / "backups"
            metadata_file = metadata_dir / "backup_metadata.json"
            
            # Load existing metadata
            all_metadata = []
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            
            # Add new backup metadata
            metadata_dict = {
                "backup_path": str(backup_metadata.backup_path),
                "source_path": str(backup_metadata.source_path),
                "backup_time": backup_metadata.backup_time.isoformat(),
                "source_size": backup_metadata.source_size,
                "backup_size": backup_metadata.backup_size,
                "checksum": backup_metadata.checksum,
                "backup_type": backup_metadata.backup_type,
                "retention_days": backup_metadata.retention_days
            }
            
            all_metadata.append(metadata_dict)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save backup metadata: {e}")
    
    def cleanup_expired_backups(self) -> Tuple[CleanupResult, Dict[str, int]]:
        """
        Clean up expired backup files based on retention policies.
        
        Implements automated backup retention management with configurable
        policies. Removes backups that exceed their retention period while
        maintaining metadata consistency.
        
        Returns:
            Tuple[CleanupResult, Dict[str, int]]: Cleanup result and statistics
        """
        try:
            self.logger.info("Starting expired backup cleanup")
            
            cleanup_stats = {
                "backups_processed": 0,
                "backups_deleted": 0,
                "space_freed_bytes": 0,
                "errors": 0
            }
            
            metadata_file = self.app_data_root / "backups" / "backup_metadata.json"
            
            if not metadata_file.exists():
                return CleanupResult.NO_ACTION_NEEDED, cleanup_stats
            
            # Load backup metadata
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            updated_metadata = []
            
            for metadata_dict in all_metadata:
                cleanup_stats["backups_processed"] += 1
                
                try:
                    # Parse backup metadata
                    backup_time = datetime.fromisoformat(metadata_dict["backup_time"])
                    retention_days = metadata_dict["retention_days"]
                    backup_path = Path(metadata_dict["backup_path"])
                    backup_size = metadata_dict["backup_size"]
                    
                    # Check if backup is expired
                    expiry_date = backup_time + timedelta(days=retention_days)
                    
                    if datetime.now() > expiry_date:
                        # Delete expired backup
                        if backup_path.exists():
                            backup_path.unlink()
                            cleanup_stats["backups_deleted"] += 1
                            cleanup_stats["space_freed_bytes"] += backup_size
                            self.logger.debug(f"Deleted expired backup: {backup_path}")
                        
                        # Don't add to updated metadata (remove from tracking)
                    else:
                        # Keep non-expired backup in metadata
                        updated_metadata.append(metadata_dict)
                        
                except Exception as e:
                    self.logger.error(f"Error processing backup metadata: {e}")
                    cleanup_stats["errors"] += 1
                    # Keep metadata entry in case of error
                    updated_metadata.append(metadata_dict)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(updated_metadata, f, indent=2)
            
            # Determine cleanup result
            if cleanup_stats["errors"] > 0:
                result = CleanupResult.PARTIAL_SUCCESS if cleanup_stats["backups_deleted"] > 0 else CleanupResult.FAILED
            else:
                result = CleanupResult.SUCCESS if cleanup_stats["backups_deleted"] > 0 else CleanupResult.NO_ACTION_NEEDED
            
            self.logger.info(f"Backup cleanup completed: {result.value}, "
                           f"deleted {cleanup_stats['backups_deleted']} backups, "
                           f"freed {cleanup_stats['space_freed_bytes'] / (1024*1024):.2f} MB")
            
            return result, cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
            return CleanupResult.FAILED, {"errors": 1}
    
    def rotate_log_files(self, max_size_mb: int = 10, max_files: int = 10) -> Tuple[CleanupResult, Dict[str, int]]:
        """
        Perform log file rotation with compression and cleanup.
        
        Implements professional log rotation with size-based triggers and
        compression of archived logs. Maintains configurable number of
        rotated log files for debugging and audit purposes.
        
        Args:
            max_size_mb: Maximum size in MB before rotation
            max_files: Maximum number of rotated log files to keep
            
        Returns:
            Tuple[CleanupResult, Dict[str, int]]: Rotation result and statistics
        """
        try:
            self.logger.info("Starting log file rotation")
            
            rotation_stats = {
                "files_processed": 0,
                "files_rotated": 0,
                "files_compressed": 0,
                "files_deleted": 0,
                "space_freed_bytes": 0,
                "errors": 0
            }
            
            logs_dir = self.directories[DirectoryType.LOGS]
            
            if not logs_dir.exists():
                return CleanupResult.NO_ACTION_NEEDED, rotation_stats
            
            # Find all log files
            log_files = []
            for ext in self.file_extensions["logs"]:
                log_files.extend(logs_dir.glob(f"*{ext}"))
            
            max_size_bytes = max_size_mb * 1024 * 1024
            
            for log_file in log_files:
                try:
                    rotation_stats["files_processed"] += 1
                    
                    # Check if file needs rotation
                    if log_file.stat().st_size > max_size_bytes:
                        # Perform rotation
                        if self._rotate_single_log_file(log_file, max_files):
                            rotation_stats["files_rotated"] += 1
                        else:
                            rotation_stats["errors"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing log file {log_file}: {e}")
                    rotation_stats["errors"] += 1
            
            # Compress old rotated logs
            compressed_count, deleted_count, space_freed = self._compress_old_logs(logs_dir, max_files)
            rotation_stats["files_compressed"] += compressed_count
            rotation_stats["files_deleted"] += deleted_count
            rotation_stats["space_freed_bytes"] += space_freed
            
            # Determine rotation result
            if rotation_stats["errors"] > 0:
                result = CleanupResult.PARTIAL_SUCCESS if rotation_stats["files_rotated"] > 0 else CleanupResult.FAILED
            else:
                result = CleanupResult.SUCCESS if rotation_stats["files_rotated"] > 0 else CleanupResult.NO_ACTION_NEEDED
            
            self.logger.info(f"Log rotation completed: {result.value}, "
                           f"rotated {rotation_stats['files_rotated']} files, "
                           f"compressed {rotation_stats['files_compressed']} files")
            
            return result, rotation_stats
            
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
            return CleanupResult.FAILED, {"errors": 1}
    
    def _rotate_single_log_file(self, log_file: Path, max_files: int) -> bool:
        """Rotate a single log file, shifting existing rotated files."""
        try:
            base_name = log_file.stem
            extension = log_file.suffix
            log_dir = log_file.parent
            
            # Shift existing rotated files
            for i in range(max_files - 1, 0, -1):
                old_rotated = log_dir / f"{base_name}.{i}{extension}"
                new_rotated = log_dir / f"{base_name}.{i + 1}{extension}"
                
                if old_rotated.exists():
                    if i == max_files - 1:
                        # Delete the oldest file
                        old_rotated.unlink()
                    else:
                        # Rename to next number
                        old_rotated.rename(new_rotated)
            
            # Move current log to .1
            rotated_name = log_dir / f"{base_name}.1{extension}"
            log_file.rename(rotated_name)
            
            # Create new empty log file
            log_file.touch()
            
            self.logger.debug(f"Rotated log file: {log_file} -> {rotated_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate log file {log_file}: {e}")
            return False
    
    def _compress_old_logs(self, logs_dir: Path, max_files: int) -> Tuple[int, int, int]:
        """Compress old rotated log files and delete excess files."""
        compressed_count = 0
        deleted_count = 0
        space_freed = 0
        
        try:
            # Find rotated log files (ending with .1, .2, etc.)
            rotated_logs = []
            for log_file in logs_dir.glob("*.*.log"):
                if log_file.suffix == ".log":
                    rotated_logs.append(log_file)
            
            for log_file in rotated_logs:
                try:
                    # Extract rotation number
                    parts = log_file.stem.split('.')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        rotation_num = int(parts[-1])
                        
                        if rotation_num > max_files:
                            # Delete excess files
                            original_size = log_file.stat().st_size
                            log_file.unlink()
                            deleted_count += 1
                            space_freed += original_size
                            
                        elif rotation_num > 1:  # Compress files older than .1
                            compressed_path = log_file.with_suffix(log_file.suffix + ".gz")
                            
                            if not compressed_path.exists():
                                original_size = log_file.stat().st_size
                                
                                # Compress the file
                                with open(log_file, 'rb') as f_in:
                                    with gzip.open(compressed_path, 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                                
                                # Delete original after successful compression
                                log_file.unlink()
                                compressed_count += 1
                                
                                # Calculate space saved
                                compressed_size = compressed_path.stat().st_size
                                space_freed += (original_size - compressed_size)
                
                except Exception as e:
                    self.logger.warning(f"Error processing rotated log {log_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error compressing old logs: {e}")
        
        return compressed_count, deleted_count, space_freed
    
    def validate_configuration_files(self) -> Dict[str, bool]:
        """
        Validate integrity and accessibility of configuration files.
        
        Performs comprehensive validation of critical configuration files
        including JSON schema validation, file corruption detection using
        SHA-256 checksums, and automatic restoration from backup files.
        
        Returns:
            Dict[str, bool]: Validation status for each configuration file
        """
        validation_results = {}
        config_dir = self.directories[DirectoryType.CONFIG]
        
        # Configuration files to validate
        config_files = {
            "default_settings.json": self._validate_default_settings,
            "api_keys.json": self._validate_api_keys,
        }
        
        for filename, validator_func in config_files.items():
            file_path = config_dir / filename
            
            try:
                if not file_path.exists():
                    self.logger.warning(f"Configuration file missing: {filename}")
                    validation_results[filename] = False
                    continue
                
                # Check file accessibility
                if not os.access(file_path, os.R_OK):
                    self.logger.error(f"Configuration file not readable: {filename}")
                    validation_results[filename] = False
                    continue
                
                # Validate file content using specific validator
                is_valid = validator_func(file_path)
                validation_results[filename] = is_valid
                
                if is_valid:
                    self.logger.debug(f"Configuration file validated: {filename}")
                else:
                    self.logger.error(f"Configuration file validation failed: {filename}")
                    # Attempt restoration from backup
                    self._attempt_config_restoration(file_path)
                
            except Exception as e:
                self.logger.error(f"Error validating configuration file {filename}: {e}")
                validation_results[filename] = False
        
        return validation_results
    
    def _validate_default_settings(self, file_path: Path) -> bool:
        """Validate default_settings.json file structure and content."""
        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
            
            # Check required sections
            required_sections = ["data_retention", "cache_settings", "backup_settings", "monitoring"]
            
            for section in required_sections:
                if section not in settings:
                    self.logger.error(f"Missing required section in default_settings.json: {section}")
                    return False
            
            # Validate data retention settings
            data_retention = settings["data_retention"]
            if not isinstance(data_retention.get("market_data_cache"), int):
                return False
            if data_retention.get("trade_history") != "permanent":
                return False
            if not isinstance(data_retention.get("logs"), int):
                return False
            
            # Validate cache settings
            cache_settings = settings["cache_settings"]
            if not isinstance(cache_settings.get("max_cache_size_mb"), int):
                return False
            if not isinstance(cache_settings.get("cleanup_threshold_percent"), int):
                return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Invalid default_settings.json format: {e}")
            return False
    
    def _validate_api_keys(self, file_path: Path) -> bool:
        """Validate api_keys.json file structure and content."""
        try:
            with open(file_path, 'r') as f:
                api_keys = json.load(f)
            
            # Check required providers
            required_providers = ["yahoo_finance", "alpha_vantage", "iex_cloud"]
            
            for provider in required_providers:
                if provider not in api_keys:
                    self.logger.error(f"Missing provider in api_keys.json: {provider}")
                    return False
                
                provider_config = api_keys[provider]
                if not isinstance(provider_config.get("enabled"), bool):
                    return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Invalid api_keys.json format: {e}")
            return False
    
    def _attempt_config_restoration(self, config_file: Path) -> bool:
        """Attempt to restore configuration file from backup."""
        try:
            backup_file = config_file.with_suffix(config_file.suffix + ".backup")
            
            if backup_file.exists():
                self.logger.info(f"Attempting to restore {config_file.name} from backup")
                
                # Validate backup file first
                if config_file.name == "default_settings.json":
                    is_valid = self._validate_default_settings(backup_file)
                elif config_file.name == "api_keys.json":
                    is_valid = self._validate_api_keys(backup_file)
                else:
                    is_valid = True
                
                if is_valid:
                    # Create backup of corrupted file
                    corrupted_backup = config_file.with_suffix(f".corrupted_{int(time.time())}")
                    shutil.copy2(config_file, corrupted_backup)
                    
                    # Restore from backup
                    shutil.copy2(backup_file, config_file)
                    
                    self.logger.info(f"Successfully restored {config_file.name} from backup")
                    return True
                
            self.logger.error(f"No valid backup available for {config_file.name}")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration file {config_file.name}: {e}")
            return False
    
    def create_configuration_backup(self) -> bool:
        """
        Create backup copies of all configuration files.
        
        Creates backup copies of critical configuration files to enable
        automatic restoration in case of corruption or accidental deletion.
        
        Returns:
            bool: True if all configuration files backed up successfully
        """
        try:
            config_dir = self.directories[DirectoryType.CONFIG]
            config_files = ["default_settings.json", "api_keys.json"]
            
            backup_success = True
            
            for filename in config_files:
                file_path = config_dir / filename
                backup_path = config_dir / f"{filename}.backup"
                
                try:
                    if file_path.exists():
                        shutil.copy2(file_path, backup_path)
                        self.logger.debug(f"Created backup for {filename}")
                    else:
                        self.logger.warning(f"Configuration file not found for backup: {filename}")
                        backup_success = False
                        
                except Exception as e:
                    self.logger.error(f"Failed to backup configuration file {filename}: {e}")
                    backup_success = False
            
            return backup_success
            
        except Exception as e:
            self.logger.error(f"Configuration backup operation failed: {e}")
            return False
    
    def monitor_storage_health(self) -> Dict[str, Any]:
        """
        Comprehensive storage health monitoring for alerting integration.
        
        Provides detailed storage health analysis including disk space monitoring,
        directory size tracking, and automated cleanup trigger recommendations.
        Supports integration with the monitoring framework from section 6.5.2.1.
        
        Returns:
            Dict[str, Any]: Complete storage health report
        """
        try:
            storage_metrics = self.get_storage_metrics()
            permission_status = self.validate_directory_permissions()
            config_validation = self.validate_configuration_files()
            
            # Calculate health scores
            permission_health = sum(permission_status.values()) / len(permission_status) * 100
            config_health = sum(config_validation.values()) / len(config_validation) * 100 if config_validation else 100
            
            # Determine overall health status
            overall_health = "healthy"
            if storage_metrics.is_space_critical:
                overall_health = "critical"
            elif storage_metrics.is_cache_oversized or storage_metrics.utilization_percent > 85:
                overall_health = "degraded"
            elif permission_health < 100 or config_health < 100:
                overall_health = "warning"
            
            # Generate recommendations
            recommendations = []
            if storage_metrics.is_space_critical:
                recommendations.append("Critical: Less than 500MB free space available")
            if storage_metrics.is_cache_oversized:
                recommendations.append("Cache directory exceeds 500MB - cleanup recommended")
            if permission_health < 100:
                recommendations.append("Directory permission issues detected")
            if config_health < 100:
                recommendations.append("Configuration file validation errors detected")
            
            health_report = {
                "timestamp": datetime.now().isoformat(),
                "overall_health": overall_health,
                "storage_metrics": {
                    "total_space_gb": storage_metrics.total_space / (1024**3),
                    "available_space_gb": storage_metrics.available_space / (1024**3),
                    "utilization_percent": storage_metrics.utilization_percent,
                    "cache_size_mb": storage_metrics.cache_size / (1024**2),
                    "logs_size_mb": storage_metrics.logs_size / (1024**2),
                    "database_size_mb": storage_metrics.database_size / (1024**2),
                    "exports_size_mb": storage_metrics.exports_size / (1024**2),
                },
                "directory_permissions": {
                    "permission_health_percent": permission_health,
                    "failed_directories": [dir_type.value for dir_type, status in permission_status.items() if not status]
                },
                "configuration_status": {
                    "config_health_percent": config_health,
                    "failed_files": [filename for filename, status in config_validation.items() if not status]
                },
                "recommendations": recommendations,
                "cleanup_needed": storage_metrics.is_cache_oversized or storage_metrics.utilization_percent > 85,
                "last_cache_cleanup": storage_metrics.last_cleanup.isoformat() if storage_metrics.last_cleanup else None
            }
            
            self.logger.debug(f"Storage health monitoring completed: {overall_health}")
            return health_report
            
        except Exception as e:
            self.logger.error(f"Storage health monitoring failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_health": "error",
                "error": str(e),
                "recommendations": ["Storage health monitoring system failure - manual inspection required"]
            }
    
    def get_directory_path(self, directory_type: DirectoryType) -> Path:
        """
        Get the path for a specific directory type.
        
        Args:
            directory_type: The type of directory to retrieve
            
        Returns:
            Path: The path to the requested directory
        """
        return self.directories.get(directory_type, self.app_data_root)
    
    def ensure_directory_exists(self, directory_type: DirectoryType) -> bool:
        """
        Ensure a specific directory exists and is accessible.
        
        Args:
            directory_type: The type of directory to ensure exists
            
        Returns:
            bool: True if directory exists and is accessible
        """
        try:
            directory_path = self.get_directory_path(directory_type)
            directory_path.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = directory_path / f".access_test_{int(time.time())}"
            test_file.touch()
            test_file.unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ensure directory {directory_type.value}: {e}")
            return False


# Convenience functions for common operations
def get_app_data_root() -> Path:
    """Get the application data root directory path."""
    manager = FileSystemManager()
    return manager.app_data_root


def ensure_app_directories() -> bool:
    """Ensure all application directories exist and are accessible."""
    manager = FileSystemManager()
    return manager.initialize_directory_structure()


def cleanup_application_cache(force: bool = False) -> Tuple[CleanupResult, Dict[str, int]]:
    """Perform application cache cleanup with TTL-based policies."""
    manager = FileSystemManager()
    return manager.cleanup_cache_by_ttl(force_cleanup=force)


def create_database_backup(database_path: Union[str, Path]) -> Optional[BackupMetadata]:
    """Create a backup of the application database."""
    manager = FileSystemManager()
    return manager.create_database_backup(Path(database_path))


def monitor_application_storage() -> Dict[str, Any]:
    """Monitor application storage health and return comprehensive report."""
    manager = FileSystemManager()
    return manager.monitor_storage_health()


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    manager = FileSystemManager()
    
    # Initialize directory structure
    if manager.initialize_directory_structure():
        print("Directory structure initialized successfully")
    
    # Monitor storage health
    health_report = manager.monitor_storage_health()
    print(f"Storage health: {health_report['overall_health']}")
    
    # Perform cache cleanup
    cleanup_result, stats = manager.cleanup_cache_by_ttl()
    print(f"Cache cleanup: {cleanup_result.value}, deleted {stats.get('files_deleted', 0)} files")