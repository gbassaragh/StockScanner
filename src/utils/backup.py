"""
Backup and Restoration Utilities for Trading Platform

This module provides comprehensive backup and restoration capabilities for the trading platform,
including automated database backups, configuration file versioning, SHA-256 integrity verification,
and recovery procedures for trading data and application settings with retention management.

Key Features:
- Automated database backup with SHA-256 checksum verification
- Configuration file backup system with versioning
- Trading journal backup operations with ACID transaction rollback
- Scheduled backup creation with configurable retention policies
- Backup compression and archival functionality
- Restoration utilities with validation and rollback capabilities

Author: Blitzy Development Team
Version: 1.0.0
"""

import os
import sys
import json
import sqlite3
import hashlib
import shutil
import zipfile
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Configure logging for backup operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata structure for backup operations"""
    backup_id: str
    timestamp: datetime
    backup_type: str  # 'full', 'incremental', 'config_only', 'database_only'
    files_included: List[str]
    sha256_checksums: Dict[str, str]
    compression_enabled: bool
    backup_size_bytes: int
    retention_days: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class BackupConfig:
    """Configuration for backup operations"""
    # Retention policies
    daily_retention_days: int = 30
    weekly_retention_weeks: int = 12
    monthly_retention_months: int = 6
    
    # Backup types
    enable_compression: bool = True
    enable_encryption: bool = False
    
    # Scheduling
    auto_backup_enabled: bool = True
    auto_backup_interval_hours: int = 6
    
    # Storage settings
    max_backup_size_mb: int = 1024  # 1GB default
    backup_location: Optional[str] = None
    
    # Recovery settings
    enable_journal_backup: bool = True
    journal_backup_interval_minutes: int = 5


class BackupError(Exception):
    """Custom exception for backup operations"""
    pass


class BackupIntegrityError(BackupError):
    """Exception for backup integrity validation failures"""
    pass


class BackupUtility:
    """
    Comprehensive backup and restoration utility for the trading platform.
    
    Provides automated database backups, configuration file versioning, SHA-256 integrity
    verification, and recovery procedures for trading data and application settings.
    """
    
    def __init__(self, config: Optional[BackupConfig] = None):
        """
        Initialize backup utility with configuration.
        
        Args:
            config: Backup configuration settings. If None, uses default configuration.
        """
        self.config = config or BackupConfig()
        self._setup_paths()
        self._lock = threading.Lock()
        self._running_backups: Dict[str, bool] = {}
        
        # Initialize logging for backup operations
        self.logger = logging.getLogger(f"{__name__}.BackupUtility")
        
        # Ensure backup directories exist
        self._ensure_backup_directories()
        
        # Load existing backup metadata
        self._backup_metadata: Dict[str, BackupMetadata] = self._load_backup_metadata()
    
    def _setup_paths(self) -> None:
        """Setup application and backup directory paths"""
        # Application data directory
        self.app_data_dir = Path(os.environ.get('APPDATA', '')) / 'Blitzy'
        
        # Primary database and configuration paths
        self.database_path = self.app_data_dir / 'database' / 'trading_app.db'
        self.metrics_db_path = self.app_data_dir / 'metrics.db'
        self.config_dir = self.app_data_dir / 'config'
        self.default_settings_path = self.config_dir / 'default_settings.json'
        self.api_keys_path = self.config_dir / 'api_keys.json'
        self.cache_dir = self.app_data_dir / 'cache'
        self.exports_dir = self.app_data_dir / 'exports'
        self.logs_dir = self.app_data_dir / 'logs'
        
        # Backup directory structure
        if self.config.backup_location:
            self.backup_root = Path(self.config.backup_location)
        else:
            self.backup_root = self.app_data_dir / 'backups'
        
        self.backup_database_dir = self.backup_root / 'database'
        self.backup_config_dir = self.backup_root / 'config'
        self.backup_full_dir = self.backup_root / 'full'
        self.backup_metadata_file = self.backup_root / 'backup_metadata.json'
        
        # Journal backup path (for automatic recovery)
        self.journal_backup_path = self.backup_database_dir / 'journal.backup'
    
    def _ensure_backup_directories(self) -> None:
        """Ensure all backup directories exist"""
        directories = [
            self.backup_root,
            self.backup_database_dir,
            self.backup_config_dir,
            self.backup_full_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured backup directory exists: {directory}")
    
    def _load_backup_metadata(self) -> Dict[str, BackupMetadata]:
        """Load existing backup metadata from storage"""
        if not self.backup_metadata_file.exists():
            return {}
        
        try:
            with open(self.backup_metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for backup_id, meta_dict in data.items():
                # Convert timestamp string back to datetime
                meta_dict['timestamp'] = datetime.fromisoformat(meta_dict['timestamp'])
                metadata[backup_id] = BackupMetadata(**meta_dict)
            
            self.logger.info(f"Loaded metadata for {len(metadata)} backups")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load backup metadata: {e}")
            return {}
    
    def _save_backup_metadata(self) -> None:
        """Save backup metadata to storage"""
        try:
            # Convert datetime objects to ISO strings for JSON serialization
            serializable_metadata = {}
            for backup_id, metadata in self._backup_metadata.items():
                meta_dict = asdict(metadata)
                meta_dict['timestamp'] = metadata.timestamp.isoformat()
                serializable_metadata[backup_id] = meta_dict
            
            with open(self.backup_metadata_file, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            
            self.logger.debug("Backup metadata saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save backup metadata: {e}")
            raise BackupError(f"Failed to save backup metadata: {e}")
    
    def calculate_sha256(self, file_path: Union[str, Path]) -> str:
        """
        Calculate SHA-256 checksum for a file.
        
        Args:
            file_path: Path to the file to checksum
            
        Returns:
            SHA-256 checksum as hexadecimal string
            
        Raises:
            BackupError: If file cannot be read or checksum fails
        """
        try:
            file_path = Path(file_path)
            hash_sha256 = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            checksum = hash_sha256.hexdigest()
            self.logger.debug(f"Calculated SHA-256 for {file_path}: {checksum}")
            return checksum
            
        except Exception as e:
            self.logger.error(f"Failed to calculate SHA-256 for {file_path}: {e}")
            raise BackupError(f"Failed to calculate SHA-256 checksum: {e}")
    
    def verify_file_integrity(self, file_path: Union[str, Path], expected_checksum: str) -> bool:
        """
        Verify file integrity using SHA-256 checksum.
        
        Args:
            file_path: Path to the file to verify
            expected_checksum: Expected SHA-256 checksum
            
        Returns:
            True if file integrity is valid, False otherwise
        """
        try:
            actual_checksum = self.calculate_sha256(file_path)
            is_valid = actual_checksum == expected_checksum
            
            if is_valid:
                self.logger.debug(f"File integrity verified: {file_path}")
            else:
                self.logger.warning(f"File integrity check failed for {file_path}")
                self.logger.warning(f"Expected: {expected_checksum}")
                self.logger.warning(f"Actual: {actual_checksum}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Failed to verify file integrity for {file_path}: {e}")
            return False
    
    @contextmanager
    def _database_connection(self, db_path: Union[str, Path], read_only: bool = False):
        """
        Context manager for database connections with proper error handling.
        
        Args:
            db_path: Path to the database file
            read_only: If True, opens database in read-only mode
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise BackupError(f"Database file not found: {db_path}")
        
        connection = None
        try:
            uri = f"file:{db_path}?mode=ro" if read_only else str(db_path)
            connection = sqlite3.connect(uri, uri=read_only)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            yield connection
            
        except Exception as e:
            self.logger.error(f"Database connection error for {db_path}: {e}")
            raise BackupError(f"Database connection failed: {e}")
        finally:
            if connection:
                connection.close()
    
    def backup_database(self, backup_id: Optional[str] = None, include_metrics: bool = True) -> str:
        """
        Create a backup of the trading database with integrity verification.
        
        Args:
            backup_id: Optional custom backup identifier
            include_metrics: Whether to include metrics database in backup
            
        Returns:
            Backup identifier for the created backup
            
        Raises:
            BackupError: If backup operation fails
        """
        if backup_id is None:
            backup_id = f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            if backup_id in self._running_backups:
                raise BackupError(f"Backup {backup_id} is already in progress")
            self._running_backups[backup_id] = True
        
        try:
            self.logger.info(f"Starting database backup: {backup_id}")
            
            backup_files = []
            checksums = {}
            
            # Create timestamped backup directory
            backup_dir = self.backup_database_dir / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup primary trading database
            if self.database_path.exists():
                backup_db_path = backup_dir / 'trading_app.db'
                self._backup_sqlite_database(self.database_path, backup_db_path)
                backup_files.append(str(backup_db_path))
                checksums[str(backup_db_path)] = self.calculate_sha256(backup_db_path)
                self.logger.info(f"Backed up primary database to: {backup_db_path}")
            
            # Backup metrics database if requested
            if include_metrics and self.metrics_db_path.exists():
                backup_metrics_path = backup_dir / 'metrics.db'
                self._backup_sqlite_database(self.metrics_db_path, backup_metrics_path)
                backup_files.append(str(backup_metrics_path))
                checksums[str(backup_metrics_path)] = self.calculate_sha256(backup_metrics_path)
                self.logger.info(f"Backed up metrics database to: {backup_metrics_path}")
            
            # Calculate total backup size
            total_size = sum(Path(file_path).stat().st_size for file_path in backup_files)
            
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type='database_only',
                files_included=backup_files,
                sha256_checksums=checksums,
                compression_enabled=False,  # SQLite backups are not compressed
                backup_size_bytes=total_size,
                retention_days=self.config.daily_retention_days,
                success=True
            )
            
            # Store metadata
            self._backup_metadata[backup_id] = metadata
            self._save_backup_metadata()
            
            # Create journal backup for automatic recovery
            if self.config.enable_journal_backup:
                self._create_journal_backup()
            
            self.logger.info(f"Database backup completed successfully: {backup_id}")
            self.logger.info(f"Backup size: {total_size:,} bytes")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            # Create failed backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type='database_only',
                files_included=[],
                sha256_checksums={},
                compression_enabled=False,
                backup_size_bytes=0,
                retention_days=self.config.daily_retention_days,
                success=False,
                error_message=str(e)
            )
            self._backup_metadata[backup_id] = metadata
            self._save_backup_metadata()
            
            raise BackupError(f"Database backup failed: {e}")
            
        finally:
            with self._lock:
                self._running_backups.pop(backup_id, None)
    
    def _backup_sqlite_database(self, source_path: Path, backup_path: Path) -> None:
        """
        Backup SQLite database using BACKUP API for consistency.
        
        Args:
            source_path: Source database file path
            backup_path: Destination backup file path
        """
        try:
            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use SQLite BACKUP API for consistent backup
            with self._database_connection(source_path, read_only=True) as source_conn:
                with sqlite3.connect(str(backup_path)) as backup_conn:
                    source_conn.backup(backup_conn)
            
            self.logger.debug(f"SQLite backup completed: {source_path} -> {backup_path}")
            
        except Exception as e:
            self.logger.error(f"SQLite backup failed: {source_path} -> {backup_path}: {e}")
            raise BackupError(f"SQLite backup failed: {e}")
    
    def _create_journal_backup(self) -> None:
        """
        Create journal backup for automatic recovery with SHA-256 validation.
        
        This creates a special backup file that can be used for automatic recovery
        in case of journal corruption or application crash.
        """
        try:
            if not self.database_path.exists():
                self.logger.warning("Primary database not found, skipping journal backup")
                return
            
            # Create journal backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_backup_path = self.journal_backup_path.with_suffix(f'.{timestamp}.tmp')
            
            # Backup database
            self._backup_sqlite_database(self.database_path, temp_backup_path)
            
            # Calculate checksum
            checksum = self.calculate_sha256(temp_backup_path)
            
            # Create checksum file
            checksum_file = temp_backup_path.with_suffix('.sha256')
            with open(checksum_file, 'w') as f:
                f.write(f"{checksum}  {temp_backup_path.name}\n")
            
            # Atomically replace journal backup
            if self.journal_backup_path.exists():
                # Remove old checksum file
                old_checksum_file = self.journal_backup_path.with_suffix('.sha256')
                if old_checksum_file.exists():
                    old_checksum_file.unlink()
                
                # Remove old backup
                self.journal_backup_path.unlink()
            
            # Move new backup into place
            temp_backup_path.rename(self.journal_backup_path)
            checksum_file.rename(self.journal_backup_path.with_suffix('.sha256'))
            
            self.logger.debug(f"Journal backup created: {self.journal_backup_path}")
            self.logger.debug(f"Journal backup checksum: {checksum}")
            
        except Exception as e:
            self.logger.error(f"Failed to create journal backup: {e}")
            # Clean up temporary files
            for temp_file in [temp_backup_path, checksum_file]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
    
    def backup_configuration(self, backup_id: Optional[str] = None, create_versioned: bool = True) -> str:
        """
        Backup configuration files with versioning and integrity verification.
        
        Args:
            backup_id: Optional custom backup identifier
            create_versioned: Whether to create versioned backup
            
        Returns:
            Backup identifier for the created backup
            
        Raises:
            BackupError: If configuration backup fails
        """
        if backup_id is None:
            backup_id = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"Starting configuration backup: {backup_id}")
            
            backup_files = []
            checksums = {}
            
            # Create backup directory
            backup_dir = self.backup_config_dir / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration files
            config_files = [
                (self.default_settings_path, 'default_settings.json'),
                (self.api_keys_path, 'api_keys.json')
            ]
            
            for source_path, filename in config_files:
                if source_path.exists():
                    backup_file_path = backup_dir / filename
                    
                    # Copy file with metadata preservation
                    shutil.copy2(source_path, backup_file_path)
                    
                    # Calculate checksum
                    checksum = self.calculate_sha256(backup_file_path)
                    checksums[str(backup_file_path)] = checksum
                    backup_files.append(str(backup_file_path))
                    
                    self.logger.debug(f"Backed up config file: {source_path} -> {backup_file_path}")
                    
                    # Create versioned backup if requested
                    if create_versioned:
                        self._create_versioned_config_backup(source_path, filename)
            
            # Calculate total backup size
            total_size = sum(Path(file_path).stat().st_size for file_path in backup_files)
            
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type='config_only',
                files_included=backup_files,
                sha256_checksums=checksums,
                compression_enabled=False,
                backup_size_bytes=total_size,
                retention_days=self.config.daily_retention_days,
                success=True
            )
            
            # Store metadata
            self._backup_metadata[backup_id] = metadata
            self._save_backup_metadata()
            
            self.logger.info(f"Configuration backup completed successfully: {backup_id}")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
            raise BackupError(f"Configuration backup failed: {e}")
    
    def _create_versioned_config_backup(self, source_path: Path, filename: str) -> None:
        """
        Create versioned backup of configuration file.
        
        Args:
            source_path: Source configuration file path
            filename: Configuration filename for versioning
        """
        try:
            # Create versioned backup directory
            version_dir = self.backup_config_dir / 'versions' / filename.replace('.json', '')
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate version filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_filename = f"{filename.replace('.json', '')}_{timestamp}.json"
            version_path = version_dir / version_filename
            
            # Copy file
            shutil.copy2(source_path, version_path)
            
            # Calculate checksum for version
            checksum = self.calculate_sha256(version_path)
            
            # Create checksum file for version
            checksum_file = version_path.with_suffix('.sha256')
            with open(checksum_file, 'w') as f:
                f.write(f"{checksum}  {version_filename}\n")
            
            # Cleanup old versions (keep last 10)
            self._cleanup_old_config_versions(version_dir, max_versions=10)
            
            self.logger.debug(f"Created versioned config backup: {version_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create versioned config backup: {e}")
    
    def _cleanup_old_config_versions(self, version_dir: Path, max_versions: int = 10) -> None:
        """
        Cleanup old configuration versions, keeping only the most recent.
        
        Args:
            version_dir: Directory containing versioned configuration files
            max_versions: Maximum number of versions to retain
        """
        try:
            # Get all version files (exclude checksum files)
            version_files = [f for f in version_dir.glob('*.json') if f.is_file()]
            
            if len(version_files) <= max_versions:
                return
            
            # Sort by modification time (newest first)
            version_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old versions
            for old_file in version_files[max_versions:]:
                try:
                    # Remove version file
                    old_file.unlink()
                    
                    # Remove associated checksum file
                    checksum_file = old_file.with_suffix('.sha256')
                    if checksum_file.exists():
                        checksum_file.unlink()
                    
                    self.logger.debug(f"Removed old config version: {old_file}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to remove old config version {old_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old config versions: {e}")
    
    def create_full_backup(self, backup_id: Optional[str] = None, include_cache: bool = False) -> str:
        """
        Create a complete backup of all application data with compression.
        
        Args:
            backup_id: Optional custom backup identifier
            include_cache: Whether to include cache directories in backup
            
        Returns:
            Backup identifier for the created backup
            
        Raises:
            BackupError: If full backup operation fails
        """
        if backup_id is None:
            backup_id = f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            if backup_id in self._running_backups:
                raise BackupError(f"Backup {backup_id} is already in progress")
            self._running_backups[backup_id] = True
        
        try:
            self.logger.info(f"Starting full backup: {backup_id}")
            
            # Create backup archive path
            archive_name = f"{backup_id}.zip" if self.config.enable_compression else backup_id
            archive_path = self.backup_full_dir / archive_name
            
            backup_files = []
            checksums = {}
            
            # Define directories and files to backup
            backup_sources = [
                (self.database_path, 'database/trading_app.db'),
                (self.metrics_db_path, 'metrics.db'),
                (self.default_settings_path, 'config/default_settings.json'),
                (self.api_keys_path, 'config/api_keys.json'),
                (self.exports_dir, 'exports/'),
                (self.logs_dir, 'logs/')
            ]
            
            # Add cache directories if requested
            if include_cache:
                backup_sources.extend([
                    (self.cache_dir / 'market_data', 'cache/market_data/'),
                    (self.cache_dir / 'metadata', 'cache/metadata/'),
                    (self.cache_dir / 'charts', 'cache/charts/')
                ])
            
            if self.config.enable_compression:
                # Create compressed backup
                self._create_compressed_backup(backup_sources, archive_path, checksums)
                backup_files = [str(archive_path)]
            else:
                # Create uncompressed directory backup
                backup_dir = self.backup_full_dir / backup_id
                self._create_directory_backup(backup_sources, backup_dir, backup_files, checksums)
            
            # Calculate total backup size
            total_size = sum(Path(file_path).stat().st_size for file_path in backup_files)
            
            # Verify backup size limits
            if total_size > (self.config.max_backup_size_mb * 1024 * 1024):
                raise BackupError(f"Backup size ({total_size:,} bytes) exceeds limit")
            
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type='full',
                files_included=backup_files,
                sha256_checksums=checksums,
                compression_enabled=self.config.enable_compression,
                backup_size_bytes=total_size,
                retention_days=self.config.daily_retention_days,
                success=True
            )
            
            # Store metadata
            self._backup_metadata[backup_id] = metadata
            self._save_backup_metadata()
            
            self.logger.info(f"Full backup completed successfully: {backup_id}")
            self.logger.info(f"Backup size: {total_size:,} bytes")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Full backup failed: {e}")
            # Create failed backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type='full',
                files_included=[],
                sha256_checksums={},
                compression_enabled=self.config.enable_compression,
                backup_size_bytes=0,
                retention_days=self.config.daily_retention_days,
                success=False,
                error_message=str(e)
            )
            self._backup_metadata[backup_id] = metadata
            self._save_backup_metadata()
            
            raise BackupError(f"Full backup failed: {e}")
            
        finally:
            with self._lock:
                self._running_backups.pop(backup_id, None)
    
    def _create_compressed_backup(self, backup_sources: List[Tuple[Path, str]], 
                                archive_path: Path, checksums: Dict[str, str]) -> None:
        """
        Create compressed ZIP backup archive.
        
        Args:
            backup_sources: List of (source_path, archive_path) tuples
            archive_path: Path for the backup archive
            checksums: Dictionary to store file checksums
        """
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for source_path, archive_name in backup_sources:
                    if not source_path.exists():
                        self.logger.warning(f"Backup source not found: {source_path}")
                        continue
                    
                    if source_path.is_file():
                        # Add single file to archive
                        archive.write(source_path, archive_name)
                        self.logger.debug(f"Added file to archive: {source_path} -> {archive_name}")
                    
                    elif source_path.is_dir():
                        # Add directory recursively
                        self._add_directory_to_archive(archive, source_path, archive_name)
            
            # Calculate archive checksum
            checksums[str(archive_path)] = self.calculate_sha256(archive_path)
            
            self.logger.info(f"Compressed backup created: {archive_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create compressed backup: {e}")
            raise BackupError(f"Compressed backup creation failed: {e}")
    
    def _add_directory_to_archive(self, archive: zipfile.ZipFile, source_dir: Path, archive_prefix: str) -> None:
        """
        Recursively add directory contents to ZIP archive.
        
        Args:
            archive: ZIP archive object
            source_dir: Source directory to add
            archive_prefix: Prefix for files in archive
        """
        try:
            for item_path in source_dir.rglob('*'):
                if item_path.is_file():
                    # Calculate relative path from source directory
                    relative_path = item_path.relative_to(source_dir)
                    archive_path = f"{archive_prefix.rstrip('/')}/{relative_path}"
                    
                    # Add file to archive
                    archive.write(item_path, archive_path)
                    self.logger.debug(f"Added to archive: {item_path} -> {archive_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to add directory to archive: {source_dir}: {e}")
            raise
    
    def _create_directory_backup(self, backup_sources: List[Tuple[Path, str]], 
                               backup_dir: Path, backup_files: List[str], 
                               checksums: Dict[str, str]) -> None:
        """
        Create uncompressed directory backup.
        
        Args:
            backup_sources: List of (source_path, archive_path) tuples
            backup_dir: Backup directory path
            backup_files: List to store backed up file paths
            checksums: Dictionary to store file checksums
        """
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            for source_path, relative_path in backup_sources:
                if not source_path.exists():
                    self.logger.warning(f"Backup source not found: {source_path}")
                    continue
                
                dest_path = backup_dir / relative_path
                
                if source_path.is_file():
                    # Ensure destination directory exists
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(source_path, dest_path)
                    
                    # Calculate checksum and add to tracking
                    checksums[str(dest_path)] = self.calculate_sha256(dest_path)
                    backup_files.append(str(dest_path))
                    
                    self.logger.debug(f"Copied file: {source_path} -> {dest_path}")
                
                elif source_path.is_dir():
                    # Copy directory recursively
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    
                    shutil.copytree(source_path, dest_path)
                    
                    # Calculate checksums for all copied files
                    for copied_file in dest_path.rglob('*'):
                        if copied_file.is_file():
                            checksums[str(copied_file)] = self.calculate_sha256(copied_file)
                            backup_files.append(str(copied_file))
                    
                    self.logger.debug(f"Copied directory: {source_path} -> {dest_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create directory backup: {e}")
            raise BackupError(f"Directory backup creation failed: {e}")
    
    def restore_from_backup(self, backup_id: str, restore_options: Optional[Dict[str, bool]] = None) -> bool:
        """
        Restore application data from a backup with validation and rollback capabilities.
        
        Args:
            backup_id: Identifier of the backup to restore
            restore_options: Dictionary of restore options:
                - restore_database: Whether to restore database files
                - restore_config: Whether to restore configuration files
                - restore_cache: Whether to restore cache directories
                - verify_integrity: Whether to verify backup integrity before restore
                - create_restore_point: Whether to create restore point before restore
                
        Returns:
            True if restore completed successfully, False otherwise
            
        Raises:
            BackupError: If restore operation fails
        """
        # Default restore options
        default_options = {
            'restore_database': True,
            'restore_config': True,
            'restore_cache': False,
            'verify_integrity': True,
            'create_restore_point': True
        }
        
        if restore_options:
            default_options.update(restore_options)
        
        restore_options = default_options
        
        try:
            self.logger.info(f"Starting restore from backup: {backup_id}")
            
            # Validate backup exists
            if backup_id not in self._backup_metadata:
                raise BackupError(f"Backup not found: {backup_id}")
            
            metadata = self._backup_metadata[backup_id]
            
            # Verify backup was successful
            if not metadata.success:
                raise BackupError(f"Cannot restore from failed backup: {backup_id}")
            
            # Verify backup integrity if requested
            if restore_options['verify_integrity']:
                if not self._verify_backup_integrity(backup_id, metadata):
                    raise BackupIntegrityError(f"Backup integrity verification failed: {backup_id}")
            
            # Create restore point if requested
            restore_point_id = None
            if restore_options['create_restore_point']:
                restore_point_id = self.create_full_backup(
                    backup_id=f"restore_point_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.logger.info(f"Created restore point: {restore_point_id}")
            
            try:
                # Perform restore based on backup type
                if metadata.backup_type == 'full':
                    self._restore_full_backup(backup_id, metadata, restore_options)
                elif metadata.backup_type == 'database_only':
                    if restore_options['restore_database']:
                        self._restore_database_backup(backup_id, metadata)
                elif metadata.backup_type == 'config_only':
                    if restore_options['restore_config']:
                        self._restore_config_backup(backup_id, metadata)
                else:
                    raise BackupError(f"Unsupported backup type: {metadata.backup_type}")
                
                self.logger.info(f"Restore completed successfully: {backup_id}")
                return True
                
            except Exception as restore_error:
                self.logger.error(f"Restore failed, attempting rollback: {restore_error}")
                
                # Attempt rollback using restore point
                if restore_point_id:
                    try:
                        self.logger.info(f"Rolling back using restore point: {restore_point_id}")
                        self._restore_full_backup(
                            restore_point_id, 
                            self._backup_metadata[restore_point_id],
                            {'restore_database': True, 'restore_config': True, 'restore_cache': False}
                        )
                        self.logger.info("Rollback completed successfully")
                    except Exception as rollback_error:
                        self.logger.error(f"Rollback failed: {rollback_error}")
                        raise BackupError(f"Restore failed and rollback failed: {rollback_error}")
                
                raise BackupError(f"Restore failed: {restore_error}")
                
        except Exception as e:
            self.logger.error(f"Restore operation failed: {e}")
            raise BackupError(f"Restore failed: {e}")
    
    def _verify_backup_integrity(self, backup_id: str, metadata: BackupMetadata) -> bool:
        """
        Verify backup integrity using stored checksums.
        
        Args:
            backup_id: Backup identifier
            metadata: Backup metadata containing checksums
            
        Returns:
            True if backup integrity is valid, False otherwise
        """
        try:
            self.logger.info(f"Verifying backup integrity: {backup_id}")
            
            all_files_valid = True
            
            for file_path, expected_checksum in metadata.sha256_checksums.items():
                file_path_obj = Path(file_path)
                
                if not file_path_obj.exists():
                    self.logger.error(f"Backup file missing: {file_path}")
                    all_files_valid = False
                    continue
                
                if not self.verify_file_integrity(file_path_obj, expected_checksum):
                    self.logger.error(f"Integrity check failed for: {file_path}")
                    all_files_valid = False
                else:
                    self.logger.debug(f"Integrity verified: {file_path}")
            
            if all_files_valid:
                self.logger.info(f"Backup integrity verification passed: {backup_id}")
            else:
                self.logger.error(f"Backup integrity verification failed: {backup_id}")
            
            return all_files_valid
            
        except Exception as e:
            self.logger.error(f"Backup integrity verification error: {e}")
            return False
    
    def _restore_full_backup(self, backup_id: str, metadata: BackupMetadata, 
                           restore_options: Dict[str, bool]) -> None:
        """
        Restore from full backup archive or directory.
        
        Args:
            backup_id: Backup identifier
            metadata: Backup metadata
            restore_options: Restore configuration options
        """
        try:
            if metadata.compression_enabled:
                # Restore from compressed archive
                archive_path = self.backup_full_dir / f"{backup_id}.zip"
                if not archive_path.exists():
                    raise BackupError(f"Backup archive not found: {archive_path}")
                
                self._restore_from_archive(archive_path, restore_options)
            else:
                # Restore from directory
                backup_dir = self.backup_full_dir / backup_id
                if not backup_dir.exists():
                    raise BackupError(f"Backup directory not found: {backup_dir}")
                
                self._restore_from_directory(backup_dir, restore_options)
            
            self.logger.info(f"Full backup restore completed: {backup_id}")
            
        except Exception as e:
            self.logger.error(f"Full backup restore failed: {e}")
            raise BackupError(f"Full backup restore failed: {e}")
    
    def _restore_from_archive(self, archive_path: Path, restore_options: Dict[str, bool]) -> None:
        """
        Restore application data from compressed archive.
        
        Args:
            archive_path: Path to backup archive
            restore_options: Restore configuration options
        """
        try:
            with zipfile.ZipFile(archive_path, 'r') as archive:
                # Create temporary extraction directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Extract archive
                    archive.extractall(temp_path)
                    
                    # Restore from extracted content
                    self._restore_from_directory(temp_path, restore_options)
            
        except Exception as e:
            self.logger.error(f"Archive restore failed: {e}")
            raise BackupError(f"Archive restore failed: {e}")
    
    def _restore_from_directory(self, backup_dir: Path, restore_options: Dict[str, bool]) -> None:
        """
        Restore application data from backup directory.
        
        Args:
            backup_dir: Backup directory path
            restore_options: Restore configuration options
        """
        try:
            # Define restore mappings
            restore_mappings = []
            
            if restore_options['restore_database']:
                restore_mappings.extend([
                    (backup_dir / 'database' / 'trading_app.db', self.database_path),
                    (backup_dir / 'metrics.db', self.metrics_db_path)
                ])
            
            if restore_options['restore_config']:
                restore_mappings.extend([
                    (backup_dir / 'config' / 'default_settings.json', self.default_settings_path),
                    (backup_dir / 'config' / 'api_keys.json', self.api_keys_path)
                ])
            
            if restore_options['restore_cache']:
                restore_mappings.extend([
                    (backup_dir / 'cache' / 'market_data', self.cache_dir / 'market_data'),
                    (backup_dir / 'cache' / 'metadata', self.cache_dir / 'metadata'),
                    (backup_dir / 'cache' / 'charts', self.cache_dir / 'charts')
                ])
            
            # Other directories (exports, logs) are always restored if present
            restore_mappings.extend([
                (backup_dir / 'exports', self.exports_dir),
                (backup_dir / 'logs', self.logs_dir)
            ])
            
            # Perform restore operations
            for source_path, dest_path in restore_mappings:
                if not source_path.exists():
                    self.logger.debug(f"Restore source not found (skipping): {source_path}")
                    continue
                
                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if source_path.is_file():
                    # Restore file
                    if dest_path.exists():
                        dest_path.unlink()
                    
                    shutil.copy2(source_path, dest_path)
                    self.logger.debug(f"Restored file: {source_path} -> {dest_path}")
                
                elif source_path.is_dir():
                    # Restore directory
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    
                    shutil.copytree(source_path, dest_path)
                    self.logger.debug(f"Restored directory: {source_path} -> {dest_path}")
            
        except Exception as e:
            self.logger.error(f"Directory restore failed: {e}")
            raise BackupError(f"Directory restore failed: {e}")
    
    def _restore_database_backup(self, backup_id: str, metadata: BackupMetadata) -> None:
        """
        Restore database files from database-only backup.
        
        Args:
            backup_id: Backup identifier
            metadata: Backup metadata
        """
        try:
            backup_dir = self.backup_database_dir / backup_id
            
            # Restore primary database
            source_db = backup_dir / 'trading_app.db'
            if source_db.exists():
                # Ensure destination directory exists
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Remove existing database
                if self.database_path.exists():
                    self.database_path.unlink()
                
                # Copy backup to destination
                shutil.copy2(source_db, self.database_path)
                self.logger.info(f"Restored primary database: {source_db} -> {self.database_path}")
            
            # Restore metrics database
            source_metrics = backup_dir / 'metrics.db'
            if source_metrics.exists():
                if self.metrics_db_path.exists():
                    self.metrics_db_path.unlink()
                
                shutil.copy2(source_metrics, self.metrics_db_path)
                self.logger.info(f"Restored metrics database: {source_metrics} -> {self.metrics_db_path}")
            
        except Exception as e:
            self.logger.error(f"Database restore failed: {e}")
            raise BackupError(f"Database restore failed: {e}")
    
    def _restore_config_backup(self, backup_id: str, metadata: BackupMetadata) -> None:
        """
        Restore configuration files from config-only backup.
        
        Args:
            backup_id: Backup identifier
            metadata: Backup metadata
        """
        try:
            backup_dir = self.backup_config_dir / backup_id
            
            # Restore configuration files
            config_files = [
                ('default_settings.json', self.default_settings_path),
                ('api_keys.json', self.api_keys_path)
            ]
            
            for filename, dest_path in config_files:
                source_file = backup_dir / filename
                if source_file.exists():
                    # Ensure destination directory exists
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Remove existing file
                    if dest_path.exists():
                        dest_path.unlink()
                    
                    # Copy backup to destination
                    shutil.copy2(source_file, dest_path)
                    self.logger.info(f"Restored config file: {source_file} -> {dest_path}")
            
        except Exception as e:
            self.logger.error(f"Configuration restore failed: {e}")
            raise BackupError(f"Configuration restore failed: {e}")
    
    def recover_from_journal_backup(self, validate_checksum: bool = True) -> bool:
        """
        Automatic recovery from journal backup file with SHA-256 checksum validation.
        
        This method provides emergency recovery capabilities for critical database corruption
        scenarios using the automatically maintained journal backup file.
        
        Args:
            validate_checksum: Whether to validate backup integrity before recovery
            
        Returns:
            True if recovery completed successfully, False otherwise
            
        Raises:
            BackupError: If recovery operation fails
        """
        try:
            self.logger.info("Starting automatic recovery from journal backup")
            
            # Check if journal backup exists
            if not self.journal_backup_path.exists():
                raise BackupError("Journal backup file not found")
            
            # Validate checksum if requested
            if validate_checksum:
                checksum_file = self.journal_backup_path.with_suffix('.sha256')
                if not checksum_file.exists():
                    raise BackupError("Journal backup checksum file not found")
                
                # Read expected checksum
                with open(checksum_file, 'r') as f:
                    checksum_line = f.readline().strip()
                    expected_checksum = checksum_line.split()[0]
                
                # Verify integrity
                if not self.verify_file_integrity(self.journal_backup_path, expected_checksum):
                    raise BackupIntegrityError("Journal backup integrity verification failed")
                
                self.logger.info("Journal backup integrity verified")
            
            # Create recovery backup of current state if database exists
            recovery_backup_id = None
            if self.database_path.exists():
                try:
                    recovery_backup_id = f"recovery_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_dir = self.backup_database_dir / recovery_backup_id
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    
                    recovery_path = backup_dir / 'trading_app.db'
                    shutil.copy2(self.database_path, recovery_path)
                    
                    self.logger.info(f"Created recovery backup: {recovery_backup_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to create recovery backup: {e}")
            
            # Ensure destination directory exists
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove corrupted database
            if self.database_path.exists():
                self.database_path.unlink()
            
            # Restore from journal backup
            shutil.copy2(self.journal_backup_path, self.database_path)
            
            # Verify restored database
            if not self._verify_database_integrity(self.database_path):
                # Rollback if verification fails
                if recovery_backup_id:
                    recovery_path = self.backup_database_dir / recovery_backup_id / 'trading_app.db'
                    if recovery_path.exists():
                        shutil.copy2(recovery_path, self.database_path)
                        self.logger.warning("Restored database failed verification, rolled back")
                
                raise BackupError("Restored database failed integrity verification")
            
            self.logger.info("Automatic recovery from journal backup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Journal backup recovery failed: {e}")
            raise BackupError(f"Journal backup recovery failed: {e}")
    
    def _verify_database_integrity(self, db_path: Path) -> bool:
        """
        Verify SQLite database integrity.
        
        Args:
            db_path: Path to database file to verify
            
        Returns:
            True if database integrity is valid, False otherwise
        """
        try:
            with self._database_connection(db_path, read_only=True) as conn:
                # Run PRAGMA integrity_check
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                
                if result and result[0] == 'ok':
                    self.logger.debug(f"Database integrity check passed: {db_path}")
                    return True
                else:
                    self.logger.error(f"Database integrity check failed: {db_path}")
                    if result:
                        self.logger.error(f"Integrity check result: {result[0]}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Database integrity verification error: {e}")
            return False
    
    def cleanup_old_backups(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Clean up old backups based on retention policies.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            self.logger.info("Starting backup cleanup process")
            
            cleanup_stats = {
                'total_backups': len(self._backup_metadata),
                'expired_backups': 0,
                'deleted_backups': 0,
                'freed_bytes': 0,
                'errors': 0
            }
            
            current_time = datetime.now()
            expired_backups = []
            
            # Find expired backups
            for backup_id, metadata in self._backup_metadata.items():
                retention_cutoff = current_time - timedelta(days=metadata.retention_days)
                
                if metadata.timestamp < retention_cutoff:
                    expired_backups.append((backup_id, metadata))
                    cleanup_stats['expired_backups'] += 1
            
            # Delete expired backups
            for backup_id, metadata in expired_backups:
                try:
                    if not dry_run:
                        deleted_size = self._delete_backup_files(backup_id, metadata)
                        cleanup_stats['freed_bytes'] += deleted_size
                        
                        # Remove from metadata
                        del self._backup_metadata[backup_id]
                        cleanup_stats['deleted_backups'] += 1
                    
                    self.logger.info(f"{'Would delete' if dry_run else 'Deleted'} expired backup: {backup_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to delete backup {backup_id}: {e}")
                    cleanup_stats['errors'] += 1
            
            # Save updated metadata if not dry run
            if not dry_run and cleanup_stats['deleted_backups'] > 0:
                self._save_backup_metadata()
            
            self.logger.info(f"Backup cleanup completed. Stats: {cleanup_stats}")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
            raise BackupError(f"Backup cleanup failed: {e}")
    
    def _delete_backup_files(self, backup_id: str, metadata: BackupMetadata) -> int:
        """
        Delete backup files for a specific backup.
        
        Args:
            backup_id: Backup identifier
            metadata: Backup metadata
            
        Returns:
            Total bytes freed by deletion
        """
        freed_bytes = 0
        
        try:
            if metadata.backup_type == 'full':
                # Delete full backup files
                if metadata.compression_enabled:
                    archive_path = self.backup_full_dir / f"{backup_id}.zip"
                    if archive_path.exists():
                        freed_bytes += archive_path.stat().st_size
                        archive_path.unlink()
                else:
                    backup_dir = self.backup_full_dir / backup_id
                    if backup_dir.exists():
                        freed_bytes += self._calculate_directory_size(backup_dir)
                        shutil.rmtree(backup_dir)
            
            elif metadata.backup_type == 'database_only':
                # Delete database backup directory
                backup_dir = self.backup_database_dir / backup_id
                if backup_dir.exists():
                    freed_bytes += self._calculate_directory_size(backup_dir)
                    shutil.rmtree(backup_dir)
            
            elif metadata.backup_type == 'config_only':
                # Delete config backup directory
                backup_dir = self.backup_config_dir / backup_id
                if backup_dir.exists():
                    freed_bytes += self._calculate_directory_size(backup_dir)
                    shutil.rmtree(backup_dir)
            
            return freed_bytes
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup files for {backup_id}: {e}")
            raise
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """
        Calculate total size of directory and all its contents.
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            self.logger.warning(f"Error calculating directory size {directory}: {e}")
        
        return total_size
    
    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get comprehensive backup system status and statistics.
        
        Returns:
            Dictionary containing backup system status information
        """
        try:
            current_time = datetime.now()
            
            # Calculate backup statistics
            total_backups = len(self._backup_metadata)
            successful_backups = sum(1 for meta in self._backup_metadata.values() if meta.success)
            failed_backups = total_backups - successful_backups
            
            # Calculate storage usage
            total_storage_bytes = 0
            backup_type_counts = {'full': 0, 'database_only': 0, 'config_only': 0}
            
            for metadata in self._backup_metadata.values():
                if metadata.success:
                    total_storage_bytes += metadata.backup_size_bytes
                    if metadata.backup_type in backup_type_counts:
                        backup_type_counts[metadata.backup_type] += 1
            
            # Find latest backup
            latest_backup = None
            latest_timestamp = None
            
            for backup_id, metadata in self._backup_metadata.items():
                if metadata.success and (latest_timestamp is None or metadata.timestamp > latest_timestamp):
                    latest_backup = backup_id
                    latest_timestamp = metadata.timestamp
            
            # Check journal backup status
            journal_backup_available = self.journal_backup_path.exists()
            journal_backup_age = None
            
            if journal_backup_available:
                journal_stat = self.journal_backup_path.stat()
                journal_backup_age = current_time - datetime.fromtimestamp(journal_stat.st_mtime)
            
            # Check for expired backups
            expired_count = 0
            for metadata in self._backup_metadata.values():
                retention_cutoff = current_time - timedelta(days=metadata.retention_days)
                if metadata.timestamp < retention_cutoff:
                    expired_count += 1
            
            return {
                'total_backups': total_backups,
                'successful_backups': successful_backups,
                'failed_backups': failed_backups,
                'backup_types': backup_type_counts,
                'total_storage_mb': round(total_storage_bytes / (1024 * 1024), 2),
                'latest_backup': {
                    'backup_id': latest_backup,
                    'timestamp': latest_timestamp.isoformat() if latest_timestamp else None,
                    'age_hours': round((current_time - latest_timestamp).total_seconds() / 3600, 1) if latest_timestamp else None
                },
                'journal_backup': {
                    'available': journal_backup_available,
                    'age_minutes': round(journal_backup_age.total_seconds() / 60, 1) if journal_backup_age else None
                },
                'expired_backups': expired_count,
                'retention_policy': {
                    'daily_retention_days': self.config.daily_retention_days,
                    'weekly_retention_weeks': self.config.weekly_retention_weeks,
                    'monthly_retention_months': self.config.monthly_retention_months
                },
                'auto_backup': {
                    'enabled': self.config.auto_backup_enabled,
                    'interval_hours': self.config.auto_backup_interval_hours
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get backup status: {e}")
            return {'error': str(e)}
    
    def list_backups(self, backup_type: Optional[str] = None, 
                    include_failed: bool = False) -> List[Dict[str, Any]]:
        """
        List available backups with filtering options.
        
        Args:
            backup_type: Filter by backup type ('full', 'database_only', 'config_only')
            include_failed: Whether to include failed backups in results
            
        Returns:
            List of backup information dictionaries
        """
        try:
            backups = []
            
            for backup_id, metadata in self._backup_metadata.items():
                # Apply filters
                if not include_failed and not metadata.success:
                    continue
                
                if backup_type and metadata.backup_type != backup_type:
                    continue
                
                backup_info = {
                    'backup_id': backup_id,
                    'timestamp': metadata.timestamp.isoformat(),
                    'backup_type': metadata.backup_type,
                    'success': metadata.success,
                    'size_mb': round(metadata.backup_size_bytes / (1024 * 1024), 2),
                    'compression_enabled': metadata.compression_enabled,
                    'retention_days': metadata.retention_days,
                    'files_count': len(metadata.files_included),
                    'error_message': metadata.error_message
                }
                
                backups.append(backup_info)
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return backups
            
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    def create_scheduled_backup(self) -> Optional[str]:
        """
        Create scheduled backup based on configuration.
        
        This method is designed to be called by a scheduler service for automated backups.
        
        Returns:
            Backup ID if successful, None if backup was skipped or failed
        """
        try:
            if not self.config.auto_backup_enabled:
                self.logger.debug("Auto backup is disabled, skipping scheduled backup")
                return None
            
            # Check if recent backup exists
            latest_backup_time = None
            for metadata in self._backup_metadata.values():
                if metadata.success and metadata.backup_type == 'full':
                    if latest_backup_time is None or metadata.timestamp > latest_backup_time:
                        latest_backup_time = metadata.timestamp
            
            if latest_backup_time:
                time_since_last = datetime.now() - latest_backup_time
                if time_since_last.total_seconds() < (self.config.auto_backup_interval_hours * 3600):
                    self.logger.debug("Recent backup exists, skipping scheduled backup")
                    return None
            
            # Create full backup
            backup_id = self.create_full_backup(
                backup_id=f"scheduled_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                include_cache=False  # Exclude cache for scheduled backups
            )
            
            self.logger.info(f"Scheduled backup completed: {backup_id}")
            
            # Cleanup old backups after successful backup
            try:
                cleanup_stats = self.cleanup_old_backups(dry_run=False)
                self.logger.info(f"Post-backup cleanup completed: {cleanup_stats}")
            except Exception as cleanup_error:
                self.logger.warning(f"Post-backup cleanup failed: {cleanup_error}")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Scheduled backup failed: {e}")
            return None


class BackupScheduler:
    """
    Scheduler for automated backup operations.
    
    This class provides scheduling capabilities for the backup utility,
    supporting configurable backup intervals and retention management.
    """
    
    def __init__(self, backup_utility: BackupUtility):
        """
        Initialize backup scheduler.
        
        Args:
            backup_utility: BackupUtility instance to use for scheduled operations
        """
        self.backup_utility = backup_utility
        self.logger = logging.getLogger(f"{__name__}.BackupScheduler")
        self._scheduler_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._running = False
    
    def start_scheduler(self) -> None:
        """Start the backup scheduler in background thread."""
        if self._running:
            self.logger.warning("Backup scheduler is already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="BackupScheduler",
            daemon=True
        )
        self._scheduler_thread.start()
        
        self.logger.info("Backup scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the backup scheduler."""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=10)
        
        self.logger.info("Backup scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop running in background thread."""
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    # Perform scheduled backup
                    backup_id = self.backup_utility.create_scheduled_backup()
                    
                    if backup_id:
                        self.logger.info(f"Scheduled backup completed: {backup_id}")
                    
                except Exception as e:
                    self.logger.error(f"Scheduled backup error: {e}")
                
                # Wait for next backup interval
                interval_seconds = self.backup_utility.config.auto_backup_interval_hours * 3600
                self._shutdown_event.wait(timeout=interval_seconds)
                
        except Exception as e:
            self.logger.error(f"Scheduler loop error: {e}")
        finally:
            self._running = False


# Utility functions for external use
def create_backup_utility(config_file: Optional[str] = None) -> BackupUtility:
    """
    Create and configure a BackupUtility instance.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured BackupUtility instance
    """
    config = BackupConfig()
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        except Exception as e:
            logger.warning(f"Failed to load backup config from {config_file}: {e}")
    
    return BackupUtility(config)


def emergency_database_recovery() -> bool:
    """
    Emergency database recovery function for critical situations.
    
    This function provides a simple interface for emergency recovery
    when the main application is not available.
    
    Returns:
        True if recovery was successful, False otherwise
    """
    try:
        backup_utility = create_backup_utility()
        return backup_utility.recover_from_journal_backup(validate_checksum=True)
    except Exception as e:
        logger.error(f"Emergency database recovery failed: {e}")
        return False


if __name__ == "__main__":
    # Command-line interface for backup operations
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(description="Trading Platform Backup Utility")
        parser.add_argument("--config", help="Configuration file path")
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Backup commands
        backup_parser = subparsers.add_parser("backup", help="Create backup")
        backup_parser.add_argument("--type", choices=["full", "database", "config"], 
                                 default="full", help="Backup type")
        backup_parser.add_argument("--id", help="Custom backup ID")
        
        # Restore commands
        restore_parser = subparsers.add_parser("restore", help="Restore from backup")
        restore_parser.add_argument("backup_id", help="Backup ID to restore")
        restore_parser.add_argument("--no-verify", action="store_true", 
                                  help="Skip integrity verification")
        
        # Recovery commands
        subparsers.add_parser("recover", help="Emergency recovery from journal backup")
        
        # Status commands
        subparsers.add_parser("status", help="Show backup system status")
        subparsers.add_parser("list", help="List available backups")
        
        # Cleanup commands
        cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
        cleanup_parser.add_argument("--dry-run", action="store_true", 
                                  help="Show what would be deleted without deleting")
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 1
        
        try:
            backup_utility = create_backup_utility(args.config)
            
            if args.command == "backup":
                if args.type == "full":
                    backup_id = backup_utility.create_full_backup(args.id)
                elif args.type == "database":
                    backup_id = backup_utility.backup_database(args.id)
                elif args.type == "config":
                    backup_id = backup_utility.backup_configuration(args.id)
                
                print(f"Backup completed: {backup_id}")
            
            elif args.command == "restore":
                restore_options = {"verify_integrity": not args.no_verify}
                success = backup_utility.restore_from_backup(args.backup_id, restore_options)
                print(f"Restore {'completed' if success else 'failed'}")
            
            elif args.command == "recover":
                success = backup_utility.recover_from_journal_backup()
                print(f"Recovery {'completed' if success else 'failed'}")
            
            elif args.command == "status":
                status = backup_utility.get_backup_status()
                print(json.dumps(status, indent=2))
            
            elif args.command == "list":
                backups = backup_utility.list_backups()
                for backup in backups:
                    print(f"{backup['backup_id']}: {backup['backup_type']} "
                          f"({backup['size_mb']:.1f}MB) - {backup['timestamp']}")
            
            elif args.command == "cleanup":
                stats = backup_utility.cleanup_old_backups(args.dry_run)
                print(f"Cleanup stats: {stats}")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    sys.exit(main())