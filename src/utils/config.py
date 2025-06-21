"""
Configuration Management Utilities

This module provides comprehensive configuration management for the Trading Application,
including secure JSON configuration handling, Windows Data Protection API integration,
schema validation, backup/restore capabilities, and environment variable management.

Key Features:
- Secure API key storage using Windows DPAPI encryption
- JSON schema validation with integrity checking
- Configuration backup/restore with SHA-256 verification
- Environment variable management with python-dotenv
- Configuration migration utilities for schema evolution
- Thread-safe configuration access for PyQt6 application

Author: Blitzy Development Team
Version: 1.0.0
"""

import json
import os
import shutil
import hashlib
import logging
import threading
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager

try:
    import win32crypt
    import win32security
    import win32api
    DPAPI_AVAILABLE = True
except ImportError:
    # Fallback for non-Windows platforms during development
    DPAPI_AVAILABLE = False
    logging.warning("Windows DPAPI not available - using fallback encryption")

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("Cryptography library not available")

try:
    from pydantic import BaseModel, ValidationError, Field
    from pydantic.json import pydantic_encoder
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.warning("Pydantic not available - basic validation only")

try:
    from dotenv import load_dotenv, set_key, unset_key
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not available")

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Models
# ============================================================================

@dataclass
class ConfigurationMetadata:
    """Metadata for configuration file tracking and validation."""
    file_path: str
    checksum: str
    last_modified: datetime
    version: str
    backup_count: int = 0
    encrypted: bool = False


@dataclass
class BackupInfo:
    """Information about configuration backup files."""
    backup_path: str
    original_path: str
    checksum: str
    timestamp: datetime
    version: str


@dataclass
class ValidationResult:
    """Result of configuration validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Pydantic models for configuration validation (if available)
if PYDANTIC_AVAILABLE:
    class APIKeyConfig(BaseModel):
        """Pydantic model for API key configuration validation."""
        api_key: str = Field(..., min_length=10, description="API key string")
        provider_name: str = Field(..., min_length=1, description="Provider name")
        enabled: bool = Field(default=True, description="Whether provider is enabled")
        last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }

    class DefaultSettingsConfig(BaseModel):
        """Pydantic model for default settings validation."""
        application: Dict[str, Any] = Field(default_factory=dict)
        security: Dict[str, Any] = Field(default_factory=dict)
        performance: Dict[str, Any] = Field(default_factory=dict)
        ui: Dict[str, Any] = Field(default_factory=dict)
        data_sources: Dict[str, Any] = Field(default_factory=dict)
        risk_management: Dict[str, Any] = Field(default_factory=dict)
        export_settings: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Encryption and Security Utilities
# ============================================================================

class SecurityManager:
    """Handles encryption and decryption operations for configuration data."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}
        self._cache_ttl = {}
        self.cache_duration = timedelta(minutes=30)
    
    def encrypt_data(self, data: Union[str, bytes], use_dpapi: bool = True) -> bytes:
        """
        Encrypt sensitive data using Windows DPAPI or fallback encryption.
        
        Args:
            data: Data to encrypt (string or bytes)
            use_dpapi: Whether to use Windows DPAPI (default: True)
            
        Returns:
            Encrypted data as bytes
            
        Raises:
            RuntimeError: If encryption fails or no encryption method available
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if use_dpapi and DPAPI_AVAILABLE:
            try:
                # Use Windows DPAPI for encryption
                encrypted_data = win32crypt.CryptProtectData(
                    data,
                    "TradingApp Configuration Data",
                    None,  # Optional entropy
                    None,  # Reserved
                    None,  # Prompt struct
                    win32crypt.CRYPTPROTECT_UI_FORBIDDEN
                )
                logger.debug("Data encrypted using Windows DPAPI")
                return encrypted_data
            except Exception as e:
                logger.error(f"DPAPI encryption failed: {e}")
                # Fall through to alternative encryption
        
        # Fallback encryption using cryptography library
        if CRYPTOGRAPHY_AVAILABLE:
            try:
                # Generate a key from system-specific information
                key_material = self._generate_system_key()
                fernet = Fernet(key_material)
                encrypted_data = fernet.encrypt(data)
                logger.debug("Data encrypted using Fernet encryption")
                return encrypted_data
            except Exception as e:
                logger.error(f"Fernet encryption failed: {e}")
        
        raise RuntimeError("No encryption method available")
    
    def decrypt_data(self, encrypted_data: bytes, use_dpapi: bool = True) -> bytes:
        """
        Decrypt sensitive data using Windows DPAPI or fallback decryption.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            use_dpapi: Whether to use Windows DPAPI (default: True)
            
        Returns:
            Decrypted data as bytes
            
        Raises:
            RuntimeError: If decryption fails
        """
        if use_dpapi and DPAPI_AVAILABLE:
            try:
                decrypted_data, _ = win32crypt.CryptUnprotectData(
                    encrypted_data,
                    None,  # Reserved
                    None,  # Optional entropy
                    None,  # Reserved
                    win32crypt.CRYPTPROTECT_UI_FORBIDDEN
                )
                logger.debug("Data decrypted using Windows DPAPI")
                return decrypted_data
            except Exception as e:
                logger.warning(f"DPAPI decryption failed: {e}")
                # Fall through to alternative decryption
        
        # Fallback decryption using cryptography library
        if CRYPTOGRAPHY_AVAILABLE:
            try:
                key_material = self._generate_system_key()
                fernet = Fernet(key_material)
                decrypted_data = fernet.decrypt(encrypted_data)
                logger.debug("Data decrypted using Fernet encryption")
                return decrypted_data
            except Exception as e:
                logger.error(f"Fernet decryption failed: {e}")
        
        raise RuntimeError("Failed to decrypt data")
    
    def _generate_system_key(self) -> bytes:
        """Generate a system-specific encryption key for fallback encryption."""
        # Use system-specific information to generate a consistent key
        system_info = f"{os.environ.get('COMPUTERNAME', 'default')}-{os.environ.get('USERNAME', 'user')}"
        system_info += "-TradingApp-Config-Key"
        
        # Generate key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'TradingApp-Salt-2024',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(system_info.encode()))
        return key
    
    def cache_decrypted_value(self, key: str, value: Any) -> None:
        """Cache a decrypted value with TTL expiration."""
        with self._lock:
            self._cache[key] = value
            self._cache_ttl[key] = datetime.now() + self.cache_duration
    
    def get_cached_value(self, key: str) -> Optional[Any]:
        """Retrieve a cached decrypted value if not expired."""
        with self._lock:
            if key in self._cache:
                if datetime.now() < self._cache_ttl.get(key, datetime.min):
                    return self._cache[key]
                else:
                    # Remove expired entry
                    self._cache.pop(key, None)
                    self._cache_ttl.pop(key, None)
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached decrypted values."""
        with self._lock:
            self._cache.clear()
            self._cache_ttl.clear()


# ============================================================================
# Configuration File Manager
# ============================================================================

class ConfigurationManager:
    """
    Comprehensive configuration management for the Trading Application.
    
    Provides secure JSON configuration handling, schema validation,
    backup/restore capabilities, and configuration migration support.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the Configuration Manager.
        
        Args:
            config_dir: Optional custom configuration directory path
        """
        # Determine configuration directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Use standard Windows application data directory
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            self.config_dir = Path(app_data) / 'TradingApp' / 'config'
        
        # Create configuration directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-managers
        self.security_manager = SecurityManager()
        
        # Configuration paths
        self.default_settings_path = self.config_dir / 'default_settings.json'
        self.api_keys_path = self.config_dir / 'api_keys.json'
        self.backup_dir = self.config_dir / 'backups'
        self.metadata_path = self.config_dir / 'metadata.json'
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration schemas
        self._schemas = self._load_schemas()
        
        # Environment file path
        self.env_file_path = self.config_dir / '.env'
        
        # Load environment variables if available
        if DOTENV_AVAILABLE and self.env_file_path.exists():
            load_dotenv(self.env_file_path)
        
        logger.info(f"Configuration manager initialized with directory: {self.config_dir}")
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for configuration validation."""
        schemas = {
            'default_settings': {
                "type": "object",
                "properties": {
                    "application": {"type": "object"},
                    "security": {
                        "type": "object",
                        "properties": {
                            "dpapi_encryption": {"type": "object"},
                            "tls_version": {"type": "string"},
                            "audit_logging": {"type": "object"}
                        }
                    },
                    "performance": {
                        "type": "object",
                        "properties": {
                            "ui_response_budget_ms": {"type": "integer", "minimum": 50, "maximum": 500},
                            "security_validation_timeout_ms": {"type": "integer"},
                            "async_logging_enabled": {"type": "boolean"}
                        }
                    },
                    "ui": {"type": "object"},
                    "data_sources": {"type": "object"},
                    "risk_management": {
                        "type": "object",
                        "properties": {
                            "max_position_risk_percent": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                            "account_risk_percent": {"type": "number", "minimum": 1.0, "maximum": 5.0}
                        }
                    },
                    "export_settings": {"type": "object"}
                },
                "required": ["application", "security", "performance"]
            },
            'api_keys': {
                "type": "object",
                "patternProperties": {
                    ".*": {
                        "type": "object",
                        "properties": {
                            "api_key": {"type": "string", "minLength": 10},
                            "provider_name": {"type": "string", "minLength": 1},
                            "enabled": {"type": "boolean"},
                            "last_updated": {"type": "string", "format": "date-time"}
                        },
                        "required": ["api_key", "provider_name", "enabled"]
                    }
                }
            }
        }
        return schemas
    
    def calculate_checksum(self, file_path: Union[str, Path]) -> str:
        """
        Calculate SHA-256 checksum for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 checksum as hexadecimal string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def validate_json_schema(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """
        Validate configuration data against JSON schema.
        
        Args:
            data: Configuration data to validate
            schema_name: Name of the schema to use for validation
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        if schema_name not in self._schemas:
            errors.append(f"Unknown schema: {schema_name}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        schema = self._schemas[schema_name]
        
        # Basic type validation
        if not isinstance(data, dict):
            errors.append("Configuration data must be a dictionary")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Validate required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate field types and constraints
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in data:
                value = data[field]
                field_type = field_schema.get('type')
                
                if field_type == 'object' and not isinstance(value, dict):
                    errors.append(f"Field '{field}' must be an object")
                elif field_type == 'string' and not isinstance(value, str):
                    errors.append(f"Field '{field}' must be a string")
                elif field_type == 'integer' and not isinstance(value, int):
                    errors.append(f"Field '{field}' must be an integer")
                elif field_type == 'number' and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field}' must be a number")
                elif field_type == 'boolean' and not isinstance(value, bool):
                    errors.append(f"Field '{field}' must be a boolean")
                
                # Check numeric constraints
                if field_type in ['integer', 'number'] and isinstance(value, (int, float)):
                    minimum = field_schema.get('minimum')
                    maximum = field_schema.get('maximum')
                    if minimum is not None and value < minimum:
                        errors.append(f"Field '{field}' value {value} is below minimum {minimum}")
                    if maximum is not None and value > maximum:
                        errors.append(f"Field '{field}' value {value} is above maximum {maximum}")
                
                # Check string constraints
                if field_type == 'string' and isinstance(value, str):
                    min_length = field_schema.get('minLength')
                    max_length = field_schema.get('maxLength')
                    if min_length is not None and len(value) < min_length:
                        errors.append(f"Field '{field}' length {len(value)} is below minimum {min_length}")
                    if max_length is not None and len(value) > max_length:
                        errors.append(f"Field '{field}' length {len(value)} is above maximum {max_length}")
        
        # Pydantic validation if available
        if PYDANTIC_AVAILABLE:
            try:
                if schema_name == 'default_settings':
                    DefaultSettingsConfig(**data)
                elif schema_name == 'api_keys':
                    for key, value in data.items():
                        APIKeyConfig(**value)
            except ValidationError as e:
                for error in e.errors():
                    field_path = " -> ".join(str(x) for x in error['loc'])
                    errors.append(f"Pydantic validation error in {field_path}: {error['msg']}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    @contextmanager
    def _file_lock(self, operation: str):
        """Context manager for thread-safe file operations."""
        with self._lock:
            logger.debug(f"Acquired lock for {operation}")
            try:
                yield
            finally:
                logger.debug(f"Released lock for {operation}")
    
    def create_backup(self, file_path: Union[str, Path], backup_reason: str = "") -> Optional[str]:
        """
        Create a backup of a configuration file with checksum verification.
        
        Args:
            file_path: Path to the file to backup
            backup_reason: Optional reason for the backup
            
        Returns:
            Path to the backup file if successful, None otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Cannot backup non-existent file: {file_path}")
            return None
        
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{file_path.stem}_{timestamp}.backup"
            backup_path = self.backup_dir / backup_filename
            
            # Copy file to backup location
            shutil.copy2(file_path, backup_path)
            
            # Calculate checksums
            original_checksum = self.calculate_checksum(file_path)
            backup_checksum = self.calculate_checksum(backup_path)
            
            # Verify backup integrity
            if original_checksum != backup_checksum:
                logger.error(f"Backup checksum mismatch for {file_path}")
                backup_path.unlink(missing_ok=True)
                return None
            
            # Create backup metadata
            backup_info = BackupInfo(
                backup_path=str(backup_path),
                original_path=str(file_path),
                checksum=backup_checksum,
                timestamp=datetime.now(),
                version="1.0"
            )
            
            # Save backup metadata
            metadata_path = backup_path.with_suffix('.metadata')
            with open(metadata_path, 'w') as f:
                json.dump(asdict(backup_info), f, indent=2, default=str)
            
            logger.info(f"Created backup: {backup_path} (reason: {backup_reason})")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def restore_from_backup(self, backup_path: str, verify_checksum: bool = True) -> bool:
        """
        Restore a configuration file from backup with integrity verification.
        
        Args:
            backup_path: Path to the backup file
            verify_checksum: Whether to verify backup integrity
            
        Returns:
            True if restoration was successful, False otherwise
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Load backup metadata
            metadata_path = backup_path.with_suffix('.metadata')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    backup_info = BackupInfo(**metadata)
                    original_path = Path(backup_info.original_path)
            else:
                logger.warning(f"No metadata found for backup: {backup_path}")
                # Try to infer original path from backup filename
                original_name = backup_path.stem.rsplit('_', 2)[0] + '.json'
                original_path = self.config_dir / original_name
            
            # Verify backup integrity if requested
            if verify_checksum and metadata_path.exists():
                current_checksum = self.calculate_checksum(backup_path)
                if current_checksum != backup_info.checksum:
                    logger.error(f"Backup integrity check failed for {backup_path}")
                    return False
            
            # Create backup of current file before restoration
            if original_path.exists():
                self.create_backup(original_path, "pre-restoration")
            
            # Restore the file
            shutil.copy2(backup_path, original_path)
            
            # Verify restoration
            if verify_checksum and metadata_path.exists():
                restored_checksum = self.calculate_checksum(original_path)
                if restored_checksum != backup_info.checksum:
                    logger.error(f"Restoration verification failed for {original_path}")
                    return False
            
            logger.info(f"Successfully restored {original_path} from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_path}: {e}")
            return False
    
    def load_configuration(self, config_type: str, decrypt_sensitive: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load and validate a configuration file.
        
        Args:
            config_type: Type of configuration ('default_settings' or 'api_keys')
            decrypt_sensitive: Whether to decrypt sensitive data
            
        Returns:
            Configuration data as dictionary, or None if loading failed
        """
        if config_type == 'default_settings':
            file_path = self.default_settings_path
        elif config_type == 'api_keys':
            file_path = self.api_keys_path
        else:
            logger.error(f"Unknown configuration type: {config_type}")
            return None
        
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return self._create_default_configuration(config_type)
        
        try:
            with self._file_lock(f"load_{config_type}"):
                # Calculate current checksum
                current_checksum = self.calculate_checksum(file_path)
                
                # Load configuration data
                with open(file_path, 'r', encoding='utf-8') as f:
                    if config_type == 'api_keys' and decrypt_sensitive:
                        # Handle encrypted API keys
                        encrypted_data = f.read()
                        if encrypted_data.startswith('{'):
                            # File is not encrypted (backward compatibility)
                            config_data = json.loads(encrypted_data)
                        else:
                            # File is encrypted
                            try:
                                decrypted_data = self.security_manager.decrypt_data(
                                    base64.b64decode(encrypted_data.encode())
                                )
                                config_data = json.loads(decrypted_data.decode('utf-8'))
                            except Exception as e:
                                logger.error(f"Failed to decrypt API keys: {e}")
                                return None
                    else:
                        config_data = json.load(f)
                
                # Validate configuration
                validation_result = self.validate_json_schema(config_data, config_type)
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed for {config_type}: {validation_result.errors}")
                    # Try to restore from backup
                    if self._restore_from_latest_backup(file_path):
                        return self.load_configuration(config_type, decrypt_sensitive)
                    return None
                
                # Update metadata
                metadata = ConfigurationMetadata(
                    file_path=str(file_path),
                    checksum=current_checksum,
                    last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                    version="1.0",
                    encrypted=(config_type == 'api_keys' and decrypt_sensitive)
                )
                self._update_file_metadata(config_type, metadata)
                
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        logger.warning(f"Configuration warning for {config_type}: {warning}")
                
                logger.debug(f"Successfully loaded configuration: {config_type}")
                return config_data
                
        except Exception as e:
            logger.error(f"Failed to load configuration {config_type}: {e}")
            # Try to restore from backup
            if self._restore_from_latest_backup(file_path):
                return self.load_configuration(config_type, decrypt_sensitive)
            return None
    
    def save_configuration(self, config_type: str, config_data: Dict[str, Any], 
                          encrypt_sensitive: bool = True, create_backup: bool = True) -> bool:
        """
        Save configuration data with validation and optional encryption.
        
        Args:
            config_type: Type of configuration ('default_settings' or 'api_keys')
            config_data: Configuration data to save
            encrypt_sensitive: Whether to encrypt sensitive data
            create_backup: Whether to create backup before saving
            
        Returns:
            True if save was successful, False otherwise
        """
        if config_type == 'default_settings':
            file_path = self.default_settings_path
        elif config_type == 'api_keys':
            file_path = self.api_keys_path
        else:
            logger.error(f"Unknown configuration type: {config_type}")
            return False
        
        try:
            with self._file_lock(f"save_{config_type}"):
                # Validate configuration before saving
                validation_result = self.validate_json_schema(config_data, config_type)
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed for {config_type}: {validation_result.errors}")
                    return False
                
                # Create backup if requested and file exists
                if create_backup and file_path.exists():
                    backup_path = self.create_backup(file_path, f"pre-save {config_type}")
                    if not backup_path:
                        logger.warning(f"Failed to create backup before saving {config_type}")
                
                # Prepare data for saving
                if config_type == 'api_keys' and encrypt_sensitive:
                    # Encrypt API keys
                    json_data = json.dumps(config_data, indent=2, default=str)
                    encrypted_data = self.security_manager.encrypt_data(json_data.encode('utf-8'))
                    file_content = base64.b64encode(encrypted_data).decode('ascii')
                    write_mode = 'w'
                    encoding = 'ascii'
                else:
                    # Save as plain JSON
                    file_content = config_data
                    write_mode = 'w'
                    encoding = 'utf-8'
                
                # Write configuration file
                if isinstance(file_content, dict):
                    with open(file_path, write_mode, encoding=encoding) as f:
                        json.dump(file_content, f, indent=2, default=str)
                else:
                    with open(file_path, write_mode, encoding=encoding) as f:
                        f.write(file_content)
                
                # Calculate checksum and update metadata
                current_checksum = self.calculate_checksum(file_path)
                metadata = ConfigurationMetadata(
                    file_path=str(file_path),
                    checksum=current_checksum,
                    last_modified=datetime.now(),
                    version="1.0",
                    encrypted=(config_type == 'api_keys' and encrypt_sensitive)
                )
                self._update_file_metadata(config_type, metadata)
                
                logger.info(f"Successfully saved configuration: {config_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save configuration {config_type}: {e}")
            return False
    
    def _create_default_configuration(self, config_type: str) -> Dict[str, Any]:
        """Create default configuration data for a given type."""
        if config_type == 'default_settings':
            default_config = {
                "application": {
                    "name": "Trading Application",
                    "version": "1.0.0",
                    "auto_start": False,
                    "minimize_to_tray": True
                },
                "security": {
                    "dpapi_encryption": {
                        "enabled": True,
                        "key_rotation_interval": "90d",
                        "backup_encryption_keys": 3
                    },
                    "tls_version": "1.3",
                    "audit_logging": {
                        "enabled": True,
                        "log_level": "INFO",
                        "retention_days": 180
                    }
                },
                "performance": {
                    "ui_response_budget_ms": 100,
                    "security_validation_timeout_ms": 50,
                    "async_logging_enabled": True,
                    "cache_warm_up_enabled": True
                },
                "ui": {
                    "theme": "light",
                    "font_size": 12,
                    "auto_refresh_interval": 30
                },
                "data_sources": {
                    "primary_provider": "yahoo_finance",
                    "fallback_providers": ["alpha_vantage", "iex_cloud"],
                    "rate_limit_management": True
                },
                "risk_management": {
                    "max_position_risk_percent": 2.0,
                    "account_risk_percent": 2.0,
                    "max_positions": 10,
                    "risk_alerts_enabled": True
                },
                "export_settings": {
                    "default_format": "excel",
                    "auto_backup": True,
                    "export_location": "%APPDATA%/TradingApp/exports"
                }
            }
        elif config_type == 'api_keys':
            default_config = {
                "alpha_vantage": {
                    "api_key": "",
                    "provider_name": "Alpha Vantage",
                    "enabled": False,
                    "last_updated": datetime.now().isoformat()
                },
                "iex_cloud": {
                    "api_key": "",
                    "provider_name": "IEX Cloud",
                    "enabled": False,
                    "last_updated": datetime.now().isoformat()
                },
                "polygon_io": {
                    "api_key": "",
                    "provider_name": "Polygon.io",
                    "enabled": False,
                    "last_updated": datetime.now().isoformat()
                }
            }
        else:
            return {}
        
        # Save the default configuration
        self.save_configuration(config_type, default_config)
        logger.info(f"Created default configuration for {config_type}")
        return default_config
    
    def _update_file_metadata(self, config_type: str, metadata: ConfigurationMetadata) -> None:
        """Update metadata for a configuration file."""
        try:
            metadata_file = self.metadata_path
            
            # Load existing metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # Update metadata for this config type
            all_metadata[config_type] = asdict(metadata)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to update metadata for {config_type}: {e}")
    
    def _restore_from_latest_backup(self, file_path: Path) -> bool:
        """Restore a file from the most recent backup."""
        try:
            # Find the most recent backup for this file
            backup_pattern = f"{file_path.stem}_*.backup"
            backup_files = list(self.backup_dir.glob(backup_pattern))
            
            if not backup_files:
                logger.warning(f"No backup files found for {file_path}")
                return False
            
            # Sort by modification time (most recent first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_backup = backup_files[0]
            
            logger.info(f"Attempting to restore {file_path} from {latest_backup}")
            return self.restore_from_backup(str(latest_backup))
            
        except Exception as e:
            logger.error(f"Failed to restore from latest backup for {file_path}: {e}")
            return False


# ============================================================================
# Environment Variable Management
# ============================================================================

class EnvironmentManager:
    """Manages environment variables and .env file operations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the Environment Manager."""
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            self.config_dir = Path(app_data) / 'TradingApp' / 'config'
        
        self.env_file_path = self.config_dir / '.env'
        self._lock = threading.Lock()
        
        # Load environment variables if file exists
        if DOTENV_AVAILABLE and self.env_file_path.exists():
            load_dotenv(self.env_file_path)
    
    def get_env_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable value."""
        return os.environ.get(key, default)
    
    def set_env_variable(self, key: str, value: str, persist: bool = True) -> bool:
        """
        Set an environment variable value.
        
        Args:
            key: Environment variable name
            value: Environment variable value
            persist: Whether to persist to .env file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set in current environment
            os.environ[key] = value
            
            # Persist to .env file if requested
            if persist and DOTENV_AVAILABLE:
                with self._lock:
                    set_key(str(self.env_file_path), key, value)
            
            logger.debug(f"Set environment variable: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set environment variable {key}: {e}")
            return False
    
    def unset_env_variable(self, key: str, persist: bool = True) -> bool:
        """
        Unset an environment variable.
        
        Args:
            key: Environment variable name
            persist: Whether to remove from .env file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from current environment
            os.environ.pop(key, None)
            
            # Remove from .env file if requested
            if persist and DOTENV_AVAILABLE:
                with self._lock:
                    unset_key(str(self.env_file_path), key)
            
            logger.debug(f"Unset environment variable: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unset environment variable {key}: {e}")
            return False
    
    def load_env_file(self, env_file_path: Optional[str] = None) -> bool:
        """
        Load environment variables from a .env file.
        
        Args:
            env_file_path: Optional path to .env file
            
        Returns:
            True if successful, False otherwise
        """
        if not DOTENV_AVAILABLE:
            logger.warning("python-dotenv not available")
            return False
        
        if env_file_path:
            target_path = Path(env_file_path)
        else:
            target_path = self.env_file_path
        
        if not target_path.exists():
            logger.warning(f"Environment file not found: {target_path}")
            return False
        
        try:
            load_dotenv(target_path)
            logger.info(f"Loaded environment variables from: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load environment file {target_path}: {e}")
            return False


# ============================================================================
# Configuration Migration Utilities
# ============================================================================

class ConfigurationMigrator:
    """Handles configuration schema migration and upgrades."""
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize the Configuration Migrator."""
        self.config_manager = config_manager
        self.current_version = "1.0"
        
        # Migration functions for different versions
        self.migrations = {
            "0.9": self._migrate_from_0_9,
            "1.0": self._migrate_from_1_0,
        }
    
    def migrate_configuration(self, config_type: str, from_version: str = None) -> bool:
        """
        Migrate configuration to the current version.
        
        Args:
            config_type: Type of configuration to migrate
            from_version: Source version (auto-detected if None)
            
        Returns:
            True if migration was successful, False otherwise
        """
        try:
            # Load current configuration
            config_data = self.config_manager.load_configuration(config_type, decrypt_sensitive=False)
            if not config_data:
                logger.warning(f"No configuration data found for migration: {config_type}")
                return False
            
            # Detect version if not provided
            if from_version is None:
                from_version = self._detect_version(config_data)
            
            logger.info(f"Migrating {config_type} from version {from_version} to {self.current_version}")
            
            # Apply migrations sequentially
            migrated_data = config_data.copy()
            for version, migration_func in self.migrations.items():
                if self._version_compare(from_version, version) < 0:
                    migrated_data = migration_func(migrated_data, config_type)
                    if migrated_data is None:
                        logger.error(f"Migration failed at version {version}")
                        return False
            
            # Add version information
            migrated_data['_metadata'] = {
                'version': self.current_version,
                'migrated_at': datetime.now().isoformat(),
                'original_version': from_version
            }
            
            # Save migrated configuration
            success = self.config_manager.save_configuration(
                config_type, migrated_data, create_backup=True
            )
            
            if success:
                logger.info(f"Successfully migrated {config_type} to version {self.current_version}")
            else:
                logger.error(f"Failed to save migrated configuration: {config_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"Configuration migration failed for {config_type}: {e}")
            return False
    
    def _detect_version(self, config_data: Dict[str, Any]) -> str:
        """Detect the version of configuration data."""
        # Check for version metadata
        if '_metadata' in config_data and 'version' in config_data['_metadata']:
            return config_data['_metadata']['version']
        
        # Check for version indicators in structure
        if 'security' in config_data and 'dpapi_encryption' in config_data['security']:
            return "1.0"
        
        # Default to earliest version
        return "0.9"
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad to same length
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        return 0
    
    def _migrate_from_0_9(self, config_data: Dict[str, Any], config_type: str) -> Optional[Dict[str, Any]]:
        """Migrate configuration from version 0.9."""
        try:
            if config_type == 'default_settings':
                # Add new security section if missing
                if 'security' not in config_data:
                    config_data['security'] = {
                        "dpapi_encryption": {
                            "enabled": True,
                            "key_rotation_interval": "90d",
                            "backup_encryption_keys": 3
                        },
                        "tls_version": "1.3",
                        "audit_logging": {
                            "enabled": True,
                            "log_level": "INFO",
                            "retention_days": 180
                        }
                    }
                
                # Update performance section
                if 'performance' not in config_data:
                    config_data['performance'] = {}
                
                performance = config_data['performance']
                if 'ui_response_budget_ms' not in performance:
                    performance['ui_response_budget_ms'] = 100
                if 'async_logging_enabled' not in performance:
                    performance['async_logging_enabled'] = True
            
            elif config_type == 'api_keys':
                # Ensure all API key entries have required fields
                for provider, data in config_data.items():
                    if isinstance(data, dict):
                        if 'enabled' not in data:
                            data['enabled'] = False
                        if 'last_updated' not in data:
                            data['last_updated'] = datetime.now().isoformat()
                        if 'provider_name' not in data:
                            data['provider_name'] = provider.replace('_', ' ').title()
            
            return config_data
            
        except Exception as e:
            logger.error(f"Migration from 0.9 failed: {e}")
            return None
    
    def _migrate_from_1_0(self, config_data: Dict[str, Any], config_type: str) -> Optional[Dict[str, Any]]:
        """Migrate configuration from version 1.0 (no changes needed)."""
        return config_data


# ============================================================================
# Main Configuration Interface
# ============================================================================

class Config:
    """
    Main configuration interface for the Trading Application.
    
    Provides a unified interface for all configuration management operations
    including loading, saving, validation, backup/restore, and encryption.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the main configuration interface.
        
        Args:
            config_dir: Optional custom configuration directory
        """
        self.config_manager = ConfigurationManager(config_dir)
        self.env_manager = EnvironmentManager(config_dir)
        self.migrator = ConfigurationMigrator(self.config_manager)
        
        # Cache for frequently accessed configurations
        self._config_cache = {}
        self._cache_timestamps = {}
        self.cache_ttl = timedelta(minutes=5)
    
    def get_default_settings(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get default application settings.
        
        Args:
            use_cache: Whether to use cached configuration
            
        Returns:
            Default settings configuration
        """
        cache_key = 'default_settings'
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.config_manager.load_configuration('default_settings')
        if config:
            self._update_cache(cache_key, config)
            return config
        
        # Return minimal safe defaults if loading fails
        return {
            "application": {"name": "Trading Application"},
            "security": {"dpapi_encryption": {"enabled": True}},
            "performance": {"ui_response_budget_ms": 100}
        }
    
    def get_api_keys(self, decrypt: bool = True, use_cache: bool = False) -> Dict[str, Any]:
        """
        Get API keys configuration.
        
        Args:
            decrypt: Whether to decrypt sensitive data
            use_cache: Whether to use cached configuration (not recommended for sensitive data)
            
        Returns:
            API keys configuration
        """
        cache_key = f'api_keys_decrypt_{decrypt}'
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.config_manager.load_configuration('api_keys', decrypt_sensitive=decrypt)
        if config:
            if use_cache:
                self._update_cache(cache_key, config)
            return config
        
        return {}
    
    def update_default_settings(self, settings: Dict[str, Any], create_backup: bool = True) -> bool:
        """
        Update default application settings.
        
        Args:
            settings: Settings to update
            create_backup: Whether to create backup before update
            
        Returns:
            True if update was successful
        """
        current_settings = self.get_default_settings()
        
        # Deep merge settings
        updated_settings = self._deep_merge(current_settings, settings)
        
        success = self.config_manager.save_configuration(
            'default_settings', updated_settings, create_backup=create_backup
        )
        
        if success:
            self._invalidate_cache('default_settings')
        
        return success
    
    def update_api_key(self, provider: str, api_key: str, enabled: bool = True, 
                      create_backup: bool = True) -> bool:
        """
        Update an API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'alpha_vantage', 'iex_cloud')
            api_key: API key value
            enabled: Whether the provider is enabled
            create_backup: Whether to create backup before update
            
        Returns:
            True if update was successful
        """
        current_keys = self.get_api_keys()
        
        # Update the specific provider
        current_keys[provider] = {
            "api_key": api_key,
            "provider_name": provider.replace('_', ' ').title(),
            "enabled": enabled,
            "last_updated": datetime.now().isoformat()
        }
        
        success = self.config_manager.save_configuration(
            'api_keys', current_keys, encrypt_sensitive=True, create_backup=create_backup
        )
        
        if success:
            self._invalidate_cache('api_keys')
        
        return success
    
    def get_env_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable value."""
        return self.env_manager.get_env_variable(key, default)
    
    def set_env_variable(self, key: str, value: str, persist: bool = True) -> bool:
        """Set an environment variable value."""
        return self.env_manager.set_env_variable(key, value, persist)
    
    def backup_configuration(self, config_type: str = None) -> List[str]:
        """
        Create backups of configuration files.
        
        Args:
            config_type: Specific configuration type, or None for all
            
        Returns:
            List of backup file paths created
        """
        backup_paths = []
        
        if config_type is None:
            # Backup all configurations
            config_types = ['default_settings', 'api_keys']
        else:
            config_types = [config_type]
        
        for cfg_type in config_types:
            if cfg_type == 'default_settings':
                file_path = self.config_manager.default_settings_path
            elif cfg_type == 'api_keys':
                file_path = self.config_manager.api_keys_path
            else:
                continue
            
            backup_path = self.config_manager.create_backup(file_path, f"manual backup {cfg_type}")
            if backup_path:
                backup_paths.append(backup_path)
        
        return backup_paths
    
    def validate_configuration(self, config_type: str = None) -> Dict[str, ValidationResult]:
        """
        Validate configuration files.
        
        Args:
            config_type: Specific configuration type, or None for all
            
        Returns:
            Dictionary of validation results by configuration type
        """
        results = {}
        
        if config_type is None:
            config_types = ['default_settings', 'api_keys']
        else:
            config_types = [config_type]
        
        for cfg_type in config_types:
            config_data = self.config_manager.load_configuration(cfg_type, decrypt_sensitive=False)
            if config_data:
                results[cfg_type] = self.config_manager.validate_json_schema(config_data, cfg_type)
            else:
                results[cfg_type] = ValidationResult(
                    is_valid=False,
                    errors=[f"Failed to load configuration: {cfg_type}"],
                    warnings=[]
                )
        
        return results
    
    def migrate_configuration(self, config_type: str = None) -> bool:
        """
        Migrate configuration files to current version.
        
        Args:
            config_type: Specific configuration type, or None for all
            
        Returns:
            True if all migrations were successful
        """
        if config_type is None:
            config_types = ['default_settings', 'api_keys']
        else:
            config_types = [config_type]
        
        success = True
        for cfg_type in config_types:
            if not self.migrator.migrate_configuration(cfg_type):
                success = False
        
        if success:
            self._clear_cache()
        
        return success
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached configuration is still valid."""
        if cache_key not in self._config_cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key)
        if not timestamp:
            return False
        
        return datetime.now() - timestamp < self.cache_ttl
    
    def _update_cache(self, cache_key: str, config: Dict[str, Any]) -> None:
        """Update configuration cache."""
        self._config_cache[cache_key] = config.copy()
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _invalidate_cache(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        keys_to_remove = [key for key in self._config_cache.keys() if pattern in key]
        for key in keys_to_remove:
            self._config_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._config_cache.clear()
        self._cache_timestamps.clear()
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# ============================================================================
# Module Exports
# ============================================================================

# Main interface for configuration management
__all__ = [
    'Config',
    'ConfigurationManager',
    'EnvironmentManager',
    'SecurityManager',
    'ConfigurationMigrator',
    'ValidationResult',
    'ConfigurationMetadata',
    'BackupInfo'
]

# Create default instance for easy importing
config = Config()

# Convenience functions for common operations
def get_default_settings() -> Dict[str, Any]:
    """Get default application settings."""
    return config.get_default_settings()

def get_api_keys(decrypt: bool = True) -> Dict[str, Any]:
    """Get API keys configuration."""
    return config.get_api_keys(decrypt=decrypt)

def update_api_key(provider: str, api_key: str, enabled: bool = True) -> bool:
    """Update an API key for a specific provider."""
    return config.update_api_key(provider, api_key, enabled)

def get_env_variable(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable value."""
    return config.get_env_variable(key, default)

def set_env_variable(key: str, value: str, persist: bool = True) -> bool:
    """Set an environment variable value."""
    return config.set_env_variable(key, value, persist)

if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    # Initialize configuration
    config_mgr = Config()
    
    print("Configuration Management Utility")
    print("=" * 40)
    
    # Test default settings
    print("\n1. Loading default settings...")
    settings = config_mgr.get_default_settings()
    print(f"Loaded {len(settings)} setting categories")
    
    # Test API keys
    print("\n2. Loading API keys...")
    api_keys = config_mgr.get_api_keys()
    print(f"Loaded {len(api_keys)} API provider configurations")
    
    # Test validation
    print("\n3. Validating configurations...")
    validation_results = config_mgr.validate_configuration()
    for config_type, result in validation_results.items():
        status = "VALID" if result.is_valid else "INVALID"
        print(f"  {config_type}: {status}")
        if result.errors:
            for error in result.errors:
                print(f"    ERROR: {error}")
        if result.warnings:
            for warning in result.warnings:
                print(f"    WARNING: {warning}")
    
    # Test backup
    print("\n4. Creating configuration backups...")
    backup_paths = config_mgr.backup_configuration()
    print(f"Created {len(backup_paths)} backup files")
    for path in backup_paths:
        print(f"  {path}")
    
    print("\nConfiguration management test completed successfully!")