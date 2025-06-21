#!/usr/bin/env python3
"""
Blitzy Trading Platform - Main Application Entry Point

This module serves as the primary entry point for the native Windows desktop trading platform,
replacing the previous Streamlit web interface with a comprehensive PyQt6-based application.

The application provides:
- Native Windows desktop experience with multi-tab interface
- Real-time market scanning across 100+ securities
- Multi-source data integration with failover capabilities
- Risk management and position sizing tools
- Paper trading simulation with P/L tracking
- Trading journal with searchable notes
- Configurable alert system with Windows notifications

Author: Blitzy Development Team
Version: 1.0.0
License: Proprietary
"""

import sys
import os
import logging
import signal
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from decimal import Decimal, getcontext

# PyQt6 imports for desktop application framework
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSystemTrayIcon, QMenu, QMessageBox
)
from PyQt6.QtCore import (
    QTimer, QThread, QThreadPool, QSettings, QStandardPaths, 
    QCoreApplication, pyqtSignal, QObject
)
from PyQt6.QtGui import QIcon, QAction, QPixmap
import qasync
import asyncio

# Application imports (these will be created by other agents)
try:
    from src.gui.main_window import MainWindow
    from src.core.data_integration import DataIntegrationService
    from src.models.database import DatabaseManager
except ImportError as e:
    # During development, modules may not exist yet
    print(f"Warning: Some modules not yet available: {e}")
    MainWindow = None
    DataIntegrationService = None
    DatabaseManager = None


class TradingPlatformApp(QObject):
    """
    Main application controller that coordinates all desktop trading platform components.
    
    This class manages the application lifecycle, initializes core services, and coordinates
    communication between the GUI, data services, and background processing components.
    """
    
    # Application signals for inter-component communication
    shutdown_requested = pyqtSignal()
    data_service_ready = pyqtSignal()
    database_ready = pyqtSignal()
    
    def __init__(self, app: QApplication):
        """
        Initialize the trading platform application.
        
        Args:
            app: The QApplication instance
        """
        super().__init__()
        self.app = app
        self.main_window: Optional[MainWindow] = None
        self.data_service: Optional[DataIntegrationService] = None
        self.database_manager: Optional[DatabaseManager] = None
        self.system_tray: Optional[QSystemTrayIcon] = None
        self.settings: Optional[QSettings] = None
        self.thread_pool: Optional[QThreadPool] = None
        
        # Setup application metadata
        self._setup_application_metadata()
        
        # Setup logging
        self._setup_logging()
        
        # Setup application directories
        self._setup_application_directories()
        
        # Setup configuration management
        self._setup_configuration()
        
        # Setup financial precision
        self._setup_financial_precision()
        
        # Connect shutdown signals
        self._setup_signal_handlers()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trading Platform Application initialized")
    
    def _setup_application_metadata(self) -> None:
        """Configure application metadata for PyQt6."""
        QCoreApplication.setOrganizationName("Blitzy Trading")
        QCoreApplication.setOrganizationDomain("blitzy.com")
        QCoreApplication.setApplicationName("Trading Platform")
        QCoreApplication.setApplicationVersion("1.0.0")
    
    def _setup_logging(self) -> None:
        """Configure comprehensive logging for the application."""
        # Create logs directory
        logs_dir = Path.home() / "AppData" / "Local" / "TradingApp" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_file = logs_dir / "trading_platform.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set up specific loggers for different components
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('PyQt6').setLevel(logging.WARNING)
    
    def _setup_application_directories(self) -> None:
        """Create necessary application directories."""
        app_data_dir = Path.home() / "AppData" / "Local" / "TradingApp"
        
        # Create directory structure
        directories = [
            app_data_dir / "config",
            app_data_dir / "data" / "cache",
            app_data_dir / "data" / "database",
            app_data_dir / "logs",
            app_data_dir / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def _setup_configuration(self) -> None:
        """Initialize configuration management using QSettings."""
        self.settings = QSettings()
        
        # Set default configuration values if not present
        default_config = {
            'scanner/refresh_interval': 30,
            'scanner/max_securities': 100,
            'scanner/default_filters': {
                'min_volume': 1000000,
                'min_price': 1.0,
                'max_price': 1000.0
            },
            'risk_management/max_position_size': 0.1,  # 10% of account
            'risk_management/max_risk_per_trade': 0.02,  # 2% of account
            'alerts/enabled': True,
            'alerts/desktop_notifications': True,
            'ui/theme': 'light',
            'ui/save_window_state': True,
            'api/providers': ['yahoo', 'iex', 'alphavantage'],
            'api/failover_enabled': True,
            'data/cache_expiry_minutes': 5
        }
        
        for key, value in default_config.items():
            if not self.settings.contains(key):
                self.settings.setValue(key, value)
        
        self.settings.sync()
        self.logger.info("Configuration initialized")
    
    def _setup_financial_precision(self) -> None:
        """Configure decimal precision for financial calculations."""
        # Set decimal precision context for financial accuracy
        getcontext().prec = 10  # 10 decimal places for financial precision
        getcontext().rounding = 'ROUND_HALF_UP'  # Standard financial rounding
        
        self.logger.info("Financial precision configured: 10 decimal places")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested.emit()
        QApplication.quit()
    
    async def initialize_async_components(self) -> None:
        """Initialize components that require async setup."""
        try:
            self.logger.info("Initializing async components...")
            
            # Initialize database manager
            if DatabaseManager:
                self.database_manager = DatabaseManager()
                await self.database_manager.initialize()
                self.database_ready.emit()
                self.logger.info("Database manager initialized")
            
            # Initialize data integration service
            if DataIntegrationService:
                config = {
                    'providers': self.settings.value('api/providers', ['yahoo', 'iex', 'alphavantage']),
                    'failover_enabled': self.settings.value('api/failover_enabled', True),
                    'cache_expiry_minutes': self.settings.value('data/cache_expiry_minutes', 5)
                }
                
                self.data_service = DataIntegrationService(config)
                await self.data_service.initialize()
                self.data_service_ready.emit()
                self.logger.info("Data integration service initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing async components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def initialize_ui_components(self) -> None:
        """Initialize the user interface components."""
        try:
            self.logger.info("Initializing UI components...")
            
            # Setup thread pool for background operations
            self.thread_pool = QThreadPool.globalInstance()
            self.thread_pool.setMaxThreadCount(8)  # Optimize for concurrent operations
            
            # Initialize main window
            if MainWindow:
                self.main_window = MainWindow(
                    data_service=self.data_service,
                    database_manager=self.database_manager,
                    settings=self.settings,
                    thread_pool=self.thread_pool
                )
                
                # Connect window signals
                self.main_window.close_requested.connect(self._on_close_requested)
                
                # Restore window state if enabled
                if self.settings.value('ui/save_window_state', True):
                    self._restore_window_state()
                
                self.logger.info("Main window initialized")
            
            # Setup system tray
            self._setup_system_tray()
            
        except Exception as e:
            self.logger.error(f"Error initializing UI components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _setup_system_tray(self) -> None:
        """Setup system tray functionality for background operation."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.logger.warning("System tray not available")
            return
        
        # Create system tray icon
        self.system_tray = QSystemTrayIcon(self.app)
        
        # Set icon (placeholder - will be replaced with actual icon)
        icon = self.app.style().standardIcon(self.app.style().StandardPixmap.SP_ComputerIcon)
        self.system_tray.setIcon(icon)
        
        # Create tray menu
        tray_menu = QMenu()
        
        # Show/Hide action
        show_action = QAction("Show", self.app)
        show_action.triggered.connect(self._show_main_window)
        tray_menu.addAction(show_action)
        
        # Exit action
        exit_action = QAction("Exit", self.app)
        exit_action.triggered.connect(self._exit_application)
        tray_menu.addAction(exit_action)
        
        self.system_tray.setContextMenu(tray_menu)
        self.system_tray.show()
        
        # Connect tray signals
        self.system_tray.activated.connect(self._on_tray_activated)
        
        self.logger.info("System tray initialized")
    
    def _show_main_window(self) -> None:
        """Show and activate the main window."""
        if self.main_window:
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()
    
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle system tray activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_main_window()
    
    def _on_close_requested(self) -> None:
        """Handle main window close request."""
        if self.system_tray and self.system_tray.isVisible():
            # Minimize to tray instead of closing
            self.main_window.hide()
            self.system_tray.showMessage(
                "Trading Platform",
                "Application minimized to system tray",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        else:
            self._exit_application()
    
    def _restore_window_state(self) -> None:
        """Restore saved window state."""
        if self.main_window and self.settings:
            geometry = self.settings.value('window/geometry')
            if geometry:
                self.main_window.restoreGeometry(geometry)
            
            state = self.settings.value('window/state')
            if state:
                self.main_window.restoreState(state)
    
    def _save_window_state(self) -> None:
        """Save current window state."""
        if self.main_window and self.settings:
            self.settings.setValue('window/geometry', self.main_window.saveGeometry())
            self.settings.setValue('window/state', self.main_window.saveState())
            self.settings.sync()
    
    def _exit_application(self) -> None:
        """Perform clean shutdown of the application."""
        self.logger.info("Initiating application shutdown...")
        
        try:
            # Save window state
            if self.settings.value('ui/save_window_state', True):
                self._save_window_state()
            
            # Cleanup async components
            if self.data_service:
                # Note: Actual cleanup will be implemented in DataIntegrationService
                pass
            
            if self.database_manager:
                # Note: Actual cleanup will be implemented in DatabaseManager
                pass
            
            # Hide system tray
            if self.system_tray:
                self.system_tray.hide()
            
            # Close main window
            if self.main_window:
                self.main_window.close()
            
            self.logger.info("Application shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            QApplication.quit()
    
    def run(self) -> None:
        """Run the application with all components initialized."""
        try:
            self.logger.info("Starting Trading Platform Application...")
            
            # Initialize UI components
            self.initialize_ui_components()
            
            # Show main window
            if self.main_window:
                self.main_window.show()
            
            self.logger.info("Application started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            self.logger.error(traceback.format_exc())
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText("Failed to start the Trading Platform application.")
            error_dialog.setDetailedText(str(e))
            error_dialog.exec()
            
            self._exit_application()


async def async_main() -> None:
    """
    Async main function that handles the application lifecycle with proper async/await support.
    
    This function integrates qasync to bridge PyQt6's event loop with asyncio, enabling
    seamless coordination between GUI operations and asynchronous data processing.
    """
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setQuitOnLastWindowClosed(False)  # Keep running when main window is closed
    app.setDesktopSettingsAware(True)     # Use system desktop settings
    
    # Create application instance
    trading_app = TradingPlatformApp(app)
    
    try:
        # Initialize async components first
        await trading_app.initialize_async_components()
        
        # Start the main application
        trading_app.run()
        
        # Keep the event loop running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in main application: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Ensure cleanup
        try:
            trading_app._exit_application()
        except:
            pass


def main() -> None:
    """
    Main entry point for the Trading Platform desktop application.
    
    This function sets up the asyncio event loop integration with PyQt6 and starts
    the application with full async/await support for data operations.
    """
    try:
        # Setup asyncio event loop integration with PyQt6
        import qasync
        
        # Create the application
        app = QApplication(sys.argv)
        
        # Set up the async event loop
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # Run the async main function
        with loop:
            loop.run_until_complete(async_main())
            
    except ImportError as e:
        print(f"Missing required dependencies: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install PyQt6 qasync aiohttp pandas numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', 
                         os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'PyQt6', 'Qt6', 'plugins'))
    
    # Start the application
    main()