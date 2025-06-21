# PyQt6 Desktop Application Development Container
# Based on python:3.11-slim for enhanced PyQt6 GUI development
# Supporting Qt6 system dependencies, OpenGL, and Linux display server compatibility

FROM python:3.11-slim

# Metadata
LABEL maintainer="Blitzy Platform Development Team"
LABEL description="PyQt6 desktop application development environment with Qt6 dependencies"
LABEL version="1.0.0"

# Environment variables for development
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/workspace/src"
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for PyQt6 development
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core build tools and development utilities
    build-essential \
    git \
    curl \
    wget \
    unzip \
    # Qt6 development packages for PyQt6 compilation and runtime
    qt6-base-dev \
    qt6-tools-dev \
    qt6-tools-dev-tools \
    libqt6sql6 \
    # OpenGL support for Qt6 graphics rendering and advanced UI components
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    # X11 and display server support for containerized GUI development
    libxkbcommon-x11-0 \
    libxcb1-dev \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libx11-xcb1 \
    libglib2.0-0 \
    libfontconfig1 \
    libdbus-1-3 \
    # Additional dependencies for Qt Designer and resource compilation
    qttools5-dev-tools \
    # System libraries for Python package compilation
    pkg-config \
    cmake \
    # Audio support for potential multimedia features
    libasound2-dev \
    # Network and SSL support
    libssl-dev \
    libffi-dev \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    # Cleanup in same layer to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install Python development tools
RUN python -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspace

# Copy and install Python development dependencies
# Note: requirements-dev.txt will be created by another agent
COPY requirements-dev.txt* ./
RUN if [ -f requirements-dev.txt ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    else \
        echo "requirements-dev.txt not found, installing common development dependencies"; \
        pip install --no-cache-dir \
            # PyQt6 framework and tools
            PyQt6>=6.4.0 \
            PyQt6-tools>=6.4.0 \
            # Development and testing tools
            pytest>=7.0.0 \
            pytest-qt>=4.0.0 \
            pytest-cov>=4.0.0 \
            pytest-mock>=3.10.0 \
            # Code quality tools
            black>=23.0.0 \
            flake8>=6.0.0 \
            mypy>=1.0.0 \
            isort>=5.12.0 \
            # Additional development utilities
            ipython>=8.0.0 \
            jupyter>=1.0.0 \
            # Database and async support
            aiohttp>=3.8.0 \
            qasync>=0.28.0 \
            # Data processing libraries
            pandas>=2.0.0 \
            numpy>=1.24.0 \
            # Financial data providers
            yfinance>=0.2.28 \
            # Export functionality
            openpyxl>=3.1.0 \
            # Configuration management
            pydantic>=1.10.9 \
            python-dotenv>=1.0.0; \
    fi

# Set up development environment
# Create common directories for project structure
RUN mkdir -p /workspace/src \
    && mkdir -p /workspace/tests \
    && mkdir -p /workspace/docs \
    && mkdir -p /workspace/config \
    && mkdir -p /workspace/assets \
    && mkdir -p /workspace/ui \
    && mkdir -p /workspace/data \
    && mkdir -p /workspace/logs

# Configure Git for development (will be overridden by user config)
RUN git config --global --add safe.directory /workspace

# Set proper permissions for workspace
RUN chmod -R 755 /workspace

# Configure Qt platform for headless development
ENV QT_QPA_PLATFORM=offscreen
ENV QT_LOGGING_RULES="*.debug=false"

# Display server configuration for GUI development
ENV DISPLAY=:0
ENV LIBGL_ALWAYS_INDIRECT=1

# Create user for development (optional, can be overridden)
RUN groupadd -r developer && useradd -r -g developer -d /workspace -s /bin/bash developer

# Set ownership of workspace
RUN chown -R developer:developer /workspace

# Development container health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; print(f'Python {sys.version}'); import PyQt6; print('PyQt6 available')" || exit 1

# Default command for development
CMD ["/bin/bash"]

# Development Notes:
# 1. This container supports PyQt6 GUI development with Qt Designer
# 2. OpenGL support is included for advanced graphics rendering
# 3. X11 forwarding is configured for GUI display in containerized environment
# 4. PYTHONPATH is set to /workspace/src for proper module imports
# 5. Common development tools and dependencies are pre-installed
# 6. Container can be used with VS Code Dev Containers extension
# 7. For GUI display, ensure X11 forwarding is configured on host system