# PQC-FHE Integration Portfolio
# Multi-stage Docker build for production deployment
#
# Build: docker build -t pqc-fhe-api .
# Run:   docker run -p 8000:8000 pqc-fhe-api

# =============================================================================
# Stage 1: Build liboqs from source
# =============================================================================
FROM python:3.11-slim as liboqs-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    gcc \
    g++ \
    git \
    ninja-build \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build liboqs
WORKDIR /build
RUN git clone --depth 1 --branch 0.10.1 https://github.com/open-quantum-safe/liboqs.git && \
    cd liboqs && \
    mkdir build && cd build && \
    cmake -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DOQS_BUILD_ONLY_LIB=ON \
        .. && \
    ninja && \
    ninja install

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="PQC-FHE Portfolio"
LABEL version="1.0.0"
LABEL description="Post-Quantum Cryptography + FHE REST API"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy liboqs from builder
COPY --from=liboqs-builder /usr/local/lib/liboqs* /usr/local/lib/
COPY --from=liboqs-builder /usr/local/include/oqs /usr/local/include/oqs
RUN ldconfig

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    pydantic>=2.0.0 \
    cryptography>=41.0.0 \
    liboqs-python>=0.10.0

# Copy application code
COPY pqc_fhe_integration.py .
COPY api/ ./api/
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
