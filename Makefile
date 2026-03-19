# PQC-FHE Integration Portfolio
# Makefile for common development tasks

.PHONY: help install install-dev install-all test lint format clean docker-build docker-run docker-stop api benchmark docs

# Default target
help:
	@echo "PQC-FHE Integration - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make install-all   Install all dependencies"
	@echo "  make install-liboqs Install liboqs from source (requires sudo)"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Check code style"
	@echo "  make format        Format code with black and isort"
	@echo "  make typecheck     Run mypy type checking"
	@echo ""
	@echo "Running:"
	@echo "  make api           Start the REST API server"
	@echo "  make api-dev       Start API in development mode (auto-reload)"
	@echo "  make benchmark     Run performance benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-stop   Stop Docker containers"
	@echo "  make docker-gpu    Build GPU-enabled image"
	@echo "  make compose-up    Start all services with docker-compose"
	@echo "  make compose-down  Stop all docker-compose services"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make docs-serve    Serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make clean-all     Remove all generated files"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

install-liboqs:
	@echo "Installing liboqs from source..."
	@if [ ! -d "liboqs" ]; then \
		git clone --depth 1 --branch 0.10.1 https://github.com/open-quantum-safe/liboqs.git; \
	fi
	cd liboqs && mkdir -p build && cd build && \
		cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON .. && \
		ninja && sudo ninja install && sudo ldconfig
	@echo "liboqs installed successfully"

# =============================================================================
# Development
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	@echo "Running linters..."
	black --check .
	isort --check-only .
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Code formatted successfully"

typecheck:
	mypy pqc_fhe_integration.py api/ --ignore-missing-imports

# =============================================================================
# Running
# =============================================================================

api:
	uvicorn api.server:app --host 0.0.0.0 --port 8000

api-dev:
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	python -c "from benchmarks import run_full_benchmark; run_full_benchmark()"

demo-financial:
	python examples/financial_demo.py

demo-healthcare:
	python examples/healthcare_demo.py

# =============================================================================
# Docker
# =============================================================================

DOCKER_IMAGE = pqc-fhe-api
DOCKER_TAG = latest

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-gpu:
	docker build -f Dockerfile.gpu -t $(DOCKER_IMAGE):gpu .

docker-run:
	docker run -d -p 8000:8000 --name pqc-fhe-api $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-gpu:
	docker run -d --gpus all -p 8000:8000 --name pqc-fhe-api-gpu $(DOCKER_IMAGE):gpu

docker-stop:
	docker stop pqc-fhe-api 2>/dev/null || true
	docker rm pqc-fhe-api 2>/dev/null || true

docker-logs:
	docker logs -f pqc-fhe-api

compose-up:
	docker compose up -d

compose-up-monitoring:
	docker compose --profile monitoring up -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve -a 0.0.0.0:8080

# =============================================================================
# Maintenance
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/

clean-all: clean
	rm -rf .venv/ venv/ liboqs/ benchmark_results/
	docker compose down -v 2>/dev/null || true

# =============================================================================
# Security
# =============================================================================

security-scan:
	@echo "Running security scans..."
	bandit -r . -x ./tests -ll
	safety check

audit:
	pip-audit

# =============================================================================
# Release
# =============================================================================

version:
	@python -c "import pqc_fhe_integration; print(pqc_fhe_integration.__version__ if hasattr(pqc_fhe_integration, '__version__') else '1.0.0')"

build-package:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*
