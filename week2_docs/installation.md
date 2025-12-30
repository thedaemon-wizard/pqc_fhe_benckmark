# Installation Guide

This guide covers various installation methods for the PQC-FHE Integration library.

## Requirements

- Python 3.9 or higher
- pip 21.0 or higher

### Optional Dependencies

- **GPU Acceleration**: NVIDIA GPU with CUDA 11.0+ for GPU-accelerated FHE operations
- **Docker**: For containerized deployment
- **Kubernetes**: For orchestrated production deployment

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Install from PyPI
pip install pqc-fhe-integration

# Install with optional dependencies
pip install pqc-fhe-integration[gpu,dev,docs]
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/your-username/pqc-fhe-portfolio.git
cd pqc-fhe-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Method 3: Docker

```bash
# Pull the official image
docker pull pqc-fhe/integration:latest

# Run the container
docker run -p 8000:8000 pqc-fhe/integration:latest
```

### Method 4: Docker Compose

```bash
# Clone the repository
git clone https://github.com/your-username/pqc-fhe-portfolio.git
cd pqc-fhe-portfolio

# Start all services
docker-compose up -d
```

### Method 5: Kubernetes (Helm)

```bash
# Add the Helm repository
helm repo add pqc-fhe https://your-username.github.io/pqc-fhe-charts
helm repo update

# Install the chart
helm install pqc-fhe pqc-fhe/pqc-fhe \
  --namespace pqc-fhe \
  --create-namespace \
  --values custom-values.yaml
```

## Verification

After installation, verify that everything is working:

```python
from pqc_fhe_integration import PQCKeyManager, FHEEngine

# Test PQC
pqc = PQCKeyManager()
pk, sk = pqc.generate_kem_keypair("ML-KEM-768")
print(f"PQC OK: Generated ML-KEM-768 keypair")

# Test FHE
fhe = FHEEngine()
ct = fhe.encrypt([1.0, 2.0, 3.0])
pt = fhe.decrypt(ct)
print(f"FHE OK: Encrypted and decrypted {pt}")
```

Or using the CLI:

```bash
pqc-fhe info
pqc-fhe keygen --algorithm ML-KEM-768 --output ./test-keys
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pqc_fhe_integration'

Ensure the package is installed in your active Python environment:

```bash
pip list | grep pqc-fhe
```

#### GPU not detected

Install the GPU dependencies:

```bash
pip install pqc-fhe-integration[gpu]
```

Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())
```

#### Permission denied on key files

Ensure proper file permissions:

```bash
chmod 600 keys/*_secret.*
chmod 644 keys/*_public.*
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get up and running quickly
- [Configuration](configuration.md) - Configure the library for your environment
- [API Reference](../api/overview.md) - Explore the API documentation
