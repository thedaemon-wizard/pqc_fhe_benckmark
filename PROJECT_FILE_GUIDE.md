# PQC-FHE Portfolio v2.1.0 - Complete File Guide

## Overview

This document provides a complete guide to all files in the PQC-FHE Integration Portfolio v2.1.0, organized by development week and component.

**Version**: 2.1.0  
**Date**: December 27, 2025  
**Author**: Amon (Quantum Computing Specialist)

## Key Changes in v2.1.0

1. **Fixed Import Inconsistencies**: `PQCFHEIntegration` now properly available as main class alias
2. **Scientifically Computed Chebyshev Coefficients**: All activation function coefficients computed using NumPy polynomial fitting
3. **Updated Documentation**: README and all docs now use correct imports
4. **Added Convenience Functions**: Factory functions for common configurations

---

## Quick Start

```python
# CORRECT IMPORTS (v2.1.0)
from pqc_fhe_integration import PQCFHEIntegration

# Initialize system
system = PQCFHEIntegration()

# PQC operations
kem_pk, kem_sk = system.pqc.generate_kem_keypair()
sig_pk, sig_sk = system.pqc.generate_sig_keypair()

# FHE operations
ct = system.fhe.encrypt([1.0, 2.0, 3.0])
ct_squared = system.fhe.square(ct)
result = system.fhe.decrypt(ct_squared, length=3)
```

---

## File Structure Overview

```
pqc_fhe_portfolio_v2.1.0/
├── pqc_fhe_integration.py     # Main library (v2.1.0)
├── __init__.py                # Package exports
├── README.md                  # Project documentation
├── CHANGELOG.md               # Version history
├── requirements.txt           # Dependencies
├── pyproject.toml             # Package configuration
├── quickstart.py              # Demo script
│
├── src/                       # Source modules
│   ├── pqc_fhe_integration.py # Library copy
│   ├── desilo_fhe_engine.py   # DESILO FHE engine (v2.1.0)
│   ├── chebyshev_coefficients.py  # Coefficient computation
│   └── chebyshev_calculator.py    # Validation tools
│
├── api/                       # REST/WebSocket APIs
│   ├── server.py              # FastAPI server
│   └── websocket_server.py    # WebSocket server
│
├── cli/                       # Command-line interface
│   └── main.py                # CLI commands
│
├── tests/                     # Test suite
│   └── test_pqc_fhe.py        # Unit/integration tests
│
├── examples/                  # Demo applications
│   ├── healthcare_demo.py
│   ├── financial_demo.py
│   ├── blockchain_demo.py
│   └── iot_demo.py
│
├── docs/                      # Documentation
│   ├── api/                   # API reference
│   ├── guides/                # User guides
│   ├── tutorials/             # Tutorials
│   └── security/              # Security docs
│
├── kubernetes/                # K8s deployment
│   └── helm/pqc-fhe/          # Helm charts
│
├── week1_core/                # Week 1 deliverables
├── week2_docs/                # Week 2 deliverables
├── week3_advanced/            # Week 3 deliverables
└── week4_production/          # Week 4 deliverables
```

---

## Week 1: Core Implementation

**Purpose**: Core cryptographic functionality

| File | Size | Description |
|------|------|-------------|
| `pqc_fhe_integration.py` | 51KB | **Main library** - PQC + FHE integration |
| `__init__.py` | 3KB | Package exports with correct aliases |
| `desilo_fhe_engine.py` | 12KB | DESILO FHE engine with computed coefficients |
| `chebyshev_coefficients.py` | 23KB | Scientific coefficient computation |
| `quickstart.py` | 10KB | Quick start demo script |
| `requirements.txt` | 3KB | Python dependencies |
| `pyproject.toml` | 5KB | Package configuration |
| `server.py` | 29KB | FastAPI REST server |
| `websocket_server.py` | 47KB | WebSocket server |
| `cli_main.py` | 44KB | CLI tool |
| `test_pqc_fhe.py` | 28KB | Test suite |
| `benchmarks.py` | 22KB | Performance benchmarks |

### Main Classes (Correct API)

```python
# Main integration class (use either)
from pqc_fhe_integration import PQCFHEIntegration
from pqc_fhe_integration import PQCFHESystem  # Same class

# Component classes
from pqc_fhe_integration import PQCKeyManager
from pqc_fhe_integration import FHEEngine

# Configuration
from pqc_fhe_integration import PQCConfig, FHEConfig, IntegrationConfig

# Enums
from pqc_fhe_integration import SecurityLevel, KEMAlgorithm, SignatureAlgorithm, BootstrapMethod

# Constants
from pqc_fhe_integration import PQC_KEM_ALGORITHMS, PQC_SIGN_ALGORITHMS

# Factory functions
from pqc_fhe_integration import (
    create_default_system,
    create_high_security_system,
    create_gpu_accelerated_system
)
```

---

## Week 2: Documentation Foundation

**Purpose**: API reference and basic guides

| File | Description |
|------|-------------|
| `mkdocs.yml` | MkDocs configuration |
| `index.md` | Documentation home |
| `API.md` | API overview |
| `ARCHITECTURE.md` | System architecture |
| `DESILO_API_COMPLIANCE.md` | DESILO API compliance details |
| `installation.md` | Installation guide |
| `quickstart.md` | Quick start guide |
| `configuration.md` | Configuration guide |
| `pqc_key_exchange.md` | PQC tutorial |
| `fhe_computation.md` | FHE tutorial |
| `hybrid_workflow.md` | Hybrid mode tutorial |
| `api_overview.md` | API components overview |
| `pqc_manager.md` | PQCKeyManager reference |
| `fhe_engine.md` | FHEEngine reference |
| `hybrid_manager.md` | HybridCryptoManager reference |
| `rest_api.md` | REST API reference |
| `cli.md` | CLI reference |
| `security_overview.md` | Security overview |
| `best_practices.md` | Security best practices |

---

## Week 3: Advanced Documentation

**Purpose**: Advanced features and use cases

| File | Description |
|------|-------------|
| `websocket_api.md` | WebSocket API reference |
| `threat_model.md` | Security threat model |
| `nist_compliance.md` | NIST standards compliance |
| `use_cases.md` | Real-world use cases |
| `enterprise_integration.md` | Enterprise integration patterns |
| `benchmarks.md` | Benchmark methodology |

---

## Week 4: Production Guides

**Purpose**: Deployment and operations

| File | Description |
|------|-------------|
| `performance_optimization.md` | Performance tuning |
| `deployment.md` | Production deployment |
| `migration.md` | Migration guide |
| `helm-pqc-fhe/` | Kubernetes Helm charts |
| `Dockerfile.gpu` | GPU-enabled Docker image |
| `prometheus.yml` | Monitoring configuration |
| `SECURITY.md` | Security policy |
| `CONTRIBUTING.md` | Contribution guide |

---

## Chebyshev Coefficients (Scientific Computation)

### Computation Method

```python
import numpy as np
from numpy.polynomial import chebyshev as cheb

def compute_chebyshev_coefficients(func, degree=8, scale=3.0, n_points=10000):
    """
    Compute Chebyshev polynomial coefficients using NumPy fitting.
    
    References:
    - Press et al., "Numerical Recipes", Chapter 5.8
    - Mason & Handscomb (2003), "Chebyshev Polynomials"
    """
    k = np.arange(n_points)
    x_cheb = np.cos(np.pi * (2*k + 1) / (2*n_points))  # Chebyshev nodes
    y = func(scale * x_cheb)
    coeffs = cheb.chebfit(x_cheb, y, degree)
    return coeffs
```

### Computed Coefficients Summary

| Activation | c₀ | Scale | Max Error |
|------------|-----|-------|-----------|
| GELU | 0.8951939622570262 | 3.0 | 2.19e-03 |
| Sigmoid | 0.5000000000000003 | 3.0 | 2.89e-04 |
| Swish | 0.7581537374154663 | 3.0 | 3.69e-04 |
| Tanh | ~0.0 | 3.0 | 1.95e-02 |

### References

1. Hendrycks, D., Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)". arXiv:1606.08415
2. Ramachandran, P., et al. (2017). "Searching for Activation Functions". arXiv:1710.05941
3. Lee, J., et al. (2022). "Privacy-Preserving ML with FHE". arXiv:2106.07229
4. Cheon, J.H., et al. (2018). "A Full RNS Variant of Approximate HE". ePrint 2018/931
5. Kim, M., et al. (2018). "Logistic Regression Model Training based on Approximate HE". BMC Medical Genomics

---

## Installation

```bash
# Clone or extract
cd pqc_fhe_portfolio_v2.1.0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PQC library
pip install liboqs-python

# Optional: Install DESILO FHE
pip install desilofhe

# Run quickstart demo
python quickstart.py
```

---

## API Server

```bash
# REST API
uvicorn api.server:app --host 0.0.0.0 --port 8000

# WebSocket API
python -m api.websocket_server

# Swagger docs: http://localhost:8000/docs
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

---

## Version History

- **v2.1.0** (2025-12-27): Scientific Chebyshev coefficients, import fixes
- **v2.0.0** (2025-12-25): DESILO API compliance update
- **v1.0.0** (2025-12-20): Initial release

---

*Generated: December 27, 2025*
