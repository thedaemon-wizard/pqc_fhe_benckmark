# PQC-FHE Integration Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NIST PQC](https://img.shields.io/badge/NIST-PQC%20Standardized-green.svg)](https://csrc.nist.gov/projects/post-quantum-cryptography)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Helm%20Chart-326ce5.svg)](https://helm.sh/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-e6522c.svg)](https://prometheus.io/)

Production-ready framework combining **Post-Quantum Cryptography (PQC)** with **Fully Homomorphic Encryption (FHE)** for enterprise security applications.

## What's New in v2.3.5 Complete

🔐 **Hybrid X25519 + ML-KEM Key Exchange**
- Defense-in-depth security combining classical and post-quantum cryptography
- IETF draft-ietf-tls-ecdhe-mlkem compliant

☸️ **Kubernetes Deployment**
- Production-ready Helm chart
- Horizontal Pod Autoscaling (2-10 replicas)
- GPU worker support with NVIDIA device plugin

📊 **Monitoring & Observability**
- Prometheus ServiceMonitor integration
- Pre-configured alerting rules
- Grafana dashboard support

📝 **File-Based Logging**
- Rotating log files (10MB max, 5 backups)
- Separate error and access logs
- Configurable log levels

## Key Features

| Feature | Technology | Status |
|---------|------------|--------|
| Post-Quantum KEM | ML-KEM-768 (FIPS 203) | ✅ Production |
| Post-Quantum Signatures | ML-DSA-65 (FIPS 204) | ✅ Production |
| Hybrid Key Exchange | X25519 + ML-KEM-768 | ✅ Production |
| Homomorphic Encryption | CKKS (DESILO FHE) | ✅ Production |
| Kubernetes Deployment | Helm Chart | ✅ Production |
| Monitoring | Prometheus + Grafana | ✅ Production |
| File Logging | RotatingFileHandler | ✅ Production |

## Quick Start

### Prerequisites

```bash
# Debian/Ubuntu
sudo apt install -y cmake gcc g++ libssl-dev python3-dev git

# Fedora/RHEL
sudo dnf install -y cmake gcc gcc-c++ openssl-devel python3-devel git

# macOS
brew install cmake openssl@3 git
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/thedaemon-wizard/pqc_fhe_benckmark.git
cd pqc_fhe_benckmark

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install liboqs-python (required for PQC)
## 3.1  Install liboqs from github
git clone --depth=1 https://github.com/open-quantum-safe/liboqs
cmake -S liboqs -B liboqs/build -DBUILD_SHARED_LIBS=ON
cmake --build liboqs/build --parallel 8
cmake --build liboqs/build --target install


## 3.2 Install liboqs-python from github
git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python
cd liboqs-python && pip install . && cd ..

# 4. Install DESILO FHE
pip install desilofhe          # CPU mode
# pip install desilofhe-cu130  # GPU mode (CUDA 13.0)

# 5. Optional: Live data libraries
pip install yfinance vitaldb

# 6. Start server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Access Points

- **Web UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics
- **Health**: http://localhost:8000/health

## Kubernetes Deployment

### Helm Chart Installation

```bash
# Add dependencies
helm dependency update ./kubernetes/helm/pqc-fhe

# Install
helm install pqc-fhe ./kubernetes/helm/pqc-fhe \
  --namespace pqc-fhe --create-namespace

# With GPU workers
helm install pqc-fhe ./kubernetes/helm/pqc-fhe \
  --set gpuWorker.enabled=true \
  --set gpuWorker.replicaCount=2
```

### Helm Chart Features

| Feature | Description |
|---------|-------------|
| **Auto-scaling** | HPA with 2-10 replicas |
| **GPU Workers** | NVIDIA device plugin support |
| **Redis Cache** | Optional caching layer |
| **Prometheus** | ServiceMonitor + alerts |
| **NetworkPolicy** | Security isolation |
| **PodDisruptionBudget** | High availability |
| **Ingress** | TLS termination |

### Key Configuration Values

```yaml
api:
  replicaCount: 2
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
  autoscaling:
    enabled: true
    maxReplicas: 10

gpuWorker:
  enabled: false
  resources:
    limits:
      nvidia.com/gpu: 1

crypto:
  pqc:
    kemAlgorithm: ML-KEM-768
    signatureAlgorithm: ML-DSA-65
  fhe:
    useBootstrap: true
    mode: cpu

prometheus:
  enabled: true
```

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request latency |
| `fhe_encryption_duration_seconds` | Histogram | FHE encrypt time |
| `pqc_keygen_duration_seconds` | Histogram | Key generation |

### Pre-configured Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| PQCFHEHighErrorRate | Error rate > 5% | Critical |
| PQCFHEHighLatency | p95 > 5s | Warning |
| PQCFHEPodNotReady | Pods not ready | Warning |
| PQCFHESlowEncryption | Encrypt > 10s | Warning |

## Logging

### Log Files

| File | Max Size | Backups | Content |
|------|----------|---------|---------|
| `logs/pqc_fhe_server.log` | 10 MB | 5 | All server logs |
| `logs/pqc_fhe_error.log` | 10 MB | 3 | Errors only |
| `logs/pqc_fhe_access.log` | 10 MB | 5 | HTTP access |

### Configuration

```bash
# Set log level via environment variable
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
python -m uvicorn api.server:app
```

### Log Format

```
# File format (includes source location)
2025-12-30 12:00:00 - api.server - INFO - [server.py:123] - Message

# Console format
2025-12-30 12:00:00 - api.server - INFO - Message
```

## Hybrid Migration Strategy

| Phase | Timeline | Strategy | Algorithms |
|-------|----------|----------|------------|
| 1. Assessment | 2024-2025 | Inventory | RSA, ECDSA, X25519 |
| **2. Hybrid** | **2025-2027** | **Deploy hybrid** | **X25519 + ML-KEM-768** |
| 3. PQC Primary | 2027-2030 | PQC first | ML-KEM-768 |
| 4. PQC Only | 2030-2035 | Full migration | ML-KEM-1024 |

## API Endpoints

### Hybrid Key Exchange (v2.3.5)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pqc/hybrid/keypair` | POST | Generate hybrid keypair |
| `/pqc/hybrid/encapsulate` | POST | Hybrid encapsulation |
| `/pqc/hybrid/decapsulate` | POST | Hybrid decapsulation |
| `/pqc/hybrid/compare` | GET | Algorithm comparison |
| `/pqc/hybrid/migration-strategy` | GET | Migration roadmap |

### PQC Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pqc/kem/keypair` | POST | Generate ML-KEM keypair |
| `/pqc/kem/encapsulate` | POST | Encapsulate secret |
| `/pqc/sig/sign` | POST | Sign message |
| `/pqc/sig/verify` | POST | Verify signature |

### FHE Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fhe/encrypt` | POST | Encrypt data |
| `/fhe/decrypt` | POST | Decrypt ciphertext |
| `/fhe/add` | POST | Homomorphic addition |
| `/fhe/multiply` | POST | Homomorphic multiplication |

## Project Structure

```
pqc_fhe_benckmark/
├── api/
│   └── server.py              # FastAPI server (v2.3.5)
├── kubernetes/
│   └── helm/pqc-fhe/          # Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── hpa.yaml
│           ├── servicemonitor.yaml
│           └── ...
├── monitoring/
│   └── prometheus.yml         # Prometheus config
├── web_ui/
│   └── index.html             # React Web UI
├── docs/
│   ├── PQC_FHE_Technical_Report_v2.3.5_Complete.pdf
│   └── PQC_FHE_Technical_Report_v2.3.5_Complete.docx
├── logs/                      # Log files (auto-created)
│   ├── pqc_fhe_server.log
│   ├── pqc_fhe_error.log
│   └── pqc_fhe_access.log
└── README.md
```

## Documentation

- [Technical Report (PDF)](docs/PQC_FHE_Technical_Report_v2.3.5_Complete.pdf)
- [Technical Report (Word)](docs/PQC_FHE_Technical_Report_v2.3.5_Complete.docx)
- [API Documentation](http://localhost:8000/docs)
- [CHANGELOG](CHANGELOG.md)

## Requirements

- Python 3.9+
- liboqs-python (build from source)
- cryptography (for X25519)
- desilofhe (FHE library)
- Kubernetes 1.24+ (for Helm deployment)

## References

1. NIST FIPS 203: ML-KEM Standard (August 2024)
2. NIST FIPS 204: ML-DSA Standard (August 2024)
3. IETF draft-ietf-tls-ecdhe-mlkem: Hybrid Key Exchange
4. NIST IR 8547: PQC Migration Guidelines
5. DESILO FHE Library: https://fhe.desilo.dev/
6. Kubernetes Helm: https://helm.sh/docs/

## Version History

### v2.3.5 Complete (2025-12-30)
- X25519 + ML-KEM hybrid key exchange
- Kubernetes Helm chart with GPU support
- Prometheus monitoring and alerting
- File-based logging with rotation
- Web UI Hybrid Migration tab

### v2.3.4 (2025-12-30)
- Fixed numpy array handling
- Enhanced live data fetching

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Security Notice**: This platform implements NIST-standardized post-quantum cryptography with Kubernetes deployment support and comprehensive monitoring.
