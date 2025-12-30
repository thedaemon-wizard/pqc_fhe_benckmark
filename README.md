# PQC-FHE Integration Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NIST PQC](https://img.shields.io/badge/NIST-PQC%20Standardized-green.svg)](https://csrc.nist.gov/projects/post-quantum-cryptography)

Production-ready framework combining **Post-Quantum Cryptography (PQC)** with **Fully Homomorphic Encryption (FHE)** for enterprise security applications.

## Overview

This platform addresses the emerging threat of quantum computers to current cryptographic systems while enabling privacy-preserving computation on sensitive data.

### Key Features

- **NIST-Standardized PQC**: ML-KEM (FIPS 203), ML-DSA (FIPS 204)
- **Homomorphic Encryption**: CKKS scheme via DESILO FHE
- **Live Data Integration**: VitalDB, Yahoo Finance, Ethereum RPC
- **REST API**: FastAPI with Swagger documentation
- **Web UI**: React-based interactive demonstrations
- **GPU Acceleration**: CUDA 12.x/13.x support

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

### Access

- **Web UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs (Swagger)
- **Health Check**: http://localhost:8000/health

## Supported Algorithms

### Post-Quantum Cryptography (NIST Standardized)

| Algorithm | Type | Security Level | Use Case |
|-----------|------|----------------|----------|
| ML-KEM-512 | KEM | Level 1 | IoT/Embedded |
| ML-KEM-768 | KEM | Level 3 | General Purpose |
| ML-KEM-1024 | KEM | Level 5 | High Security |
| ML-DSA-44 | Signature | Level 2 | Fast Signing |
| ML-DSA-65 | Signature | Level 3 | Balanced |
| ML-DSA-87 | Signature | Level 5 | Maximum Security |

### Fully Homomorphic Encryption

| Parameter | Value | Description |
|-----------|-------|-------------|
| Scheme | CKKS | Approximate arithmetic |
| poly_degree | 16,384 | Ring dimension |
| scale | 2^40 | Encoding precision |
| slot_count | 8,192 | Parallel operations |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/pqc/algorithms` | GET | List available algorithms |
| `/pqc/kem/keypair` | POST | Generate KEM keypair |
| `/pqc/kem/encapsulate` | POST | Encapsulate shared secret |
| `/pqc/kem/decapsulate` | POST | Decapsulate shared secret |
| `/pqc/sig/keypair` | POST | Generate signature keypair |
| `/pqc/sig/sign` | POST | Sign message |
| `/pqc/sig/verify` | POST | Verify signature |
| `/fhe/encrypt` | POST | Encrypt data (CKKS) |
| `/fhe/decrypt` | POST | Decrypt ciphertext |
| `/fhe/add` | POST | Homomorphic addition |
| `/fhe/multiply` | POST | Homomorphic multiplication |
| `/enterprise/healthcare` | GET | Healthcare demo (VitalDB) |
| `/enterprise/finance` | GET | Finance demo (Yahoo Finance) |
| `/enterprise/iot` | GET | IoT demo (UCI Dataset) |
| `/enterprise/blockchain` | GET | Blockchain demo (Ethereum) |

## Enterprise Use Cases

### Healthcare (HIPAA-Compliant Analytics)
Analyze encrypted patient vital signs without exposing PHI. Compute statistics on blood pressure readings while maintaining full regulatory compliance.

### Finance (Confidential Portfolio Analysis)
Perform growth projections on encrypted portfolio values. Client holdings remain confidential during third-party analysis.

### IoT (Secure Smart Grid)
Aggregate encrypted smart meter readings for demand forecasting without accessing individual consumption patterns.

### Blockchain (Quantum-Resistant Transactions)
Migrate from ECDSA to ML-DSA signatures, protecting transaction integrity against future quantum attacks.

## Live Data Sources

| Domain | Source | License |
|--------|--------|---------|
| Healthcare | [VitalDB](https://vitaldb.net/) | CC BY-NC-SA 4.0 |
| Finance | [Yahoo Finance](https://finance.yahoo.com/) | Terms of Service |
| IoT | [UCI ML Repository](https://archive.ics.uci.edu/) | CC BY 4.0 |
| Blockchain | Ethereum RPC (Ankr, PublicNode) | Free/Public |

## Documentation

- [Technical Report (PDF)](docs/PQC_FHE_Technical_Report_v2.3.4.pdf) - 18 pages
- [Technical Report (Word)](docs/PQC_FHE_Technical_Report_v2.3.4.docx) - Editable version
- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI

## Requirements

- Python 3.11+
- liboqs-python (build from source)
- DESILO FHE library
- FastAPI, uvicorn
- Optional: CUDA 12.x/13.x for GPU acceleration(Use at least VRAM 30GB GPU)



## References

1. NIST FIPS 203: ML-KEM Standard (August 2024)
2. NIST FIPS 204: ML-DSA Standard (August 2024)
3. NIST FIPS 205: SLH-DSA Standard (August 2024)
4. Cheon et al. CKKS: Homomorphic Encryption for Approximate Numbers (ASIACRYPT 2017)
5. DESILO FHE Library: https://fhe.desilo.dev/latest/
6. Open Quantum Safe: https://openquantumsafe.org/

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## Author

Created for quantum-safe cryptography research and enterprise security applications.

---

**Note**: This platform is intended for research and development purposes. For production deployment, ensure proper security audits and compliance verification.
