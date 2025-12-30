# PQC-FHE Integration Platform v2.3.4

**Post-Quantum Cryptography + Fully Homomorphic Encryption Integration**

A production-ready platform combining NIST-standardized post-quantum cryptography with homomorphic encryption for quantum-resistant secure computation.

## What's New in v2.3.4

### Bug Fixes
- **Healthcare Demo**: Fixed numpy array boolean check error (ValueError)
- **Web UI**: Added missing `addLog` function in Healthcare example
- **Ethereum RPC**: Migrated to Ankr as primary endpoint (no API key required)
- **VitalDB**: Improved library-based data fetching with robust fallback

### Live Data Sources
Enterprise examples fetch **real-time data** from external APIs:

| Domain | Data Source | Method | API Key Required |
|--------|-------------|--------|------------------|
| Healthcare | VitalDB Open Dataset | `vitaldb` library | No |
| Finance | Yahoo Finance API | `yfinance` library | No |
| IoT | UCI ML Repository | HTTP download | No |
| Blockchain | Ethereum RPC | Ankr/PublicNode | No |

**Blockchain RPC Endpoints (Fallback Chain):**
1. `https://rpc.ankr.com/eth` (Primary)
2. `https://ethereum-rpc.publicnode.com`
3. `https://cloudflare-eth.com`
4. `https://eth.drpc.org`
5. `https://1rpc.io/eth`

## Features

### Post-Quantum Cryptography (NIST FIPS 203/204)
- **ML-KEM** (Kyber): Key encapsulation - 512/768/1024 variants
- **ML-DSA** (Dilithium): Digital signatures - 44/65/87 variants
- **SLH-DSA** (SPHINCS+): Hash-based signatures (future)

### Fully Homomorphic Encryption (DESILO FHE)
- CKKS scheme for approximate arithmetic
- GPU acceleration via CUDA 12.x/13.x
- Encrypted computation: add, multiply, bootstrap

### Enterprise Use Cases
- **Healthcare**: HIPAA-compliant BP analytics on encrypted patient data
- **Finance**: Confidential portfolio projections without exposing positions
- **IoT**: Privacy-preserving sensor calibration and aggregation
- **Blockchain**: Quantum-resistant transaction signing

## Installation

### Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 8 GB | 32 GB |
| GPU (optional) | CUDA 11.x | CUDA 12.x/13.x |
| Storage | 1 GB | 5 GB |
| OS | Ubuntu 20.04+ | Ubuntu 22.04+ |

### Quick Start

```bash
# 1. Extract and setup
unzip pqc_fhe_portfolio_v2.3.4_final.zip
cd pqc_fhe_v2.3.0
pip install -r requirements.txt
```

### Installing liboqs-python (Required for PQC)

**IMPORTANT**: liboqs-python is NOT available via `pip install`. It must be built from source.

#### Option A: Automatic Build (Recommended)

```bash
# Clone and install - liboqs C library is built automatically
git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python
cd liboqs-python
pip install .
cd ..
```

#### Option B: Manual Build (If Option A fails)

```bash
# Step 1: Install build dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y cmake gcc ninja-build libssl-dev python3-dev

# Step 2: Build liboqs C library
git clone --depth=1 https://github.com/open-quantum-safe/liboqs
cd liboqs
mkdir build && cd build
cmake -GNinja -DBUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install
cd ../..

# Step 3: Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc

# Step 4: Install liboqs-python
git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python
cd liboqs-python
pip install .
cd ..
```

#### Verify Installation

```bash
python3 -c "import oqs; print('liboqs-python version:', oqs.oqs_version())"
```

### Installing DESILO FHE

```bash
# CPU mode
pip install desilofhe

# GPU mode (choose one based on your CUDA version)
pip install desilofhe-cu130   # CUDA 13.0
pip install desilofhe-cu124   # CUDA 12.4
pip install desilofhe-cu121   # CUDA 12.1
```

### Optional: Live Data Libraries

```bash
# For live financial data
pip install yfinance

# For live healthcare data (large download on first use)
pip install vitaldb
```

### Start the Server

```bash
# From project root
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

### Access the Interfaces
- **Web UI**: http://127.0.0.1:8000/ui
- **API Docs (Swagger)**: http://127.0.0.1:8000/docs
- **API Docs (ReDoc)**: http://127.0.0.1:8000/redoc

## API Endpoints

### PQC Key Encapsulation
```bash
# Generate ML-KEM keypair
POST /pqc/kem/keypair
{"algorithm": "ML-KEM-768"}

# Encapsulate shared secret
POST /pqc/kem/encapsulate
{"public_key": "...", "algorithm": "ML-KEM-768"}

# Decapsulate shared secret
POST /pqc/kem/decapsulate
{"secret_key": "...", "ciphertext": "...", "algorithm": "ML-KEM-768"}
```

### PQC Digital Signatures
```bash
# Generate ML-DSA keypair
POST /pqc/sig/keypair
{"algorithm": "ML-DSA-65"}

# Sign message
POST /pqc/sig/sign
{"message": "Hello, quantum-safe world!", "secret_key": "...", "algorithm": "ML-DSA-65"}

# Verify signature
POST /pqc/sig/verify
{"message": "...", "signature": "...", "public_key": "...", "algorithm": "ML-DSA-65"}
```

### FHE Operations
```bash
# Encrypt data
POST /fhe/encrypt
{"data": [120, 125, 118, 122, 130]}

# Compute on encrypted data
POST /fhe/multiply
{"ciphertext_id": "...", "scalar": 0.1}

# Add ciphertexts
POST /fhe/add
{"ciphertext_id_1": "...", "ciphertext_id_2": "..."}

# Decrypt result
POST /fhe/decrypt
{"ciphertext_id": "..."}
```

### Enterprise Data
```bash
# Healthcare (VitalDB vital signs)
GET /enterprise/healthcare

# Finance (Yahoo Finance stock prices)
GET /enterprise/finance

# IoT (UCI sensor data)
GET /enterprise/iot

# Blockchain (Ethereum transactions)
GET /enterprise/blockchain
```

## Algorithm Support

### Key Encapsulation (FIPS 203)
| Algorithm | NIST Level | Public Key | Ciphertext | Security |
|-----------|------------|------------|------------|----------|
| ML-KEM-512 | 1 | 800 B | 768 B | AES-128 equivalent |
| ML-KEM-768 | 3 | 1,184 B | 1,088 B | AES-192 equivalent |
| ML-KEM-1024 | 5 | 1,568 B | 1,568 B | AES-256 equivalent |

### Digital Signatures (FIPS 204)
| Algorithm | NIST Level | Public Key | Signature | Security |
|-----------|------------|------------|-----------|----------|
| ML-DSA-44 | 2 | 1,312 B | 2,420 B | SHA-256 equivalent |
| ML-DSA-65 | 3 | 1,952 B | 3,309 B | AES-192 equivalent |
| ML-DSA-87 | 5 | 2,592 B | 4,627 B | AES-256 equivalent |

## Project Structure

```
pqc_fhe_v2.3.0/
├── api/
│   ├── server.py              # FastAPI server (2400+ lines)
│   ├── live_data_fetcher.py   # External API integration
│   ├── algorithm_config.py    # PQC algorithm definitions
│   └── enterprise_data.py     # Enterprise data module
├── web_ui/
│   └── index.html             # React-based Web UI
├── docs/
│   ├── PQC_FHE_Technical_Report_v2.3.4.pdf  # Technical Report (18 pages)
│   └── generate_report_final.py             # Report generator
├── src/
│   ├── desilo_fhe_engine.py   # FHE engine wrapper
│   └── pqc_fhe_integration.py # Core integration
├── examples/
│   ├── healthcare_demo.py     # Healthcare FHE demo
│   ├── financial_demo.py      # Finance FHE demo
│   ├── iot_demo.py            # IoT FHE demo
│   └── blockchain_demo.py     # Blockchain PQC demo
├── tests/
│   └── test_pqc_fhe.py        # Unit tests
├── README.md
├── CHANGELOG.md
├── requirements.txt
└── Dockerfile
```

## References

### NIST Post-Quantum Cryptography Standards

1. **FIPS 203** - Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM)
   - URL: https://csrc.nist.gov/pubs/fips/203/final
   - Published: August 13, 2024

2. **FIPS 204** - Module-Lattice-Based Digital Signature Algorithm (ML-DSA)
   - URL: https://csrc.nist.gov/pubs/fips/204/final
   - Published: August 13, 2024

3. **FIPS 205** - Stateless Hash-Based Digital Signature Algorithm (SLH-DSA)
   - URL: https://csrc.nist.gov/pubs/fips/205/final
   - Published: August 13, 2024

### Homomorphic Encryption

4. **CKKS Scheme** - Cheon, Kim, Kim, Song (2017)
   - Paper: "Homomorphic Encryption for Arithmetic of Approximate Numbers"
   - DOI: https://doi.org/10.1007/978-3-319-70694-8_15

5. **DESILO FHE Library**
   - Documentation: https://fhe.desilo.dev/latest/
   - GPU acceleration with CUDA support

### Data Sources

6. **VitalDB** - Surgical Patient Vital Signs Database
   - Lee HC, et al. "VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients."
   - Scientific Data 9, 279 (2022)
   - DOI: https://doi.org/10.1038/s41597-022-01411-5
   - License: CC BY-NC-SA 4.0

7. **UCI Machine Learning Repository** - Household Electric Power Consumption
   - Hebrail G, Berard A. "Individual household electric power consumption Data Set"
   - DOI: https://doi.org/10.24432/C52G6F
   - License: CC BY 4.0

8. **Yahoo Finance API**
   - Library: yfinance (https://github.com/ranaroussi/yfinance)
   - License: Apache 2.0
   - Disclaimer: Data for educational/research purposes only

9. **Ethereum Mainnet**
   - Public RPC Endpoints: Ankr, PublicNode, Cloudflare
   - Blockchain: Ethereum Mainnet (Chain ID: 1)

### Implementation Libraries

10. **Open Quantum Safe (OQS) Project**
    - liboqs: https://github.com/open-quantum-safe/liboqs
    - liboqs-python: https://github.com/open-quantum-safe/liboqs-python
    - License: MIT

11. **FastAPI**
    - Documentation: https://fastapi.tiangolo.com/
    - License: MIT

### Clinical Guidelines (Healthcare Demo)

12. **American Heart Association (AHA)** - Blood Pressure Categories
    - URL: https://www.heart.org/en/health-topics/high-blood-pressure

## License

MIT License - See [LICENSE](LICENSE) for details.

**Data Source Licenses:**
- VitalDB: CC BY-NC-SA 4.0 (non-commercial)
- UCI Repository: CC BY 4.0
- Yahoo Finance: Personal use only
- Ethereum: Public blockchain data

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

### v2.3.4 (2025-12-30)
- Fixed Healthcare demo numpy array boolean check error
- Fixed Web UI missing addLog function in Healthcare example
- Added Ankr as primary Ethereum RPC endpoint
- Technical Report v2.3.4 (18 pages) with:
  - Multi-platform installation guide (Debian, Fedora, Arch, macOS)
  - Complete References section (12 academic citations)
  - Proper table formatting with KeepTogether

### v2.3.3 (2025-12-30)
- Fixed VitalDB UTF-8 BOM handling
- Migrated from Etherscan V1 API to Ethereum RPC
- Added 5 redundant RPC endpoints for blockchain data

### v2.3.0 (2025-12-30)
- Real data sources with full academic citations
- Live data fetching from external APIs
- Technical report with 11 sections
