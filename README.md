# PQC-FHE Integration Platform v3.2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![NIST PQC](https://img.shields.io/badge/NIST-PQC%20Standardized-green.svg)](https://csrc.nist.gov/projects/post-quantum-cryptography)
[![NIST IR 8547](https://img.shields.io/badge/NIST%20IR%208547-PQC%20Migration-orange.svg)](https://csrc.nist.gov/pubs/ir/8547/final)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Helm%20Chart-326ce5.svg)](https://helm.sh/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-e6522c.svg)](https://prometheus.io/)

Production-ready framework combining **Post-Quantum Cryptography (PQC)** with **Fully Homomorphic Encryption (FHE)** for enterprise security applications, featuring quantum threat simulation, security scoring, and multi-party computation.

## What's New in v3.2.0

### 2026 Research-Based Accuracy Corrections
- **BKZ Block Size**: Reference values from NIST FIPS 203/204 security analyses (ML-KEM-768: 633, ML-KEM-1024: 870)
- **CBD Sigma Fix**: Corrected from `eta*sqrt(2/3)` to `sqrt(eta/2)` per FIPS 203
- **Quantum Sieve**: Updated to 0.257 (Dutch team Oct 2025, 8% improvement over Laarhoven 2015)
- **BKZ Improvement**: -3.5 bit correction per Zhao & Ding (2025)
- **Core-SVP Thresholds**: Calibrated to match NIST acceptance criteria

### Multi-Era Shor Resource Estimates
- 4 generations compared: Gidney-Ekera 2021 (20M) → Chevignard 2024 (4M) → Gidney 2025 (1M) → Pinnacle 2026 (100K qubits)
- Extended factorizations: N=143 (11x13), N=221 (13x17) with 8-bit quantum circuits
- Configurable error correction overhead (updated default: 500, down from 1000)

### Side-Channel Risk Assessment (UPDATED)
- ML-KEM: **CRITICAL** - SPA key recovery in 30 seconds (Berzati et al. 2025)
- ML-KEM: **HIGH** - EM fault injection 89.5% success on ARM (2025)
- ML-DSA: **HIGH** - Signing leakage via profiling attacks (2025)
- SLH-DSA: **LOW** - Hash-based design inherently resistant
- CKKS-FHE: **CRITICAL** - Neural network extracts secret key from single NTT trace, 98.6% accuracy (arXiv:2505.11058)
- Mitigation recommendations and implementation hardening status (liboqs, pqm4, OpenFHE v1.5.0, SEAL)

### Noise-Aware Quantum Simulation
- Depolarizing error channels at multiple error rates (10^-3 to 5x10^-2)
- Ideal vs noisy comparison for Grover and QFT circuits
- Error threshold estimation for algorithm reliability

### CKKS / FHE Quantum Security Verification (NEW)
- **Ring-LWE security assessment** for CKKS FHE parameters against HE Standard bounds
- **Lattice monoculture risk**: CKKS shares Ring-LWE with ML-KEM/ML-DSA — single failure point
- **MPC-HE parameter validation**: Default `num_scales=40` at `log_n=15` exceeds 128-bit bound
- **Business impact analysis** per sector (healthcare, finance, IoT, MPC-FHE)
- 7 predefined CKKS configurations verified (Light/Medium/Standard/Heavy/MPC-HE variants)

### Algorithm Diversity & CNSA 2.0
- Algorithm diversity assessment (lattice/hash/code family coverage)
- CNSA 2.0 5-phase readiness assessment (2025-2035)
- Masking verification (liboqs/pqcrystals/pqm4)
- HQC (code-based KEM) integration for lattice diversification

### 13 New API Endpoints (total)
- `GET /quantum/shor-resources/multi-era` - 4-generation Shor resource comparison
- `POST /quantum/simulate/noisy` - Noise-aware quantum simulation
- `GET /security/side-channel/{algorithm}` - Per-algorithm side-channel assessment
- `GET /security/side-channel/all` - All-algorithm side-channel assessment
- `GET /security/algorithm-diversity` - PQC family diversity scoring
- `GET /security/cnsa-readiness` - CNSA 2.0 phase gate assessment
- `GET /security/masking-verification` - SPA masking deployment check
- `GET /quantum/ckks-security` - CKKS Ring-LWE security verification
- `GET /quantum/ckks-security/all-configs` - All CKKS config security comparison
- `GET /security/fhe-quantum-risk` - FHE quantum risk with business context

### Dynamic Version Management (NEW)
- Centralized `version.json` configuration for all module versions
- `src/version_loader.py` shared utility with caching
- All 14 source files dynamically load version from `version.json`
- Eliminates version string hardcoding (fixed api/server.py 2.3.0/2.3.5, pqc_fhe_integration.py 2.1.2)

### GL Scheme Integration (NEW)
- **GL (Gentry-Lee) 5th Generation FHE**: ePrint 2025/1935, DESILO GLEngine API
- **Native matrix multiplication**: O(1) homomorphic operations vs CKKS O(n) rotations
- **GLSchemeEngine wrapper**: encrypt, matrix_multiply, hadamard, transpose, conjugate
- **GL Private Inference**: 2-party inference using GL native matrix operations
- **GL+CKKS Hybrid Engine**: Matrix ops via GL, vector/activation ops via CKKS
- **Security assessment**: GL inherits CKKS NTT side-channel surface (arXiv:2505.11058)
- Referenced: RhombusEnd2End_HEonGPU GPU-accelerated 2PC architecture

### 3 New API Endpoints (GL Scheme)
- `GET /fhe/gl-scheme/info` - GL scheme capabilities and status
- `GET /fhe/gl-scheme/security` - GL security info and known vulnerabilities
- `GET /mpc-he/gl-inference/info` - GL private inference capabilities

### 2026 Research Updates (March 2026, Updated)

#### Critical Security Updates
- **CKKS NTT SPA (CRITICAL)**: Single-trace neural network attack achieves 98.6% key extraction accuracy (arXiv:2505.11058). Random delay insertion INEFFECTIVE. Requires combined masking+shuffling+constant-time NTT or hardware isolation (TEE/SGX).
- **Threshold FHE CPAD (CRITICAL)**: Full key recovery in < 1 hour without smudging noise (CEA 2025). MPC-HE `individual_decrypt()` enforces smudging noise by default.
- **ML-DSA Rejected Signatures Attack (TCHES 2025)**: First practical side-channel key recovery targeting ML-DSA rejection sampling. Private key recovered in seconds with 100% success on ARM Cortex-M4.
- **ML-DSA Public-Based Template Attack (IEEE DATE 2026)**: No clone device needed; templates built from publicly available data only (ePrint 2026/056).
- **GlitchFHE (USENIX Security 2025)**: Fault injection on CKKS targeting non-NTT domain operations in Microsoft SEAL.

#### NIST Standards & Guidance
- **FIPS 204 Errata**: Updated February 23, 2026. Minor corrections to ML-DSA specification.
- **SP 800-227 Finalized**: September 18, 2025. KEM guidance covering composite KEM, KEM-DEM, authenticated key establishment.
- **NIST IR 8547**: Still in draft (IPD Nov 2024). Deprecation targets: 112-bit by 2031, all quantum-vulnerable by 2035.
- **FIPS 206 (FN-DSA/FALCON)**: IPD pending internal clearance. Final standard expected late 2026/early 2027. Compact signatures (~666 bytes).
- **HQC Draft Standard (2026)**: NIST draft expected 2026, final 2027. Code-based backup KEM for ML-KEM.

#### Quantum Hardware Progress
- IBM Quantum: Kookaburra (2026, qLDPC + LPU), Starling (2028-2029, 200 logical qubits), Blue Jay (2033+, 2000+ logical qubits). Flamingo removed from roadmap.
- Google Quantum AI: Quantum Echoes (Oct 2025) 13,000x faster than classical. Verifiable quantum advantage demonstrated.
- **Microsoft Majorana 1** (Feb 2025): First topological qubit chip. 8-qubit processor; millions could fit on a single wafer.
- **Quantinuum Helios** (Nov 2025/Mar 2026): 98 trapped-ion qubits, 94 logical qubits, 2:1 physical-to-logical ratio. $10B valuation.
- **Magic State Distillation**: Optimal scaling achieved (gamma=0, Nature Physics Nov 2025). QuEra/Harvard first experimental demonstration (Nature Jul 2025).

#### Lattice & Quantum Sieving
- **Quantum 3-tuple Sieve**: Exponent 0.3098 → 0.2846 (Dutch team, ePrint 2025/2189). ~8% improvement.
- **BKZ 3-4 bit loss**: Combined BKZ improvements reduce all lattice PQC security by 3-4 bits (Zhao & Ding 2025).
- **Li & Nguyen (JoC 2025)**: First rigorous dynamic analysis of real BKZ algorithm.
- **Dense Sublattice BKZ**: Cryptanalytic no-go confirmed (Ducas & Loyer, CiC 2025).

#### FHE & Migration
- **GL Scheme (FHE.org 2026, Mar 8)**: 5th generation FHE by Gentry & Lee at DESILO. Native matrix multiplication for Private AI.
- **OpenFHE v1.5.0** (Feb 26, 2026): Dev release with BFV/BGV/CKKS/TFHE/LMKCDEY.
- **FHE GPU Acceleration**: Cheddar (ASPLOS 2026, 2-4x over prior GPU), WarpDrive (HPCA 2025, 73% instruction reduction), CAT (2173x speedup on 4090).
- **EU PQC Roadmap** (Jun 2025): Critical infrastructure by 2030, full transition by 2035.
- **UK NCSC**: 3-phase migration (2028/2031/2035).
- **Japan**: 2035 PQC migration deadline aligned with US/EU.
- **Hybrid TLS**: 38% of Cloudflare HTTPS connections use X25519+ML-KEM (Mar 2025).
- **JVG Algorithm (Mar 2026)**: DISMISSED — no valid quantum speedup demonstrated.

#### Browser-Verified (March 2026)
- **151 tests** all passing (Python 3.12.11, Qiskit 2.3.1, Aer 0.17.2)
- Healthcare (5), Finance (5), IoT (10), Blockchain (5), MPC-FHE (9) benchmarks verified
- Shor N=15/143/221, Grover 4-qubit (96.6%), NIST 9 algorithms, Noise simulation all verified in browser
- Side-channel: ML-KEM CRITICAL, ML-DSA HIGH, SLH-DSA LOW, CKKS-FHE CRITICAL, GL-FHE HIGH

## What's New in v3.1.0

### Quantum Algorithm Verification (Qiskit)
- **Shor's Algorithm**: Real quantum circuit simulation for factoring 15, 21, 35 using QFT-based period finding on Qiskit AerSimulator
- **Grover's Algorithm**: Actual amplitude amplification circuits with probability evolution measurements (3-20 qubits)
- Quantum Period Finding with inverse QFT producing real measurement histograms
- Resource extrapolation from small circuits to RSA-2048 and AES-256
- **NIST Security Level Verification**: Lattice BKZ/Core-SVP analysis for ML-KEM and ML-DSA parameter sets

### Sector-Specific Benchmarks
- **Healthcare** (HIPAA): Patient record encryption, FHE vital signs analysis, medical IoT PQC key exchange
- **Finance** (PCI-DSS/SOX): Transaction encryption, ML-DSA trade settlement, key rotation
- **Blockchain**: ML-DSA-44/65/87 signature throughput, batch verification, full TX pipeline
- **IoT**: ML-KEM-512 vs ML-KEM-768 at 64B-4KB payloads, constrained device keygen
- **MPC-FHE**: CKKS engine setup, encrypted computation, 2-party inference protocol

### 5 New API Endpoints
- `POST /quantum/verify/shor` - Real Shor's quantum circuit execution
- `POST /quantum/verify/grover` - Real Grover's quantum circuit execution
- `GET /quantum/verify/nist-levels` - NIST security level lattice verification
- `GET /benchmarks/sector/{sector}` - Per-sector benchmarks
- `GET /benchmarks/sector-all` - All-sector combined benchmarks

## What's New in v3.0.0

### Quantum Threat Simulator
- **Shor's algorithm** resource estimation for RSA-2048/3072/4096 and ECC P-256/P-384/P-521
- **Grover's algorithm** resource estimation for AES-128/192/256 and SHA-256/384/512
- Quantum threat timeline with conservative/moderate/aggressive QPU growth models
- Classical vs PQC vulnerability comparison (Gidney-Ekera 2021 model)

### Security Scoring Framework
- **NIST IR 8547** compliant PQC readiness assessment (0-100 score)
- 5 weighted sub-scores: Algorithm Strength (25%), PQC Readiness (25%), Compliance (20%), Key Management (15%), Crypto Agility (15%)
- Compliance checks for NIST IR 8547, NSA CNSA 2.0, FIPS 140-3, NIST SP 800-57
- Migration plan generation (Phase 1-4: Assessment through Full Migration)
- Sample inventories: enterprise, financial, government

### MPC-HE 2-Party Private Inference
- **DESILO FHE multiparty** API integration (`use_multiparty=True`)
- ALICE (data owner) + BOB (model owner) 2-party protocol
- 4-phase protocol: Key Setup, Encryption, Computation, Decryption
- Chebyshev polynomial activation functions for encrypted neural network inference
- BFV scheme guidance via HEonGPU C++ integration reference

### Extended Benchmarks
- GPU vs CPU FHE performance comparison (RTX 6000 PRO Blackwell 96GB)
- Quantum threat estimation benchmarks
- Security scoring engine benchmarks
- MPC-HE protocol phase-by-phase timing

### 15 New API Endpoints
- Quantum Threat Assessment, Timeline, Shor/Grover simulation
- Security Assessment, Compliance check, Migration plan
- MPC-HE protocol info, demo execution
- Extended benchmark execution

### Previous Versions (v2.3.5)
- Hybrid X25519 + ML-KEM key exchange (IETF draft-ietf-tls-ecdhe-mlkem)
- Kubernetes Helm chart with HPA and GPU workers
- Prometheus monitoring with pre-configured alerts
- File-based logging with rotation

## Key Features

| Feature | Technology | Status |
|---------|------------|--------|
| Post-Quantum KEM | ML-KEM-512/768/1024 (FIPS 203) | Production |
| Post-Quantum Signatures | ML-DSA-44/65/87 (FIPS 204) | Production |
| Hybrid Key Exchange | X25519 + ML-KEM-768 | Production |
| Homomorphic Encryption | CKKS (DESILO FHE) | Production |
| Quantum Threat Simulation | Shor + Grover Estimators | **v3.0.0** |
| Security Scoring | NIST IR 8547 Compliance | **v3.0.0** |
| MPC-HE Inference | 2-Party DESILO Multiparty | **v3.0.0** |
| GPU Acceleration | CUDA 13.0 / RTX 6000 PRO | **v3.0.0** |
| Kubernetes Deployment | Helm Chart | Production |
| Monitoring | Prometheus + Grafana | Production |

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

### Quantum Threat Assessment (v3.0.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/quantum/threat-assessment` | POST | Full quantum threat assessment |
| `/quantum/threat-timeline` | GET | Quantum threat timeline |
| `/quantum/pqc-comparison` | GET | Classical vs PQC comparison |
| `/quantum/shor-simulation/{key_size}` | GET | Shor algorithm simulation |
| `/quantum/grover-simulation/{key_size}` | GET | Grover algorithm simulation |

### Security Scoring (v3.0.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/security/assess` | POST | Security assessment with scoring |
| `/security/compliance/{standard}` | GET | Compliance check (NIST/CNSA/FIPS) |
| `/security/migration-plan` | GET | PQC migration plan |
| `/security/inventory-templates` | GET | Sample crypto inventories |

### MPC-HE Inference (v3.0.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mpc-he/protocol-info` | GET | MPC-HE protocol information |
| `/mpc-he/demo/{demo_type}` | POST | Run MPC-HE demo (linear_regression/classification/statistics) |

### Hybrid Key Exchange

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

### Extended Benchmarks (v3.0.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/benchmarks/quantum-threat` | POST | Quantum threat estimation benchmark |
| `/benchmarks/security-scoring` | POST | Security scoring benchmark |
| `/benchmarks/extended` | POST | Run all extended benchmarks |

### Quantum Verification (v3.1.0 + v3.2.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/quantum/verify/shor` | POST | Real Shor's quantum circuit execution |
| `/quantum/verify/grover` | POST | Real Grover's quantum circuit execution |
| `/quantum/verify/nist-levels` | GET | NIST security level lattice verification |
| `/quantum/shor-resources/multi-era` | GET | 4-generation Shor resource comparison |
| `/quantum/simulate/noisy` | POST | Noise-aware quantum simulation |

### Side-Channel & Sector Benchmarks (v3.1.0 + v3.2.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/security/side-channel/{algorithm}` | GET | Per-algorithm side-channel assessment (incl. HQC) |
| `/security/side-channel/all` | GET | All-algorithm side-channel assessment |
| `/security/algorithm-diversity` | GET | PQC algorithm family diversity (lattice monoculture detection) |
| `/security/cnsa-readiness` | GET | CNSA 2.0 phase gate compliance assessment |
| `/security/masking-verification` | GET | SPA masking countermeasure verification |
| `/quantum/ckks-security` | GET | CKKS Ring-LWE security verification (HE Standard bounds) |
| `/quantum/ckks-security/all-configs` | GET | All CKKS configuration security comparison |
| `/security/fhe-quantum-risk` | GET | FHE deployment quantum risk with business context |
| `/benchmarks/sector/{sector}` | GET | Per-sector benchmarks |
| `/benchmarks/sector-all` | GET | All-sector combined benchmarks |

## Project Structure

```
pqc_fhe_benckmark/
├── api/
│   └── server.py              # FastAPI server (v3.2.0, 64 endpoints)
├── src/
│   ├── pqc_fhe_integration.py # Core PQC + FHE integration
│   ├── pqc_simulator.py       # ML-KEM/ML-DSA educational simulator
│   ├── desilo_fhe_engine.py   # DESILO FHE CKKS engine
│   ├── quantum_threat_simulator.py  # [v3.0.0] Shor/Grover estimator
│   ├── quantum_verification.py     # [v3.1.0] Qiskit circuit verification + noise sim
│   ├── security_scoring.py    # [v3.0.0] NIST IR 8547 scoring
│   ├── sector_benchmarks.py   # [v3.1.0] Sector-specific benchmarks
│   ├── side_channel_assessment.py   # [v3.2.0] Side-channel risk assessment
│   ├── mpc_he_inference.py    # [v3.0.0] MPC-HE 2-party protocol
│   └── version_loader.py      # [v3.2.0] Dynamic version loader
├── benchmarks/
│   └── __init__.py            # Performance benchmarks (incl. GPU)
├── tests/
│   └── test_pqc_fhe.py        # 151 tests (v3.2.0)
├── kubernetes/
│   └── helm/pqc-fhe/          # Helm chart
├── monitoring/
│   └── prometheus.yml         # Prometheus config
├── web_ui/
│   └── index.html             # React Web UI (9 tabs)
├── docs/
│   ├── PQC_FHE_Technical_Report_v3.2.0_Enterprise.docx
│   └── PQC_FHE_Technical_Report_v3.2.0_Enterprise.pdf
├── version.json               # [v3.2.0] Dynamic version configuration
├── logs/                      # Log files (auto-created)
└── README.md
```

## Documentation

- [Technical Report v3.2.0 (PDF)](docs/PQC_FHE_Technical_Report_v3.2.0_Enterprise.pdf)
- [Technical Report v3.2.0 (Word)](docs/PQC_FHE_Technical_Report_v3.2.0_Enterprise.docx)
- [API Documentation](http://localhost:8000/docs) (Swagger UI)
- [CHANGELOG](CHANGELOG.md)

## Development Environment

| Component | Specification |
|-----------|---------------|
| OS | Alma Linux 9.7 |
| CPU | Intel Core i5-13600K |
| RAM | 128GB DDR5 5200 |
| GPU | NVIDIA RTX 6000 PRO Blackwell 96GB |
| Python | 3.12 |
| CUDA | 13.0 |

## Requirements

- Python 3.12+
- liboqs-python (build from source, liboqs 0.15.0)
- cryptography >= 41.0 (for X25519 hybrid)
- desilofhe / desilofhe-cu130 (FHE, optional for GPU)
- numpy >= 1.21
- fastapi >= 0.100, uvicorn >= 0.23
- Kubernetes 1.24+ (for Helm deployment)

## References

1. NIST FIPS 203: ML-KEM Standard (August 2024)
2. NIST FIPS 204: ML-DSA Standard (August 2024)
3. NIST FIPS 205: SLH-DSA Standard (August 2024)
4. NIST IR 8547: Transition to Post-Quantum Cryptography Standards (November 2024)
5. NIST IR 8545: HQC Selection as 4th-round KEM (March 2025)
6. NIST SP 800-227: Recommendations for Key-Encapsulation Mechanisms (2025)
7. NSA CNSA 2.0: ML-KEM-1024 required by 2030 (updated May 2025)
8. IETF draft-ietf-tls-ecdhe-mlkem: Hybrid Key Exchange
9. Gidney & Ekera (2021): How to factor 2048 bit RSA integers in 8 hours
10. Gidney (May 2025): Magic state cultivation — RSA-2048 ~1M physical qubits
11. Pinnacle Architecture (Feb 2026): QLDPC codes — ~100K physical qubits
12. Albrecht et al. (2019): Lattice Estimator methodology (SCN 2018)
13. Chen & Nguyen (2011): BKZ 2.0 lattice security estimates
14. Dutch team / van Hoof et al. (Oct 2025): Quantum sieve exponent 0.257
15. Zhao & Ding (PQCrypto 2025): BKZ improvements, 3-4 bit security reduction
16. Berzati et al. (CHES 2025): ML-KEM SPA key recovery in 30 seconds
17. Grassl et al. (2016): Applying Grover's algorithm to AES
18. Cheon et al. (ASIACRYPT 2017): CKKS Homomorphic Encryption
19. DESILO FHE Library: https://fhe.desilo.dev/
20. Open Quantum Safe liboqs: https://github.com/open-quantum-safe/liboqs
21. arXiv:2505.11058 (May 2025): CKKS NTT neural network side-channel (98.6% key extraction)
22. DESILO GL scheme (ePrint 2025/1935): 5th generation FHE by Gentry & Lee
23. OpenFHE v1.5.0 (Feb 2026): BFV/BGV/CKKS with bootstrapping
24. NIST SP 800-227 (Sep 2025): KEM operational guidance (finalized)
25. BDGL sieve optimality (Jan 2026): NNS paradigm proven optimal for lattice sieving
26. IBM Quantum Roadmap 2026: Kookaburra (4,158 qubits with qLDPC memory)

## Version History

### v3.2.0 (2026-03-19)
- BKZ/Core-SVP accuracy fixes (NIST reference lookup table, CBD sigma, sieve constant 0.257)
- Multi-era Shor resource estimation (20M → 100K physical qubits, 4 generations)
- Side-channel risk assessment: ML-KEM critical (SPA+EM), CKKS-FHE critical (NTT neural network 98.6%)
- Noise-aware quantum simulation (depolarizing error channels)
- CKKS/FHE Ring-LWE security verification against HE Standard bounds
- Dynamic version management via `version.json` (eliminates hardcoded version strings)
- IBM quantum roadmap update: Kookaburra 4,158 qubits, qLDPC real-time decoding
- 13 new API endpoints, 151 tests (up from 88), 50+ academic references

### v3.1.0 (2026-03-18)
- Quantum circuit verification via Qiskit AerSimulator (Shor/Grover/NIST levels)
- Sector-specific benchmarks (Healthcare, Finance, Blockchain, IoT, MPC-FHE)
- 5 new API endpoints, 21 new tests (88 total)

### v3.0.0 (2026-03-18)
- Shor/Grover quantum threat simulator with resource estimation
- NIST IR 8547 security scoring framework (enterprise/financial/government)
- MPC-HE 2-party private inference protocol (DESILO multiparty)
- Extended GPU benchmarks (RTX 6000 PRO Blackwell)
- 15 new API endpoints (Quantum, Security, MPC-HE, Benchmarks)
- 3 new Web UI tabs (Quantum Threat, Security Scoring, MPC-HE Demo)
- Chebyshev polynomial activation functions for encrypted inference
- Comprehensive test suite (65 tests)

### v2.3.5 (2025-12-30)
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

**Security Notice**: This platform implements NIST-standardized post-quantum cryptography (FIPS 203/204/205) with quantum threat assessment, NIST IR 8547 compliance scoring, and privacy-preserving multi-party computation.
