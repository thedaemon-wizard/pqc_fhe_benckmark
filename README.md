# PQC-FHE Integration Platform v3.5.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![NIST PQC](https://img.shields.io/badge/NIST-PQC%20Standardized-green.svg)](https://csrc.nist.gov/projects/post-quantum-cryptography)
[![NIST IR 8547](https://img.shields.io/badge/NIST%20IR%208547-PQC%20Migration-orange.svg)](https://csrc.nist.gov/pubs/ir/8547/final)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Helm%20Chart-326ce5.svg)](https://helm.sh/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-e6522c.svg)](https://prometheus.io/)

Production-ready framework combining **Post-Quantum Cryptography (PQC)** with **Fully Homomorphic Encryption (FHE)** for enterprise security applications, featuring quantum threat simulation, security scoring, and multi-party computation.

## Documentation

- [Technical Report v3.5.0 (PDF)](docs/PQC_FHE_Technical_Report_v3.5.0_Enterprise.pdf)
- [Technical Report v3.5.0 (Word)](docs/PQC_FHE_Technical_Report_v3.5.0_Enterprise.docx)
- [CHANGELOG](CHANGELOG.md)
- [Summary Page(infographic)](https://thedaemon-wizard.github.io/pqc_fhe_benckmark/infographic.html)


## What's New in v3.5.0

### Accurate Hardware Discovery (FIX)
- **ibm_torino corrected**: Heron r1 (133Q, Dec 2023), was incorrectly listed as Heron R2 (156Q)
- **3 new Heron r2 backends**: `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` (156Q each, Jul 2024)
- **KNOWN_PROCESSORS expanded**: 3 → 6 backends with correct processor types and coherence times
- **HERON_R2_FALLBACK fixed**: Now references `ibm_fez` (actual Heron r2) instead of `ibm_torino`

### Benchmark Results Persistence (NEW)
- **BenchmarkResultsManager**: Saves benchmark results as timestamped JSON files
- **Circuit diagram auto-save**: Shor/Grover/ECC diagram endpoints persist PNG files
- **3 new API endpoints**: `GET /benchmarks/results`, `GET /benchmarks/results/{filename}`, `GET /benchmarks/diagrams`
- **All-Sector circuit diagrams**: Button 5 now displays circuit diagrams (previously only Button 4)

### 2026 PQC Research Updates
- **NIST IR 8547**: Classical algorithms deprecated by 2030, disallowed by 2035
- **HQC selection**: 5th NIST PQC algorithm (code-based KEM, March 2025) — lattice monoculture mitigation
- **CRQC estimate compression**: 20M (2021) → 1M (2025) → ~100K physical qubits (QLDPC, Feb 2026)
- **Hybrid TLS default**: ML-KEM + X25519 now default in Chrome, Firefox, Cloudflare, Akamai
- **AI-assisted side-channel**: Single-trace key recovery attacks on ML-KEM demonstrated (2026)

### Infrastructure & Monitoring (NEW)
- **Prometheus `/metrics` endpoint**: Zero-dependency exposition format (uptime, memory, HTTP stats, GC, app info)
- **Docker image**: `pqc-fhe-api:v3.5.0` (420MB) — multi-stage build with liboqs 0.14.0
- **Docker Compose monitoring**: API + Prometheus + Grafana with `--profile monitoring`
- **Helm chart validated**: `helm lint` passed, `helm template` renders 8 Kubernetes manifests
- **Shor test stabilization**: Probabilistic algorithm retry (max 3 attempts) for N=143, N=221

### 17 New Tests (221 → 238 total)
- Heron r1/r2 correctness: ibm_torino=r1(133Q), ibm_fez/kingston/marrakesh=r2(156Q)
- BenchmarkResultsManager: save/load/list, path traversal prevention, PNG save

## What's New in v3.4.0

### Dynamic QPU Backend Discovery
- **Server startup discovery**: Background thread connects to IBM Quantum API and discovers all operational QPU backends
- **JSON file cache**: Dynamically fetched backends persisted to `data/ibm_backends_cache.json` for offline fallback
- **3-tier fallback chain**: IBM Quantum Runtime API → JSON cache → KNOWN_PROCESSORS (hardcoded)
- **Processor-specific basis_gates**: Heron=CZ, Eagle=ECR, Nighthawk=CZ
- 15 new tests (206 → 221 total)

## What's New in v3.3.0

### IBM Quantum Hardware Noise Integration
- **Dynamic QPU backend discovery**: Fetches real T1/T2/gate errors/readout errors from IBM Quantum Platform via `qiskit-ibm-runtime`
- **IBM Heron r1 (ibm_torino, 133Q)** and **Heron r2 (ibm_fez, 156Q)**: CZ-based processors with published noise parameters
- **Fallback-safe design**: Uses published QPU specs when API unavailable (6 backends: torino, brisbane, sherbrooke, fez, kingston, marrakesh)
- **NoiseModel construction**: Thermal relaxation + depolarizing + readout errors from real hardware parameters
- **WebUI backend selector**: Choose noise model for circuit benchmarks (Default/IBM Heron r2/discovered QPU backends)
- **IBM QPU noise comparison**: Side-by-side sector profile vs IBM QPU noise fidelity in benchmark results

### FHE Bootstrap Key Memory Optimization
- **Deferred bootstrap loading**: `defer_bootstrap=True` skips ~24GB key creation at server startup
- **Server memory reduced**: ~28GB → ~3.7GB (core keys only) at startup
- **On-demand creation**: `ensure_bootstrap_keys()` creates keys when needed, auto-triggered by bootstrap operations
- **Memory release**: `release_bootstrap_keys()` frees ~24GB with garbage collection
- **API endpoints**: `GET /fhe/memory-status` and `POST /fhe/release-bootstrap-keys` for memory management

### Agentic AI × PQC Research
- wolfSSL SLIM (2025): MLS-based PQ channel binding for AI agents
- IETF draft-mpsb-agntcy-messaging-01: Multi-agent PQ messaging protocol
- IBM Pinnacle Architecture (Feb 2026): qLDPC codes, RSA-2048 ~100K physical qubits
- Google Quantum AI revised Q-Day to ~2029
- Q-Fusion diffusion model: Circuit layout from natural language
- NVIDIA GQE: GPU-accelerated quantum error decoding

### 6 API Endpoints (v3.3.0)
- `GET /quantum/ibm/backends` - List IBM QPU backends
- `GET /quantum/ibm/backend/{name}/noise-params` - QPU noise parameters
- `POST /quantum/simulate/noisy-ibm` - IBM QPU noise model simulation
- `GET /fhe/memory-status` - FHE bootstrap memory status
- `POST /fhe/release-bootstrap-keys` - Release bootstrap keys

### 28 New Tests (178 → 206 total)
- IBM Quantum backend: 14 tests (fallback, cache, NoiseModel, profile compatibility)
- FHE bootstrap deferred: 8 tests (defer, ensure, release, encrypt/decrypt without bootstrap)
- FHE bootstrap config: 3 tests
- Sector circuit IBM noise: 3 tests

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

### Real Quantum Circuit Sector Benchmarks (NEW)
- **Actual Qiskit circuit execution** per sector — not mathematical estimates
- **Shor's algorithm circuits**: N=15, 21, 35 real QFT-based factoring on AerSimulator
- **ECC discrete log circuits**: GF(2^4) quantum period finding + P-256/P-384/Ed25519 extrapolation
- **Grover's search circuits**: 4-16 qubit real amplitude amplification on AerSimulator
- **Regev vs Shor comparison**: O(n^{3/2}) gates vs O(n^2 log n) resource analysis (JACM 2025)
- **Enhanced noise models**: 5 sector-specific profiles (medical IoT, datacenter, adversarial, constrained device, lattice correlated)
- **GPU acceleration**: RTX 6000 PRO Blackwell 96GB via cuStateVec (32 qubits max)
- **HNDL circuit demonstration**: Shor proof-of-concept attack sequence
- **Quantum Security Infographic**: Standalone HTML with CSS animations (`docs/infographic.html`)

### Qiskit Pass Manager Circuit Optimization (NEW)
- **`generate_preset_pass_manager()`** replaces `transpile()` per IBM Quantum Learning recommendations
- **Shor circuits**: `optimization_level=2` (IBM Quantum Learning Shor's algorithm recommendation)
- **Grover circuits**: `optimization_level=3` (IBM Quantum Learning Grover's algorithm — deep circuits)
- **ECC/Noise circuits**: `optimization_level=2` with standard basis gate set
- **optimization_info**: Original vs optimized gate count/depth with reduction percentage in all results
- **Circuit visualization**: `generate_circuit_diagram()` renders Qiskit circuits as base64 PNG diagrams
- **WebUI diagrams**: Shor N=15, Grover 4-qubit, ECC GF(2^4) circuit diagrams displayed after benchmark

### 30 API Endpoints (total, v3.2.0 + v3.3.0)
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
- `POST /benchmarks/sector/{sector}/circuit-benchmark` - Per-sector real Qiskit circuit benchmark
- `POST /benchmarks/sector-all/circuit-benchmark` - All 5 sectors circuit comparison
- `POST /quantum/circuit/shor-demo` - Shor factoring real circuit demo
- `POST /quantum/circuit/ecc-dlog-demo` - ECC discrete log circuit demo
- `POST /quantum/circuit/grover-demo` - Grover search real circuit demo
- `GET /quantum/circuit/regev-comparison` - Regev vs Shor resource comparison
- `GET /quantum/circuit/gpu-status` - GPU/CPU quantum simulation backend status
- `GET /quantum/circuit/shor-diagram` - Shor circuit diagram (base64 PNG)
- `GET /quantum/circuit/grover-diagram` - Grover circuit diagram (base64 PNG)
- `GET /quantum/circuit/ecc-diagram` - ECC discrete log circuit diagram (base64 PNG)

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

### Sector Quantum Security Simulator (NEW)
- **Per-sector quantum security analysis** for 5 industry sectors: Healthcare (HIPAA), Finance (PCI-DSS/CNSA 2.0), Blockchain, IoT/Edge, MPC-FHE
- **7 simulation types per sector**:
  - Shor vs RSA/ECC: 4-generation qubit estimates for current keys
  - Shor vs Hybrid (RSA+PQC): Transitional security analysis
  - Shor vs PQC Primary (ML-KEM/ML-DSA): Post-migration lattice security
  - Shor vs PQC Only: Full migration residual risks (lattice monoculture, CPAD)
  - Grover vs AES-128: 64-bit PQ security — insufficient for CNSA 2.0
  - Grover vs AES-256: 128-bit PQ security — quantum-safe
  - HNDL threat window: Data retention vs Q-Day scenarios
- **Migration urgency scoring** (0-100): SNDL risk (30%), compliance proximity (25%), side-channel (20%), FHE lattice risk (15%), data retention (10%)
- **Cross-sector comparison**: Urgency ranking, HNDL critical sector identification
- **Key findings**: Healthcare CRITICAL (87/100, 45yr HNDL exposure), Blockchain HIGH (73/100, 994yr HNDL), Finance HIGH (70.5/100, CNSA 2.0 2030 deadline)

### 2 New API Endpoints (Sector Quantum Security)
- `GET /benchmarks/sector/{sector}/quantum-security` - Single sector full simulation (Shor×4 + Grover×2 + HNDL)
- `GET /benchmarks/sector-all/quantum-security` - All 5 sectors with cross-sector comparison and urgency ranking

### 2026 Research Updates (March 2026, Updated)

#### Critical Security Updates
- **CKKS NTT SPA (CRITICAL)**: Single-trace neural network attack achieves 98.6% key extraction accuracy (arXiv:2505.11058). Random delay insertion INEFFECTIVE. Requires combined masking+shuffling+constant-time NTT or hardware isolation (TEE/SGX).
- **Threshold FHE CPAD (CRITICAL)**: Full key recovery in < 1 hour without smudging noise (CEA 2025). MPC-HE `individual_decrypt()` enforces smudging noise by default.
- **CPAD Impossibility for HELLHO schemes** (ePrint 2026/203): Proves NO BFV/BGV/CKKS basic variant can achieve IND-CPA^D security. Fundamental limitation.
- **ML-DSA Rejected Signatures Attack (TCHES 2025)**: Rejection sampling leakage enables full key recovery (TCHES Vol. 2025 No. 4, pp. 817-847).
- **ML-DSA**: 6+ attack papers in 2025-2026 — rejection sampling (TCHES 2025), factor-graph (ePrint 2025/582), masked y leakage (ePrint 2025/276), hardware CPA (HOST 2025), implicit hint (SAC 2025), template attack (DATE 2026, ePrint 2026/056).
- **GlitchFHE (USENIX Security 2025)**: Single corrupted RNS limb breaks FHE confidentiality. Covers CKKS and BFV on SEAL.
- **SNDL/HNDL Active Threat**: DHS, UK NCSC, ENISA, Australian ACSC all confirm adversaries currently harvesting encrypted data.

#### NIST Standards & Guidance
- **FIPS 203/204 Errata**: Both updated February 2026. Minor corrections.
- **SP 800-227 Finalized**: September 18, 2025. Covers composite KEM (X-Wing: ML-KEM + X25519), ephemeral key one-time use.
- **NIST IR 8547**: Still in draft (IPD Nov 2024). Deprecation targets: 112-bit by 2031, all quantum-vulnerable by 2035.
- **FIPS 206 (FN-DSA/FALCON)**: IPD pending internal clearance. Final standard expected late 2026/early 2027. Compact signatures (~666 bytes).
- **HQC Draft Standard**: Expected early 2026, final 2027. Code-based backup KEM for ML-KEM (NIST IR 8545).
- **CSWP 39 (Crypto Agility)**: Finalized December 19, 2025. Maturity model, CBOM practices, policy-mechanism separation.
- **CSWP 48 (PQC Migration Mappings)**: IPD September 2025. Maps to CSF 2.0 and SP 800-53.
- **Additional Signature Onramp Round 2**: 14 candidates — CROSS, FAEST, HAWK, LESS, MAYO, Mirath, MQOM, PERK, QR-UOV, RYDE, SDitH, SNOVA, SQIsign, UOV. Round 3 down-select in 2026.

#### Quantum Hardware Progress
- **IBM Kookaburra** (Mar 2026): 1,386-qubit processor, 4,158 qubits via 3-chip link. qLDPC codes reduce overhead up to 90%. Starling (2029, 200 logical qubits, 100M gates), Blue Jay (2033, 2000+ logical qubits).
- Google Quantum AI: Willow (105 qubits, below-threshold QEC). Quantum Echoes (Oct 2025) 13,000x faster than classical.
- **Microsoft Majorana 1** (Feb 2025): 8 topological qubits via topoconductors. Unvalidated — 2018 paper retracted. High-risk/high-reward.
- **Quantinuum Helios** (Nov 2025): 98 physical qubits → 94 logical GHZ, 48 fully error-corrected (2:1), 50 error-detected. Sol (2027), Apollo (2029, fully fault-tolerant).
- **Magic State Distillation**: Optimal scaling gamma=0 (Nature Physics Nov 2025). Low-cost 53-qubit distillation (npj QI 2026). Constant-overhead injection into qLDPC codes (arXiv:2505.06981).
- **Q-Day Median Estimate**: 2029-2032. ECC likely falls before RSA.

#### Lattice & Quantum Sieving
- **Quantum 3-tuple Sieve**: Exponent 0.3098 → 0.2846 (Dutch team, ePrint 2025/2189). ~8% improvement, ~2^25 theoretical speedup at d=1000.
- **BKZ 3-4 bit loss**: Combined BKZ improvements reduce all lattice PQC security by 3-4 bits (Zhao & Ding 2025).
- **Li & Nguyen (JoC 2025)**: First rigorous dynamic analysis of real BKZ algorithm.
- **Dense Sublattice BKZ**: Cryptanalytic no-go confirmed (Ducas & Loyer, CiC 2025).
- **VERDE AI Cryptanalysis** (2025): Transformer-based red-team attack on lattice schemes. 30% faster training vs prior SALSA approach.

#### FHE & Migration
- **GL Scheme (FHE.org 2026, Mar 8)**: 5th generation FHE by Gentry & Lee at DESILO. Native matrix multiplication for Private AI.
- **OpenFHE v1.5.0** (Feb 26, 2026): Dev release with BFV/BGV/CKKS/TFHE/LMKCDEY.
- **TFHE-rs v1.5.0** (Jan 2026): 42% faster ZK verification, MultiBit blind rotation.
- **FHE GPU Acceleration**: Cheddar (ASPLOS 2026), WarpDrive (HPCA 2025, 73% instruction reduction), CAT (2173x speedup over CPU), Theodosian (Dec 2025, 1.45-1.83x over Cheddar).
- **Arbitrary-Threshold FHE** (USENIX Security 2025): O(N^2+K) complexity, 3.83-15.4x speedup for 1000-party systems.
- **NIST MPTS 2026** (Jan 2026): Threshold FHE standardization session (NIST IR 8214C category S5).
- **EU PQC Roadmap** (Jun 2025): National plans by end 2026, hybrid pilots 2026-2027, critical infra by 2030, full by 2035. EU Quantum Act expected Q2 2026.
- **UK NCSC** (Mar 2025): 3-phase migration — discover (by 2028), upgrade (2028-2031), complete (2031-2035).
- **Japan CRYPTREC**: 2035 target. NEDO PQC program with PQShield/AIST/Mitsubishi/UTokyo.
- **Hybrid TLS Adoption**: Only 8.6% of top 1M websites (F5 Labs, Jun 2025). Top 100: 42%. Banking: only 3%.
- **JVG Algorithm (Mar 2026)**: DISMISSED — no valid quantum speedup demonstrated.

#### Browser-Verified (March 2026)
- **206 tests** all passing (Python 3.12.11, Qiskit 2.3.1, Aer 0.17.2, qiskit-ibm-runtime 0.46.1)
- Healthcare (7), Finance (8), IoT (6), Blockchain (7), MPC-FHE (7) circuit benchmarks — all **🟢 GPU accelerated**
- Shor N=15/21/35/143/221, Grover 4-16 qubit, ECC GF(2^4) — all on `AerSimulator(device='GPU')`
- **GPU backend**: NVIDIA RTX PRO 6000 Blackwell 96GB via cuStateVec — ~25% speedup (Healthcare: 119.9s→89.5s)
- Pass Manager: Shor level=2, Grover level=3, ECC/Noise level=2 (`generate_preset_pass_manager`)
- Circuit diagrams: Shor, Grover, ECC rendered as base64 PNG via matplotlib
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
=======
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
| Post-Quantum KEM | ML-KEM-512/768/1024 (FIPS 203) | Production |
| Post-Quantum Signatures | ML-DSA-44/65/87 (FIPS 204) | Production |
| Hybrid Key Exchange | X25519 + ML-KEM-768 | Production |
| Homomorphic Encryption | CKKS (DESILO FHE) | Production |
| Dynamic QPU Discovery | API → JSON cache → fallback, least_busy, basis_gates | **v3.4.0** |
| IBM Quantum QPU Noise | Real T1/T2/gate errors from IBM Quantum | **v3.3.0** |
| FHE Bootstrap Optimization | Deferred ~24GB key loading | **v3.3.0** |
| Quantum Circuit Benchmarks | Shor/Grover/ECC on AerSimulator GPU | **v3.2.0** |
| Qiskit Pass Manager | Level 2/3 circuit optimization | **v3.2.0** |
| Circuit Visualization | matplotlib PNG diagrams via API | **v3.2.0** |
| Quantum Threat Simulation | Shor + Grover Estimators | **v3.0.0** |
| Security Scoring | NIST IR 8547 Compliance | **v3.0.0** |
| MPC-HE Inference | 2-Party DESILO Multiparty | **v3.0.0** |
| GPU Acceleration | cuStateVec / RTX 6000 PRO Blackwell 96GB | **v3.2.0** |
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
| `/fhe/memory-status` | GET | FHE bootstrap key memory status (loaded/deferred) |
| `/fhe/release-bootstrap-keys` | POST | Release bootstrap keys (~24GB freed) |

### IBM Quantum Hardware (v3.3.0 + v3.4.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/quantum/ibm/backends` | GET | List available IBM QPU backends (API/cache/fallback) |
| `/quantum/ibm/backend/{name}/noise-params` | GET | QPU noise parameters (T1/T2/gate errors/basis_gates) |
| `/quantum/ibm/least-busy` | GET | **[v3.4.0]** Get least busy IBM QPU backend |
| `/quantum/simulate/noisy-ibm` | POST | IBM QPU noise model circuit simulation |

### Monitoring & Health (v3.5.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check (status, components, version) |
| `/version` | GET | Version information from `version.json` |
| `/metrics` | GET | **[v3.5.0]** Prometheus exposition format metrics (uptime, memory, HTTP stats, GC) |

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

### Quantum Circuit Benchmarks (v3.2.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/benchmarks/sector/{sector}/circuit-benchmark` | POST | Per-sector Qiskit circuit benchmark (Shor+Grover+ECC, GPU) |
| `/benchmarks/sector-all/circuit-benchmark` | POST | All 5 sectors circuit comparison |
| `/quantum/circuit/shor-demo` | POST | Shor factoring real circuit demo |
| `/quantum/circuit/grover-demo` | POST | Grover search real circuit demo |
| `/quantum/circuit/ecc-dlog-demo` | POST | ECC discrete log circuit demo |
| `/quantum/circuit/regev-comparison` | GET | Regev vs Shor resource comparison |
| `/quantum/circuit/gpu-status` | GET | GPU/CPU quantum simulation backend status |
| `/quantum/circuit/shor-diagram` | GET | Shor circuit diagram (base64 PNG) |
| `/quantum/circuit/grover-diagram` | GET | Grover circuit diagram (base64 PNG) |
| `/quantum/circuit/ecc-diagram` | GET | ECC discrete log circuit diagram (base64 PNG) |

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
│   └── server.py              # FastAPI server (v3.5.0, 90 routes, /metrics)
├── src/
│   ├── pqc_fhe_integration.py # Core PQC + FHE integration (deferred bootstrap)
│   ├── ibm_quantum_backend.py # [v3.5.0] IBM Quantum QPU noise fetcher + dynamic discovery + singleton
│   ├── pqc_simulator.py       # ML-KEM/ML-DSA educational simulator
│   ├── desilo_fhe_engine.py   # DESILO FHE CKKS engine
│   ├── quantum_threat_simulator.py  # [v3.0.0] Shor/Grover estimator
│   ├── quantum_verification.py     # [v3.1.0] Qiskit circuit verification + noise sim (GPU)
│   ├── sector_quantum_circuit_benchmark.py  # [v3.2.0] Sector circuit benchmarks, ECC, noise, diagrams (GPU)
│   ├── security_scoring.py    # [v3.0.0] NIST IR 8547 scoring
│   ├── sector_benchmarks.py   # [v3.1.0] Sector-specific benchmarks
│   ├── side_channel_assessment.py   # [v3.2.0] Side-channel risk assessment
│   ├── mpc_he_inference.py    # [v3.0.0] MPC-HE 2-party protocol
│   └── version_loader.py      # [v3.2.0] Dynamic version loader
├── benchmarks/
│   └── __init__.py            # Performance benchmarks (incl. GPU)
├── tests/
│   └── test_pqc_fhe.py        # 238 tests (v3.5.0)
├── kubernetes/
│   └── helm/pqc-fhe/          # Helm chart (Deployment, Service, Ingress, HPA, NetworkPolicy, RBAC)
├── monitoring/
│   └── prometheus.yml         # Prometheus scrape config (api, gpu, redis targets)
├── docker-compose.yml         # API + Prometheus + Grafana (monitoring profile)
├── Dockerfile                 # Multi-stage build (liboqs 0.14.0, python:3.12-slim, 420MB)
├── Dockerfile.gpu             # GPU build (CUDA 12.2, liboqs 0.14.0)
├── web_ui/
│   └── index.html             # React Web UI (9 tabs)
├── docs/
│   ├── PQC_FHE_Technical_Report_v3.5.0_Enterprise.docx
│   ├── PQC_FHE_Technical_Report_v3.5.0_Enterprise.pdf
│   └── infographic.html       # Project infographic (v3.5.0)
├── data/
│   ├── ibm_backends_cache.json # [v3.5.0] Dynamic QPU backend cache (6 backends)
│   ├── benchmark_results/     # [v3.5.0] Saved benchmark results (JSON)
│   └── circuit_diagrams/      # [v3.5.0] Saved circuit diagrams (PNG)
├── version.json               # [v3.5.0] Dynamic version configuration
├── logs/                      # Log files (auto-created)
└── README.md
```
## Development Environment

| Component | Specification |
|-----------|---------------|
| OS | Alma Linux 9.7 |
| CPU | Intel Core i5-13600K |
| RAM | 128GB DDR5 5200 |
| GPU | NVIDIA RTX PRO 6000 Blackwell 96GB (cuStateVec) |
| Python | 3.12.11 |
| CUDA Driver | 580.105.08 |
| Qiskit | 2.3.1 + Aer 0.17.2 (GPU) |

## Requirements

- Python 3.12+
- liboqs-python (build from source, liboqs 0.15.0)
- cryptography >= 41.0 (for X25519 hybrid)
- desilofhe / desilofhe-cu130 (FHE, optional for GPU)
- qiskit >= 2.3.1, qiskit-aer >= 0.17.2 (circuit benchmarks)
- qiskit-aer-gpu-cu11 (optional, GPU acceleration via cuStateVec)
- pylatexenc (circuit diagram rendering)
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
27. IBM Quantum Docs — Processor Types: https://quantum.cloud.ibm.com/docs/guides/processor-types
28. IBM Quantum Docs — Build Noise Models: https://quantum.cloud.ibm.com/docs/guides/build-noise-models
29. IBM Quantum Docs — Error Mitigation Overview: https://quantum.cloud.ibm.com/docs/guides/error-mitigation-overview
30. IBM Quantum Docs — QPU Information: https://quantum.cloud.ibm.com/docs/guides/qpu-information
31. IBM Quantum Learning — Shor's Algorithm: https://quantum.cloud.ibm.com/learning/ja/modules/computer-science/shors-algorithm
32. IBM Quantum Learning — Grover's Algorithm: https://quantum.cloud.ibm.com/learning/ja/modules/computer-science/grovers
33. NIST IR 8547: Transition to Post-Quantum Cryptography Standards (November 2024) — 2030 deprecated, 2035 disallowed
34. IBM Quantum Open Plan Updates (March 2026): Heron r3 (156Q), Flamingo (1,386Q announced), ibm_kingston 2Q error 2.03e-3
35. NIST IR 8545: HQC Selection as 5th PQC Algorithm (March 2025) — code-based KEM for lattice monoculture mitigation
36. Iceberg Quantum (Feb 2026): QLDPC codes enable RSA-2048 attack with <100K physical qubits
37. Security Boulevard: Enterprise PQC Migration Guide 2026 — HNDL risk assessment, sector-specific timelines
38. MOZAIK (Jan 2026): Open-source MPC+FHE IoT Platform — privacy-preserving ML on constrained devices
39. MDPI: HNDL Temporal Cybersecurity Risk Model — Mosca theorem formalization for data retention vs Q-Day
40. Microsoft Majorana 1 (Feb 2025): Topological qubits — potential for lower error rates in future CRQC

## Platform Validation & Limitations

### Why Small-Scale Quantum Circuits Are Valid

This platform uses 8-24 qubit circuits for Shor, Grover, and ECC demonstrations. This approach is standard across quantum computing research:

- **IBM Quantum Learning**: Official tutorials use N=15 factoring (8 qubits) for Shor's algorithm
- **Google Cirq**: Tutorials use 4-8 qubit Grover search demonstrations
- **Academic papers**: Gidney & Ekera (2021), Chevignard et al. (2024), Pinnacle (2026) all provide extrapolation models from small demonstrations

### Extrapolation Model Sources

- **RSA-2048 resource estimates**: ~20M physical qubits (Gidney 2021) → ~1M (Gidney 2025, magic state cultivation) → ~100K (Pinnacle 2026, QLDPC codes)
- **AES-128/256 Grover**: 2,953-6,681 logical qubits (Grassl 2016), optimized T-depth=30 (ASIACRYPT 2025)
- **ECC P-256**: 2,330 logical qubits (Roetteler 2017), exact gate counts (arXiv:2503.02984, March 2025)

### Known Limitations

1. **Scale gap**: RSA-2048 requires ~4,000+ logical qubits; current max is 156 physical qubits (no error correction)
2. **Extrapolation uncertainty**: Resource estimates depend on theoretical models; implementation overhead not included
3. **Lattice monoculture risk**: ML-KEM, ML-DSA, FN-DSA, and CKKS all rely on lattice assumptions — HQC (code-based) mitigates this but is not yet standardized (expected 2027)
4. **Side-channel attacks**: Platform evaluates but does not implement hardware-level countermeasures
5. **AI-assisted attacks**: ML-KEM single-trace key recovery demonstrated in 2026 — constant-time implementation required

### NIST IR 8547 Migration Timeline

| Year | Status | Action Required |
|------|--------|----------------|
| 2024 | Current | Begin PQC evaluation and planning |
| 2030 | Deprecated | RSA, ECDSA, DH deprecated; PQC preferred |
| 2035 | Disallowed | Classical-only algorithms prohibited |

## Version History

### v3.5.0 (2026-03-29)
- **Accurate Hardware Discovery**: ibm_torino corrected to Heron r1 (133Q), added ibm_fez/kingston/marrakesh as Heron r2 (156Q)
- **Benchmark Results Persistence**: BenchmarkResultsManager saves results and circuit diagrams to files
- **Prometheus `/metrics` endpoint**: Zero-dependency exposition format (uptime, memory, HTTP stats, GC, app info)
- **Docker image verified**: `pqc-fhe-api:v3.5.0` (420MB), multi-stage build with liboqs 0.14.0
- **Docker Compose monitoring**: API + Prometheus + Grafana stack verified end-to-end
- **Helm chart validated**: `helm lint` passed, `helm template` renders 8 Kubernetes manifests
- **All-Sector circuit diagrams**: Button 5 now displays Shor/Grover/ECC circuit diagrams
- **KNOWN_PROCESSORS expanded**: 3 → 6 backends, HERON_R2_FALLBACK corrected to ibm_fez
- **2026 PQC research**: NIST IR 8547 timeline, HQC, CRQC ~100K qubits, Hybrid TLS default
- 17 new tests (238 total), 90 API routes, Platform Validation section added

### v3.4.0 (2026-03-29)
- **Dynamic QPU backend discovery**: Server startup connects to IBM Quantum API, persists to JSON cache
- **3-tier fallback chain**: API → JSON cache file → KNOWN_PROCESSORS (hardcoded)
- **Processor-specific basis_gates**: Heron=CZ, Eagle=ECR, Nighthawk=CZ
- 15 new tests (221 total), 70 endpoints

### v3.2.0 (2026-03-19)
- **GPU acceleration**: All circuit verifiers use `AerSimulator(device='GPU')` with CPU fallback (~25% speedup)
- **Qiskit Pass Manager**: `generate_preset_pass_manager()` replaces `transpile()` (Shor level=2, Grover level=3)
- **Circuit visualization**: `generate_circuit_diagram()` renders Shor/Grover/ECC circuits as base64 PNG
- **WebUI GPU indicators**: 🟢 GPU / CPU device display for all circuit results
- BKZ/Core-SVP accuracy fixes (NIST reference lookup table, CBD sigma, sieve constant 0.257)
- Multi-era Shor resource estimation (20M → 100K physical qubits, 4 generations)
- Extended factorizations: N=143 (11×13), N=221 (13×17) with 24-qubit circuits
- Side-channel risk assessment: ML-KEM critical (SPA+EM), CKKS-FHE critical (NTT neural network 98.6%)
- Noise-aware quantum simulation (depolarizing error channels)
- CKKS/FHE Ring-LWE security verification against HE Standard bounds
- Dynamic version management via `version.json` (eliminates hardcoded version strings)
- IBM quantum roadmap update: Kookaburra 4,158 qubits, qLDPC real-time decoding
- Bug fixes: Shor trivial GCD UnboundLocalError, WebUI sector switch clearing, null safety
- 25 API endpoints, 178 tests (up from 88), 60+ academic references

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
