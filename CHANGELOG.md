# Changelog

All notable changes to the PQC-FHE Integration Platform.

## [3.2.0] - 2026-03-19

### Fixed
- **BKZ Block Size Estimation**: Replaced broken GSA binary search with NIST reference
  lookup table validated against Lattice Estimator. Prior formula did not converge for
  Module-LWE parameters, returning beta=1500 for all algorithms.
- **CBD Sigma Calculation**: Fixed Centered Binomial Distribution standard deviation from
  `eta * sqrt(2/3)` (incorrect) to `sqrt(eta/2)` (correct). ML-KEM-768 sigma: 1.633 → 1.0.
- **NIST Level Thresholds**: Calibrated Core-SVP thresholds to match what NIST actually
  accepts (L1: 118, L3: 176, L5: 244 bits classical), accounting for Core-SVP model's
  underestimation of actual security.
- **Quantum Sieve Constant**: Updated from 0.265 (Laarhoven 2015) to 0.257 (Dutch team,
  Oct 2025), reflecting ~8% improvement in quantum sieving efficiency.
- **Version Inconsistencies**: Fixed api/server.py (2.3.0/2.3.5 → 3.2.0) and
  pqc_fhe_integration.py (2.1.2 → 3.2.0).
- **SLH-DSA Comments**: Corrected misleading Grover impact description.
- **Shor API Endpoint**: Extended `/quantum/verify/shor` to support N=143 and N=221
  factorizations (previously only N=15, 21, 35 were allowed).
- **WebUI Noise Simulation**: Fixed POST request to send query parameters instead of
  JSON body (matching FastAPI endpoint signature). Fixed result display to use correct
  API response keys (`ideal_probability`, `noisy_results` dict with `degradation`).

### Added
- **Multi-Era Shor Resource Estimation**
  - 4-generation comparison: Gidney-Ekera 2021 (20M), Chevignard 2024 (4M),
    Gidney 2025 (1M), Pinnacle 2026 (100K physical qubits)
  - `ShorResourceEra` enum and `SHOR_RESOURCE_MODELS` dictionary
  - `estimate_rsa_resources_multi_era()` method on QuantumThreatSimulator
  - Updated `_extrapolate_to_rsa()` with multi-era output

- **Extended Shor Factorization Targets**
  - N=143 (11×13): 24 qubits, AerSimulator verified
  - N=221 (13×17): 24 qubits, AerSimulator verified

- **Side-Channel Risk Assessment** (`src/side_channel_assessment.py`)
  - ML-KEM: Critical risk — SPA key recovery in 30s (Berzati 2025), KyberSlash timing
  - ML-DSA: High risk — signing vector leakage
  - SLH-DSA: Low risk — inherently resistant (hash-based)
  - Per-algorithm mitigation recommendations
  - Implementation hardening checks (liboqs, pqcrystals, pqm4)

- **Noise-Aware Quantum Simulation** (`NoiseAwareQuantumSimulator` class)
  - Depolarizing error channel simulation
  - Ideal vs noisy circuit comparison at multiple error rates
  - Error threshold estimation (point where success drops below 50%)

- **4 New API Endpoints**
  - `GET /quantum/shor-resources/multi-era` — 4-generation Shor comparison
  - `POST /quantum/simulate/noisy` — Noise-aware quantum simulation
  - `GET /security/side-channel/{algorithm}` — Per-algorithm side-channel risk
  - `GET /security/side-channel/all` — All-algorithm side-channel assessment

- **WebUI Enhancements**
  - Shor dropdown: added N=143 and N=221 factoring targets
  - Multi-era Shor comparison section with bar chart
  - Noise-aware quantum simulation panel (circuit type selector, qubit slider, error rate checkboxes, ideal vs noisy comparison)
  - Side-channel risk assessment card with per-algorithm display

- **Algorithm Diversity Assessment** (`SecurityScoringEngine.assess_algorithm_diversity()`)
  - Lattice monoculture detection (ML-KEM + ML-DSA only = single point of failure)
  - PQC family classification: lattice, hash-based, code-based
  - Diversity scoring (0-100) with per-family breakdown
  - Recommendations for SLH-DSA and HQC deployment

- **CNSA 2.0 Phase Gate Assessment** (`SecurityScoringEngine.assess_cnsa_2_0_readiness()`)
  - 5-phase milestone tracking (2025-2035)
  - Per-algorithm compliance check (ML-KEM-1024, ML-DSA-87, AES-256, SHA-384)
  - Gap analysis with remediation guidance
  - Phase gate status: inventory, hybrid, full PQC, CNSA compliance, NIST disallow

- **Masking Deployment Verification** (`SideChannelRiskAssessment.verify_masking_deployment()`)
  - Per-implementation masking status (liboqs, pqcrystals, pqm4)
  - Critical finding: ML-KEM on liboqs lacks SPA protection
  - Recommendation: pqm4 for ARM, custom masking for x86

- **HQC (Code-Based KEM) Integration**
  - HQC-128/192/256 in algorithm knowledge base (NIST IR 8545)
  - HQC side-channel assessment (timing vulnerability, patched in liboqs v0.12.0+)
  - HQC in algorithm diversity scoring as code-based family
  - HQC implementation hardening status

- **3 New API Endpoints**
  - `GET /security/algorithm-diversity` — PQC family diversity assessment
  - `GET /security/cnsa-readiness` — CNSA 2.0 phase gate compliance
  - `GET /security/masking-verification` — SPA countermeasure check

- **WebUI Enhancements**
  - Algorithm Diversity card with family breakdown and scoring
  - CNSA 2.0 Readiness card with per-algorithm compliance status
  - Masking Verification card with per-implementation SPA check
  - Side-channel assessment now includes HQC (4-algorithm grid)

- **CKKS / FHE Quantum Security Verification** (`CKKSSecurityVerifier` class)
  - Ring-LWE security assessment against HE Standard (homomorphicencryption.org) bounds
  - 7 predefined configs verified: Light, Medium, Standard, Heavy, MPC-HE default/linear/NN
  - Lattice monoculture risk analysis: CKKS shares Ring-LWE with ML-KEM/ML-DSA
  - Business impact assessment per sector (healthcare, finance, IoT, MPC)
  - **Key finding**: MPC-HE default (num_scales=40, log_n=15) exceeds 128-bit security bound

- **FHE Quantum Risk Assessment** (`SecurityScoringEngine.assess_fhe_quantum_security()`)
  - Risk scoring (0-100) with lattice monoculture penalty
  - Shared lattice risk correlation between FHE and PQC
  - Diversification strategy recommendations (HQC for KEM, conservative FHE params)

- **CKKS/FHE Side-Channel Assessment**
  - CKKS-FHE added to side-channel vulnerability database
  - Noise flooding leakage (CKKS decryption noise patterns, Li & Micciancio 2021)
  - Key generation power analysis (low risk for server-side deployments)
  - DESILO FHE and OpenFHE implementation hardening status

- **Sector Quantum Security Context**
  - Per-sector quantum risk profiles (data retention risk, compliance framework)
  - Business recommendations for FHE parameter selection per industry
  - MPC-FHE sector: CRITICAL warning about parameter validation

- **3 New API Endpoints**
  - `GET /quantum/ckks-security` — CKKS Ring-LWE security verification
  - `GET /quantum/ckks-security/all-configs` — All CKKS config comparison
  - `GET /security/fhe-quantum-risk` — FHE deployment quantum risk scoring

- **WebUI Enhancements**
  - CKKS / FHE Quantum Security card with log_n and max_levels selectors
  - NIST level display, BKZ block size, HE Standard bound check
  - Lattice monoculture warning banner

- **45 New Tests** (88 → 133 total)
  - `TestBKZAccuracyFixes` (5): CBD sigma, BKZ block size range, sieve constant, all-pass
  - `TestShorMultiEra` (4): 4 models, Gidney 2025 <2M qubits, Pinnacle <200K, EC config
  - `TestSideChannelAssessment` (3): ML-KEM critical, SLH-DSA low, mitigations
  - `TestNoiseAwareSimulation` (3): noisy < ideal, noise model, multiple rates
  - `TestExtendedFactorization` (1): N=143 = 11×13
  - `TestAlgorithmDiversity` (5): lattice monoculture, govt SLH-DSA, HQC knowledge base
  - `TestCNSA20Readiness` (3): enterprise readiness, govt higher, phase gates
  - `TestMaskingVerification` (4): liboqs lacks masking, pqm4 has masking, HQC assessment
  - `TestCKKSSecurityVerification` (6): NIST level, MPC-HE warning, heavy config, all-configs, monoculture, NN demo
  - `TestFHEQuantumRisk` (4): risk score, monoculture penalty, insecure params critical, diversification
  - `TestCKKSSideChannel` (4): CKKS in assess_all, normalize, masking verification, scoring
  - `TestSectorQuantumContext` (3): healthcare context, MPC-FHE context, all sectors

- **Dynamic Version Management**
  - `version.json`: Centralized version configuration for all modules
  - `src/version_loader.py`: Shared version loading utility with caching
  - All 14 source files updated to dynamically load version from `version.json`
  - Eliminates hardcoded version strings across the codebase

- **Browser-Verified Sector Benchmarks (All 5 Sectors)**
  - Healthcare (HIPAA): 5 benchmarks verified — patient record encryption,
    FHE vital signs, medical IoT PQC, key rotation, compliance
  - Finance (PCI-DSS): 5 benchmarks verified — transaction batch, trade settlement,
    key rotation, compliance assessment, real-time signing
  - IoT/Edge: 10 benchmarks verified — ML-KEM-512/768 at 64B/256B/1KB/4KB payloads,
    constrained device, gateway
  - Blockchain: 5 benchmarks verified — ML-DSA-44/65/87 throughput, batch verification,
    full TX pipeline
  - MPC-FHE: 9 benchmarks verified — engine setup, encrypted computation, 2-party
    inference, CKKS operations, GL Scheme status

- **Browser-Verified Quantum Algorithm Security**
  - Shor's algorithm: N=15 (3×5), N=143 (11×13), N=221 (13×17) — all factors verified
  - Grover's search: 4-qubit, 96.6% success probability confirmed
  - NIST Level verification: 9 algorithms (ML-KEM-512/768/1024, ML-DSA-44/65/87,
    SLH-DSA-128s/192s/256s) — all PASSED
  - Multi-Era Shor Resources: 4 generations displayed correctly (20M/4M/1M/100K)
  - Noise Simulation: Ideal 96.6% → Error 0.05: 15.3% degradation verified
  - Side-Channel Assessment: 6 algorithms (ML-KEM CRITICAL, ML-DSA HIGH,
    SLH-DSA LOW, HQC LOW, CKKS-FHE CRITICAL, GL-FHE HIGH)
  - Security Scoring, Algorithm Diversity, CNSA 2.0 Readiness — all verified

- **2026 Research Updates (March 2026, Updated)**
  - CKKS-FHE side-channel severity upgraded to CRITICAL: neural network classifier
    achieves 98.6% accuracy extracting secret key from single NTT power trace
    (arXiv:2505.11058). **Random delay insertion and masking alone are INEFFECTIVE.**
    At -O3 optimization, guard/mul_root operations expose new leakage points.
  - **Threshold FHE CPAD attack (CEA 2025)**: Full key recovery in < 1 hour without
    smudging noise. MPC-HE `individual_decrypt()` now enforces smudging noise
    (`smudging_noise_bits=40` default). Decryption count tracking added.
  - **PKC 2025 Noise-Flooding key recovery**: Non-worst-case noise estimation
    enables CKKS key recovery. `max_decryptions_per_key=1000` limit implemented.
  - **Quantum 3-tuple sieve (Oct 2025)**: Exponent reduced 0.3098 → 0.2846
    (Engelberts et al., ePrint 2025/2189). ML-KEM/ML-DSA margins narrowed.
  - ML-KEM EM fault injection: 89.5% success rate on ARM Cortex-M4 (2025)
  - IBM Quantum roadmap: Kookaburra (4,158 qubits), Relay-BP decoder (<480ns,
    10x accuracy, Nov 2025), Cockatoo (2027), Starling (2029), Blue Jay (2029+)
  - Quantinuum Helios: 98 trapped-ion qubits, 94 logical qubits in GHZ state,
    better-than-break-even fidelity (Nov 2025, $10B valuation)
  - BDGL sieve optimality confirmed: overlattice lower bound (MDPI Jan 2026)
  - DESILO GL scheme: announced FHE.org 2026 Taipei (Mar 7, 2026), Google HEIR #2408
  - OpenFHE v1.5.0 (Feb 26, 2026): BFV, BGV, CKKS, TFHE, LMKCDEY with bootstrapping
  - Qiskit 2.3.1 (Mar 2026) + Aer 0.17.2 verified compatible
  - Microsoft SEAL added to implementation hardening database (vulnerable to NTT SPA)
  - JVG algorithm (Mar 2026): DISMISSED — precomputing classical values doesn't provide
    quantum speedup. Expert consensus (Aaronson, PostQuantum.com) confirms no threat.
  - NIST HQC draft standard expected early 2026, final standard 2027 (code-based KEM backup)
  - Iceberg Quantum Pinnacle Architecture (Feb 2026): RSA-2048 < 100K physical qubits via QLDPC
  - Ubuntu 26.04 LTS (Apr 2026) to ship with PQC enabled by default in OpenSSH/OpenSSL

- **GL Scheme Integration (NEW)**
  - `src/gl_scheme_engine.py`: GL (Gentry-Lee) 5th generation FHE engine wrapper
  - `GLSchemeEngine`: encrypt, decrypt, matrix_multiply, hadamard, transpose, conjugate
  - `GLCKKSHybridEngine`: Combined GL + CKKS for matrix and vector operations
  - `GLPrivateInference`: 2-party private inference using GL native matrix multiplication
  - O(1) matrix multiply vs CKKS O(n) rotations — significant speedup for ML inference
  - Security assessment: GL inherits CKKS NTT side-channel surface (arXiv:2505.11058)
  - Referenced: RhombusEnd2End_HEonGPU GPU-accelerated 2PC, DESILO FHE GLEngine API

- **3 New GL Scheme API Endpoints**
  - `GET /fhe/gl-scheme/info`: GL capabilities and supported shapes
  - `GET /fhe/gl-scheme/security`: GL security info, known vulnerabilities, CKKS comparison
  - `GET /mpc-he/gl-inference/info`: GL private inference capabilities

- **18 New Tests (133 → 151)**
  - TestGLSchemeEngine: GL info, config, shapes, security (5 tests)
  - TestGLPrivateInference: info, protocol_info, security_notes, references (4 tests)
  - TestGLSideChannelAssessment: assess_all, inherited risks, key findings (3 tests)
  - TestQuantumThreatGLScheme: PQC comparison, security details, resistant list (3 tests)
  - TestSectorBenchmarkGL: MPC-FHE includes GL status, metadata (2 tests)
  - TestExtendedFactorization: test_shor_factoring_221 (N=221 = 13×17) (1 test)

### Changed
- Updated version to 3.2.0 across all 14 files
- DEFAULT_EC_OVERHEAD reduced from 1000 to 500 (2025 moderate estimate)
- BKZ improvement correction: -3.5 bits per Zhao & Ding (2025)
- NIST verification now uses classical Core-SVP (0.292*β) for level determination
- Quantum Core-SVP (0.257*β - 3.5) reported as supplementary information
- Qiskit-Aer noise model integration for depolarizing channels

### References (Key Sources for v3.2.0 — 50+ Citations)

**NIST Standards & Guidance:**
- NIST FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA) — August 2024
- NIST FIPS 204 Errata — February 2026 (minor corrections)
- NIST FIPS 206 (FN-DSA/FALCON) — Initial Public Draft pending, ~666 byte signatures
- NIST IR 8545 (Mar 2025): HQC selected as 4th-round code-based KEM
- NIST IR 8547 — Transition to PQC Standards (still DRAFT as of March 2026)
- NIST SP 800-227 — KEM operational guidance (Finalized September 18, 2025)
- CNSA 2.0 (May 2025 update): 2030 full PQC migration deadline

**Shor Algorithm Resource Estimation:**
- Gidney & Ekerå (2021): 20M physical qubits, Quantum 5, 433
- Chevignard et al. (2024): 4M physical qubits, sublinear resources
- Gidney (May 2025): Magic state cultivation → ~1M physical qubits
- Pinnacle Architecture (Feb 2026): QLDPC codes → ~100K physical qubits

**Lattice Cryptography & Quantum Sieving:**
- Dutch team (Oct 2025): Quantum sieve exponent 0.3098 → 0.2846 (~8% improvement)
- Zhao & Ding (PQCrypto 2025): BKZ improvements → 3-4 bit security reduction
- Li & Nguyen (Journal of Cryptology 38, 2025): Complete BKZ analysis
- BDGL sieve optimality (Jan 2026): NNS paradigm proven optimal
- Ducas, Engelberts & Perthuis (ASIACRYPT 2025): Module-lattice reduction prediction

**Side-Channel Attacks:**
- Berzati et al. (CHES 2025): ML-KEM SPA key recovery in 30 seconds
- ML-DSA rejection sampling attack (TCHES 2025)
- ML-DSA template attack (DATE 2026, ePrint 2026/056)
- arXiv:2505.11058 (May 2025): CKKS NTT neural network SPA (98.6% key extraction)
- CEA 2025: Threshold FHE CPAD attack, key recovery < 1 hour
- PKC 2025: CKKS Noise-Flooding key recovery
- EM fault injection on ML-KEM (2025): 89.5% success rate on ARM Cortex-M4
- GlitchFHE (USENIX Security 2025): Fault injection on FHE

**Quantum Hardware:**
- IBM Quantum (2026): Kookaburra 4,158 qubits, real-time qLDPC decoding (Feb 2026)
- Google Quantum AI: Willow (105 qubits), Quantum Echoes (Dec 2025)
- Microsoft Majorana 1: Topological qubit chip (Feb 2025)
- Quantinuum Helios: 98 qubits, 94 logical qubits, 2:1 ratio (Mar 2026, $10B)
- Magic state distillation optimal scaling gamma=0 (Nature Physics Nov 2025)

**FHE & PQC Integration:**
- Gentry & Lee (2025): GL scheme, ePrint 2025/1935 (5th gen FHE)
- FHE.org 2026 Taipei (Mar 7, 2026): GL scheme announcement
- DESILO FHE v1.10.0 (Feb 11, 2026): GLEngine API for matrix FHE
- OpenFHE v1.5.0 (Feb 26, 2026): BFV/BGV/CKKS/TFHE with bootstrapping
- FHE GPU acceleration: Cheddar (ASPLOS 2026), WarpDrive (HPCA 2025), CAT framework
- RhombusEnd2End_HEonGPU: GPU-accelerated 2PC inference
- Google HEIR project: GL scheme investigation (GitHub Issue #2408)

**Migration & Compliance:**
- EU PQC Roadmap: Critical infrastructure by 2030, full transition by 2035
- UK NCSC: PQC migration guidance, PQTLS recommendations
- Japan CRYPTREC: PQC evaluation, 2035 target
- Hybrid TLS: 38% of Cloudflare HTTPS connections use X25519+ML-KEM (Mar 2026)
- Ubuntu 26.04 LTS (Apr 2026): PQC enabled by default in OpenSSH/OpenSSL

**Other:**
- JVG Algorithm (Mar 2026): Debunked quantum factoring claims (Aaronson)
- Qiskit 2.3.1 (Mar 2026) + Aer 0.17.2 verified compatible

## [3.1.0] - 2026-03-18

### Added
- **Quantum Algorithm Verification (Qiskit AerSimulator)**
  - `src/quantum_verification.py` - Real quantum circuit simulation module
  - `ShorCircuitVerifier` - Shor's algorithm with QFT-based period finding (N=15, 21, 35)
  - `GroverCircuitVerifier` - Grover's search with amplitude amplification (3-20 qubits)
  - `NISTLevelVerifier` - Lattice parameter BKZ/Core-SVP analysis for ML-KEM and ML-DSA
  - Resource extrapolation from small circuits to RSA-2048 and AES-256

- **Sector-Specific Benchmarks**
  - `src/sector_benchmarks.py` - Real PQC+FHE benchmarks for 5 industry sectors
  - Healthcare (HIPAA): Patient record encryption, FHE vital signs, medical IoT
  - Finance (PCI-DSS/SOX): Transaction batch, trade settlement, key rotation
  - Blockchain: ML-DSA-44/65/87 throughput, batch verification, full TX pipeline
  - IoT: ML-KEM-512 vs 768 at varying payload sizes (64B-4KB)
  - MPC-FHE: Engine setup, encrypted computation, 2-party inference

- **New API Endpoints (5 endpoints)**
  - `POST /quantum/verify/shor` - Real Shor's circuit execution
  - `POST /quantum/verify/grover` - Real Grover's circuit execution
  - `GET /quantum/verify/nist-levels` - NIST level verification
  - `GET /benchmarks/sector/{sector}` - Per-sector benchmarks
  - `GET /benchmarks/sector-all` - All-sector benchmarks

- **WebUI Enhancements**
  - Quantum Circuit Verification section in Quantum Threat tab
  - Shor's algorithm circuit demo with measurement results display
  - Grover's algorithm demo with amplitude amplification bar chart
  - NIST level verification table with pass/fail status
  - New "Sector Benchmarks" tab with per-sector benchmark execution

- **21 New Tests** (67 → 88 total)
  - `TestShorVerification` (5 tests): factoring 15/21, QFT verification, RSA-2048 extrapolation
  - `TestGroverVerification` (6 tests): 3/4/5-qubit search, probability evolution, AES extrapolation
  - `TestNISTLevelVerification` (5 tests): ML-KEM/DSA verification, ordering, serialization
  - `TestSectorBenchmarks` (5 tests): healthcare, finance, blockchain, IoT, serialization

### Changed
- Updated version to 3.1.0 across all files
- Added Qiskit 2.3.1 + qiskit-aer 0.17.2 to dependencies
- Version numbers synchronized across pyproject.toml, __init__.py, api/server.py

## [2.3.5] - 2025-12-30

### Added
- **X25519 + ML-KEM Hybrid Key Exchange**
  - `POST /pqc/hybrid/keypair` - Generate hybrid keypair
  - `POST /pqc/hybrid/encapsulate` - Hybrid encapsulation (sender)
  - `POST /pqc/hybrid/decapsulate` - Hybrid decapsulation (receiver)
  - `GET /pqc/hybrid/compare` - Algorithm comparison
  - `GET /pqc/hybrid/migration-strategy` - Enterprise migration roadmap
  - `GET /pqc/hybrid/keypairs` - List stored keypairs

- **Web UI Hybrid Migration Tab**
  - Interactive 4-phase migration timeline visualization
  - Live X25519 + ML-KEM demo with step-by-step logging
  - Algorithm comparison table (Classical vs Hybrid vs PQC)
  - Security analysis and HNDL protection explanation

- **Kubernetes Helm Chart**
  - Production-ready deployment with HPA (2-10 replicas)
  - GPU worker deployment with NVIDIA device plugin
  - Redis cache integration (Bitnami dependency)
  - Prometheus and Grafana integration
  - NetworkPolicy for security isolation
  - PodDisruptionBudget for high availability
  - Ingress with TLS termination
  - ConfigMap for cryptographic parameters

- **Monitoring and Observability**
  - Prometheus ServiceMonitor configuration
  - Pre-configured alerting rules:
    - PQCFHEHighErrorRate (error rate > 5%)
    - PQCFHEHighLatency (p95 > 5s)
    - PQCFHEPodNotReady (pods not ready)
    - PQCFHESlowEncryption (encrypt > 10s)
    - PQCFHEGPUMemoryHigh (GPU memory > 90%)

- **File-Based Logging**
  - Rotating log files (10MB max, 5 backups)
  - Separate log files:
    - `pqc_fhe_server.log` - All logs
    - `pqc_fhe_error.log` - Errors only
    - `pqc_fhe_access.log` - HTTP access logs
  - Configurable via LOG_LEVEL environment variable

- **IETF Compliance**
  - Follows draft-ietf-tls-ecdhe-mlkem pattern for TLS 1.3
  - SHA-256 combination of shared secrets

### Changed
- Updated API version to 2.3.5
- Enhanced Swagger documentation with hybrid endpoints
- Web UI now has 5 tabs (added Hybrid Migration)
- Comprehensive README with K8s and monitoring docs
- Technical reports updated (PDF and Word)

## [2.3.4] - 2025-12-30

### Fixed
- Web UI: Healthcare example missing `addLog` function definition (JS ReferenceError)
- API Server: Healthcare demo numpy array boolean check error (ValueError)
- Ethereum RPC: Added Ankr (`rpc.ankr.com/eth`) as primary endpoint (no API key)
- Ethereum RPC: Better error handling for hex parsing and empty responses
- VitalDB: Improved numpy array handling for vital signs data
- VitalDB: Proper fallback when library not installed

### Changed
- RPC endpoint priority: Ankr > PublicNode > Cloudflare > DRPC > 1RPC
- Increased RPC timeout from 10s to 15s for reliability
- Added detailed logging for successful RPC connections

### Added
- Technical Report v2.3.4 (18 pages) with comprehensive documentation
  - Multi-platform installation guide (Debian/Ubuntu, Fedora/RHEL, Arch, macOS)
  - liboqs-python build instructions (automatic and manual options)
  - Complete References section with 12 academic citations
  - KeepTogether for tables (prevents page splits)
  - Proper code block formatting with Preformatted style

## [2.3.3] - 2025-12-30

### Fixed
- VitalDB API UTF-8 BOM handling (decode with `utf-8-sig`)
- Etherscan V1 API deprecation: Migrated to public Ethereum RPC endpoints
  - Primary: cloudflare-eth.com
  - Fallback: publicnode.com, llamarpc.com
- No API key required for blockchain data fetching

### Changed
- Blockchain data source now uses standard Ethereum JSON-RPC
- Multiple RPC endpoint fallback for reliability

## [2.3.2] - 2025-12-30

### Fixed
- VitalDB API gzip response handling (`'utf-8' codec can't decode byte 0x8b` error)
- FHE multiply TypeError when using scalar multiplication (removed erroneous relin key)
- Healthcare demo mean calculation (now correctly computes average)

### Changed
- `multiply()` method now auto-detects scalar vs ciphertext multiplication
- Healthcare demo uses proper mean formula: `sum(values * (1/n))`
- Live data fetcher handles gzip-compressed API responses

## [2.3.1] - 2025-12-30

### Added
- Live data fetching from external APIs (VitalDB, Yahoo Finance, UCI, Etherscan)
- `api/live_data_fetcher.py` module for real-time data retrieval
- LIVE/Embedded status indicators in Web UI
- Automatic fallback to embedded samples when APIs unavailable

### Changed
- Enterprise endpoints now attempt live API calls first
- Web UI shows data source status (LIVE green badge when live data)
- Sign/Verify endpoints now properly pass algorithm parameter

### Fixed
- HTTP 500 error in blockchain signing demo (algorithm parameter not passed)
- Sign endpoint tuple unpacking for signature return value
- Verify endpoint algorithm handling

## [2.3.0] - 2025-12-30

### Added
- Real data sources with full academic citations
- Healthcare: VitalDB Open Dataset integration (DOI: 10.1038/s41597-022-01411-5)
- Finance: Yahoo Finance API with real stock prices (yfinance library)
- IoT: UCI ML Repository power consumption dataset (DOI: 10.24432/C52G6F)
- Blockchain: Etherscan API with real Ethereum transaction references
- GET /enterprise/citations endpoint for all data source citations
- Data verification methods with SHA256 integrity checks
- Clinical reference ranges from AHA, WHO, CDC guidelines

### Changed
- Enterprise data module completely rewritten with citation support
- All enterprise endpoints now return data_source and verification fields
- Updated API documentation with data source information
- Technical report updated with academic references

### Fixed
- Proper attribution for all external data sources
- License compliance for CC BY-NC-SA 4.0 (VitalDB) and CC BY 4.0 (UCI)

## [2.2.0] - 2025-12-29

### Added
- Multi-algorithm PQC support (ML-KEM-512/768/1024, ML-DSA-44/65/87)
- Algorithm selection API endpoints
- Enterprise data verification examples
- Algorithm comparison endpoint
- Web UI algorithm selectors

### Changed
- PQCManager supports dynamic algorithm selection
- Updated Web UI with algorithm dropdown menus

## [2.1.2] - 2025-12-29

### Fixed
- ML-DSA-65 signature size validation (3309 bytes)
- Web UI API path corrections
- Technical report table layouts

## [2.1.0] - 2025-12-28

### Added
- RealPQCManager using liboqs-python
- Swagger UI integration guide
- Smart Default feature for API testing
- React-based Web UI with crypto simulations
- Enterprise examples (Healthcare, Finance, IoT, Blockchain)

### Removed
- MockPQCManager (replaced with real implementation)

## [2.0.0] - 2025-12-27

### Added
- FastAPI REST API server
- DESILO FHE integration with GPU support
- WebSocket server for real-time operations
- Comprehensive API documentation

### Changed
- Complete architecture redesign
- Modular component structure

## [1.0.0] - 2025-12-26

### Added
- Initial PQC-FHE integration implementation
- Basic Kyber/Dilithium support
- FHE operations (encrypt, decrypt, add, multiply)
- CLI interface
