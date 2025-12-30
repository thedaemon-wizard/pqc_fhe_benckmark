# PQC-FHE Integration Architecture

## System Overview

This document describes the architecture of the Post-Quantum Cryptography and Fully Homomorphic Encryption integration system.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PQC-FHE Integration System                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Client Apps   │    │   REST API      │    │  Background     │         │
│  │                 │───▶│   (FastAPI)     │───▶│  Workers        │         │
│  │ - Web UI        │    │                 │    │                 │         │
│  │ - Mobile        │    │ - /pqc/*        │    │ - Key rotation  │         │
│  │ - IoT Devices   │    │ - /fhe/*        │    │ - Batch encrypt │         │
│  │ - CLI Tools     │    │ - /hybrid/*     │    │ - Monitoring    │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Core Cryptographic Layer                       │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │  │
│  │  │   PQCKeyManager     │  │     FHEEngine       │  │   Hybrid     │  │  │
│  │  │                     │  │                     │  │   Manager    │  │  │
│  │  │ • ML-KEM-512/768    │  │ • CKKS Encryption   │  │              │  │  │
│  │  │ • ML-KEM-1024       │  │ • Homomorphic Ops   │  │ • Secure     │  │  │
│  │  │ • ML-DSA-44/65/87   │  │ • Bootstrap         │  │   Channel    │  │  │
│  │  │ • SLH-DSA (planned) │  │ • GPU Acceleration  │  │ • Key Trans  │  │  │
│  │  └─────────┬───────────┘  └──────────┬──────────┘  └──────┬───────┘  │  │
│  │            │                         │                     │          │  │
│  │            ▼                         ▼                     ▼          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Cryptographic Backends                        │ │  │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │ │  │
│  │  │  │   liboqs    │  │ DESILO FHE  │  │   Hardware Backends     │  │ │  │
│  │  │  │  (v0.10.1)  │  │  (v5.5.0)   │  │  • GPU (CUDA/cuQuantum) │  │ │  │
│  │  │  │             │  │             │  │  • HSM (planned)        │  │ │  │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Post-Quantum Cryptography Layer

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PQCKeyManager                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Key Encapsulation (KEM)                       │   │
│  │                                                                   │   │
│  │   ML-KEM-512          ML-KEM-768           ML-KEM-1024           │   │
│  │   ┌──────────┐       ┌──────────┐         ┌──────────┐          │   │
│  │   │ Level 1  │       │ Level 3  │         │ Level 5  │          │   │
│  │   │ 128-bit  │       │ 192-bit  │         │ 256-bit  │          │   │
│  │   │ pk: 800B │       │ pk: 1184B│         │ pk: 1568B│          │   │
│  │   │ ct: 768B │       │ ct: 1088B│         │ ct: 1568B│          │   │
│  │   └──────────┘       └──────────┘         └──────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Digital Signatures                            │   │
│  │                                                                   │   │
│  │   ML-DSA-44           ML-DSA-65            ML-DSA-87             │   │
│  │   ┌──────────┐       ┌──────────┐         ┌──────────┐          │   │
│  │   │ Level 2  │       │ Level 3  │         │ Level 5  │          │   │
│  │   │ pk: 1312B│       │ pk: 1952B│         │ pk: 2592B│          │   │
│  │   │ sig:2420B│       │ sig:3293B│         │ sig:4595B│          │   │
│  │   └──────────┘       └──────────┘         └──────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Methods:                                                               │
│  • generate_keypair(algorithm) → (public_key, secret_key)              │
│  • encapsulate(public_key, algorithm) → (ciphertext, shared_secret)    │
│  • decapsulate(ciphertext, secret_key, algorithm) → shared_secret      │
│  • sign(message, secret_key, algorithm) → signature                    │
│  • verify(message, signature, public_key, algorithm) → bool            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 2. Fully Homomorphic Encryption Layer

```
┌────────────────────────────────────────────────────────────────────────┐
│                           FHEEngine (CKKS)                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Key Generation                              │   │
│  │                                                                   │   │
│  │   Secret Key ──┬──▶ Public Key                                   │   │
│  │                ├──▶ Relinearization Key                          │   │
│  │                ├──▶ Rotation Key                                 │   │
│  │                ├──▶ Conjugation Key                              │   │
│  │                └──▶ Bootstrap Key                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Homomorphic Operations                         │   │
│  │                                                                   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│  │   │   Add    │  │ Multiply │  │  Square  │  │  Rotate  │       │   │
│  │   │  0 level │  │  1 level │  │  1 level │  │  0 level │       │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │   │
│  │                                                                   │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                      │   │
│  │   │  Negate  │  │Polynomial│  │Bootstrap │                      │   │
│  │   │  0 level │  │  N level │  │Level Ref │                      │   │
│  │   └──────────┘  └──────────┘  └──────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Level Management                              │   │
│  │                                                                   │   │
│  │   Max Level (60) ──▶ ... ──▶ Level 8 ──▶ Bootstrap ──▶ Max      │   │
│  │                                                                   │   │
│  │   CRITICAL: Bootstrap requires values in [-1, 1] range!          │   │
│  │   Use FHEValueScaler for automatic scaling                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 3. Hybrid Cryptography Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Secure Channel Establishment                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│     Client (Bob)                              Server (Alice)            │
│     ────────────                              ──────────────            │
│                                                                         │
│  1. Generate ML-KEM keypair                                            │
│     (pk_kem, sk_kem) ← KeyGen()                                        │
│                                                                         │
│  2. Generate ML-DSA keypair                                            │
│     (pk_sig, sk_sig) ← KeyGen()                                        │
│                                                                         │
│  3. ──────────────── pk_kem, pk_sig ────────────────▶                  │
│                                                                         │
│                                               4. Encapsulate           │
│                                                  (ct, ss) ← Encaps(pk) │
│                                                                         │
│                                               5. Sign ciphertext       │
│                                                  sig ← Sign(ct, sk)    │
│                                                                         │
│  6. ◀──────────────── ct, sig ─────────────────────                    │
│                                                                         │
│  7. Verify signature                                                   │
│     Verify(ct, sig, pk_sig) = true                                     │
│                                                                         │
│  8. Decapsulate                                                        │
│     ss ← Decaps(ct, sk_kem)                                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                     Shared Secret Established                           │
│             (Both parties have same 32-byte shared secret)             │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Encrypted Computation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FHE Computation Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Data Preparation (Client Side)                           │  │
│  │                                                                   │  │
│  │   Raw Data ──▶ Normalize ──▶ Encode ──▶ Encrypt ──▶ Ciphertext   │  │
│  │   [100, 200]    [0.1, 0.2]   [slots]    CKKS      ct_data        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: Secure Transport (PQC Protected)                         │  │
│  │                                                                   │  │
│  │   ct_data ──▶ ML-KEM Encrypt ──▶ Sign ──▶ Transport ──▶ Verify   │  │
│  │              (shared secret)    ML-DSA    TLS 1.3     signature  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: Homomorphic Computation (Server Side)                    │  │
│  │                                                                   │  │
│  │   ct_data ──▶ Add ──▶ Multiply ──▶ Bootstrap ──▶ ct_result       │  │
│  │              (+10)    (*weights)   (if needed)                   │  │
│  │                                                                   │  │
│  │   Level: 60 ──▶ 60 ──▶ 59 ──▶ ... ──▶ 8 ──▶ 60 (bootstrap)      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 4: Result Retrieval (Client Side)                           │  │
│  │                                                                   │  │
│  │   ct_result ──▶ Decrypt ──▶ Decode ──▶ Denormalize ──▶ Result    │  │
│  │                CKKS       [slots]     [scaled]       [110, 220]  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Docker Compose Services

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Docker Compose Deployment                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Load Balancer / Ingress                       │   │
│  │                       (nginx / traefik)                          │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         │                    │                    │                    │
│         ▼                    ▼                    ▼                    │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │   api:8000  │      │ api-gpu:8001│      │   redis     │            │
│  │             │      │             │      │   :6379     │            │
│  │  FastAPI    │      │  FastAPI    │      │             │            │
│  │  CPU Mode   │      │  GPU Mode   │      │  Session    │            │
│  │             │      │  CUDA 12.x  │      │  Cache      │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
│         │                    │                    │                    │
│         └────────────────────┴────────────────────┘                    │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Monitoring Stack                              │  │
│  │                                                                   │  │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │  │
│  │   │ prometheus  │──────│   grafana   │      │   alertmgr  │      │  │
│  │   │    :9090    │      │    :3000    │      │    :9093    │      │  │
│  │   └─────────────┘      └─────────────┘      └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Network: pqc-fhe-network (bridge)                                     │
│  Volumes: redis-data, prometheus-data, grafana-data                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Kubernetes Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Namespace: pqc-fhe                          │    │
│  │                                                                 │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │    │
│  │  │  Deployment  │    │  Deployment  │    │ StatefulSet  │     │    │
│  │  │   api (3)    │    │  api-gpu (2) │    │  redis (1)   │     │    │
│  │  │              │    │              │    │              │     │    │
│  │  │ replicas: 3  │    │ replicas: 2  │    │ replicas: 1  │     │    │
│  │  │ cpu: 2000m   │    │ gpu: 1       │    │ storage: 10G │     │    │
│  │  │ mem: 4Gi     │    │ mem: 8Gi     │    │ mem: 2Gi     │     │    │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │    │
│  │         │                   │                   │              │    │
│  │         ▼                   ▼                   ▼              │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │    │
│  │  │   Service    │    │   Service    │    │   Service    │     │    │
│  │  │  ClusterIP   │    │  ClusterIP   │    │  ClusterIP   │     │    │
│  │  │   :8000      │    │   :8001      │    │   :6379      │     │    │
│  │  └──────┬───────┘    └──────┬───────┘    └──────────────┘     │    │
│  │         │                   │                                  │    │
│  │         └─────────┬─────────┘                                  │    │
│  │                   ▼                                            │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │                    Ingress Controller                    │  │    │
│  │  │                                                          │  │    │
│  │  │   api.pqc-fhe.example.com ──────▶ api:8000              │  │    │
│  │  │   gpu.pqc-fhe.example.com ──────▶ api-gpu:8001          │  │    │
│  │  │                                                          │  │    │
│  │  │   TLS: Let's Encrypt (cert-manager)                     │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │                    ConfigMaps & Secrets                  │  │    │
│  │  │                                                          │  │    │
│  │  │   • pqc-fhe-config (API settings)                       │  │    │
│  │  │   • pqc-fhe-secrets (API keys, TLS certs)              │  │    │
│  │  │   • fhe-keys (FHE key material - encrypted)            │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Threat Model & Mitigations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Security Layers                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Threat: Quantum Computer Attack                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Mitigation: Post-Quantum Cryptography                           │   │
│  │                                                                   │   │
│  │   • ML-KEM for key exchange (lattice-based, quantum-resistant)  │   │
│  │   • ML-DSA for signatures (lattice-based, quantum-resistant)    │   │
│  │   • Hybrid mode: X25519 + ML-KEM (defense in depth)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Threat: Data Exposure During Processing                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Mitigation: Fully Homomorphic Encryption                        │   │
│  │                                                                   │   │
│  │   • Data encrypted at rest, in transit, AND during computation  │   │
│  │   • Server never sees plaintext data                            │   │
│  │   • Results remain encrypted until client decrypts              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Threat: API Abuse                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Mitigation: Defense in Depth                                    │   │
│  │                                                                   │   │
│  │   • API key authentication (X-API-Key header)                   │   │
│  │   • Rate limiting (configurable per endpoint)                   │   │
│  │   • Input validation (Pydantic schemas)                         │   │
│  │   • TLS 1.3 encryption (with PQC key exchange planned)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Threat: Side-Channel Attacks                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Mitigation: Constant-Time Operations                            │   │
│  │                                                                   │   │
│  │   • liboqs provides constant-time implementations               │   │
│  │   • DESILO FHE uses noise for security                          │   │
│  │   • Avoid data-dependent branching in critical paths            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Operation Latency (Typical)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Performance Benchmarks                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Post-Quantum Cryptography (liboqs)                                    │
│  ─────────────────────────────────                                     │
│                                                                         │
│  Operation          ML-KEM-512   ML-KEM-768   ML-KEM-1024             │
│  ──────────────────────────────────────────────────────────            │
│  Key Generation     0.03 ms      0.05 ms      0.07 ms                 │
│  Encapsulation      0.04 ms      0.06 ms      0.08 ms                 │
│  Decapsulation      0.04 ms      0.05 ms      0.07 ms                 │
│                                                                         │
│  Operation          ML-DSA-44    ML-DSA-65    ML-DSA-87               │
│  ──────────────────────────────────────────────────────────            │
│  Key Generation     0.08 ms      0.13 ms      0.20 ms                 │
│  Signing            0.15 ms      0.28 ms      0.40 ms                 │
│  Verification       0.06 ms      0.10 ms      0.15 ms                 │
│                                                                         │
│  Fully Homomorphic Encryption (DESILO CKKS)                            │
│  ──────────────────────────────────────────                            │
│                                                                         │
│  Operation              CPU (ms)     GPU (ms)     Speedup             │
│  ──────────────────────────────────────────────────────────            │
│  Encryption (8K slots)     50           10          5x                │
│  Decryption (8K slots)     30            8          3.75x             │
│  Add (ct + ct)              2           0.5         4x                │
│  Multiply (ct * ct)       100           15          6.67x             │
│  Bootstrap              2000          300          6.67x             │
│                                                                         │
│  Note: GPU performance requires NVIDIA CUDA 12.x compatible GPU        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Extension Points

### Plugin Architecture (Planned)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Extensibility Framework                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Plugin Interface                             │   │
│  │                                                                   │   │
│  │   class CryptoBackend(Protocol):                                 │   │
│  │       def generate_keypair() -> tuple[bytes, bytes]              │   │
│  │       def encrypt(data: bytes, key: bytes) -> bytes              │   │
│  │       def decrypt(data: bytes, key: bytes) -> bytes              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         │                    │                    │                    │
│         ▼                    ▼                    ▼                    │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │   liboqs    │      │   PQClean   │      │ BouncyCastle│            │
│  │   Backend   │      │   Backend   │      │   Backend   │            │
│  │  (default)  │      │  (planned)  │      │  (planned)  │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     FHE Backend Interface                        │   │
│  │                                                                   │   │
│  │   class FHEBackend(Protocol):                                    │   │
│  │       def encrypt(data: list[float]) -> Ciphertext               │   │
│  │       def compute(ct: Ciphertext, op: str) -> Ciphertext         │   │
│  │       def decrypt(ct: Ciphertext) -> list[float]                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         │                    │                    │                    │
│         ▼                    ▼                    ▼                    │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │  DESILO FHE │      │   OpenFHE   │      │   SEAL      │            │
│  │   Backend   │      │   Backend   │      │   Backend   │            │
│  │  (default)  │      │  (planned)  │      │  (planned)  │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. **NIST Post-Quantum Cryptography Standards**
   - FIPS 203: ML-KEM (Module-Lattice Key Encapsulation Mechanism)
   - FIPS 204: ML-DSA (Module-Lattice Digital Signature Algorithm)
   - FIPS 205: SLH-DSA (Stateless Hash-Based Digital Signature Algorithm)

2. **Homomorphic Encryption Standards**
   - CKKS: Cheon-Kim-Kim-Song scheme for approximate arithmetic
   - DESILO FHE: https://fhe.desilo.dev/latest/

3. **Libraries**
   - liboqs v0.10.1: https://openquantumsafe.org/liboqs/
   - DESILO FHE v5.5.0: https://fhe.desilo.dev/

---

*Architecture Document v1.0.0 - Last Updated: 2025-01*
