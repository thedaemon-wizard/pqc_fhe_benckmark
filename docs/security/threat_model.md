# Threat Model

Comprehensive security analysis for PQC-FHE integrated systems.

## Executive Summary

This threat model analyzes security risks for systems combining Post-Quantum 
Cryptography (PQC) with Fully Homomorphic Encryption (FHE), identifying attack 
vectors, mitigations, and residual risks.

### Scope

| Component | Coverage |
|-----------|----------|
| PQC Key Management | ML-KEM, ML-DSA, SLH-DSA |
| FHE Operations | CKKS scheme encryption/computation |
| Network Communication | REST API, WebSocket, TLS |
| Key Storage | Memory, disk, HSM integration |
| Deployment | Kubernetes, Docker, bare metal |

### Security Objectives

1. **Confidentiality**: Data remains encrypted at rest, in transit, and during computation
2. **Integrity**: Detect unauthorized modifications to data and cryptographic keys
3. **Authenticity**: Verify identity of communicating parties
4. **Availability**: Maintain service under attack conditions
5. **Quantum Resistance**: Protect against both classical and quantum adversaries

## Threat Actors

### T1: Nation-State Adversary

| Attribute | Description |
|-----------|-------------|
| **Capability** | Unlimited computational resources, quantum computers |
| **Motivation** | Intelligence gathering, sabotage |
| **Attack Methods** | Cryptanalysis, supply chain compromise, zero-days |
| **Resources** | $1B+ budget, dedicated teams |

**Relevant Attacks:**
- Harvest Now, Decrypt Later (HNDL)
- Side-channel attacks with physical access
- Supply chain infiltration

### T2: Organized Crime

| Attribute | Description |
|-----------|-------------|
| **Capability** | Significant but not unlimited resources |
| **Motivation** | Financial gain, data theft |
| **Attack Methods** | Ransomware, credential theft, API abuse |
| **Resources** | $10M budget, contracted expertise |

**Relevant Attacks:**
- Credential stuffing and phishing
- API abuse for data exfiltration
- DDoS for extortion

### T3: Malicious Insider

| Attribute | Description |
|-----------|-------------|
| **Capability** | Authorized access to systems |
| **Motivation** | Financial gain, revenge, coercion |
| **Attack Methods** | Privilege abuse, data exfiltration |
| **Resources** | Legitimate access credentials |

**Relevant Attacks:**
- Key material theft
- Audit log manipulation
- Backdoor installation

### T4: Opportunistic Attacker

| Attribute | Description |
|-----------|-------------|
| **Capability** | Public tools, limited expertise |
| **Motivation** | Easy targets, curiosity |
| **Attack Methods** | Known CVEs, misconfigurations |
| **Resources** | Minimal investment |

**Relevant Attacks:**
- Exploitation of unpatched vulnerabilities
- Credential reuse attacks
- Misconfiguration exploitation

## Attack Surface Analysis

### AS1: Network Endpoints

```
┌─────────────────────────────────────────────────────────────┐
│                     Attack Surface                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Internet ──▶ [Load Balancer] ──▶ [API Gateway]             │
│                     │                    │                   │
│                     │                    ▼                   │
│                     │              [REST API]                │
│                     │                    │                   │
│                     ▼                    ▼                   │
│              [WebSocket Server] ──▶ [PQC-FHE Core]          │
│                                          │                   │
│                                          ▼                   │
│                                   [Key Storage]              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Surface | Exposure | Risk Level |
|---------|----------|------------|
| REST API | Public | High |
| WebSocket | Public | High |
| Admin API | Internal | Medium |
| Key Storage | Internal | Critical |
| Metrics | Internal | Low |

### AS2: Data Assets

| Asset | Classification | Impact if Compromised |
|-------|---------------|----------------------|
| FHE Secret Keys | Critical | Complete data exposure |
| PQC Private Keys | Critical | Identity theft, MITM |
| Session Keys | High | Session hijacking |
| Encrypted Data | Medium | Privacy breach (if decrypted) |
| Audit Logs | Medium | Evidence tampering |
| Configuration | Medium | Service compromise |

### AS3: Dependencies

| Dependency | Risk | Mitigation |
|------------|------|------------|
| liboqs | High | Vendor security updates, code review |
| DESILO FHE | High | Regular updates, parameter validation |
| OpenSSL | High | Patch management, CVE monitoring |
| Python runtime | Medium | Container isolation, minimal base |
| OS libraries | Medium | Minimal container images |

## Threat Scenarios

### TS1: Harvest Now, Decrypt Later (HNDL)

**Description:** Adversary captures encrypted traffic today to decrypt with future quantum computers.

**Attack Flow:**
```
1. Adversary performs passive network interception
2. Stores all encrypted key exchanges and data
3. Waits for cryptographically-relevant quantum computer
4. Decrypts historical RSA/ECDH key exchanges
5. Recovers session keys and decrypts data
```

**Likelihood:** Certain (for valuable data)
**Impact:** Critical (complete historical data exposure)
**Timeline:** 10-15 years (estimated)

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| ML-KEM-768 key exchange | Prevents quantum decryption | ✅ Implemented |
| Hybrid PQC+Classical | Defense in depth | ✅ Implemented |
| Forward secrecy | Limits exposure window | ✅ Implemented |
| Key rotation | Reduces key lifetime | ✅ Implemented |

**Residual Risk:** Low (with PQC implementation)

### TS2: FHE Noise Budget Exploitation

**Description:** Attacker manipulates inputs to exhaust FHE noise budget, causing decryption errors.

**Attack Flow:**
```
1. Attacker identifies FHE-enabled computation endpoint
2. Crafts inputs that maximize noise growth
3. Submits computation requests
4. FHE operations accumulate excessive noise
5. Decryption produces incorrect results
```

**Likelihood:** Medium
**Impact:** High (data integrity compromise)

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| Noise budget monitoring | Early detection | ✅ Implemented |
| Input validation | Prevents malicious inputs | ✅ Implemented |
| Bootstrap threshold | Automatic noise refresh | ✅ Implemented |
| Rate limiting | Limits attack frequency | ✅ Implemented |

**Residual Risk:** Low

### TS3: Side-Channel Attack on PQC Implementation

**Description:** Attacker extracts key material through timing, power, or electromagnetic analysis.

**Attack Flow:**
```
1. Attacker gains physical or VM co-location access
2. Monitors timing variations during key operations
3. Performs statistical analysis of measurements
4. Recovers partial or complete key material
5. Impersonates victim or decrypts data
```

**Likelihood:** Low (requires proximity)
**Impact:** Critical (key compromise)

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| Constant-time implementation | Prevents timing leaks | ✅ Verified |
| Memory protection | Prevents cache attacks | ✅ Implemented |
| Key blinding | Masks operations | ⚠️ Partial |
| HSM integration | Hardware isolation | 📋 Planned |

**Residual Risk:** Medium (until HSM deployment)

### TS4: API Authentication Bypass

**Description:** Attacker bypasses authentication to access protected endpoints.

**Attack Flow:**
```
1. Attacker probes API for authentication weaknesses
2. Discovers JWT validation flaw or default credentials
3. Forges authentication tokens
4. Accesses protected resources
5. Exfiltrates data or compromises keys
```

**Likelihood:** Medium
**Impact:** High

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| JWT with Ed25519 signatures | Strong authentication | ✅ Implemented |
| Token expiration (15 min) | Limits exposure | ✅ Implemented |
| Rate limiting | Prevents brute force | ✅ Implemented |
| IP allowlisting | Reduces attack surface | 📋 Optional |

**Residual Risk:** Low

### TS5: Key Storage Compromise

**Description:** Attacker gains access to stored cryptographic keys.

**Attack Flow:**
```
1. Attacker exploits application vulnerability
2. Gains code execution on server
3. Reads key material from memory or disk
4. Exfiltrates keys to external server
5. Decrypts all protected data
```

**Likelihood:** Medium
**Impact:** Critical

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| Memory encryption | Protects in-memory keys | ✅ Implemented |
| Key derivation | No stored master keys | ✅ Implemented |
| Secure enclave | Hardware isolation | 📋 Planned |
| Key rotation | Limits exposure window | ✅ Implemented |
| Audit logging | Detects access | ✅ Implemented |

**Residual Risk:** Medium (until secure enclave)

### TS6: Denial of Service (DoS)

**Description:** Attacker overwhelms system to prevent legitimate access.

**Attack Flow:**
```
1. Attacker identifies resource-intensive endpoints
2. Generates high volume of requests
3. System resources exhausted (CPU, memory, connections)
4. Legitimate users unable to access service
5. Business impact from service unavailability
```

**Likelihood:** High
**Impact:** Medium (availability only)

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| Rate limiting | Limits request volume | ✅ Implemented |
| Connection limits | Prevents exhaustion | ✅ Implemented |
| Auto-scaling | Absorbs spikes | ✅ Implemented |
| CDN/WAF | Filters attacks | 📋 Recommended |

**Residual Risk:** Low

### TS7: Supply Chain Attack

**Description:** Attacker compromises dependency to inject malicious code.

**Attack Flow:**
```
1. Attacker compromises upstream package
2. Malicious code included in release
3. Organization installs compromised package
4. Malicious code executes with application privileges
5. Attacker gains persistent access
```

**Likelihood:** Low
**Impact:** Critical

**Mitigations:**

| Control | Effectiveness | Status |
|---------|--------------|--------|
| Dependency pinning | Prevents auto-updates | ✅ Implemented |
| SBOM generation | Tracks components | ✅ Implemented |
| Vulnerability scanning | Detects known issues | ✅ Implemented |
| Signed packages | Verifies integrity | ⚠️ Partial |
| Private registry | Controls sources | 📋 Recommended |

**Residual Risk:** Medium

## Attack Trees

### AT1: Compromise PQC Private Key

```
[Compromise PQC Private Key]
├── [Extract from Memory]
│   ├── Memory dump via vulnerability (M)
│   ├── Cold boot attack (L)
│   └── Side-channel on VM (M)
├── [Extract from Storage]
│   ├── Disk access via RCE (M)
│   ├── Backup compromise (M)
│   └── Insider access (M)
├── [Cryptanalysis]
│   ├── Implementation flaw (L)
│   ├── Quantum computer (L, future)
│   └── Mathematical breakthrough (VL)
└── [Social Engineering]
    ├── Phishing admin credentials (M)
    └── Insider recruitment (L)

Legend: VL=Very Low, L=Low, M=Medium, H=High
```

### AT2: Decrypt FHE-Protected Data

```
[Decrypt FHE-Protected Data]
├── [Obtain Secret Key]
│   ├── Server compromise (M)
│   ├── Key storage attack (M)
│   └── Insider theft (M)
├── [Break FHE Scheme]
│   ├── Parameter weakness (VL)
│   ├── Implementation bug (L)
│   └── Cryptanalytic advance (VL)
├── [Corrupt Computation]
│   ├── Noise budget exhaustion (M)
│   ├── Input manipulation (M)
│   └── Parameter tampering (L)
└── [Bypass FHE]
    ├── Intercept before encryption (M)
    └── Access after decryption (M)
```

## Risk Assessment Matrix

| Threat | Likelihood | Impact | Risk Score | Priority |
|--------|------------|--------|------------|----------|
| TS1: HNDL | Certain | Critical | **Critical** | P1 |
| TS2: Noise Exploit | Medium | High | **High** | P2 |
| TS3: Side-Channel | Low | Critical | **Medium** | P3 |
| TS4: Auth Bypass | Medium | High | **High** | P2 |
| TS5: Key Storage | Medium | Critical | **High** | P1 |
| TS6: DoS | High | Medium | **Medium** | P3 |
| TS7: Supply Chain | Low | Critical | **Medium** | P3 |

## Security Controls

### C1: Cryptographic Controls

```python
class CryptographicControls:
    """
    Security controls for cryptographic operations
    
    References:
    - NIST FIPS 203, 204, 205
    - NIST SP 800-57 Key Management
    """
    
    # PQC Algorithm Selection
    PQC_KEM_ALGORITHM = "ML-KEM-768"      # NIST Level 3
    PQC_SIGN_ALGORITHM = "ML-DSA-65"      # NIST Level 3
    HYBRID_CLASSICAL = "X25519"            # Backup
    
    # FHE Parameters (128-bit security)
    FHE_POLY_MODULUS_DEGREE = 8192
    FHE_COEFF_MODULUS_BITS = [60, 40, 40, 60]
    FHE_SCALE = 2**40
    
    # Key Lifetimes
    SESSION_KEY_LIFETIME_SECONDS = 3600    # 1 hour
    SIGNING_KEY_LIFETIME_DAYS = 365        # 1 year
    KEM_KEY_LIFETIME_DAYS = 90             # 90 days
    
    # Secure Defaults
    MIN_ENTROPY_BITS = 256
    PBKDF2_ITERATIONS = 600000
    ARGON2_MEMORY_KB = 65536
```

### C2: Access Controls

| Control | Implementation |
|---------|---------------|
| Authentication | JWT with Ed25519 signatures |
| Authorization | RBAC with least privilege |
| Session Management | 15-minute token expiry |
| API Rate Limiting | Per-endpoint limits |
| Network Segmentation | Internal-only key storage |

### C3: Audit and Monitoring

```python
# Required Audit Events
AUDIT_EVENTS = {
    "key_generation": "CRITICAL",
    "key_access": "HIGH",
    "key_rotation": "HIGH",
    "authentication_success": "INFO",
    "authentication_failure": "WARNING",
    "authorization_failure": "WARNING",
    "fhe_encryption": "INFO",
    "fhe_decryption": "HIGH",
    "fhe_computation": "INFO",
    "configuration_change": "HIGH",
    "admin_action": "CRITICAL"
}
```

### C4: Incident Response Procedures

| Severity | Response Time | Escalation |
|----------|--------------|------------|
| Critical | 15 minutes | Immediate executive notification |
| High | 1 hour | Security team lead |
| Medium | 4 hours | On-call engineer |
| Low | 24 hours | Normal queue |

## Compliance Mapping

### NIST Cybersecurity Framework

| Function | Category | Controls |
|----------|----------|----------|
| **Identify** | Asset Management | Key inventory, SBOM |
| **Protect** | Access Control | RBAC, JWT, rate limiting |
| **Protect** | Data Security | PQC, FHE, encryption |
| **Detect** | Anomaly Detection | Audit logs, metrics |
| **Respond** | Response Planning | Incident procedures |
| **Recover** | Recovery Planning | Key rotation, backup |

### NIST PQC Standards

| Standard | Requirement | Status |
|----------|-------------|--------|
| FIPS 203 | ML-KEM implementation | ✅ Compliant |
| FIPS 204 | ML-DSA implementation | ✅ Compliant |
| FIPS 205 | SLH-DSA implementation | ✅ Compliant |
| IR 8547 | Migration timeline | ✅ On track |

## Recommendations

### Immediate (P1)

1. **Deploy Hardware Security Module (HSM)**
   - Protect PQC private keys in hardware
   - Estimated effort: 2-4 weeks

2. **Implement Key Escrow for FHE**
   - Enable key recovery for compliance
   - Estimated effort: 1-2 weeks

### Short-term (P2)

1. **Add Web Application Firewall (WAF)**
   - Filter application-layer attacks
   - Estimated effort: 1 week

2. **Enable Mutual TLS (mTLS)**
   - Client certificate authentication
   - Estimated effort: 1-2 weeks

3. **Implement Anomaly Detection**
   - ML-based threat detection
   - Estimated effort: 2-4 weeks

### Long-term (P3)

1. **Post-Quantum TLS 1.3**
   - Full transport-layer PQC
   - Waiting for standardization

2. **Secure Enclave Integration**
   - Intel SGX / AMD SEV support
   - Estimated effort: 4-8 weeks

3. **Formal Verification**
   - Cryptographic protocol proofs
   - Estimated effort: 3-6 months

## Appendix A: Vulnerability Classes

### PQC-Specific Vulnerabilities

| Class | Description | Example |
|-------|-------------|---------|
| Timing Oracle | Key-dependent timing | ML-KEM decapsulation |
| Fault Injection | Corrupted computation | ML-DSA signing |
| Parameter Misuse | Weak parameters | Reduced security level |
| Implementation Bug | Coding errors | Memory leaks |

### FHE-Specific Vulnerabilities

| Class | Description | Example |
|-------|-------------|---------|
| Noise Overflow | Excessive noise | Incorrect results |
| Parameter Leakage | Side-channel | Timing on operations |
| Ciphertext Malleability | Unauthorized modification | Integrity attacks |
| Key Reuse | Same key multiple times | Distinguishing attacks |

## Appendix B: Security Testing Checklist

### Pre-Deployment

- [ ] Static code analysis (SAST)
- [ ] Dependency vulnerability scan
- [ ] Secret scanning
- [ ] Container image scanning
- [ ] Configuration review

### Deployment

- [ ] TLS configuration verification
- [ ] Authentication testing
- [ ] Authorization testing
- [ ] Rate limiting verification
- [ ] Logging verification

### Post-Deployment

- [ ] Penetration testing
- [ ] Fuzzing of cryptographic inputs
- [ ] Side-channel analysis
- [ ] Red team exercise
- [ ] Compliance audit

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-15 | Security Team | Initial release |
| 1.1 | 2025-02-01 | Security Team | Added supply chain threats |
| 1.2 | 2025-03-01 | Security Team | Updated risk scores |

**Classification:** Internal Use Only
**Review Cycle:** Quarterly
**Next Review:** 2025-04-15
