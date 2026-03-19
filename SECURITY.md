# Security Policy

## Supported Versions

| Version | Supported          | Notes                                |
| ------- | ------------------ | ------------------------------------ |
| 1.0.x   | :white_check_mark: | Current release, actively maintained |
| < 1.0   | :x:                | Pre-release, not supported           |

## Reporting a Vulnerability

### Private Disclosure

If you discover a security vulnerability in this project, please **do not** open a public GitHub issue. Instead:

1. **Email**: Send details to the maintainers privately
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)

3. **Response Time**: 
   - Initial response within 48 hours
   - Status update within 7 days
   - Fix timeline based on severity

### What to Expect

- Acknowledgment of your report
- Regular updates on investigation progress
- Credit in release notes (unless you prefer anonymity)
- Coordinated disclosure timeline

---

## Security Architecture

### Cryptographic Standards Compliance

This library implements NIST-approved post-quantum cryptographic standards:

| Standard | Algorithm | Security Level | Status |
|----------|-----------|----------------|--------|
| FIPS 203 | ML-KEM-512/768/1024 | 1/3/5 | Implemented |
| FIPS 204 | ML-DSA-44/65/87 | 2/3/5 | Implemented |
| FIPS 205 | SLH-DSA | 1/3/5 | Planned |

### Security Levels (NIST)

| Level | Classical Security | Quantum Security |
|-------|-------------------|------------------|
| 1 | AES-128 equivalent | SIKE-p434 |
| 3 | AES-192 equivalent | SIKE-p610 |
| 5 | AES-256 equivalent | SIKE-p751 |

---

## Secure Usage Guidelines

### Key Management

```python
# GOOD: Use appropriate security level
km = PQCKeyManager()
public_key, secret_key = km.generate_keypair("ML-KEM-768")  # Level 3

# BAD: Storing keys in plaintext
with open("secret_key.txt", "w") as f:
    f.write(secret_key.hex())  # NEVER DO THIS

# GOOD: Use secure key storage
import os
# Store in secure enclave, HSM, or encrypted storage
```

### Secret Key Handling

1. **Never log secret keys**: Use redaction in logging
2. **Clear memory after use**: Zero-out sensitive data when possible
3. **Secure storage**: Use HSM, secure enclave, or encrypted storage
4. **Minimal exposure**: Keep keys in memory only when needed

```python
# GOOD: Clear sensitive data
def process_with_key(secret_key: bytes):
    try:
        result = do_operation(secret_key)
        return result
    finally:
        # Best effort memory clearing (Python limitation)
        secret_key = b'\x00' * len(secret_key)
```

### FHE Value Ranges

The CKKS scheme requires careful value management:

```python
# GOOD: Keep values in safe range for bootstrap
# DESILO FHE requires values in [-1, 1] for bootstrap
fhe = FHEEngine()
normalized_data = [x / 100.0 for x in raw_data]  # Normalize first
ct = fhe.encrypt(normalized_data)

# BAD: Large values cause bootstrap failures
ct = fhe.encrypt([1000000.0, 2000000.0])  # May cause errors
```

### API Security

When using the REST API:

```python
# GOOD: Use API key authentication
headers = {"X-API-Key": os.environ["PQC_FHE_API_KEY"]}
response = requests.post(url, json=data, headers=headers)

# GOOD: Use HTTPS in production
url = "https://api.example.com/pqc/keygen"  # HTTPS only

# BAD: Expose keys in URLs
url = f"https://api.example.com/decrypt?key={secret_key}"  # NEVER
```

---

## Known Limitations

### 1. Side-Channel Considerations

While liboqs implementations include side-channel protections, this wrapper does not add additional protections. For high-security applications:

- Use constant-time comparison for secrets
- Consider hardware-backed key storage
- Implement additional timing attack mitigations

### 2. Memory Safety

Python does not guarantee memory clearing. Sensitive data may remain in memory after deletion. For critical applications:

- Consider using `mlock` for sensitive buffers
- Use HSM for key operations
- Implement secure memory pools

### 3. Randomness

This library relies on the system's cryptographic random number generator:

```python
# We use os.urandom() which calls:
# - Linux: getrandom() syscall or /dev/urandom
# - Windows: CryptGenRandom
# - macOS: getentropy() or /dev/urandom
```

Ensure your system has sufficient entropy.

---

## Security Checklist for Deployment

### Pre-Deployment

- [ ] Dependencies updated to latest security patches
- [ ] API key authentication enabled
- [ ] HTTPS/TLS configured (TLS 1.3 recommended)
- [ ] Input validation enabled on all endpoints
- [ ] Rate limiting configured
- [ ] Logging configured (without sensitive data)

### Production Configuration

- [ ] Debug mode disabled
- [ ] Error messages sanitized
- [ ] CORS properly configured
- [ ] Security headers set (CSP, HSTS, etc.)
- [ ] Docker running as non-root user

### Monitoring

- [ ] Failed authentication alerts
- [ ] Unusual API usage patterns
- [ ] Error rate monitoring
- [ ] Resource usage anomalies

---

## Cryptographic Recommendations

### Key Sizes for Long-Term Security

For data that must remain secure beyond 2030:

| Use Case | Recommended Algorithm | Notes |
|----------|----------------------|-------|
| Key Exchange | ML-KEM-768 | Level 3, good balance |
| Digital Signatures | ML-DSA-65 | Level 3, reasonable size |
| Long-term Secrets | ML-KEM-1024 | Level 5, maximum security |
| High-security Signing | ML-DSA-87 | Level 5, larger signatures |

### Hybrid Mode Recommendations

For maximum security during transition:

```python
# Combine classical and post-quantum
hybrid = HybridCryptoManager()
channel = hybrid.establish_secure_channel(
    security_level="level3",
    use_hybrid=True  # X25519 + ML-KEM-768
)
```

---

## References

### Standards

- [NIST FIPS 203](https://csrc.nist.gov/pubs/fips/203/final) - ML-KEM Specification
- [NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final) - ML-DSA Specification
- [NIST IR 8547](https://csrc.nist.gov/pubs/ir/8547/final) - PQC Transition Guidelines

### Libraries

- [Open Quantum Safe](https://openquantumsafe.org/) - liboqs documentation
- [DESILO FHE](https://fhe.desilo.dev/) - FHE library documentation

### Research

- Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers" (CKKS)
- Bos et al., "CRYSTALS-Kyber: A CCA-Secure Module-Lattice-Based KEM"
- Ducas et al., "CRYSTALS-Dilithium: A Lattice-Based Digital Signature Scheme"

---

## Security Updates

### 2025-01-15 (v1.0.0)
- Initial release with FIPS 203/204 support
- DESILO FHE integration with safe bootstrap
- API key authentication

### Future Updates
- SLH-DSA (FIPS 205) support planned
- Hardware security module integration planned
- FIPS 140-3 validation in progress

---

*Last updated: 2025-01-15*
