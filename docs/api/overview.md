# API Overview

The PQC-FHE Integration library provides multiple interfaces for accessing cryptographic operations.

## Available Interfaces

### Python Library

Direct Python API for maximum control and performance.

```python
from pqc_fhe_integration import PQCKeyManager, FHEEngine, HybridCryptoManager
```

### REST API

HTTP-based API for language-agnostic integration.

```
Base URL: http://localhost:8000/api/v1
```

### WebSocket API

Real-time bidirectional communication for streaming operations.

```
Endpoint: ws://localhost:8765
```

### Command Line Interface

CLI for scripting and interactive use.

```bash
pqc-fhe [command] [options]
```

## Core Components

### PQCKeyManager

Handles all post-quantum cryptographic operations:

- Key generation (KEM and signatures)
- Key encapsulation/decapsulation
- Digital signatures and verification

[Full Reference →](pqc_key_manager.md)

### FHEEngine

Manages fully homomorphic encryption:

- Encryption/decryption
- Homomorphic operations
- Context management

[Full Reference →](fhe_engine.md)

### HybridCryptoManager

Combines PQC and FHE for complete workflows:

- Secure key exchange with FHE
- Signed encrypted computation
- End-to-end secure pipelines

[Full Reference →](hybrid_manager.md)

## Quick Reference

### PQC Operations

| Operation | Python | REST | CLI |
|-----------|--------|------|-----|
| KEM KeyGen | `pqc.generate_kem_keypair()` | `POST /pqc/keygen` | `pqc-fhe keygen` |
| Encapsulate | `pqc.encapsulate()` | `POST /pqc/encapsulate` | `pqc-fhe encapsulate` |
| Decapsulate | `pqc.decapsulate()` | `POST /pqc/decapsulate` | `pqc-fhe decapsulate` |
| Sign KeyGen | `pqc.generate_signature_keypair()` | `POST /pqc/keygen` | `pqc-fhe keygen` |
| Sign | `pqc.sign()` | `POST /pqc/sign` | `pqc-fhe sign` |
| Verify | `pqc.verify()` | `POST /pqc/verify` | `pqc-fhe verify` |

### FHE Operations

| Operation | Python | REST | CLI |
|-----------|--------|------|-----|
| Encrypt | `fhe.encrypt()` | `POST /fhe/encrypt` | `pqc-fhe fhe-encrypt` |
| Decrypt | `fhe.decrypt()` | `POST /fhe/decrypt` | `pqc-fhe fhe-decrypt` |
| Add | `fhe.add()` | `POST /fhe/compute` | `pqc-fhe fhe-compute` |
| Multiply | `fhe.multiply()` | `POST /fhe/compute` | `pqc-fhe fhe-compute` |
| Scalar Ops | `fhe.multiply_scalar()` | `POST /fhe/compute` | `pqc-fhe fhe-compute` |

## Error Handling

### Error Response Format

All APIs return errors in a consistent format:

```json
{
  "status": "error",
  "error_code": "INVALID_ALGORITHM",
  "message": "Algorithm 'ML-KEM-999' is not supported",
  "details": {
    "supported_algorithms": ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_ALGORITHM` | Unsupported algorithm specified |
| `INVALID_KEY` | Malformed or corrupted key |
| `DECAPSULATION_FAILED` | KEM decapsulation failed |
| `SIGNATURE_INVALID` | Signature verification failed |
| `FHE_OVERFLOW` | Value exceeded FHE precision |
| `RATE_LIMITED` | Too many requests |

## Authentication

### API Key Authentication

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/pqc/keygen
```

### JWT Authentication

```bash
curl -H "Authorization: Bearer eyJ..." http://localhost:8000/api/v1/pqc/keygen
```

## Rate Limiting

Default limits (configurable):

| Endpoint Type | Limit |
|--------------|-------|
| Key Generation | 100/minute |
| Encryption | 500/minute |
| Computation | 1000/minute |

## Versioning

The API uses URL versioning:

- Current: `/api/v1`
- Deprecated: `/api/v0` (removed after 6 months)

## SDKs

Official SDKs are available for:

- Python (native)
- JavaScript/TypeScript
- Go
- Rust

## Next Steps

- [PQC Key Manager Reference](pqc_key_manager.md)
- [FHE Engine Reference](fhe_engine.md)
- [REST API Reference](rest_api.md)
- [WebSocket API Reference](websocket_api.md)
- [CLI Reference](cli.md)
