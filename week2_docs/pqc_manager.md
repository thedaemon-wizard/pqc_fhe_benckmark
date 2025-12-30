# PQC Key Manager API Reference

Complete API reference for the Post-Quantum Cryptography Key Manager.

## Class: PQCKeyManager

The `PQCKeyManager` class provides post-quantum cryptographic operations using NIST-standardized algorithms.

### Constructor

```python
PQCKeyManager(
    kem_algorithm: str = "ML-KEM-768",
    sig_algorithm: str = "ML-DSA-65"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kem_algorithm` | `str` | `"ML-KEM-768"` | Key encapsulation algorithm |
| `sig_algorithm` | `str` | `"ML-DSA-65"` | Digital signature algorithm |

**Supported Algorithms:**

| Algorithm | Type | Security Level | Key Size | NIST Standard |
|-----------|------|----------------|----------|---------------|
| ML-KEM-512 | KEM | Level 1 (128-bit) | 800 bytes (pk) | FIPS 203 |
| ML-KEM-768 | KEM | Level 3 (192-bit) | 1,184 bytes (pk) | FIPS 203 |
| ML-KEM-1024 | KEM | Level 5 (256-bit) | 1,568 bytes (pk) | FIPS 203 |
| ML-DSA-44 | Signature | Level 2 | 1,312 bytes (pk) | FIPS 204 |
| ML-DSA-65 | Signature | Level 3 | 1,952 bytes (pk) | FIPS 204 |
| ML-DSA-87 | Signature | Level 5 | 2,592 bytes (pk) | FIPS 204 |

**Example:**

```python
from pqc_fhe import PQCKeyManager

# Default configuration (ML-KEM-768, ML-DSA-65)
manager = PQCKeyManager()

# High security configuration
manager = PQCKeyManager(
    kem_algorithm="ML-KEM-1024",
    sig_algorithm="ML-DSA-87"
)
```

---

## Key Generation Methods

### generate_kem_keypair()

Generate a key encapsulation mechanism (KEM) keypair.

```python
def generate_kem_keypair(self) -> Tuple[bytes, bytes]
```

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `public_key` | `bytes` | Public key for encapsulation |
| `secret_key` | `bytes` | Secret key for decapsulation |

**Example:**

```python
public_key, secret_key = manager.generate_kem_keypair()
print(f"Public key size: {len(public_key)} bytes")
print(f"Secret key size: {len(secret_key)} bytes")
```

**Key Sizes:**

| Algorithm | Public Key | Secret Key |
|-----------|------------|------------|
| ML-KEM-512 | 800 bytes | 1,632 bytes |
| ML-KEM-768 | 1,184 bytes | 2,400 bytes |
| ML-KEM-1024 | 1,568 bytes | 3,168 bytes |

---

### generate_sig_keypair()

Generate a digital signature keypair.

```python
def generate_sig_keypair(self) -> Tuple[bytes, bytes]
```

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `public_key` | `bytes` | Public key for verification |
| `secret_key` | `bytes` | Secret key for signing |

**Example:**

```python
public_key, secret_key = manager.generate_sig_keypair()
print(f"Signature public key: {len(public_key)} bytes")
```

**Key Sizes:**

| Algorithm | Public Key | Secret Key |
|-----------|------------|------------|
| ML-DSA-44 | 1,312 bytes | 2,528 bytes |
| ML-DSA-65 | 1,952 bytes | 4,000 bytes |
| ML-DSA-87 | 2,592 bytes | 4,864 bytes |

---

## Key Encapsulation Methods

### encapsulate()

Encapsulate a shared secret using a public key.

```python
def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `public_key` | `bytes` | Recipient's public key |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `ciphertext` | `bytes` | Encapsulated ciphertext |
| `shared_secret` | `bytes` | 32-byte shared secret |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Invalid public key |
| `RuntimeError` | Encapsulation failed |

**Example:**

```python
# Sender encapsulates
ciphertext, shared_secret = manager.encapsulate(recipient_public_key)

# shared_secret is always 32 bytes
assert len(shared_secret) == 32
```

**Ciphertext Sizes:**

| Algorithm | Ciphertext |
|-----------|------------|
| ML-KEM-512 | 768 bytes |
| ML-KEM-768 | 1,088 bytes |
| ML-KEM-1024 | 1,568 bytes |

---

### decapsulate()

Decapsulate to recover the shared secret.

```python
def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ciphertext` | `bytes` | Encapsulated ciphertext |
| `secret_key` | `bytes` | Recipient's secret key |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `shared_secret` | `bytes` | 32-byte shared secret |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Invalid ciphertext or secret key |
| `RuntimeError` | Decapsulation failed |

**Example:**

```python
# Recipient decapsulates
shared_secret = manager.decapsulate(ciphertext, secret_key)

# Verify shared secrets match
assert sender_shared_secret == shared_secret
```

---

## Digital Signature Methods

### sign()

Sign a message using the secret key.

```python
def sign(self, message: bytes, secret_key: bytes) -> bytes
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `bytes` | Message to sign |
| `secret_key` | `bytes` | Signer's secret key |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `signature` | `bytes` | Digital signature |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Invalid secret key |
| `RuntimeError` | Signing failed |

**Example:**

```python
message = b"Important document"
signature = manager.sign(message, secret_key)
print(f"Signature size: {len(signature)} bytes")
```

**Signature Sizes:**

| Algorithm | Signature Size |
|-----------|----------------|
| ML-DSA-44 | 2,420 bytes |
| ML-DSA-65 | 3,293 bytes |
| ML-DSA-87 | 4,595 bytes |

---

### verify()

Verify a signature against a message and public key.

```python
def verify(
    self, 
    message: bytes, 
    signature: bytes, 
    public_key: bytes
) -> bool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `bytes` | Original message |
| `signature` | `bytes` | Signature to verify |
| `public_key` | `bytes` | Signer's public key |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `valid` | `bool` | `True` if signature is valid |

**Example:**

```python
is_valid = manager.verify(message, signature, public_key)
if is_valid:
    print("Signature verified successfully")
else:
    print("Invalid signature!")
```

---

## Utility Methods

### get_algorithm_info()

Get information about the configured algorithms.

```python
def get_algorithm_info(self) -> Dict[str, Any]
```

**Returns:**

```python
{
    "kem": {
        "algorithm": "ML-KEM-768",
        "security_level": 3,
        "public_key_size": 1184,
        "secret_key_size": 2400,
        "ciphertext_size": 1088,
        "shared_secret_size": 32
    },
    "signature": {
        "algorithm": "ML-DSA-65",
        "security_level": 3,
        "public_key_size": 1952,
        "secret_key_size": 4000,
        "signature_size": 3293
    }
}
```

---

### export_public_keys()

Export public keys in various formats.

```python
def export_public_keys(
    self,
    kem_public_key: bytes,
    sig_public_key: bytes,
    format: str = "pem"
) -> Dict[str, str]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kem_public_key` | `bytes` | - | KEM public key |
| `sig_public_key` | `bytes` | - | Signature public key |
| `format` | `str` | `"pem"` | Output format (`"pem"`, `"base64"`, `"hex"`) |

**Returns:**

```python
{
    "kem_public_key": "-----BEGIN ML-KEM PUBLIC KEY-----\n...",
    "sig_public_key": "-----BEGIN ML-DSA PUBLIC KEY-----\n..."
}
```

---

## Error Handling

### Exception Types

```python
from pqc_fhe.exceptions import (
    PQCError,           # Base exception
    KeyGenerationError, # Key generation failed
    EncapsulationError, # Encapsulation failed
    DecapsulationError, # Decapsulation failed
    SigningError,       # Signing failed
    VerificationError,  # Verification failed
    InvalidKeyError,    # Invalid key format
)
```

### Error Handling Example

```python
from pqc_fhe import PQCKeyManager
from pqc_fhe.exceptions import PQCError, DecapsulationError

manager = PQCKeyManager()

try:
    shared_secret = manager.decapsulate(ciphertext, secret_key)
except DecapsulationError as e:
    print(f"Decapsulation failed: {e}")
except PQCError as e:
    print(f"PQC operation failed: {e}")
```

---

## Thread Safety

The `PQCKeyManager` class is thread-safe for all operations. Each method call uses independent internal state.

```python
import threading
from concurrent.futures import ThreadPoolExecutor

manager = PQCKeyManager()

def generate_keys():
    return manager.generate_kem_keypair()

# Safe to use from multiple threads
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(generate_keys) for _ in range(100)]
    results = [f.result() for f in futures]
```

---

## Performance Benchmarks

Typical performance on modern hardware (Intel Xeon, single thread):

| Operation | ML-KEM-768 | ML-DSA-65 |
|-----------|------------|-----------|
| Key Generation | ~0.1 ms | ~0.2 ms |
| Encapsulate | ~0.1 ms | N/A |
| Decapsulate | ~0.1 ms | N/A |
| Sign | N/A | ~0.5 ms |
| Verify | N/A | ~0.2 ms |

---

## Security Considerations

1. **Secret Key Storage**: Never log or expose secret keys
2. **Memory Zeroing**: Secret keys are zeroed after use
3. **Constant Time**: All operations are constant-time to prevent timing attacks
4. **Random Number Generation**: Uses OS-provided CSPRNG

---

## Related APIs

- [FHE Engine API](fhe_engine.md)
- [Hybrid Manager API](hybrid_manager.md)
- [REST API Reference](rest_api.md)
