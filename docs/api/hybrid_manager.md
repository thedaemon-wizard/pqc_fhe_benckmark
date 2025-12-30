# Hybrid Crypto Manager API Reference

Complete API reference for the HybridCryptoManager integrating PQC and FHE.

## Class: HybridCryptoManager

The `HybridCryptoManager` class combines post-quantum key exchange, digital signatures, and fully homomorphic encryption into a unified interface.

### Constructor

```python
HybridCryptoManager(
    kem_algorithm: str = "ML-KEM-768",
    sig_algorithm: str = "ML-DSA-65",
    fhe_poly_degree: int = 8192,
    fhe_scale: float = 2**40
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kem_algorithm` | `str` | `"ML-KEM-768"` | Key encapsulation algorithm |
| `sig_algorithm` | `str` | `"ML-DSA-65"` | Digital signature algorithm |
| `fhe_poly_degree` | `int` | `8192` | FHE polynomial degree |
| `fhe_scale` | `float` | `2**40` | FHE encoding scale |

**Example:**

```python
from pqc_fhe import HybridCryptoManager

# Default configuration
manager = HybridCryptoManager()

# High security configuration
manager = HybridCryptoManager(
    kem_algorithm="ML-KEM-1024",
    sig_algorithm="ML-DSA-87",
    fhe_poly_degree=16384
)
```

---

## Key Management

### generate_identity()

Generate a complete cryptographic identity.

```python
def generate_identity(self) -> HybridIdentity
```

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `identity` | `HybridIdentity` | Complete identity with all keys |

**HybridIdentity Structure:**

```python
@dataclass
class HybridIdentity:
    kem_public_key: bytes
    kem_secret_key: bytes
    sig_public_key: bytes
    sig_secret_key: bytes
    identity_id: str
    created_at: datetime
```

**Example:**

```python
# Generate identity
identity = manager.generate_identity()

print(f"Identity ID: {identity.identity_id}")
print(f"KEM Public Key: {len(identity.kem_public_key)} bytes")
print(f"Sig Public Key: {len(identity.sig_public_key)} bytes")
```

---

### export_public_identity()

Export only public keys for sharing.

```python
def export_public_identity(
    self, 
    identity: HybridIdentity
) -> PublicIdentity
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `identity` | `HybridIdentity` | Full identity |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `public_id` | `PublicIdentity` | Public keys only |

**PublicIdentity Structure:**

```python
@dataclass
class PublicIdentity:
    kem_public_key: bytes
    sig_public_key: bytes
    identity_id: str
```

**Example:**

```python
# Export for sharing
public_id = manager.export_public_identity(identity)

# Safe to share with anyone
send_to_peer(public_id)
```

---

## Session Establishment

### establish_session()

Establish a secure session with a peer.

```python
def establish_session(
    self,
    local_identity: HybridIdentity,
    peer_public_identity: PublicIdentity,
    sign_handshake: bool = True
) -> HybridSession
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_identity` | `HybridIdentity` | - | Local identity |
| `peer_public_identity` | `PublicIdentity` | - | Peer's public identity |
| `sign_handshake` | `bool` | `True` | Sign the handshake |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `session` | `HybridSession` | Established session |

**HybridSession Structure:**

```python
@dataclass
class HybridSession:
    session_id: str
    shared_secret: bytes
    ciphertext: bytes
    signature: Optional[bytes]
    local_identity_id: str
    peer_identity_id: str
    established_at: datetime
    expires_at: datetime
```

**Example:**

```python
# Alice establishes session with Bob
alice_session = manager.establish_session(
    local_identity=alice_identity,
    peer_public_identity=bob_public_identity,
    sign_handshake=True
)

# Alice sends ciphertext and signature to Bob
handshake_data = {
    "ciphertext": alice_session.ciphertext,
    "signature": alice_session.signature,
    "alice_public_identity": alice_public_identity
}
send_to_bob(handshake_data)
```

---

### accept_session()

Accept a session initiated by a peer.

```python
def accept_session(
    self,
    local_identity: HybridIdentity,
    peer_public_identity: PublicIdentity,
    ciphertext: bytes,
    signature: Optional[bytes] = None,
    verify_signature: bool = True
) -> HybridSession
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_identity` | `HybridIdentity` | - | Local identity |
| `peer_public_identity` | `PublicIdentity` | - | Peer's public identity |
| `ciphertext` | `bytes` | - | Encapsulated ciphertext |
| `signature` | `Optional[bytes]` | `None` | Handshake signature |
| `verify_signature` | `bool` | `True` | Verify the signature |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `session` | `HybridSession` | Accepted session |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `SessionError` | Decapsulation failed |
| `SignatureVerificationError` | Signature invalid |

**Example:**

```python
# Bob receives handshake from Alice
handshake_data = receive_from_alice()

# Bob accepts session
bob_session = manager.accept_session(
    local_identity=bob_identity,
    peer_public_identity=handshake_data["alice_public_identity"],
    ciphertext=handshake_data["ciphertext"],
    signature=handshake_data["signature"],
    verify_signature=True
)

# Verify shared secrets match
assert alice_session.shared_secret == bob_session.shared_secret
```

---

## Encrypted Communication

### encrypt_message()

Encrypt a message for a session.

```python
def encrypt_message(
    self,
    session: HybridSession,
    message: bytes,
    sign: bool = True,
    local_identity: Optional[HybridIdentity] = None
) -> EncryptedMessage
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `HybridSession` | - | Active session |
| `message` | `bytes` | - | Message to encrypt |
| `sign` | `bool` | `True` | Sign the message |
| `local_identity` | `Optional[HybridIdentity]` | `None` | Identity for signing |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `encrypted` | `EncryptedMessage` | Encrypted message |

**EncryptedMessage Structure:**

```python
@dataclass
class EncryptedMessage:
    ciphertext: bytes
    nonce: bytes
    signature: Optional[bytes]
    session_id: str
    timestamp: datetime
```

**Example:**

```python
# Encrypt message
encrypted = manager.encrypt_message(
    session=session,
    message=b"Hello, secure world!",
    sign=True,
    local_identity=local_identity
)

# Send to peer
send_to_peer(encrypted)
```

---

### decrypt_message()

Decrypt a message from a session.

```python
def decrypt_message(
    self,
    session: HybridSession,
    encrypted: EncryptedMessage,
    verify_signature: bool = True,
    peer_public_identity: Optional[PublicIdentity] = None
) -> bytes
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `HybridSession` | - | Active session |
| `encrypted` | `EncryptedMessage` | - | Encrypted message |
| `verify_signature` | `bool` | `True` | Verify signature |
| `peer_public_identity` | `Optional[PublicIdentity]` | `None` | Peer's identity for verification |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `message` | `bytes` | Decrypted message |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `DecryptionError` | Decryption failed |
| `SignatureVerificationError` | Signature invalid |
| `SessionExpiredError` | Session expired |

**Example:**

```python
# Receive encrypted message
encrypted = receive_from_peer()

# Decrypt
message = manager.decrypt_message(
    session=session,
    encrypted=encrypted,
    verify_signature=True,
    peer_public_identity=peer_public_identity
)

print(f"Received: {message.decode()}")
```

---

## FHE Operations

### encrypt_fhe()

Encrypt data using FHE.

```python
def encrypt_fhe(
    self,
    data: Union[float, List[float], np.ndarray]
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `float`, `List[float]`, `np.ndarray` | Data to encrypt |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `encrypted` | `EncryptedData` | FHE encrypted data |

**Example:**

```python
# Encrypt for homomorphic computation
ct_data = manager.encrypt_fhe([100.0, 200.0, 300.0])
```

---

### decrypt_fhe()

Decrypt FHE encrypted data.

```python
def decrypt_fhe(
    self,
    encrypted: EncryptedData,
    length: int = None
) -> Union[float, List[float]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encrypted` | `EncryptedData` | - | FHE encrypted data |
| `length` | `int` | `None` | Elements to return |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `decrypted` | `float` or `List[float]` | Decrypted values |

**Example:**

```python
# Decrypt result
result = manager.decrypt_fhe(ct_result, length=3)
print(f"Result: {result}")
```

---

### compute_fhe()

Perform homomorphic computation.

```python
def compute_fhe(
    self,
    operation: str,
    operands: List[EncryptedData],
    **kwargs
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `str` | Operation name |
| `operands` | `List[EncryptedData]` | Input ciphertexts |
| `**kwargs` | - | Additional parameters |

**Supported Operations:**

| Operation | Operands | Additional Args | Description |
|-----------|----------|-----------------|-------------|
| `"add"` | 2 | - | Add two ciphertexts |
| `"subtract"` | 2 | - | Subtract ciphertexts |
| `"multiply"` | 2 | - | Multiply ciphertexts |
| `"square"` | 1 | - | Square ciphertext |
| `"negate"` | 1 | - | Negate ciphertext |
| `"add_plain"` | 1 | `plain: float` | Add plaintext |
| `"multiply_plain"` | 1 | `plain: float` | Multiply by plaintext |
| `"sum"` | 1 | `length: int` | Sum elements |
| `"mean"` | 1 | `length: int` | Compute mean |
| `"polynomial"` | 1 | `coeffs: List[float]` | Evaluate polynomial |

**Example:**

```python
# Add two encrypted values
ct_sum = manager.compute_fhe("add", [ct_a, ct_b])

# Multiply by plaintext
ct_scaled = manager.compute_fhe(
    "multiply_plain", 
    [ct_data], 
    plain=2.0
)

# Compute mean
ct_mean = manager.compute_fhe(
    "mean", 
    [ct_values], 
    length=100
)
```

---

## Secure Computation Workflow

### create_computation_request()

Create a request for secure computation.

```python
def create_computation_request(
    self,
    session: HybridSession,
    encrypted_data: EncryptedData,
    computation_spec: ComputationSpec,
    sign: bool = True,
    local_identity: Optional[HybridIdentity] = None
) -> ComputationRequest
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `HybridSession` | - | Active session |
| `encrypted_data` | `EncryptedData` | - | Data to compute on |
| `computation_spec` | `ComputationSpec` | - | Computation specification |
| `sign` | `bool` | `True` | Sign the request |
| `local_identity` | `Optional[HybridIdentity]` | `None` | Signing identity |

**ComputationSpec Structure:**

```python
@dataclass
class ComputationSpec:
    operations: List[Dict[str, Any]]
    return_encrypted: bool = True
    max_depth: int = 10
```

**Example:**

```python
# Define computation: mean of squares
spec = ComputationSpec(
    operations=[
        {"op": "square"},
        {"op": "sum", "length": 100},
        {"op": "multiply_plain", "plain": 0.01}  # Divide by 100
    ],
    return_encrypted=True
)

# Create request
request = manager.create_computation_request(
    session=session,
    encrypted_data=ct_data,
    computation_spec=spec,
    sign=True,
    local_identity=local_identity
)

# Send to compute server
send_to_server(request)
```

---

### process_computation_request()

Process a computation request (server-side).

```python
def process_computation_request(
    self,
    request: ComputationRequest,
    verify_signature: bool = True,
    peer_public_identity: Optional[PublicIdentity] = None
) -> ComputationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request` | `ComputationRequest` | - | Computation request |
| `verify_signature` | `bool` | `True` | Verify signature |
| `peer_public_identity` | `Optional[PublicIdentity]` | `None` | Client's identity |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `ComputationResult` | Computation result |

**Example:**

```python
# Server processes request
result = manager.process_computation_request(
    request=request,
    verify_signature=True,
    peer_public_identity=client_public_identity
)

# Send result back to client
send_to_client(result)
```

---

### verify_computation_result()

Verify and extract computation result (client-side).

```python
def verify_computation_result(
    self,
    result: ComputationResult,
    expected_spec: ComputationSpec
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `ComputationResult` | Server's result |
| `expected_spec` | `ComputationSpec` | Expected computation |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `encrypted_result` | `EncryptedData` | Verified result |

**Example:**

```python
# Client verifies and extracts result
ct_result = manager.verify_computation_result(
    result=result,
    expected_spec=spec
)

# Decrypt locally
final_value = manager.decrypt_fhe(ct_result)
print(f"Mean of squares: {final_value}")
```

---

## Utility Methods

### get_system_info()

Get information about the hybrid system.

```python
def get_system_info(self) -> Dict[str, Any]
```

**Returns:**

```python
{
    "pqc": {
        "kem_algorithm": "ML-KEM-768",
        "sig_algorithm": "ML-DSA-65",
        "kem_security_level": 3,
        "sig_security_level": 3
    },
    "fhe": {
        "poly_modulus_degree": 8192,
        "scale": 1099511627776,
        "max_multiplicative_depth": 3,
        "max_slots": 4096
    },
    "version": "1.0.0"
}
```

---

### validate_session()

Check if a session is valid.

```python
def validate_session(self, session: HybridSession) -> bool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session` | `HybridSession` | Session to validate |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `valid` | `bool` | True if session is valid and not expired |

---

### refresh_session()

Refresh an expiring session.

```python
def refresh_session(
    self,
    session: HybridSession,
    local_identity: HybridIdentity,
    peer_public_identity: PublicIdentity
) -> HybridSession
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session` | `HybridSession` | Session to refresh |
| `local_identity` | `HybridIdentity` | Local identity |
| `peer_public_identity` | `PublicIdentity` | Peer's public identity |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `new_session` | `HybridSession` | New session with fresh keys |

---

## Error Handling

### Exception Types

```python
from pqc_fhe.exceptions import (
    HybridCryptoError,          # Base exception
    SessionError,               # Session establishment failed
    SessionExpiredError,        # Session has expired
    SignatureVerificationError, # Signature verification failed
    ComputationError,           # FHE computation failed
    IdentityError,              # Identity operation failed
)
```

### Error Handling Example

```python
from pqc_fhe import HybridCryptoManager
from pqc_fhe.exceptions import (
    SessionError, 
    SignatureVerificationError
)

manager = HybridCryptoManager()

try:
    session = manager.accept_session(
        local_identity=local_id,
        peer_public_identity=peer_id,
        ciphertext=ciphertext,
        signature=signature
    )
except SignatureVerificationError:
    print("WARNING: Handshake signature invalid!")
    print("Possible man-in-the-middle attack")
except SessionError as e:
    print(f"Session establishment failed: {e}")
```

---

## Thread Safety

The `HybridCryptoManager` is thread-safe for all operations.

```python
import threading
from concurrent.futures import ThreadPoolExecutor

manager = HybridCryptoManager()

def create_session(peer_id):
    local_id = manager.generate_identity()
    return manager.establish_session(local_id, peer_id)

# Safe to use from multiple threads
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(create_session, peer_id) for _ in range(100)]
    sessions = [f.result() for f in futures]
```

---

## Complete Workflow Example

```python
from pqc_fhe import HybridCryptoManager

# Initialize
manager = HybridCryptoManager()

# === Identity Generation ===
alice = manager.generate_identity()
bob = manager.generate_identity()

alice_public = manager.export_public_identity(alice)
bob_public = manager.export_public_identity(bob)

# === Session Establishment ===
# Alice initiates
alice_session = manager.establish_session(
    local_identity=alice,
    peer_public_identity=bob_public
)

# Bob accepts
bob_session = manager.accept_session(
    local_identity=bob,
    peer_public_identity=alice_public,
    ciphertext=alice_session.ciphertext,
    signature=alice_session.signature
)

# === Encrypted Communication ===
# Alice sends message
encrypted_msg = manager.encrypt_message(
    session=alice_session,
    message=b"Sensitive financial data",
    sign=True,
    local_identity=alice
)

# Bob receives and decrypts
decrypted = manager.decrypt_message(
    session=bob_session,
    encrypted=encrypted_msg,
    verify_signature=True,
    peer_public_identity=alice_public
)

# === FHE Computation ===
# Alice encrypts data
salary_data = [50000.0, 60000.0, 55000.0, 70000.0]
ct_salaries = manager.encrypt_fhe(salary_data)

# Server computes on encrypted data (no access to raw data)
ct_sum = manager.compute_fhe("sum", [ct_salaries], length=4)
ct_mean = manager.compute_fhe("multiply_plain", [ct_sum], plain=0.25)

# Alice decrypts result
mean_salary = manager.decrypt_fhe(ct_mean)
print(f"Average salary: ${mean_salary:.2f}")
```

---

## Related APIs

- [PQC Manager API](pqc_manager.md)
- [FHE Engine API](fhe_engine.md)
- [REST API Reference](rest_api.md)
- [WebSocket API Reference](websocket_api.md)
