# FHE Engine API Reference

Complete API reference for the Fully Homomorphic Encryption Engine using CKKS scheme.

## Class: FHEEngine

The `FHEEngine` class provides fully homomorphic encryption operations on encrypted data.

### Constructor

```python
FHEEngine(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: List[int] = None,
    scale: float = 2**40,
    security_level: int = 128
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poly_modulus_degree` | `int` | `8192` | Polynomial modulus degree (power of 2) |
| `coeff_mod_bit_sizes` | `List[int]` | `[60, 40, 40, 60]` | Coefficient modulus bit sizes |
| `scale` | `float` | `2**40` | Encoding scale for CKKS |
| `security_level` | `int` | `128` | Security level in bits |

**Supported Configurations:**

| Degree | Max Mult Depth | Security | Use Case |
|--------|----------------|----------|----------|
| 4096 | 2 | 128-bit | Simple computations |
| 8192 | 4 | 128-bit | Standard operations |
| 16384 | 8 | 128-bit | Complex computations |
| 32768 | 16 | 128-bit | Deep computations |

**Example:**

```python
from pqc_fhe import FHEEngine

# Default configuration
engine = FHEEngine()

# High-depth configuration
engine = FHEEngine(
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60],
    scale=2**40
)
```

---

## Encryption Methods

### encrypt()

Encrypt numerical data.

```python
def encrypt(
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
| `encrypted` | `EncryptedData` | Encrypted data object |

**Example:**

```python
# Encrypt single value
ct_value = engine.encrypt(42.5)

# Encrypt list
ct_list = engine.encrypt([1.0, 2.0, 3.0, 4.0])

# Encrypt numpy array
import numpy as np
ct_array = engine.encrypt(np.array([1.0, 2.0, 3.0]))
```

**Slot Capacity:**

| Degree | Max Slots |
|--------|-----------|
| 4096 | 2048 |
| 8192 | 4096 |
| 16384 | 8192 |
| 32768 | 16384 |

---

### decrypt()

Decrypt encrypted data.

```python
def decrypt(
    self, 
    encrypted: EncryptedData,
    length: int = None
) -> Union[float, List[float]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encrypted` | `EncryptedData` | - | Encrypted data |
| `length` | `int` | `None` | Number of elements to return |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `decrypted` | `float` or `List[float]` | Decrypted values |

**Example:**

```python
# Decrypt single value
value = engine.decrypt(ct_value)
print(f"Decrypted: {value}")  # ~42.5

# Decrypt with specific length
values = engine.decrypt(ct_list, length=4)
print(f"Decrypted: {values}")  # [~1.0, ~2.0, ~3.0, ~4.0]
```

---

## Arithmetic Operations

### add()

Add two encrypted values.

```python
def add(
    self, 
    a: EncryptedData, 
    b: EncryptedData
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `EncryptedData` | First operand |
| `b` | `EncryptedData` | Second operand |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Encrypted sum |

**Level Cost:** 0 (no multiplicative depth consumed)

**Example:**

```python
ct_a = engine.encrypt(10.0)
ct_b = engine.encrypt(20.0)
ct_sum = engine.add(ct_a, ct_b)

result = engine.decrypt(ct_sum)
print(f"Sum: {result}")  # ~30.0
```

---

### add_plain()

Add encrypted value and plaintext.

```python
def add_plain(
    self, 
    encrypted: EncryptedData, 
    plain: Union[float, List[float]]
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Encrypted operand |
| `plain` | `float` or `List[float]` | Plaintext operand |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Encrypted sum |

**Level Cost:** 0

**Example:**

```python
ct = engine.encrypt(10.0)
ct_result = engine.add_plain(ct, 5.0)

result = engine.decrypt(ct_result)
print(f"Result: {result}")  # ~15.0
```

---

### subtract()

Subtract encrypted values.

```python
def subtract(
    self, 
    a: EncryptedData, 
    b: EncryptedData
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `EncryptedData` | Minuend |
| `b` | `EncryptedData` | Subtrahend |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Encrypted difference (a - b) |

**Level Cost:** 0

**Example:**

```python
ct_a = engine.encrypt(30.0)
ct_b = engine.encrypt(10.0)
ct_diff = engine.subtract(ct_a, ct_b)

result = engine.decrypt(ct_diff)
print(f"Difference: {result}")  # ~20.0
```

---

### multiply()

Multiply two encrypted values.

```python
def multiply(
    self, 
    a: EncryptedData, 
    b: EncryptedData
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `EncryptedData` | First operand |
| `b` | `EncryptedData` | Second operand |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Encrypted product |

**Level Cost:** 1 (consumes one multiplicative level)

**Example:**

```python
ct_a = engine.encrypt(5.0)
ct_b = engine.encrypt(4.0)
ct_product = engine.multiply(ct_a, ct_b)

result = engine.decrypt(ct_product)
print(f"Product: {result}")  # ~20.0
```

---

### multiply_plain()

Multiply encrypted value by plaintext.

```python
def multiply_plain(
    self, 
    encrypted: EncryptedData, 
    plain: Union[float, List[float]]
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Encrypted operand |
| `plain` | `float` or `List[float]` | Plaintext multiplier |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Encrypted product |

**Level Cost:** 0-1 (depends on implementation)

**Example:**

```python
ct = engine.encrypt(10.0)
ct_result = engine.multiply_plain(ct, 3.0)

result = engine.decrypt(ct_result)
print(f"Result: {result}")  # ~30.0
```

---

### negate()

Negate encrypted value.

```python
def negate(self, encrypted: EncryptedData) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Input ciphertext |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Negated ciphertext |

**Level Cost:** 0

**Example:**

```python
ct = engine.encrypt(42.0)
ct_neg = engine.negate(ct)

result = engine.decrypt(ct_neg)
print(f"Negated: {result}")  # ~-42.0
```

---

### square()

Square an encrypted value.

```python
def square(self, encrypted: EncryptedData) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Input ciphertext |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Squared ciphertext |

**Level Cost:** 1

**Example:**

```python
ct = engine.encrypt(5.0)
ct_squared = engine.square(ct)

result = engine.decrypt(ct_squared)
print(f"Squared: {result}")  # ~25.0
```

---

## Advanced Operations

### rotate()

Rotate encrypted vector elements.

```python
def rotate(
    self, 
    encrypted: EncryptedData, 
    steps: int
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Input ciphertext |
| `steps` | `int` | Rotation steps (positive = left) |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Rotated ciphertext |

**Level Cost:** 0

**Example:**

```python
ct = engine.encrypt([1.0, 2.0, 3.0, 4.0])

# Rotate left by 1
ct_rotated = engine.rotate(ct, 1)
result = engine.decrypt(ct_rotated, length=4)
print(f"Rotated: {result}")  # [~2.0, ~3.0, ~4.0, ~1.0]
```

---

### sum_elements()

Sum all elements in encrypted vector.

```python
def sum_elements(
    self, 
    encrypted: EncryptedData,
    length: int
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Input ciphertext |
| `length` | `int` | Number of elements to sum |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Ciphertext with sum in all slots |

**Level Cost:** 0 (uses rotations)

**Example:**

```python
ct = engine.encrypt([1.0, 2.0, 3.0, 4.0])
ct_sum = engine.sum_elements(ct, length=4)

result = engine.decrypt(ct_sum)
print(f"Sum: {result}")  # ~10.0 (in all slots)
```

---

### evaluate_polynomial()

Evaluate polynomial on encrypted data.

```python
def evaluate_polynomial(
    self, 
    encrypted: EncryptedData,
    coefficients: List[float]
) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Input ciphertext (x) |
| `coefficients` | `List[float]` | Polynomial coefficients [a0, a1, a2, ...] |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | p(x) = a0 + a1*x + a2*x² + ... |

**Level Cost:** Depends on polynomial degree (approximately log2(degree))

**Example:**

```python
# Evaluate p(x) = 1 + 2x + 3x²
ct = engine.encrypt(2.0)
ct_poly = engine.evaluate_polynomial(ct, [1.0, 2.0, 3.0])

result = engine.decrypt(ct_poly)
print(f"p(2) = {result}")  # ~17.0 (1 + 4 + 12)
```

---

## Context Management

### get_context_info()

Get information about the FHE context.

```python
def get_context_info(self) -> Dict[str, Any]
```

**Returns:**

```python
{
    "poly_modulus_degree": 8192,
    "coeff_mod_bit_sizes": [60, 40, 40, 60],
    "scale": 1099511627776,
    "security_level": 128,
    "max_slots": 4096,
    "max_multiplicative_depth": 3
}
```

---

### get_noise_budget()

Get remaining noise budget of ciphertext.

```python
def get_noise_budget(self, encrypted: EncryptedData) -> int
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Ciphertext to check |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `budget` | `int` | Remaining noise budget in bits |

**Example:**

```python
ct = engine.encrypt(42.0)
budget_initial = engine.get_noise_budget(ct)
print(f"Initial budget: {budget_initial} bits")

ct_squared = engine.square(ct)
budget_after = engine.get_noise_budget(ct_squared)
print(f"After square: {budget_after} bits")
```

---

### relinearize()

Relinearize ciphertext after multiplication.

```python
def relinearize(self, encrypted: EncryptedData) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Ciphertext to relinearize |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Relinearized ciphertext |

**Note:** Called automatically after multiply() in most implementations.

---

### rescale()

Rescale ciphertext to reduce scale.

```python
def rescale(self, encrypted: EncryptedData) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Ciphertext to rescale |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `result` | `EncryptedData` | Rescaled ciphertext |

**Note:** Called automatically after multiply() in most implementations.

---

## Serialization

### serialize()

Serialize encrypted data for storage or transmission.

```python
def serialize(self, encrypted: EncryptedData) -> bytes
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `encrypted` | `EncryptedData` | Ciphertext to serialize |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `data` | `bytes` | Serialized ciphertext |

**Example:**

```python
ct = engine.encrypt(42.0)
serialized = engine.serialize(ct)

# Save to file
with open("ciphertext.bin", "wb") as f:
    f.write(serialized)
```

---

### deserialize()

Deserialize encrypted data.

```python
def deserialize(self, data: bytes) -> EncryptedData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `bytes` | Serialized ciphertext |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `encrypted` | `EncryptedData` | Deserialized ciphertext |

**Example:**

```python
# Load from file
with open("ciphertext.bin", "rb") as f:
    serialized = f.read()

ct = engine.deserialize(serialized)
value = engine.decrypt(ct)
```

---

### export_context()

Export FHE context for sharing with compute servers.

```python
def export_context(self) -> bytes
```

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `context` | `bytes` | Serialized public context |

**Example:**

```python
# Export context (no secret key)
context_bytes = engine.export_context()

# Share with compute server
# ...

# Compute server can perform operations
# but cannot decrypt
```

---

## Class: EncryptedData

Represents encrypted data with metadata.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ciphertext` | `internal` | Underlying ciphertext object |
| `scale` | `float` | Current scale |
| `chain_index` | `int` | Current level in modulus chain |
| `slot_count` | `int` | Number of slots |

### Methods

#### get_level()

```python
def get_level(self) -> int
```

Returns the remaining multiplicative depth.

#### is_transparent()

```python
def is_transparent(self) -> bool
```

Returns True if ciphertext is transparent (unencrypted).

---

## Error Handling

### Exception Types

```python
from pqc_fhe.exceptions import (
    FHEError,           # Base exception
    EncryptionError,    # Encryption failed
    DecryptionError,    # Decryption failed
    ComputationError,   # Computation failed
    NoiseBudgetError,   # Noise budget exhausted
    ContextError,       # Context mismatch
    SerializationError, # Serialization failed
)
```

### Error Handling Example

```python
from pqc_fhe import FHEEngine
from pqc_fhe.exceptions import NoiseBudgetError

engine = FHEEngine()

try:
    ct = engine.encrypt(42.0)
    
    # Multiple multiplications may exhaust noise budget
    for _ in range(10):
        ct = engine.multiply(ct, ct)
        
except NoiseBudgetError as e:
    print(f"Noise budget exhausted: {e}")
    print("Consider using deeper parameters or bootstrapping")
```

---

## Performance Tips

### Batching

Use SIMD batching for vector operations:

```python
# Efficient: Single ciphertext for 4096 values
values = [float(i) for i in range(4096)]
ct = engine.encrypt(values)

# Inefficient: 4096 separate ciphertexts
# cts = [engine.encrypt(v) for v in values]  # Don't do this!
```

### Operation Ordering

Minimize multiplicative depth:

```python
# Better: depth = 2
# (a * b) * (c * d)
ct_ab = engine.multiply(ct_a, ct_b)
ct_cd = engine.multiply(ct_c, ct_d)
ct_result = engine.multiply(ct_ab, ct_cd)

# Worse: depth = 3
# ((a * b) * c) * d
ct_ab = engine.multiply(ct_a, ct_b)
ct_abc = engine.multiply(ct_ab, ct_c)
ct_result = engine.multiply(ct_abc, ct_d)
```

### Parameter Selection

| Use Case | Degree | Depth | Notes |
|----------|--------|-------|-------|
| Simple statistics | 4096 | 2 | Mean, variance |
| Linear models | 8192 | 4 | Linear regression |
| Neural networks | 16384+ | 8+ | Deep learning |

---

## Related APIs

- [PQC Manager API](pqc_manager.md)
- [Hybrid Manager API](hybrid_manager.md)
- [REST API Reference](rest_api.md)
