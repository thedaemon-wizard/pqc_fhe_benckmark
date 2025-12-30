# DESILO FHE API Compliance - Changes and Updates

## Overview

This document details the changes made to the PQC-FHE Portfolio FHE implementation
based on analysis of the official DESILO FHE API documentation PDFs.

**Analysis Date:** December 27, 2025  
**DESILO Documentation Version:** v5.5.0+  
**Documentation URL:** https://fhe.desilo.dev/latest/

---

## Documents Analyzed

| Document | Key Information |
|----------|-----------------|
| `bootstrap.pdf` | Standard bootstrap API, level=0 support, output level=10 |
| `lossy_bootstrap.pdf` | Faster bootstrap, requires level >= stage_count, output level=16-stage_count |
| `sign_bootstrap.pdf` | High-precision bootstrap for sign values (-1, 1), 3x better precision |
| `multiply_matrix.pdf` | Matrix multiplication with rotation_key or matrix_multiplication_key |
| `multiply_pytorch_tensor.pdf` | PyTorch tensor multiplication, **0-level consumption** |
| `multiply_pytorch_tensor_matrix.pdf` | PyTorch tensor matrix multiplication, 1-level consumption |
| `Data_Structures.pdf` | Memory requirements, key sizes, ciphertext structure |
| `Advanced_Examples.pdf` | Sum, argmax, max, sign, Chebyshev approximation |
| `Matrix_Multiplication.pdf` | PlainMatrix encoding, matrix_multiplication_key |
| `Multiplication_Level.pdf` | Level management, bootstrap triggers |

---

## Critical Findings and Changes

### 1. Sign Bootstrap (NEW)

**Finding:** DESILO provides `sign_bootstrap` for bootstrapping sign values (-1, 1) with significantly higher precision.

**From `sign_bootstrap.pdf`:**
```
Average error comparison:
- Regular bootstrap: 1.60562e-3
- Lossy bootstrap:   1.60562e-3  
- Sign bootstrap:    3.10572e-10 (3x more significant digits!)
```

**Change:** Added `sign_bootstrap()` method to `DESILOFHEEngine` class.

```python
def sign_bootstrap(self, ct: Any) -> Any:
    """
    Sign bootstrap for values exactly -1 or 1.
    Achieves ~3x better precision than regular bootstrap.
    """
    if 'lossy_bootstrap' in self.keys:
        return self.engine.sign_bootstrap(
            ct,
            self.keys['relin'],
            self.keys['conjugation'],
            self.keys['lossy_bootstrap']
        )
```

**Use Case:** Comparison operations (min, max), binary classification results.

---

### 2. Bootstrap Level Requirements

**Finding:** Lossy bootstrap has specific level requirements that were not properly documented.

**From `lossy_bootstrap.pdf`:**
```
Ciphertext: The ciphertext to be bootstrapped. 
The values must be within the range [−1,1]. 
The level of the ciphertext must be at least the stage_count of the lossy bootstrap key.
```

**Changes:**
- Added level validation before lossy_bootstrap
- Clear error messages when level < stage_count
- Documented output levels:
  - Standard bootstrap: level 10
  - Lossy bootstrap: level = 16 - stage_count

```python
def lossy_bootstrap(self, ct: Any) -> Any:
    level = self.get_level(ct)
    stage_count = self.config.bootstrap_stage_count
    
    if level < stage_count:
        raise RuntimeError(
            f"Lossy bootstrap requires level >= {stage_count}, "
            f"but ciphertext is at level {level}"
        )
```

---

### 3. PyTorch Tensor Multiplication (0-Level)

**Finding:** `multiply_pytorch_tensor` does NOT consume any level, making it highly efficient for neural network weight multiplication.

**From `multiply_pytorch_tensor.pdf`:**
```
Engine.multiply_pytorch_tensor(x, y)
Returns the ciphertext resulting from the multiplication of a ciphertext 
and a PyTorch tensor. Does not consume any level.
```

**Change:** Added `multiply_pytorch_tensor()` method with clear documentation that it's a 0-level operation.

```python
def multiply_pytorch_tensor(self, ct: Any, tensor: 'torch.Tensor') -> Any:
    """
    Level consumption: 0 (DOES NOT consume levels)
    
    Note: This is a 0-level operation, making it efficient for
    weight multiplication in neural networks.
    """
```

**Impact:** Can significantly reduce level consumption in FHE neural network inference.

---

### 4. Matrix Multiplication Variants

**Finding:** Two different APIs for matrix multiplication with different key requirements.

**From `multiply_matrix.pdf`:**
```
Variant 1: multiply_matrix(matrix, ciphertext, rotation_key)
Variant 2: multiply_matrix(plain_matrix, ciphertext, matrix_multiplication_key)
           - Not supported in multiparty computation
```

**Changes:**
- Added `encode_to_plain_matrix()` method
- Added `use_matrix_multiplication_key` parameter
- Added `create_matrix_multiplication_key()` during initialization

```python
def multiply_matrix(
    self, 
    matrix: Union[np.ndarray, Any], 
    ct: Any,
    use_matrix_multiplication_key: bool = False
) -> Any:
```

---

### 5. Memory Requirements Documentation

**Finding:** DESILO documentation provides detailed memory requirements that should be documented for users.

**From `Data_Structures.pdf` (with use_bootstrap=True):**

| Key Type | Memory Size |
|----------|-------------|
| Secret Key | 16.0 MB |
| Public Key | 28.0 MB |
| Ciphertext | 28.0 MB |
| Relinearization Key | 217.0 MB |
| Rotation Key | 3.2 GB |
| Small Bootstrap Key | 223.0 MB |
| Bootstrap Key | 12.3 GB |
| Lossy Bootstrap Key | 11.3 GB |
| PlainMatrix (slot_count=1024) | 16.1 GB |

**Change:** Added comprehensive memory documentation to `DESILOFHEConfig` docstring.

---

### 6. Stage Count Support Table

**Finding:** Not all stage_count values are supported for all slot_count configurations.

**From `lossy_bootstrap.pdf` and `sign_bootstrap.pdf`:**

| key_size | slot_count | stage_count=1 | stage_count=2 | stage_count=3 | stage_count=4 | stage_count=5 |
|----------|------------|---------------|---------------|---------------|---------------|---------------|
| medium | 1-3 | ✓ | | | | |
| medium | 4-5 | ✓ | ✓ | | | |
| medium | 6 | | ✓ | | | |
| medium | 7-8 | | ✓ | ✓ | | |
| medium | 9-10 | | ✓ | ✓ | ✓ | |
| medium | 11-12 | | | ✓ | ✓ | |
| medium | 13-14 | | | ✓ | ✓ | ✓ |

**Change:** Added documentation about supported stage_count configurations.

---

### 7. Chebyshev Polynomial Evaluation

**Finding:** DESILO provides native Chebyshev polynomial evaluation for better approximation.

**From `Advanced_Examples.pdf`:**
```python
# Example: exp(x) approximation for x in [-1, 1]
# Using 8th degree Chebyshev approximation
coefficients = [1.266066, 1.130318, 0.271495, ...]
result = engine.evaluate_chebyshev_polynomial(encrypted, coefficients, relin_key)
```

**Change:** Added `evaluate_chebyshev_polynomial()` with fallback implementation using recurrence relation.

---

### 8. Sum Operation

**Finding:** DESILO may have native `sum` operation for slot aggregation.

**From `Advanced_Examples.pdf`:**
```python
# Optimized sum using O(log n) rotations
rotated_data = encrypted
added = rotated_data
for i in range(log2(n)):
    delta = 2**i
    rotated_data = engine.rotate(rotated_data, rotation_key, delta=delta)
    added = engine.add(added, rotated_data)
```

**Change:** Added `sum_slots()` method with native support check and optimized fallback.

---

### 9. Shortcut Methods

**Finding:** DESILO provides shortcut methods for common operations.

**From DESILO documentation:**
- `encorypt`: Encode + Encrypt in one step
- `decrode`: Decrypt + Decode in one step

**Change:** Added `encorypt()` and `decrode()` methods with fallback to standard operations.

---

### 10. Comparison Operations

**Finding:** Advanced examples show how to implement max, min, argmax using sign function.

**From `Advanced_Examples.pdf`:**
```python
def max(a, b, relinearization_key):
    # max(a, b) = 0.5 * (a + b + (a - b) * sign(a - b))
    subtracted = engine.subtract(a, b)
    subtracted_sign = sign(subtracted, relinearization_key)
    multiplied = engine.multiply(subtracted, subtracted_sign, relinearization_key)
    added = engine.add(multiplied, engine.add(a, b))
    result = engine.multiply(added, 0.5)
    return result
```

**Change:** Added `max_encrypted()`, `min_encrypted()`, and `sign()` methods.

---

## Summary of New Methods

| Method | Level Consumption | Description |
|--------|-------------------|-------------|
| `sign_bootstrap()` | Bootstrap | High-precision bootstrap for sign values |
| `lossy_bootstrap()` | Bootstrap | Fast bootstrap (requires level >= stage_count) |
| `multiply_pytorch_tensor()` | 0 | PyTorch tensor element-wise multiplication |
| `multiply_pytorch_tensor_matrix()` | 1 | PyTorch tensor matrix multiplication |
| `encode_to_plain_matrix()` | N/A | Encode matrix for multiplication |
| `evaluate_chebyshev_polynomial()` | Varies | Chebyshev polynomial evaluation |
| `sum_slots()` | 0 | Sum first n slots |
| `encorypt()` | N/A | Encode + Encrypt shortcut |
| `decrode()` | N/A | Decrypt + Decode shortcut |
| `sign()` | ~10 | Sign function |
| `max_encrypted()` | ~10 | Element-wise maximum |
| `min_encrypted()` | ~10 | Element-wise minimum |

---

## Breaking Changes from v1.0.0

1. **Level validation added to `lossy_bootstrap()`** - Will now raise RuntimeError if level < stage_count
2. **Bootstrap output levels clarified** - Standard=10, Lossy=16-stage_count
3. **New file structure** - `src/desilo_fhe_engine.py` contains the new implementation

---

## Migration Guide

### From v1.0.0 to v2.0.0

```python
# Old (v1.0.0)
from pqc_fhe_integration import FHEEngine, FHEConfig

config = FHEConfig(mode='cpu', use_bootstrap=True)
engine = FHEEngine(config)

# New (v2.0.0)
from src.desilo_fhe_engine import DESILOFHEEngine, DESILOFHEConfig

config = DESILOFHEConfig(mode='cpu', use_bootstrap=True)
engine = DESILOFHEEngine(config)

# New methods available:
engine.sign_bootstrap(ct)                    # High-precision sign bootstrap
engine.multiply_pytorch_tensor(ct, tensor)   # 0-level PyTorch multiplication
engine.sum_slots(ct, n)                      # Optimized slot summation
engine.max_encrypted(ct_a, ct_b)             # Encrypted comparison
```

---

## Recommendations

1. **Use `sign_bootstrap()` for comparison operations** - Achieves 3x better precision for sign values
2. **Use `multiply_pytorch_tensor()` for neural network weights** - 0-level consumption saves multiplicative depth
3. **Monitor levels before lossy bootstrap** - Ensure level >= stage_count to avoid runtime errors
4. **Consider memory requirements** - Bootstrap keys require ~12GB+ memory

---

## References

1. DESILO FHE Documentation: https://fhe.desilo.dev/latest/
2. DESILO GitHub: https://github.com/Desilo/liberate-fhe
3. CKKS Scheme: Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)
4. PrivateInference.py v5.6.1: Encrypted Transformer Inference implementation

---

*Document generated: December 27, 2025*
*Author: Amon (Quantum Computing Specialist)*
