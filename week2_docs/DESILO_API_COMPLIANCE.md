# DESILO FHE API Compliance Update

## Version 2.0.0 - DESILO API Compliance

### Overview

This document describes the changes made to ensure full compliance with the official DESILO FHE API documentation (https://fhe.desilo.dev/latest/).

### Reference Documents Analyzed

| Document | URL | Key Information |
|----------|-----|-----------------|
| Bootstrap | https://fhe.desilo.dev/latest/bootstrap/ | Output levels, key requirements |
| bootstrap API | https://fhe.desilo.dev/latest/api/engine/bootstrap/ | Regular bootstrap usage |
| lossy_bootstrap API | https://fhe.desilo.dev/latest/api/engine/lossy_bootstrap/ | Lossy bootstrap requirements |
| sign_bootstrap API | https://fhe.desilo.dev/latest/api/engine/sign_bootstrap/ | Sign value bootstrap |
| multiply_pytorch_tensor | https://fhe.desilo.dev/latest/api/engine/multiply_pytorch_tensor/ | Tensor multiplication (0 levels) |
| multiply_pytorch_tensor_matrix | https://fhe.desilo.dev/latest/api/engine/multiply_pytorch_tensor_matrix/ | Matrix multiplication (1 level) |
| multiply_matrix | https://fhe.desilo.dev/latest/api/engine/multiply_matrix/ | Matrix multiplication variants |
| Data Structures | https://fhe.desilo.dev/latest/data_structures/ | Key memory sizes |

---

## Critical Changes

### 1. Bootstrap Output Levels (FIXED)

**Previous Implementation:**
```python
# Output level was not explicitly documented
```

**Corrected Implementation (DESILO Specification):**
```python
@property
def regular_bootstrap_output_level(self) -> int:
    """
    Output level after regular bootstrap
    Reference: https://fhe.desilo.dev/latest/api/engine/bootstrap/
    """
    return 10  # Default, varies by stage_count

@property
def lossy_bootstrap_output_level(self) -> int:
    """
    Output level after lossy bootstrap
    Reference: https://fhe.desilo.dev/latest/api/engine/lossy_bootstrap/
    """
    return 16 - self.bootstrap_stage_count
```

**Scientific Basis:**
- Regular bootstrap restores to level 10 by default
- Lossy/Sign bootstrap restores to level 16 - stage_count
- stage_count=3 → output level 13
- stage_count=5 → output level 11

---

### 2. Sign Bootstrap (NEW)

**Added Implementation:**
```python
def sign_bootstrap(self, ct: Any) -> Any:
    """
    Sign bootstrap for high-precision sign value bootstrapping
    
    CRITICAL: Values MUST be exactly -1 or 1!
    CRITICAL: Level MUST be >= stage_count (default 3)!
    
    Precision comparison (average error):
        - Regular/Lossy bootstrap: ~1.60562e-3
        - Sign bootstrap: ~3.10572e-10 (3x more precision)
    
    Reference: https://fhe.desilo.dev/latest/api/engine/sign_bootstrap/
    """
```

**Scientific Basis:**
- Sign bootstrap achieves ~3x more significant digits for sign values (-1, 1)
- Critical for comparison operations (min, max, sign extraction)
- Benchmark data from DESILO documentation:

| Bootstrap Type | Average Error |
|---------------|---------------|
| Regular | 1.60562e-3 |
| Lossy | 1.60562e-3 |
| Sign | 3.10572e-10 |

---

### 3. Matrix Multiplication Operations (NEW)

**Added Operations:**

#### 3.1 multiply_pytorch_tensor (0 levels consumed)
```python
def multiply_pytorch_tensor(self, ct: Any, tensor) -> Any:
    """
    Multiply ciphertext by PyTorch tensor (0 levels consumed)
    Reference: https://fhe.desilo.dev/latest/api/engine/multiply_pytorch_tensor/
    """
    return self.engine.multiply_pytorch_tensor(ct, tensor)
```

#### 3.2 multiply_pytorch_tensor_matrix (1 level consumed)
```python
def multiply_pytorch_tensor_matrix(self, tensor_matrix, ct: Any) -> Any:
    """
    Multiply PyTorch tensor matrix by ciphertext (1 level consumed)
    Reference: https://fhe.desilo.dev/latest/api/engine/multiply_pytorch_tensor_matrix/
    """
    return self.engine.multiply_pytorch_tensor_matrix(
        tensor_matrix, ct, self.keys['rotation']
    )
```

#### 3.3 multiply_matrix (1 level consumed)
```python
def multiply_matrix(self, matrix: np.ndarray, ct: Any, use_plain_matrix: bool = False) -> Any:
    """
    Two API variants:
    1. multiply_matrix(matrix, ciphertext, rotation_key)
    2. multiply_matrix(plain_matrix, ciphertext, matrix_multiplication_key)
    """
```

**Level Consumption Summary:**

| Operation | Levels Consumed | Key Required |
|-----------|-----------------|--------------|
| multiply_pytorch_tensor | 0 | None |
| multiply_pytorch_tensor_matrix | 1 | rotation_key |
| multiply_matrix (standard) | 1 | rotation_key |
| multiply_matrix (PlainMatrix) | 1 | matrix_multiplication_key |

---

### 4. Key Memory Requirements (DOCUMENTED)

**Added Documentation:**
```python
"""
Key Memory Requirements (with use_bootstrap=True):
    - Secret Key: 16.0 MB
    - Public Key: 28.0 MB
    - Relinearization Key: 217.0 MB
    - Conjugation Key: 217.5 MB
    - Rotation Key: 3.2 GB
    - Small Bootstrap Key: 223.0 MB
    - Bootstrap Key: 12.3 GB
    - Lossy Bootstrap Key: 11.3 GB
    - Matrix Multiplication Key (slot_count=1024): 16.1 GB
"""
```

**Reference:** https://fhe.desilo.dev/latest/data_structures/

---

### 5. Level Requirements for Bootstrap (FIXED)

**Previous Implementation:**
```python
# Level check was implicit
if level >= 3 and 'lossy_bootstrap' in self.keys:
    ...
```

**Corrected Implementation:**
```python
def lossy_bootstrap(self, ct: Any) -> Any:
    """
    CRITICAL: Level MUST be >= stage_count (default 3)!
    """
    level = self.get_level(ct)
    stage_count = self.config.bootstrap_stage_count
    
    if level < stage_count:
        self.logger.warning(
            f"Lossy bootstrap requires level >= {stage_count}, but ct has level {level}. "
            f"Falling back to regular bootstrap."
        )
        return self.bootstrap(ct)
```

**Level Requirements:**

| Bootstrap Method | Minimum Level Required |
|-----------------|----------------------|
| Regular (bootstrap) | 0 |
| Lossy (lossy_bootstrap) | stage_count (default 3) |
| Sign (sign_bootstrap) | stage_count (default 3) |

---

### 6. Shortcut Methods (NEW)

**Added:**
```python
def encorypt(self, data, level=None) -> Any:
    """
    Encode + Encrypt in one step
    Reference: https://fhe.desilo.dev/latest/api/engine/encorypt/
    """

def decrode(self, ct) -> np.ndarray:
    """
    Decrypt + Decode in one step
    Reference: https://fhe.desilo.dev/latest/api/engine/decrode/
    """
```

---

### 7. Bootstrap Stage Count Support (DOCUMENTED)

**Added Configuration:**
```python
@dataclass
class FHEConfig:
    bootstrap_stage_count: int = 3  # Supported: 3, 4, 5
```

**Stage Count Effects:**

| Stage Count | Bootstrap Speed | Output Level |
|-------------|-----------------|--------------|
| 3 | Slowest | 13 (lossy/sign) |
| 4 | Medium | 12 (lossy/sign) |
| 5 | Fastest | 11 (lossy/sign) |

**Supported Combinations (from DESILO docs):**

| key_size | slot_count | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|----------|------------|---------|---------|---------|---------|---------|
| medium | 1 | ✅ | | | | |
| medium | 2 | ✅ | | | | |
| medium | 3 | ✅ | | | | |
| medium | 4 | ✅ | ✅ | | | |
| medium | 5 | ✅ | ✅ | | | |
| medium | 6 | | ✅ | | | |
| medium | 7 | | ✅ | ✅ | | |
| medium | 8 | | ✅ | ✅ | | |
| medium | 9 | | ✅ | ✅ | | |
| medium | 10 | | ✅ | ✅ | ✅ | |
| medium | 11 | | | ✅ | ✅ | |
| medium | 12 | | | ✅ | ✅ | |

---

### 8. API Feature Detection (NEW)

**Added:**
```python
def _check_api_features(self):
    """Check available DESILO API features"""
    self._api_features = {
        'multiply_pytorch_tensor': hasattr(self.engine, 'multiply_pytorch_tensor'),
        'multiply_pytorch_tensor_matrix': hasattr(self.engine, 'multiply_pytorch_tensor_matrix'),
        'multiply_matrix': hasattr(self.engine, 'multiply_matrix'),
        'evaluate_chebyshev_polynomial': hasattr(self.engine, 'evaluate_chebyshev_polynomial'),
        'sign_bootstrap': hasattr(self.engine, 'sign_bootstrap'),
        'lossy_bootstrap': hasattr(self.engine, 'lossy_bootstrap'),
        'encorypt': hasattr(self.engine, 'encorypt'),
        'decrode': hasattr(self.engine, 'decrode'),
    }
```

---

## Migration Guide

### From v1.0.0 to v2.0.0

1. **Update imports:**
```python
# Old
from pqc_fhe_integration import FHEEngine

# New (same import, but version check recommended)
from pqc_fhe_integration_v2 import FHEEngine, BootstrapMethod
```

2. **Update safe_bootstrap calls:**
```python
# Old
ct_bootstrapped = engine.safe_bootstrap(ct, context='ffn_output')

# New (with method selection)
ct_bootstrapped = engine.safe_bootstrap(
    ct, 
    context='ffn_output',
    method=BootstrapMethod.LOSSY
)
```

3. **Use sign_bootstrap for comparison operations:**
```python
# For values that are exactly -1 or 1
ct_sign = engine.sign_bootstrap(ct_sign_values)
```

4. **Check output levels:**
```python
# Get expected output level
output_level = engine.get_bootstrap_output_level(BootstrapMethod.LOSSY)
print(f"Expected output level: {output_level}")  # e.g., 13 for stage_count=3
```

---

## References

1. **DESILO FHE Documentation**: https://fhe.desilo.dev/latest/
2. **DESILO GitHub**: https://github.com/Desilo/liberate-fhe
3. **PrivateInference.py v5.6.1**: Upwork Project Reference
4. **CKKS Scheme**: Cheon, Kim, Kim, Song. "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)

---

## Changelog

- **v2.0.0** (2025-12-27):
  - Added sign_bootstrap for high-precision sign value bootstrapping
  - Added matrix multiplication operations (multiply_pytorch_tensor, multiply_pytorch_tensor_matrix, multiply_matrix)
  - Fixed bootstrap output level calculations
  - Added level requirement checks for lossy/sign bootstrap
  - Added encorypt/decrode shortcut methods
  - Added API feature detection
  - Documented key memory requirements
  - Added BootstrapMethod enum

- **v1.0.0** (2025-12-25):
  - Initial release with basic PQC + FHE integration
