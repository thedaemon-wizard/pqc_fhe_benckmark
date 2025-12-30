# DESILO FHE GPU Mode Configuration Guide

## Overview

DESILO FHE supports GPU acceleration for significantly faster homomorphic encryption operations. This guide explains how to properly configure and use GPU mode.

**Reference Documentation:**
- DESILO FHE: https://fhe.desilo.dev/latest/
- DESILO GitHub: https://github.com/Desilo/liberate-fhe
- PrivateInference.py v5.6.1

---

## Supported Execution Modes

| Mode | Description | Requirements |
|------|-------------|--------------|
| `cpu` | CPU execution (default) | None |
| `gpu` | GPU acceleration | CUDA-enabled NVIDIA GPU |
| `parallel` | Multi-threaded CPU | None (converted to `cpu` internally) |

---

## GPU Mode Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- Recommended: RTX 3090, RTX 4090, A100, or better
- Sufficient VRAM (12GB+ recommended for bootstrap operations)

### Software
- CUDA Toolkit 11.0+ (12.x recommended)
- cuDNN library
- `desilofhe` or `liberate-fhe` compiled with GPU support

---

## Basic GPU Initialization

### Using desilofhe (Simplified API)

```python
import desilofhe

# GPU mode with default settings
engine = desilofhe.Engine(mode='gpu')

# GPU mode with custom thread count
engine = desilofhe.Engine(
    mode='gpu',
    thread_count=512,        # GPU threads (default: 512)
    use_bootstrap=True       # Enable bootstrap
)

# With slot count override
engine = desilofhe.Engine(
    mode='gpu',
    thread_count=512,
    slot_count=1024,         # Custom slot count
    use_bootstrap=True
)
```

### Using liberate.fhe (Full GPU Version)

```python
from liberate import fhe
from liberate.fhe import presets

# Use preset parameters (recommended)
# Presets: bronze, silver, gold, platinum
params = presets.params["silver"].copy()
params["num_scales"] = 60

# Create CKKS engine
engine = fhe.ckks_engine(**params, verbose=True)
```

---

## Integration with PQC-FHE Portfolio

### Configuration

```python
from pqc_fhe_integration import FHEEngine, FHEConfig

# GPU mode configuration
config = FHEConfig(
    mode='gpu',              # Use GPU
    thread_count=512,        # GPU threads
    use_bootstrap=True,      # Enable bootstrap
    log_n=16,                # Ring dimension (N=65536)
    scale_bits=40,           # Encoding precision
    num_scales=60,           # Max multiplication depth
)

# Initialize engine
fhe = FHEEngine(config)

# Use FHE operations
ct = fhe.encrypt([1.0, 2.0, 3.0])
ct_squared = fhe.square(ct)
result = fhe.decrypt(ct_squared, length=3)
```

### Fallback Behavior

The `FHEEngine` automatically falls back in this order:

1. **liberate.fhe** (full GPU version)
2. **desilofhe** (CPU/GPU version)
3. **Mock engine** (for testing without DESILO)

---

## Troubleshooting

### Error: "Not supported mode"

**Cause:** `desilofhe` is not compiled with GPU support.

**Solutions:**

1. **Install GPU-enabled version:**
   ```bash
   pip install desilofhe[gpu]
   ```

2. **Build from source with CUDA:**
   ```bash
   git clone https://github.com/Desilo/liberate-fhe
   cd liberate-fhe
   mkdir build && cd build
   cmake .. -DENABLE_CUDA=ON
   make -j$(nproc)
   pip install ..
   ```

3. **Use CPU mode as fallback:**
   ```python
   config = FHEConfig(mode='cpu')  # Use CPU instead
   ```

### Error: "CUDA driver version is insufficient"

**Solution:** Update NVIDIA drivers and CUDA toolkit:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-driver-535  # or newer
sudo apt-get install cuda-toolkit-12-0
```

### Error: "Out of memory"

**Cause:** Insufficient GPU VRAM for bootstrap keys (~12GB+ needed).

**Solutions:**

1. Use smaller slot count:
   ```python
   config = FHEConfig(log_n=14)  # 8192 slots instead of 32768
   ```

2. Disable full bootstrap key:
   ```python
   config = FHEConfig(
       use_bootstrap=True,
       use_full_bootstrap_key=False,  # Only use small bootstrap
       use_lossy_bootstrap=True,      # Use lossy (faster, less memory)
   )
   ```

---

## Performance Benchmarks

From DESILO documentation (RTX 5090):

| Operation | Stage Count | CPU (s) | GPU (s) | Speedup |
|-----------|-------------|---------|---------|---------|
| Bootstrap (Small) | 3 | 31.861 | 3.139 | 10.1x |
| Bootstrap (Small) | 4 | 20.738 | 2.030 | 10.2x |
| Bootstrap (Medium) | 3 | 9.883 | 0.823 | 12.0x |
| Lossy Bootstrap | 3 | ~26 | ~0.8 | 32.5x |

---

## Reference Code from PrivateInference.py

```python
def _init_desilo(self, config):
    """
    Initialize DESILO FHE engine
    Reference: PrivateInference.py v5.6.1 lines 1039-1082
    """
    import desilofhe
    
    # Convert mode string
    mode_str = config.mode
    if mode_str == 'parallel':
        mode_str = 'cpu'  # Parallel uses CPU with multiple threads
    
    # Build engine kwargs
    engine_kwargs = {'mode': mode_str}
    
    if config.use_bootstrap:
        engine_kwargs['use_bootstrap'] = True
    
    if config.slot_count is not None:
        engine_kwargs['slot_count'] = config.slot_count
    
    # Thread count configuration
    if config.thread_count and config.thread_count > 0:
        engine_kwargs['thread_count'] = config.thread_count
    elif mode_str == 'gpu':
        engine_kwargs['thread_count'] = 512  # GPU default
    elif config.mode == 'parallel':
        engine_kwargs['thread_count'] = 4    # CPU parallel default
    
    # Create engine
    engine = desilofhe.Engine(**engine_kwargs)
    
    return engine
```

---

## Key Generation for GPU Mode

Key generation is the same for both CPU and GPU modes:

```python
# Core keys
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relin_key = engine.create_relinearization_key(secret_key)
rotation_key = engine.create_rotation_key(secret_key)
conjugation_key = engine.create_conjugation_key(secret_key)

# Bootstrap keys (GPU mode benefits most here)
small_bootstrap_key = engine.create_small_bootstrap_key(secret_key)
lossy_bootstrap_key = engine.create_lossy_bootstrap_key(secret_key, stage_count=3)
full_bootstrap_key = engine.create_bootstrap_key(secret_key, stage_count=3)
```

---

## Complete GPU Example

```python
#!/usr/bin/env python3
"""DESILO FHE GPU Mode Example"""

import numpy as np
from pqc_fhe_integration import FHEEngine, FHEConfig

def gpu_fhe_demo():
    # Configure for GPU
    config = FHEConfig(
        mode='gpu',
        thread_count=512,
        use_bootstrap=True,
        log_n=15,  # 16384 slots
        num_scales=40,
    )
    
    try:
        # Initialize
        fhe = FHEEngine(config)
        print(f"FHE Engine initialized (mode: {config.mode})")
        
        # Encrypt data
        data = np.random.randn(100).tolist()
        ct = fhe.encrypt(data)
        print(f"Encrypted at level: {fhe.get_level(ct)}")
        
        # Homomorphic operations
        ct_squared = fhe.square(ct)
        ct_added = fhe.add_scalar(ct_squared, 1.0)
        print(f"After operations: level {fhe.get_level(ct_added)}")
        
        # Decrypt
        result = fhe.decrypt(ct_added, length=len(data))
        expected = np.array(data) ** 2 + 1.0
        
        mse = np.mean((result - expected) ** 2)
        print(f"MSE: {mse:.6e}")
        
    except RuntimeError as e:
        if "Not supported mode" in str(e):
            print("GPU mode not available. Using CPU...")
            config.mode = 'cpu'
            fhe = FHEEngine(config)
            # ... continue with CPU
        else:
            raise

if __name__ == "__main__":
    gpu_fhe_demo()
```

---

## References

1. DESILO FHE Documentation: https://fhe.desilo.dev/latest/
2. DESILO GitHub: https://github.com/Desilo/liberate-fhe
3. PrivateInference.py v5.6.1 (Upwork Project)
4. CKKS Scheme: Cheon, Kim, Kim, Song (2017)

---

*Last Updated: December 2025*
