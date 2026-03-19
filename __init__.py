"""
<<<<<<< HEAD
PQC-FHE Integration Package v3.2.0
=======
PQC-FHE Integration Package v2.1.0
>>>>>>> origin/main
===================================

Post-Quantum Cryptography and Fully Homomorphic Encryption integration library.

Features:
- NIST FIPS 203 (ML-KEM) key encapsulation
- NIST FIPS 204 (ML-DSA) digital signatures
- NIST FIPS 205 (SLH-DSA) hash-based signatures
- CKKS-based fully homomorphic encryption (DESILO FHE compliant)
- Hybrid quantum-resistant cryptography
- Scientifically computed Chebyshev polynomial activations

Quick Start:
-----------
>>> from pqc_fhe_integration import PQCFHEIntegration
>>> 
>>> # Initialize the system
>>> system = PQCFHEIntegration()
>>> 
>>> # Generate PQC keys
>>> kem_pk, kem_sk = system.pqc.generate_kem_keypair()
>>> sig_pk, sig_sk = system.pqc.generate_sig_keypair()
>>>
>>> # Encrypt with FHE
>>> ciphertext = system.fhe.encrypt([1.0, 2.0, 3.0])
>>> ct_squared = system.fhe.square(ciphertext)
>>> result = system.fhe.decrypt(ct_squared, length=3)

Alternative Usage:
-----------------
>>> from pqc_fhe_integration import PQCKeyManager, FHEEngine
>>> 
>>> # Use components separately
>>> pqc = PQCKeyManager()
>>> fhe = FHEEngine()
>>>
>>> # Generate PQC keys
>>> public_key, secret_key = pqc.generate_kem_keypair()
>>>
>>> # FHE operations
>>> ciphertext = fhe.encrypt([1.0, 2.0, 3.0])
>>> result = fhe.add_scalar(ciphertext, 10.0)
>>> decrypted = fhe.decrypt(result, length=3)

References:
----------
- NIST FIPS 203: ML-KEM (Module-Lattice Key Encapsulation Mechanism)
- NIST FIPS 204: ML-DSA (Module-Lattice Digital Signature Algorithm)
- NIST FIPS 205: SLH-DSA (Stateless Hash-Based Digital Signature Algorithm)
- DESILO FHE: https://fhe.desilo.dev/latest/

License: MIT
Author: Amon (Quantum Computing Specialist)
"""

<<<<<<< HEAD
import json as _json
import os as _os

def _load_version():
    """Load version from version.json dynamically."""
    _version_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'version.json')
    try:
        with open(_version_file, 'r') as f:
            return _json.load(f)['version']
    except (FileNotFoundError, KeyError, _json.JSONDecodeError):
        return "3.2.0"

__version__ = _load_version()
=======
__version__ = "2.1.0"
>>>>>>> origin/main
__author__ = "PQC-FHE Portfolio"

# Import main classes for convenient access
try:
    from .pqc_fhe_integration import (
        # Enums
        SecurityLevel,
        KEMAlgorithm,
        SignatureAlgorithm,
        BootstrapMethod,
        
        # Configuration
        PQCConfig,
        FHEConfig,
        IntegrationConfig,
        
        # Core classes
        PQCKeyManager,
        FHEEngine,
        PQCFHESystem,
        
        # Aliases (for convenience and backward compatibility)
        PQCFHEIntegration,
        HybridCryptoManager,
        
        # Constants
        PQC_KEM_ALGORITHMS,
        PQC_SIGN_ALGORITHMS,
        
        # Utilities
        secure_random_bytes,
        
        # Factory functions
        create_default_system,
        create_high_security_system,
        create_gpu_accelerated_system,
    )
    
    __all__ = [
        # Version info
        "__version__",
        "__author__",
        
        # Enums
        "SecurityLevel",
        "KEMAlgorithm",
        "SignatureAlgorithm",
        "BootstrapMethod",
        
        # Configuration
        "PQCConfig",
        "FHEConfig",
        "IntegrationConfig",
        
        # Core classes
        "PQCKeyManager",
        "FHEEngine",
        "PQCFHESystem",
        
        # Aliases
        "PQCFHEIntegration",
        "HybridCryptoManager",
        
        # Constants
        "PQC_KEM_ALGORITHMS",
        "PQC_SIGN_ALGORITHMS",
        
        # Utilities
        "secure_random_bytes",
        
        # Factory functions
        "create_default_system",
        "create_high_security_system",
        "create_gpu_accelerated_system",
    ]

except ImportError as e:
    # Graceful degradation if dependencies missing
    import logging
    logging.warning(f"Some PQC-FHE components unavailable: {e}")
<<<<<<< HEAD

    __all__ = ["__version__", "__author__"]

# v3.0.0 modules
try:
    from .src.quantum_threat_simulator import (
        ShorSimulator,
        GroverSimulator,
        QuantumThreatTimeline,
        QuantumResourceEstimate,
    )
    __all__.extend([
        "ShorSimulator", "GroverSimulator",
        "QuantumThreatTimeline", "QuantumResourceEstimate",
    ])
except ImportError:
    pass

try:
    from .src.security_scoring import (
        SecurityScoringEngine,
        SecurityScore,
        CryptoAsset,
        ComplianceStandard,
    )
    __all__.extend([
        "SecurityScoringEngine", "SecurityScore",
        "CryptoAsset", "ComplianceStandard",
    ])
except ImportError:
    pass

try:
    from .src.mpc_he_inference import (
        MPCHEProtocol,
        SimpleMPCDemo,
        MPCConfig,
        MPCRole,
    )
    __all__.extend([
        "MPCHEProtocol", "SimpleMPCDemo",
        "MPCConfig", "MPCRole",
    ])
except ImportError:
    pass
=======
    
    __all__ = ["__version__", "__author__"]
>>>>>>> origin/main
