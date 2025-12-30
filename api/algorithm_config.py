#!/usr/bin/env python3
"""
PQC Algorithm Configuration Module v2.2.0
==========================================

Comprehensive configuration for all supported Post-Quantum Cryptography algorithms.

Supported Algorithm Families:
- ML-KEM (FIPS 203): Key Encapsulation Mechanism
- ML-DSA (FIPS 204): Digital Signature Algorithm
- SLH-DSA (FIPS 205): Stateless Hash-Based Digital Signature
- FrodoKEM: Conservative lattice-based KEM
- SPHINCS+: Stateless hash-based signatures
- Falcon: Fast Fourier lattice-based signatures
- BIKE/HQC: Code-based KEMs

References:
- NIST PQC: https://csrc.nist.gov/projects/post-quantum-cryptography
- liboqs: https://github.com/open-quantum-safe/liboqs
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class SecurityLevel(Enum):
    """NIST Security Levels"""
    LEVEL_1 = 1  # 128-bit classical / 64-bit quantum
    LEVEL_2 = 2  # 192-bit classical / 96-bit quantum (transition)
    LEVEL_3 = 3  # 192-bit classical / 128-bit quantum
    LEVEL_4 = 4  # 256-bit classical / 128-bit quantum (transition)
    LEVEL_5 = 5  # 256-bit classical / 256-bit quantum


class AlgorithmFamily(Enum):
    """Algorithm families"""
    LATTICE = "lattice"        # Module-LWE based
    HASH = "hash"              # Hash-based
    CODE = "code"              # Code-based
    ISOGENY = "isogeny"        # Isogeny-based (deprecated)


@dataclass
class KEMAlgorithm:
    """KEM Algorithm specification"""
    name: str
    display_name: str
    standard: str
    family: AlgorithmFamily
    security_level: SecurityLevel
    public_key_size: int
    secret_key_size: int
    ciphertext_size: int
    shared_secret_size: int
    description: str
    liboqs_name: str  # Name used in liboqs
    fallback_name: Optional[str] = None  # Legacy name


@dataclass
class SignatureAlgorithm:
    """Signature Algorithm specification"""
    name: str
    display_name: str
    standard: str
    family: AlgorithmFamily
    security_level: SecurityLevel
    public_key_size: int
    secret_key_size: int
    signature_size: int
    description: str
    liboqs_name: str
    fallback_name: Optional[str] = None


# =============================================================================
# KEM ALGORITHMS (Key Encapsulation Mechanisms)
# =============================================================================

KEM_ALGORITHMS: Dict[str, KEMAlgorithm] = {
    # NIST FIPS 203 - ML-KEM (Module Lattice-based KEM)
    "ML-KEM-512": KEMAlgorithm(
        name="ML-KEM-512",
        display_name="ML-KEM-512 (Kyber)",
        standard="NIST FIPS 203",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=800,
        secret_key_size=1632,
        ciphertext_size=768,
        shared_secret_size=32,
        description="Lightweight ML-KEM for resource-constrained devices. NIST Level 1 security.",
        liboqs_name="ML-KEM-512",
        fallback_name="Kyber512"
    ),
    "ML-KEM-768": KEMAlgorithm(
        name="ML-KEM-768",
        display_name="ML-KEM-768 (Kyber) - Recommended",
        standard="NIST FIPS 203",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_3,
        public_key_size=1184,
        secret_key_size=2400,
        ciphertext_size=1088,
        shared_secret_size=32,
        description="Balanced ML-KEM for general-purpose use. NIST Level 3 security. Recommended for most applications.",
        liboqs_name="ML-KEM-768",
        fallback_name="Kyber768"
    ),
    "ML-KEM-1024": KEMAlgorithm(
        name="ML-KEM-1024",
        display_name="ML-KEM-1024 (Kyber) - High Security",
        standard="NIST FIPS 203",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_5,
        public_key_size=1568,
        secret_key_size=3168,
        ciphertext_size=1568,
        shared_secret_size=32,
        description="High-security ML-KEM for sensitive applications. NIST Level 5 security.",
        liboqs_name="ML-KEM-1024",
        fallback_name="Kyber1024"
    ),
    
    # FrodoKEM - Conservative lattice-based (no NTT)
    "FrodoKEM-640-AES": KEMAlgorithm(
        name="FrodoKEM-640-AES",
        display_name="FrodoKEM-640-AES",
        standard="NIST Round 3 Alternate",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=9616,
        secret_key_size=19888,
        ciphertext_size=9720,
        shared_secret_size=16,
        description="Conservative LWE-based KEM without algebraic structure. Higher security margin.",
        liboqs_name="FrodoKEM-640-AES"
    ),
    "FrodoKEM-976-AES": KEMAlgorithm(
        name="FrodoKEM-976-AES",
        display_name="FrodoKEM-976-AES",
        standard="NIST Round 3 Alternate",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_3,
        public_key_size=15632,
        secret_key_size=31296,
        ciphertext_size=15744,
        shared_secret_size=24,
        description="Medium security FrodoKEM variant. Conservative choice for high-assurance.",
        liboqs_name="FrodoKEM-976-AES"
    ),
    
    # Code-based KEMs
    "BIKE-L1": KEMAlgorithm(
        name="BIKE-L1",
        display_name="BIKE Level 1",
        standard="NIST Round 4",
        family=AlgorithmFamily.CODE,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=1541,
        secret_key_size=3114,
        ciphertext_size=1573,
        shared_secret_size=32,
        description="Code-based KEM using quasi-cyclic MDPC codes. Compact keys.",
        liboqs_name="BIKE-L1"
    ),
    "HQC-128": KEMAlgorithm(
        name="HQC-128",
        display_name="HQC-128",
        standard="NIST Round 4",
        family=AlgorithmFamily.CODE,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=2249,
        secret_key_size=2289,
        ciphertext_size=4481,
        shared_secret_size=64,
        description="Hamming Quasi-Cyclic code-based KEM. Alternative to lattice-based.",
        liboqs_name="HQC-128"
    ),
}


# =============================================================================
# SIGNATURE ALGORITHMS (Digital Signatures)
# =============================================================================

SIGNATURE_ALGORITHMS: Dict[str, SignatureAlgorithm] = {
    # NIST FIPS 204 - ML-DSA (Module Lattice-based Digital Signature)
    "ML-DSA-44": SignatureAlgorithm(
        name="ML-DSA-44",
        display_name="ML-DSA-44 (Dilithium2)",
        standard="NIST FIPS 204",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_2,
        public_key_size=1312,
        secret_key_size=2560,
        signature_size=2420,
        description="Lightweight ML-DSA for resource-constrained devices. NIST Level 2 security.",
        liboqs_name="ML-DSA-44",
        fallback_name="Dilithium2"
    ),
    "ML-DSA-65": SignatureAlgorithm(
        name="ML-DSA-65",
        display_name="ML-DSA-65 (Dilithium3) - Recommended",
        standard="NIST FIPS 204",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_3,
        public_key_size=1952,
        secret_key_size=4032,
        signature_size=3309,
        description="Balanced ML-DSA for general-purpose use. NIST Level 3 security. Recommended for most applications.",
        liboqs_name="ML-DSA-65",
        fallback_name="Dilithium3"
    ),
    "ML-DSA-87": SignatureAlgorithm(
        name="ML-DSA-87",
        display_name="ML-DSA-87 (Dilithium5) - High Security",
        standard="NIST FIPS 204",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_5,
        public_key_size=2592,
        secret_key_size=4896,
        signature_size=4627,
        description="High-security ML-DSA for sensitive applications. NIST Level 5 security.",
        liboqs_name="ML-DSA-87",
        fallback_name="Dilithium5"
    ),
    
    # NIST FIPS 205 - SLH-DSA (Stateless Hash-based Digital Signature)
    "SLH-DSA-SHA2-128f": SignatureAlgorithm(
        name="SLH-DSA-SHA2-128f",
        display_name="SLH-DSA-SHA2-128f (SPHINCS+)",
        standard="NIST FIPS 205",
        family=AlgorithmFamily.HASH,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=32,
        secret_key_size=64,
        signature_size=17088,
        description="Fast hash-based signature. Conservative security assumptions.",
        liboqs_name="SLH-DSA-SHA2-128f",
        fallback_name="SPHINCS+-SHA2-128f-simple"
    ),
    "SLH-DSA-SHA2-128s": SignatureAlgorithm(
        name="SLH-DSA-SHA2-128s",
        display_name="SLH-DSA-SHA2-128s (SPHINCS+ Small)",
        standard="NIST FIPS 205",
        family=AlgorithmFamily.HASH,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=32,
        secret_key_size=64,
        signature_size=7856,
        description="Small signature hash-based. Slower signing but smaller signatures.",
        liboqs_name="SLH-DSA-SHA2-128s",
        fallback_name="SPHINCS+-SHA2-128s-simple"
    ),
    "SLH-DSA-SHA2-192f": SignatureAlgorithm(
        name="SLH-DSA-SHA2-192f",
        display_name="SLH-DSA-SHA2-192f (SPHINCS+)",
        standard="NIST FIPS 205",
        family=AlgorithmFamily.HASH,
        security_level=SecurityLevel.LEVEL_3,
        public_key_size=48,
        secret_key_size=96,
        signature_size=35664,
        description="Medium security fast hash-based signature.",
        liboqs_name="SLH-DSA-SHA2-192f",
        fallback_name="SPHINCS+-SHA2-192f-simple"
    ),
    "SLH-DSA-SHA2-256f": SignatureAlgorithm(
        name="SLH-DSA-SHA2-256f",
        display_name="SLH-DSA-SHA2-256f (SPHINCS+)",
        standard="NIST FIPS 205",
        family=AlgorithmFamily.HASH,
        security_level=SecurityLevel.LEVEL_5,
        public_key_size=64,
        secret_key_size=128,
        signature_size=49856,
        description="High security fast hash-based signature. Largest signatures.",
        liboqs_name="SLH-DSA-SHA2-256f",
        fallback_name="SPHINCS+-SHA2-256f-simple"
    ),
    
    # Falcon - Fast Fourier lattice-based
    "Falcon-512": SignatureAlgorithm(
        name="Falcon-512",
        display_name="Falcon-512",
        standard="NIST Round 3 Selection",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_1,
        public_key_size=897,
        secret_key_size=1281,
        signature_size=666,
        description="Compact signatures using NTRU lattices. Very small signatures.",
        liboqs_name="Falcon-512"
    ),
    "Falcon-1024": SignatureAlgorithm(
        name="Falcon-1024",
        display_name="Falcon-1024",
        standard="NIST Round 3 Selection",
        family=AlgorithmFamily.LATTICE,
        security_level=SecurityLevel.LEVEL_5,
        public_key_size=1793,
        secret_key_size=2305,
        signature_size=1280,
        description="High security Falcon variant. Compact signatures.",
        liboqs_name="Falcon-1024"
    ),
}


# =============================================================================
# ALGORITHM COMPARISON DATA
# =============================================================================

def get_algorithm_comparison() -> Dict:
    """Get comparison data for all algorithms."""
    return {
        "kem": {
            name: {
                "name": alg.display_name,
                "standard": alg.standard,
                "security_level": alg.security_level.value,
                "family": alg.family.value,
                "public_key_size": alg.public_key_size,
                "secret_key_size": alg.secret_key_size,
                "ciphertext_size": alg.ciphertext_size,
                "shared_secret_size": alg.shared_secret_size,
                "total_transmission": alg.public_key_size + alg.ciphertext_size,
                "description": alg.description
            }
            for name, alg in KEM_ALGORITHMS.items()
        },
        "signature": {
            name: {
                "name": alg.display_name,
                "standard": alg.standard,
                "security_level": alg.security_level.value,
                "family": alg.family.value,
                "public_key_size": alg.public_key_size,
                "secret_key_size": alg.secret_key_size,
                "signature_size": alg.signature_size,
                "total_transmission": alg.public_key_size + alg.signature_size,
                "description": alg.description
            }
            for name, alg in SIGNATURE_ALGORITHMS.items()
        }
    }


def get_recommended_algorithms() -> Dict:
    """Get recommended algorithms for different use cases."""
    return {
        "general_purpose": {
            "kem": "ML-KEM-768",
            "signature": "ML-DSA-65",
            "reason": "Balanced security (NIST Level 3) and performance. Recommended by NIST."
        },
        "lightweight_iot": {
            "kem": "ML-KEM-512",
            "signature": "ML-DSA-44",
            "reason": "Smaller key sizes for constrained devices. NIST Level 1-2 security."
        },
        "high_security": {
            "kem": "ML-KEM-1024",
            "signature": "ML-DSA-87",
            "reason": "Maximum security (NIST Level 5) for sensitive applications."
        },
        "conservative": {
            "kem": "FrodoKEM-976-AES",
            "signature": "SLH-DSA-SHA2-192f",
            "reason": "Conservative security assumptions. No algebraic structure (KEM) or lattice assumptions (Sig)."
        },
        "compact_signatures": {
            "kem": "ML-KEM-768",
            "signature": "Falcon-512",
            "reason": "Very compact signatures. Good for bandwidth-constrained applications."
        }
    }


# =============================================================================
# NIST STANDARD INFORMATION
# =============================================================================

NIST_STANDARDS = {
    "FIPS 203": {
        "title": "Module-Lattice-Based Key-Encapsulation Mechanism Standard",
        "date": "August 13, 2024",
        "algorithms": ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"],
        "url": "https://csrc.nist.gov/pubs/fips/203/final"
    },
    "FIPS 204": {
        "title": "Module-Lattice-Based Digital Signature Standard",
        "date": "August 13, 2024",
        "algorithms": ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"],
        "url": "https://csrc.nist.gov/pubs/fips/204/final"
    },
    "FIPS 205": {
        "title": "Stateless Hash-Based Digital Signature Standard",
        "date": "August 13, 2024",
        "algorithms": ["SLH-DSA-SHA2-128f", "SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-192f", "SLH-DSA-SHA2-256f"],
        "url": "https://csrc.nist.gov/pubs/fips/205/final"
    }
}
