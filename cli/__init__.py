"""
PQC-FHE Command Line Interface

Provides command-line access to Post-Quantum Cryptography and
Fully Homomorphic Encryption operations.

Usage:
    pqc-fhe keygen --algorithm ML-KEM-768 --output keys/
    pqc-fhe encrypt --input data.txt --output encrypted.bin
    pqc-fhe benchmark --suite full
"""

from .main import main, CLI

__all__ = ["main", "CLI"]
__version__ = "1.0.0"
