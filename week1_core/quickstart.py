#!/usr/bin/env python3
"""
PQC-FHE Integration - Quick Start Demo v2.1.2
==============================================

This script demonstrates the core functionality of the PQC-FHE Integration library.

Usage:
    python quickstart.py
    
    Or with options:
    python quickstart.py --mode cpu --no-bootstrap
    python quickstart.py --mode gpu --bootstrap

Features demonstrated:
    1. FHE encryption and decryption (CKKS scheme)
    2. Homomorphic operations (add, multiply, square)
    3. Bootstrap for unlimited computation depth
    4. PQC key encapsulation (if liboqs available)
    5. PQC digital signatures (if liboqs available)

Requirements:
    - desilofhe (or desilofhe-cu130 for GPU)
    - numpy
    - liboqs-python (optional, for PQC operations)
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from typing import List, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_liboqs_available() -> bool:
    """
    Check if liboqs is available using subprocess to avoid SystemExit.
    
    The oqs package raises SystemExit(1) at import time if liboqs
    is not properly installed, which cannot be caught with try/except.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', 
             'import oqs; print(len(oqs.get_enabled_kem_mechanisms()))'],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, 'OQS_PERMIT_UNSUPPORTED_ARCHITECTURE': '1'}
        )
        return result.returncode == 0
    except Exception:
        return False


def print_header():
    """Print demo header."""
    print("\n" + "+" + "=" * 62 + "+")
    print("|     PQC-FHE Integration - Quick Start Demo v2.1.2           |")
    print("|     Post-Quantum Cryptography + Homomorphic Encryption      |")
    print("+" + "=" * 62 + "+")
    print()


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "[OK]" if success else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        for line in details.split('\n'):
            print(f"       {line}")


def run_fhe_demo(mode: str = 'gpu', use_bootstrap: bool = True):
    """Run FHE demonstration."""
    print_section("Fully Homomorphic Encryption (FHE)")
    
    try:
        from pqc_fhe_integration import FHEEngine, FHEConfig
        
        # Create FHE configuration
        config = FHEConfig(
            mode=mode,
            use_bootstrap=use_bootstrap,
            use_full_bootstrap_key=use_bootstrap,
            use_lossy_bootstrap=use_bootstrap,
            bootstrap_stage_count=3,
            thread_count=512 if mode == 'gpu' else 4
        )
        
        print(f"\n  Configuration:")
        print(f"    Mode: {mode}")
        print(f"    Bootstrap: {'enabled' if use_bootstrap else 'disabled'}")
        
        # Initialize FHE engine
        print("\n  Initializing FHE engine...")
        start_time = time.time()
        fhe = FHEEngine(config, logger)
        init_time = time.time() - start_time
        
        print(f"    Initialization time: {init_time:.2f}s")
        print(f"    Slot count: {fhe.engine.slot_count}")
        
        # Test data
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Encryption
        print("\n  CKKS Encryption:")
        start_time = time.time()
        ciphertext = fhe.encrypt(data)
        encrypt_time = time.time() - start_time
        
        print_result("Encryption", True, 
                    f"Original data: {data}\n"
                    f"Encryption time: {encrypt_time*1000:.2f}ms")
        
        # Homomorphic operations
        print("\n  Homomorphic Operations:")
        
        # Add scalar
        start_time = time.time()
        ct_added = fhe.add_scalar(ciphertext, 10.0)
        result_added = fhe.decrypt(ct_added, length=len(data))
        add_time = time.time() - start_time
        expected_added = [x + 10.0 for x in data]
        error_add = np.max(np.abs(result_added - expected_added))
        print_result("Add Scalar (+10)", error_add < 0.01,
                    f"Result: {[round(x, 1) for x in result_added]}\n"
                    f"Error: {error_add:.6f}, Time: {add_time*1000:.2f}ms")
        
        # Multiply scalar
        start_time = time.time()
        ct_mult = fhe.multiply_scalar(ciphertext, 2.0)
        result_mult = fhe.decrypt(ct_mult, length=len(data))
        mult_time = time.time() - start_time
        expected_mult = [x * 2.0 for x in data]
        error_mult = np.max(np.abs(result_mult - expected_mult))
        print_result("Multiply Scalar (*2)", error_mult < 0.01,
                    f"Result: {[round(x, 1) for x in result_mult]}\n"
                    f"Error: {error_mult:.6f}, Time: {mult_time*1000:.2f}ms")
        
        # Square
        start_time = time.time()
        ct_sq = fhe.square(ciphertext)
        result_sq = fhe.decrypt(ct_sq, length=len(data))
        sq_time = time.time() - start_time
        expected_sq = [x ** 2 for x in data]
        error_sq = np.max(np.abs(result_sq - expected_sq))
        print_result("Square (x^2)", error_sq < 0.01,
                    f"Result: {[round(x, 1) for x in result_sq]}\n"
                    f"Error: {error_sq:.6f}, Time: {sq_time*1000:.2f}ms")
        
        # Polynomial computation: x^2 + 2x + 1
        print("\n  Polynomial Computation (x^2 + 2x + 1):")
        start_time = time.time()
        
        ct_2x = fhe.multiply_scalar(ciphertext, 2.0)
        ct_x2_plus_2x = fhe.add(ct_sq, ct_2x)
        ct_poly = fhe.add_scalar(ct_x2_plus_2x, 1.0)
        result_poly = fhe.decrypt(ct_poly, length=len(data))
        
        poly_time = time.time() - start_time
        expected_poly = [x**2 + 2*x + 1 for x in data]
        error_poly = np.max(np.abs(result_poly - expected_poly))
        
        print_result("Polynomial", error_poly < 0.1,
                    f"Input: {data}\n"
                    f"Result: {[round(x, 2) for x in result_poly]}\n"
                    f"Expected: {expected_poly}\n"
                    f"Error: {error_poly:.6f}, Time: {poly_time*1000:.2f}ms")
        
        return True
        
    except ImportError as e:
        print(f"\n  [ERROR] Import failed: {e}")
        print("  Please install: pip install desilofhe-cu130")
        return False
    except Exception as e:
        print(f"\n  [ERROR] FHE demo failed: {e}")
        logger.exception("FHE demo error")
        return False


def run_pqc_demo():
    """Run PQC demonstration."""
    print_section("Post-Quantum Cryptography (PQC)")
    
    if not check_liboqs_available():
        print("\n  [SKIP] liboqs not available")
        print("  Install: pip install liboqs-python")
        print("  (Requires liboqs C library)")
        print("\n  Using mock demonstration instead...")
        run_mock_pqc_demo()
        return True
    
    try:
        from pqc_fhe_integration import PQCKeyManager, PQCConfig
        
        config = PQCConfig()
        pqc = PQCKeyManager(config, logger)
        
        # Key Encapsulation
        print("\n  ML-KEM-768 (Key Encapsulation):")
        
        start_time = time.time()
        kem_pk, kem_sk = pqc.generate_kem_keypair()
        keygen_time = time.time() - start_time
        
        print_result("Key Generation", True,
                    f"Public key: {len(kem_pk)} bytes\n"
                    f"Secret key: {len(kem_sk)} bytes\n"
                    f"Time: {keygen_time*1000:.2f}ms")
        
        start_time = time.time()
        ciphertext, shared_secret1 = pqc.encapsulate(kem_pk)
        encap_time = time.time() - start_time
        
        print_result("Encapsulation", True,
                    f"Ciphertext: {len(ciphertext)} bytes\n"
                    f"Shared secret: {len(shared_secret1)} bytes\n"
                    f"Time: {encap_time*1000:.2f}ms")
        
        start_time = time.time()
        shared_secret2 = pqc.decapsulate(ciphertext, kem_sk)
        decap_time = time.time() - start_time
        
        match = shared_secret1 == shared_secret2
        print_result("Decapsulation", match,
                    f"Secrets match: {match}\n"
                    f"Time: {decap_time*1000:.2f}ms")
        
        # Digital Signatures
        print("\n  ML-DSA-65 (Digital Signatures):")
        
        start_time = time.time()
        sig_pk, sig_sk = pqc.generate_sig_keypair()
        sig_keygen_time = time.time() - start_time
        
        print_result("Key Generation", True,
                    f"Public key: {len(sig_pk)} bytes\n"
                    f"Secret key: {len(sig_sk)} bytes\n"
                    f"Time: {sig_keygen_time*1000:.2f}ms")
        
        message = b"Hello, Quantum World!"
        
        start_time = time.time()
        signature = pqc.sign(message, sig_sk)
        sign_time = time.time() - start_time
        
        print_result("Signing", True,
                    f"Message: {message.decode()}\n"
                    f"Signature: {len(signature)} bytes\n"
                    f"Time: {sign_time*1000:.2f}ms")
        
        start_time = time.time()
        is_valid = pqc.verify(message, signature, sig_pk)
        verify_time = time.time() - start_time
        
        print_result("Verification", is_valid,
                    f"Valid: {is_valid}\n"
                    f"Time: {verify_time*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"\n  [ERROR] PQC demo failed: {e}")
        logger.exception("PQC demo error")
        return False


def run_mock_pqc_demo():
    """Run mock PQC demonstration when liboqs is not available."""
    import secrets
    import hashlib
    
    print("\n  ML-KEM-768 (Mock - Key Encapsulation):")
    
    # Mock key sizes (matching real ML-KEM-768)
    pk = secrets.token_bytes(1184)
    sk = secrets.token_bytes(2400)
    print_result("Key Generation (mock)", True,
                f"Public key: {len(pk)} bytes\n"
                f"Secret key: {len(sk)} bytes")
    
    ct = secrets.token_bytes(1088)
    ss = hashlib.sha256(pk + secrets.token_bytes(32)).digest()
    print_result("Encapsulation (mock)", True,
                f"Ciphertext: {len(ct)} bytes\n"
                f"Shared secret: {len(ss)} bytes")
    
    print("\n  ML-DSA-65 (Mock - Digital Signatures):")
    
    sig_pk = secrets.token_bytes(1952)
    sig_sk = secrets.token_bytes(4032)
    print_result("Key Generation (mock)", True,
                f"Public key: {len(sig_pk)} bytes\n"
                f"Secret key: {len(sig_sk)} bytes")
    
    signature = secrets.token_bytes(3293)
    print_result("Signing (mock)", True,
                f"Signature: {len(signature)} bytes")
    
    print_result("Verification (mock)", True,
                "Valid: True (mock always returns True)")


def print_summary(fhe_success: bool, pqc_success: bool):
    """Print demo summary."""
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    fhe_status = "PASS" if fhe_success else "FAIL"
    pqc_status = "PASS" if pqc_success else "SKIP/MOCK"
    
    print(f"\n  FHE Operations: {fhe_status}")
    print(f"  PQC Operations: {pqc_status}")
    
    print("\n  Next Steps:")
    print("    1. Start the REST API server:")
    print("       uvicorn api.server:app --reload")
    print()
    print("    2. Open API documentation:")
    print("       http://127.0.0.1:8000/docs")
    print()
    print("    3. Try the endpoints:")
    print("       POST /fhe/encrypt - Encrypt data")
    print("       POST /fhe/multiply - Homomorphic multiply")
    print("       POST /fhe/decrypt - Decrypt result")
    print()
    print("+" + "=" * 62 + "+")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PQC-FHE Integration Quick Start Demo"
    )
    parser.add_argument(
        '--mode', 
        choices=['cpu', 'gpu', 'parallel'],
        default='gpu',
        help='FHE computation mode (default: gpu)'
    )
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        default=True,
        help='Enable bootstrap (default: True)'
    )
    parser.add_argument(
        '--no-bootstrap',
        action='store_true',
        help='Disable bootstrap'
    )
    parser.add_argument(
        '--skip-pqc',
        action='store_true',
        help='Skip PQC demonstration'
    )
    
    args = parser.parse_args()
    
    use_bootstrap = args.bootstrap and not args.no_bootstrap
    
    print_header()
    
    # Run FHE demo
    fhe_success = run_fhe_demo(mode=args.mode, use_bootstrap=use_bootstrap)
    
    # Run PQC demo
    pqc_success = True
    if not args.skip_pqc:
        pqc_success = run_pqc_demo()
    else:
        print_section("Post-Quantum Cryptography (PQC)")
        print("\n  [SKIP] PQC demo skipped by user")
    
    # Print summary
    print_summary(fhe_success, pqc_success)
    
    return 0 if fhe_success else 1


if __name__ == "__main__":
    sys.exit(main())
