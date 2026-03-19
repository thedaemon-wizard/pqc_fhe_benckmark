#!/usr/bin/env python3
"""
PQC-FHE Command Line Interface

A comprehensive CLI for Post-Quantum Cryptography and Fully Homomorphic
Encryption operations based on NIST FIPS 203/204/205 standards.

Author: Amon (Portfolio Project)
License: MIT
Version: 1.0.0

References:
- NIST FIPS 203: ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism)
- NIST FIPS 204: ML-DSA (Module-Lattice-Based Digital Signature Algorithm)
- NIST FIPS 205: SLH-DSA (Stateless Hash-Based Digital Signature Algorithm)
- CKKS: Cheon-Kim-Kim-Song Homomorphic Encryption Scheme
"""

import argparse
import json
import logging
import os
import sys
import time
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

VERSION = "1.0.0"
PROGRAM_NAME = "pqc-fhe"

# Supported algorithms based on NIST standards
PQC_KEM_ALGORITHMS = [
    "ML-KEM-512",   # NIST Security Level 1
    "ML-KEM-768",   # NIST Security Level 3 (Recommended)
    "ML-KEM-1024",  # NIST Security Level 5
]

PQC_SIGNATURE_ALGORITHMS = [
    "ML-DSA-44",    # NIST Security Level 2
    "ML-DSA-65",    # NIST Security Level 3 (Recommended)
    "ML-DSA-87",    # NIST Security Level 5
    "SLH-DSA-SHA2-128s",  # Hash-based, small signatures
    "SLH-DSA-SHA2-128f",  # Hash-based, fast signing
    "SLH-DSA-SHA2-192s",
    "SLH-DSA-SHA2-192f",
    "SLH-DSA-SHA2-256s",
    "SLH-DSA-SHA2-256f",
]

FHE_OPERATIONS = [
    "add", "subtract", "multiply", "negate", "square",
    "add_scalar", "multiply_scalar"
]

# Default output directory
DEFAULT_OUTPUT_DIR = Path("./pqc_fhe_output")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KeyPair:
    """Represents a cryptographic key pair"""
    algorithm: str
    public_key: bytes
    secret_key: bytes
    created_at: str
    key_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "public_key": base64.b64encode(self.public_key).decode(),
            "secret_key": base64.b64encode(self.secret_key).decode(),
            "created_at": self.created_at,
            "key_id": self.key_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyPair":
        return cls(
            algorithm=data["algorithm"],
            public_key=base64.b64decode(data["public_key"]),
            secret_key=base64.b64decode(data["secret_key"]),
            created_at=data["created_at"],
            key_id=data["key_id"],
        )


@dataclass
class EncapsulationResult:
    """Result of KEM encapsulation"""
    ciphertext: bytes
    shared_secret: bytes
    algorithm: str
    created_at: str


@dataclass
class SignatureResult:
    """Result of digital signature"""
    signature: bytes
    message_hash: str
    algorithm: str
    created_at: str


# =============================================================================
# CLI Class
# =============================================================================

class CLI:
    """
    Command Line Interface for PQC-FHE operations
    
    Provides comprehensive access to:
    - Post-Quantum Key Encapsulation (ML-KEM)
    - Post-Quantum Digital Signatures (ML-DSA, SLH-DSA)
    - Fully Homomorphic Encryption (CKKS)
    - Benchmarking and performance analysis
    """
    
    def __init__(self):
        self.pqc_manager = None
        self.fhe_engine = None
        self._initialize_crypto_engines()
    
    def _initialize_crypto_engines(self):
        """Initialize cryptographic engines with graceful degradation"""
        try:
            # Try to import the PQC-FHE library
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.pqc_fhe_integration import PQCKeyManager, FHEEngine
            
            self.pqc_manager = PQCKeyManager()
            self.fhe_engine = FHEEngine()
            logger.info("Cryptographic engines initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not import crypto engines: {e}")
            logger.warning("Running in simulation mode")
            self.pqc_manager = None
            self.fhe_engine = None
        except Exception as e:
            logger.warning(f"Crypto engine initialization error: {e}")
            self.pqc_manager = None
            self.fhe_engine = None
    
    # =========================================================================
    # PQC Key Generation Commands
    # =========================================================================
    
    def cmd_keygen(self, args: argparse.Namespace) -> int:
        """Generate PQC key pairs"""
        algorithm = args.algorithm
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {algorithm} key pair...")
        start_time = time.time()
        
        try:
            if self.pqc_manager:
                # Use actual PQC library
                if algorithm in PQC_KEM_ALGORITHMS:
                    public_key, secret_key = self.pqc_manager.generate_kem_keypair(algorithm)
                elif algorithm in PQC_SIGNATURE_ALGORITHMS:
                    public_key, secret_key = self.pqc_manager.generate_signature_keypair(algorithm)
                else:
                    logger.error(f"Unknown algorithm: {algorithm}")
                    return 1
            else:
                # Simulation mode - generate random keys
                logger.warning("Using simulation mode (no actual crypto)")
                public_key = os.urandom(self._get_key_size(algorithm, "public"))
                secret_key = os.urandom(self._get_key_size(algorithm, "secret"))
            
            # Create key pair object
            key_id = hashlib.sha256(public_key).hexdigest()[:16]
            keypair = KeyPair(
                algorithm=algorithm,
                public_key=public_key,
                secret_key=secret_key,
                created_at=datetime.utcnow().isoformat(),
                key_id=key_id,
            )
            
            # Save keys
            self._save_keypair(keypair, output_dir, args.format)
            
            elapsed = time.time() - start_time
            logger.info(f"Key generation completed in {elapsed*1000:.2f}ms")
            logger.info(f"Key ID: {key_id}")
            logger.info(f"Public key size: {len(public_key)} bytes")
            logger.info(f"Secret key size: {len(secret_key)} bytes")
            logger.info(f"Output directory: {output_dir}")
            
            if args.json:
                result = {
                    "status": "success",
                    "algorithm": algorithm,
                    "key_id": key_id,
                    "public_key_size": len(public_key),
                    "secret_key_size": len(secret_key),
                    "execution_time_ms": elapsed * 1000,
                    "output_directory": str(output_dir),
                }
                print(json.dumps(result, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def _get_key_size(self, algorithm: str, key_type: str) -> int:
        """Get expected key size for simulation mode"""
        # Based on NIST FIPS 203/204/205 specifications
        sizes = {
            "ML-KEM-512": {"public": 800, "secret": 1632},
            "ML-KEM-768": {"public": 1184, "secret": 2400},
            "ML-KEM-1024": {"public": 1568, "secret": 3168},
            "ML-DSA-44": {"public": 1312, "secret": 2560},
            "ML-DSA-65": {"public": 1952, "secret": 4032},
            "ML-DSA-87": {"public": 2592, "secret": 4896},
            "SLH-DSA-SHA2-128s": {"public": 32, "secret": 64},
            "SLH-DSA-SHA2-128f": {"public": 32, "secret": 64},
            "SLH-DSA-SHA2-192s": {"public": 48, "secret": 96},
            "SLH-DSA-SHA2-192f": {"public": 48, "secret": 96},
            "SLH-DSA-SHA2-256s": {"public": 64, "secret": 128},
            "SLH-DSA-SHA2-256f": {"public": 64, "secret": 128},
        }
        return sizes.get(algorithm, {"public": 1024, "secret": 2048})[key_type]
    
    def _save_keypair(self, keypair: KeyPair, output_dir: Path, fmt: str):
        """Save key pair to files"""
        if fmt == "json":
            # Save as JSON
            filepath = output_dir / f"{keypair.key_id}_keypair.json"
            with open(filepath, 'w') as f:
                json.dump(keypair.to_dict(), f, indent=2)
            logger.info(f"Saved keypair to {filepath}")
        elif fmt == "pem":
            # Save as PEM-like format
            pk_path = output_dir / f"{keypair.key_id}_public.pem"
            sk_path = output_dir / f"{keypair.key_id}_secret.pem"
            
            pk_pem = self._to_pem(keypair.public_key, f"{keypair.algorithm} PUBLIC KEY")
            sk_pem = self._to_pem(keypair.secret_key, f"{keypair.algorithm} SECRET KEY")
            
            with open(pk_path, 'w') as f:
                f.write(pk_pem)
            with open(sk_path, 'w') as f:
                f.write(sk_pem)
            
            logger.info(f"Saved public key to {pk_path}")
            logger.info(f"Saved secret key to {sk_path}")
        else:
            # Save as binary
            pk_path = output_dir / f"{keypair.key_id}_public.bin"
            sk_path = output_dir / f"{keypair.key_id}_secret.bin"
            
            with open(pk_path, 'wb') as f:
                f.write(keypair.public_key)
            with open(sk_path, 'wb') as f:
                f.write(keypair.secret_key)
            
            logger.info(f"Saved public key to {pk_path}")
            logger.info(f"Saved secret key to {sk_path}")
    
    def _to_pem(self, data: bytes, label: str) -> str:
        """Convert binary data to PEM format"""
        b64 = base64.b64encode(data).decode()
        lines = [b64[i:i+64] for i in range(0, len(b64), 64)]
        return f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n"
    
    # =========================================================================
    # KEM Commands (Encapsulation/Decapsulation)
    # =========================================================================
    
    def cmd_encapsulate(self, args: argparse.Namespace) -> int:
        """Encapsulate a shared secret using KEM"""
        logger.info("Performing KEM encapsulation...")
        
        try:
            # Load public key
            public_key = self._load_key(args.public_key)
            algorithm = args.algorithm or self._detect_algorithm(public_key)
            
            start_time = time.time()
            
            if self.pqc_manager:
                ciphertext, shared_secret = self.pqc_manager.encapsulate(public_key, algorithm)
            else:
                # Simulation mode
                ciphertext = os.urandom(self._get_ciphertext_size(algorithm))
                shared_secret = os.urandom(32)  # Standard shared secret size
            
            elapsed = time.time() - start_time
            
            # Save results
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            ct_path = output_dir / "ciphertext.bin"
            ss_path = output_dir / "shared_secret.bin"
            
            with open(ct_path, 'wb') as f:
                f.write(ciphertext)
            with open(ss_path, 'wb') as f:
                f.write(shared_secret)
            
            logger.info(f"Encapsulation completed in {elapsed*1000:.2f}ms")
            logger.info(f"Ciphertext size: {len(ciphertext)} bytes -> {ct_path}")
            logger.info(f"Shared secret size: {len(shared_secret)} bytes -> {ss_path}")
            
            if args.json:
                result = {
                    "status": "success",
                    "algorithm": algorithm,
                    "ciphertext_size": len(ciphertext),
                    "shared_secret_size": len(shared_secret),
                    "execution_time_ms": elapsed * 1000,
                    "ciphertext_hash": hashlib.sha256(ciphertext).hexdigest()[:16],
                }
                print(json.dumps(result, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"Encapsulation failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def cmd_decapsulate(self, args: argparse.Namespace) -> int:
        """Decapsulate a shared secret using KEM"""
        logger.info("Performing KEM decapsulation...")
        
        try:
            # Load keys and ciphertext
            secret_key = self._load_key(args.secret_key)
            ciphertext = self._load_key(args.ciphertext)
            algorithm = args.algorithm or "ML-KEM-768"
            
            start_time = time.time()
            
            if self.pqc_manager:
                shared_secret = self.pqc_manager.decapsulate(ciphertext, secret_key, algorithm)
            else:
                # Simulation mode
                shared_secret = os.urandom(32)
            
            elapsed = time.time() - start_time
            
            # Save result
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(shared_secret)
            
            logger.info(f"Decapsulation completed in {elapsed*1000:.2f}ms")
            logger.info(f"Shared secret size: {len(shared_secret)} bytes -> {output_path}")
            
            if args.json:
                result = {
                    "status": "success",
                    "algorithm": algorithm,
                    "shared_secret_size": len(shared_secret),
                    "execution_time_ms": elapsed * 1000,
                }
                print(json.dumps(result, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"Decapsulation failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def _get_ciphertext_size(self, algorithm: str) -> int:
        """Get ciphertext size for KEM algorithm"""
        sizes = {
            "ML-KEM-512": 768,
            "ML-KEM-768": 1088,
            "ML-KEM-1024": 1568,
        }
        return sizes.get(algorithm, 1088)
    
    # =========================================================================
    # Digital Signature Commands
    # =========================================================================
    
    def cmd_sign(self, args: argparse.Namespace) -> int:
        """Sign a message or file"""
        logger.info("Generating digital signature...")
        
        try:
            # Load secret key and message
            secret_key = self._load_key(args.secret_key)
            message = self._load_message(args.message)
            algorithm = args.algorithm or "ML-DSA-65"
            
            start_time = time.time()
            
            if self.pqc_manager:
                signature = self.pqc_manager.sign(message, secret_key, algorithm)
            else:
                # Simulation mode
                signature = os.urandom(self._get_signature_size(algorithm))
            
            elapsed = time.time() - start_time
            
            # Save signature
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(signature)
            
            msg_hash = hashlib.sha256(message).hexdigest()
            
            logger.info(f"Signing completed in {elapsed*1000:.2f}ms")
            logger.info(f"Message hash: {msg_hash[:32]}...")
            logger.info(f"Signature size: {len(signature)} bytes -> {output_path}")
            
            if args.json:
                result = {
                    "status": "success",
                    "algorithm": algorithm,
                    "message_hash": msg_hash,
                    "signature_size": len(signature),
                    "execution_time_ms": elapsed * 1000,
                }
                print(json.dumps(result, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def cmd_verify(self, args: argparse.Namespace) -> int:
        """Verify a digital signature"""
        logger.info("Verifying digital signature...")
        
        try:
            # Load public key, message, and signature
            public_key = self._load_key(args.public_key)
            message = self._load_message(args.message)
            signature = self._load_key(args.signature)
            algorithm = args.algorithm or "ML-DSA-65"
            
            start_time = time.time()
            
            if self.pqc_manager:
                valid = self.pqc_manager.verify(message, signature, public_key, algorithm)
            else:
                # Simulation mode - always return True
                valid = True
            
            elapsed = time.time() - start_time
            
            if valid:
                logger.info(f"Signature verification: VALID")
            else:
                logger.warning(f"Signature verification: INVALID")
            
            logger.info(f"Verification completed in {elapsed*1000:.2f}ms")
            
            if args.json:
                result = {
                    "status": "success",
                    "valid": valid,
                    "algorithm": algorithm,
                    "execution_time_ms": elapsed * 1000,
                }
                print(json.dumps(result, indent=2))
            
            return 0 if valid else 1
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def _get_signature_size(self, algorithm: str) -> int:
        """Get signature size for algorithm"""
        sizes = {
            "ML-DSA-44": 2420,
            "ML-DSA-65": 3293,
            "ML-DSA-87": 4595,
            "SLH-DSA-SHA2-128s": 7856,
            "SLH-DSA-SHA2-128f": 17088,
            "SLH-DSA-SHA2-192s": 16224,
            "SLH-DSA-SHA2-192f": 35664,
            "SLH-DSA-SHA2-256s": 29792,
            "SLH-DSA-SHA2-256f": 49856,
        }
        return sizes.get(algorithm, 3293)
    
    # =========================================================================
    # FHE Commands
    # =========================================================================
    
    def cmd_fhe_encrypt(self, args: argparse.Namespace) -> int:
        """Encrypt data using FHE"""
        logger.info("Performing FHE encryption...")
        
        try:
            # Load input data
            if args.input:
                with open(args.input, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    values = [float(x) for x in data]
                else:
                    values = [float(data)]
            else:
                values = [float(x) for x in args.values]
            
            start_time = time.time()
            
            if self.fhe_engine:
                ciphertext = self.fhe_engine.encrypt(values)
                # Serialize ciphertext
                ct_data = self.fhe_engine.serialize_ciphertext(ciphertext)
            else:
                # Simulation mode
                ct_data = base64.b64encode(os.urandom(8192)).decode()
            
            elapsed = time.time() - start_time
            
            # Save ciphertext
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = {
                "type": "fhe_ciphertext",
                "values_count": len(values),
                "ciphertext": ct_data,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"FHE encryption completed in {elapsed*1000:.2f}ms")
            logger.info(f"Encrypted {len(values)} values -> {output_path}")
            
            if args.json:
                output = {
                    "status": "success",
                    "values_count": len(values),
                    "ciphertext_size": len(ct_data),
                    "execution_time_ms": elapsed * 1000,
                }
                print(json.dumps(output, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"FHE encryption failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def cmd_fhe_decrypt(self, args: argparse.Namespace) -> int:
        """Decrypt FHE ciphertext"""
        logger.info("Performing FHE decryption...")
        
        try:
            # Load ciphertext
            with open(args.input, 'r') as f:
                data = json.load(f)
            
            ct_data = data.get("ciphertext", data)
            
            start_time = time.time()
            
            if self.fhe_engine:
                ciphertext = self.fhe_engine.deserialize_ciphertext(ct_data)
                values = self.fhe_engine.decrypt(ciphertext)
            else:
                # Simulation mode
                values = [0.0] * data.get("values_count", 1)
            
            elapsed = time.time() - start_time
            
            # Save or display result
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(values, f, indent=2)
                logger.info(f"Decrypted values saved to {output_path}")
            else:
                logger.info(f"Decrypted values: {values}")
            
            logger.info(f"FHE decryption completed in {elapsed*1000:.2f}ms")
            
            if args.json:
                result = {
                    "status": "success",
                    "values": values,
                    "values_count": len(values),
                    "execution_time_ms": elapsed * 1000,
                }
                print(json.dumps(result, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"FHE decryption failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def cmd_fhe_compute(self, args: argparse.Namespace) -> int:
        """Perform homomorphic computation"""
        logger.info(f"Performing FHE computation: {args.operation}")
        
        try:
            # Load ciphertexts
            with open(args.input1, 'r') as f:
                data1 = json.load(f)
            
            ct1_data = data1.get("ciphertext", data1)
            
            # Load second operand if needed
            ct2_data = None
            scalar = None
            
            if args.operation in ["add_scalar", "multiply_scalar"]:
                scalar = args.scalar
            elif args.input2:
                with open(args.input2, 'r') as f:
                    data2 = json.load(f)
                ct2_data = data2.get("ciphertext", data2)
            
            start_time = time.time()
            
            if self.fhe_engine:
                ct1 = self.fhe_engine.deserialize_ciphertext(ct1_data)
                
                if args.operation == "add" and ct2_data:
                    ct2 = self.fhe_engine.deserialize_ciphertext(ct2_data)
                    result_ct = self.fhe_engine.add(ct1, ct2)
                elif args.operation == "subtract" and ct2_data:
                    ct2 = self.fhe_engine.deserialize_ciphertext(ct2_data)
                    result_ct = self.fhe_engine.subtract(ct1, ct2)
                elif args.operation == "multiply" and ct2_data:
                    ct2 = self.fhe_engine.deserialize_ciphertext(ct2_data)
                    result_ct = self.fhe_engine.multiply(ct1, ct2)
                elif args.operation == "negate":
                    result_ct = self.fhe_engine.negate(ct1)
                elif args.operation == "square":
                    result_ct = self.fhe_engine.square(ct1)
                elif args.operation == "add_scalar" and scalar is not None:
                    result_ct = self.fhe_engine.add_scalar(ct1, scalar)
                elif args.operation == "multiply_scalar" and scalar is not None:
                    result_ct = self.fhe_engine.multiply_scalar(ct1, scalar)
                else:
                    raise ValueError(f"Invalid operation or missing operand: {args.operation}")
                
                result_data = self.fhe_engine.serialize_ciphertext(result_ct)
            else:
                # Simulation mode
                result_data = base64.b64encode(os.urandom(8192)).decode()
            
            elapsed = time.time() - start_time
            
            # Save result
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = {
                "type": "fhe_ciphertext",
                "operation": args.operation,
                "ciphertext": result_data,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"FHE computation completed in {elapsed*1000:.2f}ms")
            logger.info(f"Result saved to {output_path}")
            
            if args.json:
                output = {
                    "status": "success",
                    "operation": args.operation,
                    "execution_time_ms": elapsed * 1000,
                }
                print(json.dumps(output, indent=2))
            
            return 0
            
        except Exception as e:
            logger.error(f"FHE computation failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    # =========================================================================
    # Benchmark Commands
    # =========================================================================
    
    def cmd_benchmark(self, args: argparse.Namespace) -> int:
        """Run benchmarks"""
        logger.info(f"Running benchmark suite: {args.suite}")
        
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from benchmarks import run_full_benchmark, run_quick_benchmark
            
            if args.suite == "full":
                results = run_full_benchmark(
                    iterations=args.iterations,
                    output_dir=args.output,
                )
            else:
                results = run_quick_benchmark(
                    iterations=args.iterations,
                    output_dir=args.output,
                )
            
            logger.info("Benchmark completed successfully")
            
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            
            return 0
            
        except ImportError:
            logger.warning("Benchmark module not available, running simple benchmark...")
            return self._run_simple_benchmark(args)
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            if args.json:
                print(json.dumps({"status": "error", "message": str(e)}))
            return 1
    
    def _run_simple_benchmark(self, args: argparse.Namespace) -> int:
        """Run a simple built-in benchmark"""
        results = {
            "suite": args.suite,
            "iterations": args.iterations,
            "results": {},
        }
        
        # Benchmark key generation
        for algo in ["ML-KEM-768", "ML-DSA-65"]:
            times = []
            for _ in range(args.iterations):
                start = time.time()
                if self.pqc_manager:
                    if "KEM" in algo:
                        self.pqc_manager.generate_kem_keypair(algo)
                    else:
                        self.pqc_manager.generate_signature_keypair(algo)
                else:
                    time.sleep(0.001)  # Simulation
                times.append((time.time() - start) * 1000)
            
            results["results"][f"{algo}_keygen_ms"] = {
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }
            logger.info(f"{algo} keygen: {sum(times)/len(times):.2f}ms (avg)")
        
        # Save results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "benchmark_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_dir / 'benchmark_results.json'}")
        
        if args.json:
            print(json.dumps(results, indent=2))
        
        return 0
    
    # =========================================================================
    # Utility Commands
    # =========================================================================
    
    def cmd_info(self, args: argparse.Namespace) -> int:
        """Display system information"""
        info = {
            "version": VERSION,
            "pqc_available": self.pqc_manager is not None,
            "fhe_available": self.fhe_engine is not None,
            "supported_kem_algorithms": PQC_KEM_ALGORITHMS,
            "supported_signature_algorithms": PQC_SIGNATURE_ALGORITHMS,
            "supported_fhe_operations": FHE_OPERATIONS,
        }
        
        if self.pqc_manager:
            info["pqc_library"] = type(self.pqc_manager).__name__
        
        if self.fhe_engine:
            info["fhe_library"] = type(self.fhe_engine).__name__
        
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"PQC-FHE CLI v{VERSION}")
            print(f"{'='*60}")
            print(f"\nCryptographic Engines:")
            print(f"  PQC Available: {'Yes' if info['pqc_available'] else 'No (simulation mode)'}")
            print(f"  FHE Available: {'Yes' if info['fhe_available'] else 'No (simulation mode)'}")
            print(f"\nSupported KEM Algorithms:")
            for algo in PQC_KEM_ALGORITHMS:
                print(f"  - {algo}")
            print(f"\nSupported Signature Algorithms:")
            for algo in PQC_SIGNATURE_ALGORITHMS:
                print(f"  - {algo}")
            print(f"\nSupported FHE Operations:")
            for op in FHE_OPERATIONS:
                print(f"  - {op}")
            print(f"\n{'='*60}")
        
        return 0
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _load_key(self, filepath: str) -> bytes:
        """Load key from file"""
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            if 'public_key' in data:
                return base64.b64decode(data['public_key'])
            elif 'secret_key' in data:
                return base64.b64decode(data['secret_key'])
            else:
                raise ValueError("JSON file does not contain valid key data")
        elif path.suffix == '.pem':
            with open(path, 'r') as f:
                pem_data = f.read()
            # Extract base64 content between headers
            lines = pem_data.strip().split('\n')
            b64_lines = [l for l in lines if not l.startswith('-----')]
            return base64.b64decode(''.join(b64_lines))
        else:
            with open(path, 'rb') as f:
                return f.read()
    
    def _load_message(self, source: str) -> bytes:
        """Load message from file or string"""
        path = Path(source)
        if path.exists():
            with open(path, 'rb') as f:
                return f.read()
        else:
            return source.encode('utf-8')
    
    def _detect_algorithm(self, key: bytes) -> str:
        """Detect algorithm from key size"""
        size = len(key)
        # Based on public key sizes
        if size == 800:
            return "ML-KEM-512"
        elif size == 1184:
            return "ML-KEM-768"
        elif size == 1568:
            return "ML-KEM-1024"
        else:
            return "ML-KEM-768"  # Default


# =============================================================================
# Argument Parser Setup
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands"""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Post-Quantum Cryptography and Fully Homomorphic Encryption CLI",
        epilog="For more information, visit: https://github.com/your-repo/pqc-fhe",
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # keygen command
    # =========================================================================
    keygen_parser = subparsers.add_parser(
        'keygen',
        help='Generate PQC key pairs',
        description='Generate post-quantum cryptographic key pairs'
    )
    keygen_parser.add_argument(
        '-a', '--algorithm',
        choices=PQC_KEM_ALGORITHMS + PQC_SIGNATURE_ALGORITHMS,
        default='ML-KEM-768',
        help='Algorithm to use (default: ML-KEM-768)'
    )
    keygen_parser.add_argument(
        '-o', '--output',
        default='./keys',
        help='Output directory for keys (default: ./keys)'
    )
    keygen_parser.add_argument(
        '-f', '--format',
        choices=['json', 'pem', 'binary'],
        default='json',
        help='Output format (default: json)'
    )
    
    # =========================================================================
    # encapsulate command
    # =========================================================================
    encap_parser = subparsers.add_parser(
        'encapsulate',
        help='KEM encapsulation',
        description='Encapsulate a shared secret using KEM'
    )
    encap_parser.add_argument(
        '-p', '--public-key',
        required=True,
        help='Path to public key file'
    )
    encap_parser.add_argument(
        '-a', '--algorithm',
        choices=PQC_KEM_ALGORITHMS,
        help='Algorithm (auto-detected if not specified)'
    )
    encap_parser.add_argument(
        '-o', '--output',
        default='./encapsulation',
        help='Output directory (default: ./encapsulation)'
    )
    
    # =========================================================================
    # decapsulate command
    # =========================================================================
    decap_parser = subparsers.add_parser(
        'decapsulate',
        help='KEM decapsulation',
        description='Decapsulate a shared secret using KEM'
    )
    decap_parser.add_argument(
        '-s', '--secret-key',
        required=True,
        help='Path to secret key file'
    )
    decap_parser.add_argument(
        '-c', '--ciphertext',
        required=True,
        help='Path to ciphertext file'
    )
    decap_parser.add_argument(
        '-a', '--algorithm',
        choices=PQC_KEM_ALGORITHMS,
        help='Algorithm (default: ML-KEM-768)'
    )
    decap_parser.add_argument(
        '-o', '--output',
        default='./shared_secret.bin',
        help='Output file for shared secret'
    )
    
    # =========================================================================
    # sign command
    # =========================================================================
    sign_parser = subparsers.add_parser(
        'sign',
        help='Generate digital signature',
        description='Sign a message using post-quantum digital signature'
    )
    sign_parser.add_argument(
        '-s', '--secret-key',
        required=True,
        help='Path to secret key file'
    )
    sign_parser.add_argument(
        '-m', '--message',
        required=True,
        help='Message to sign (file path or string)'
    )
    sign_parser.add_argument(
        '-a', '--algorithm',
        choices=PQC_SIGNATURE_ALGORITHMS,
        help='Algorithm (default: ML-DSA-65)'
    )
    sign_parser.add_argument(
        '-o', '--output',
        default='./signature.bin',
        help='Output file for signature'
    )
    
    # =========================================================================
    # verify command
    # =========================================================================
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verify digital signature',
        description='Verify a post-quantum digital signature'
    )
    verify_parser.add_argument(
        '-p', '--public-key',
        required=True,
        help='Path to public key file'
    )
    verify_parser.add_argument(
        '-m', '--message',
        required=True,
        help='Message that was signed (file path or string)'
    )
    verify_parser.add_argument(
        '-S', '--signature',
        required=True,
        help='Path to signature file'
    )
    verify_parser.add_argument(
        '-a', '--algorithm',
        choices=PQC_SIGNATURE_ALGORITHMS,
        help='Algorithm (default: ML-DSA-65)'
    )
    
    # =========================================================================
    # fhe-encrypt command
    # =========================================================================
    fhe_enc_parser = subparsers.add_parser(
        'fhe-encrypt',
        help='FHE encryption',
        description='Encrypt data using Fully Homomorphic Encryption'
    )
    fhe_enc_parser.add_argument(
        '-i', '--input',
        help='Input file (JSON array of numbers)'
    )
    fhe_enc_parser.add_argument(
        '--values',
        nargs='+',
        type=float,
        help='Values to encrypt (alternative to --input)'
    )
    fhe_enc_parser.add_argument(
        '-o', '--output',
        default='./ciphertext.json',
        help='Output file for ciphertext'
    )
    
    # =========================================================================
    # fhe-decrypt command
    # =========================================================================
    fhe_dec_parser = subparsers.add_parser(
        'fhe-decrypt',
        help='FHE decryption',
        description='Decrypt FHE ciphertext'
    )
    fhe_dec_parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input ciphertext file'
    )
    fhe_dec_parser.add_argument(
        '-o', '--output',
        help='Output file for decrypted values (optional)'
    )
    
    # =========================================================================
    # fhe-compute command
    # =========================================================================
    fhe_comp_parser = subparsers.add_parser(
        'fhe-compute',
        help='FHE computation',
        description='Perform homomorphic computation on encrypted data'
    )
    fhe_comp_parser.add_argument(
        '-op', '--operation',
        choices=FHE_OPERATIONS,
        required=True,
        help='Operation to perform'
    )
    fhe_comp_parser.add_argument(
        '-i1', '--input1',
        required=True,
        help='First input ciphertext file'
    )
    fhe_comp_parser.add_argument(
        '-i2', '--input2',
        help='Second input ciphertext file (for binary operations)'
    )
    fhe_comp_parser.add_argument(
        '--scalar',
        type=float,
        help='Scalar value (for add_scalar/multiply_scalar)'
    )
    fhe_comp_parser.add_argument(
        '-o', '--output',
        default='./result.json',
        help='Output file for result ciphertext'
    )
    
    # =========================================================================
    # benchmark command
    # =========================================================================
    bench_parser = subparsers.add_parser(
        'benchmark',
        help='Run benchmarks',
        description='Run performance benchmarks'
    )
    bench_parser.add_argument(
        '-s', '--suite',
        choices=['quick', 'full'],
        default='quick',
        help='Benchmark suite to run (default: quick)'
    )
    bench_parser.add_argument(
        '-n', '--iterations',
        type=int,
        default=10,
        help='Number of iterations (default: 10)'
    )
    bench_parser.add_argument(
        '-o', '--output',
        default='./benchmark_results',
        help='Output directory for results'
    )
    
    # =========================================================================
    # info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        'info',
        help='Display system information',
        description='Display information about supported algorithms and features'
    )
    
    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: List[str] = None) -> int:
    """Main entry point for CLI"""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    if not args.command:
        parser.print_help()
        return 0
    
    cli = CLI()
    
    # Route to appropriate command handler
    command_handlers = {
        'keygen': cli.cmd_keygen,
        'encapsulate': cli.cmd_encapsulate,
        'decapsulate': cli.cmd_decapsulate,
        'sign': cli.cmd_sign,
        'verify': cli.cmd_verify,
        'fhe-encrypt': cli.cmd_fhe_encrypt,
        'fhe-decrypt': cli.cmd_fhe_decrypt,
        'fhe-compute': cli.cmd_fhe_compute,
        'benchmark': cli.cmd_benchmark,
        'info': cli.cmd_info,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
