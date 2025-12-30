#!/usr/bin/env python3
"""
PQC-FHE Integration REST API Server v2.1.2
==========================================

FastAPI-based REST API for Post-Quantum Cryptography and 
Fully Homomorphic Encryption operations.

Usage:
    uvicorn api.server:app --reload
    
    Or:
    python -m api.server

Features:
    - PQC key generation (ML-KEM-768, ML-DSA-65) using liboqs-python
    - FHE encryption/decryption (CKKS scheme) with bootstrap support
    - Homomorphic operations (add, multiply, square)
    - Health check and system status
    - Ciphertext management with listing
    - Web UI for interactive crypto simulation

Requirements:
    - liboqs-python: Required for PQC operations
    - desilofhe: Required for FHE operations
    
Installation:
    git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python
    cd liboqs-python && pip install .
    pip install desilofhe
"""

import os
import sys
import time
import secrets
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Suppress liboqs auto-install behavior
os.environ['OQS_PERMIT_UNSUPPORTED_ARCHITECTURE'] = '1'

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL STATE
# =============================================================================

# Lazy-loaded components
_fhe_engine = None
_pqc_manager = None
_liboqs_available = None

# Ciphertext store with metadata
_ciphertext_store: Dict[str, Dict[str, Any]] = {}

# PQC Key stores for Smart Default feature
_kem_keypair_store: Dict[str, Dict[str, Any]] = {}  # Stores KEM keypairs
_sig_keypair_store: Dict[str, Dict[str, Any]] = {}  # Stores SIG keypairs
_signature_store: Dict[str, Dict[str, Any]] = {}     # Stores signatures

# Example placeholder values for Smart Default detection
_EXAMPLE_HEX_PLACEHOLDERS = {
    "string",
    "your_secret_key_here",
    "your_public_key_here", 
    "your_signature_here",
}


def _is_placeholder_value(value: str) -> bool:
    """Check if a value is a placeholder/example that should trigger Smart Default."""
    if not value:
        return False
    # Check known placeholders
    if value.lower() in _EXAMPLE_HEX_PLACEHOLDERS:
        return True
    # Check if it's clearly not hex (contains non-hex chars)
    if not all(c in '0123456789abcdefABCDEF' for c in value):
        return True
    # Check if it's too short to be a valid key (real keys are much longer)
    if len(value) < 100:
        return True
    return False


def _check_liboqs_available() -> bool:
    """
    Check if liboqs is available without triggering auto-install.
    
    IMPORTANT: We cannot use `import oqs` directly because the oqs package
    will attempt to auto-install liboqs at module load time, which can
    raise SystemExit(1) that cannot be caught with try/except.
    
    Instead, we use subprocess to safely check availability.
    """
    global _liboqs_available
    
    if _liboqs_available is not None:
        return _liboqs_available
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', 
             'import oqs; print(len(oqs.get_enabled_kem_mechanisms()))'],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, 'OQS_PERMIT_UNSUPPORTED_ARCHITECTURE': '1'}
        )
        
        if result.returncode == 0:
            num_kems = result.stdout.strip()
            logger.info(f"liboqs is available ({num_kems} KEMs)")
            _liboqs_available = True
            return True
        else:
            error_lines = result.stderr.strip().split('\n') if result.stderr else []
            error_msg = error_lines[-1] if error_lines else "Unknown error"
            if len(error_msg) > 80:
                error_msg = error_msg[:77] + "..."
            logger.info(f"liboqs not available: {error_msg}")
            _liboqs_available = False
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("liboqs check timed out")
        _liboqs_available = False
        return False
    except Exception as e:
        logger.warning(f"liboqs check failed: {e}")
        _liboqs_available = False
        return False


def _get_fhe_engine():
    """Lazy initialization of FHE engine with bootstrap enabled."""
    global _fhe_engine
    
    if _fhe_engine is None:
        try:
            from pqc_fhe_integration import FHEEngine, FHEConfig
            
            # Enable bootstrap for unlimited computation depth
            config = FHEConfig(
                mode='gpu',
                use_bootstrap=True,
                use_full_bootstrap_key=True,
                use_lossy_bootstrap=True,
                bootstrap_stage_count=3,
                thread_count=512
            )
            _fhe_engine = FHEEngine(config, logger)
            logger.info("FHE engine initialized successfully (bootstrap enabled)")
            logger.info(f"  Mode: gpu")
            logger.info(f"  Slot count: {_fhe_engine.engine.slot_count}")
        except Exception as e:
            logger.error(f"Failed to initialize FHE engine: {e}")
            logger.error("Trying CPU mode...")
            try:
                from pqc_fhe_integration import FHEEngine, FHEConfig
                config = FHEConfig(
                    mode='cpu',
                    use_bootstrap=True,
                    use_full_bootstrap_key=False,  # Save memory in CPU mode
                    use_lossy_bootstrap=True,
                    bootstrap_stage_count=3
                )
                _fhe_engine = FHEEngine(config, logger)
                logger.info("FHE engine initialized in CPU mode (bootstrap enabled)")
            except Exception as e2:
                logger.error(f"Failed to initialize FHE engine in CPU mode: {e2}")
                _fhe_engine = None
    
    return _fhe_engine


def _get_pqc_manager():
    """Lazy initialization of PQC manager (only if liboqs available)."""
    global _pqc_manager
    
    if _pqc_manager is not None:
        return _pqc_manager
    
    if not _check_liboqs_available():
        logger.error("PQC manager not initialized - liboqs-python required!")
        return None
    
    try:
        from pqc_fhe_integration import PQCKeyManager, PQCConfig
        config = PQCConfig()
        _pqc_manager = PQCKeyManager(config, logger)
        logger.info("PQC manager initialized successfully")
        return _pqc_manager
    except Exception as e:
        logger.error(f"Failed to initialize PQC manager: {e}")
        return None


# =============================================================================
# PQC IMPLEMENTATIONS (liboqs-python required)
# =============================================================================

"""
Post-Quantum Cryptography Implementation
========================================

This library requires liboqs-python for real PQC operations.
No mock/simulation mode is provided to ensure cryptographic integrity.

Supported Algorithms (NIST Standardized):
- ML-KEM-768 (FIPS 203): Key Encapsulation Mechanism
- ML-DSA-65 (FIPS 204): Digital Signature Algorithm

Installation:
    git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python
    cd liboqs-python
    pip install .

References:
- NIST FIPS 203: https://csrc.nist.gov/pubs/fips/203/final
- NIST FIPS 204: https://csrc.nist.gov/pubs/fips/204/final
- liboqs: https://github.com/open-quantum-safe/liboqs
"""


class PQCManager:
    """
    Post-Quantum Cryptography Manager using liboqs-python.
    
    Implements NIST standardized algorithms:
    - ML-KEM-768 (FIPS 203): Post-quantum Key Encapsulation Mechanism
    - ML-DSA-65 (FIPS 204): Post-quantum Digital Signature Algorithm
    
    This provides actual cryptographic security based on Module-LWE
    (Learning With Errors) problem hardness.
    
    Key Sizes (ML-KEM-768):
    - Public Key: 1184 bytes
    - Secret Key: 2400 bytes  
    - Ciphertext: 1088 bytes
    - Shared Secret: 32 bytes
    
    Key Sizes (ML-DSA-65):
    - Public Key: 1952 bytes
    - Secret Key: 4032 bytes
    - Signature: 3309 bytes (average)
    """
    
    # Algorithm names (NIST standardized)
    KEM_ALG = "ML-KEM-768"
    SIG_ALG = "ML-DSA-65"
    
    # Fallback algorithm names (older liboqs versions)
    KEM_ALG_FALLBACK = "Kyber768"
    SIG_ALG_FALLBACK = "Dilithium3"
    
    def __init__(self):
        """Initialize with liboqs."""
        if not _check_liboqs_available():
            raise RuntimeError(
                "liboqs-python is required but not installed.\n"
                "Please install it:\n"
                "  git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python\n"
                "  cd liboqs-python && pip install .\n"
                "See: https://github.com/open-quantum-safe/liboqs-python"
            )
        
        import oqs
        self.oqs = oqs
        
        # Check available algorithms
        self.kem_algs = oqs.get_enabled_kem_mechanisms()
        self.sig_algs = oqs.get_enabled_sig_mechanisms()
        
        # Select KEM algorithm
        if self.KEM_ALG in self.kem_algs:
            self.kem_alg = self.KEM_ALG
        elif self.KEM_ALG_FALLBACK in self.kem_algs:
            self.kem_alg = self.KEM_ALG_FALLBACK
            logger.warning(f"Using fallback KEM: {self.KEM_ALG_FALLBACK}")
        else:
            raise RuntimeError(f"No supported KEM algorithm found. Available: {self.kem_algs}")
        
        # Select Signature algorithm
        if self.SIG_ALG in self.sig_algs:
            self.sig_alg = self.SIG_ALG
        elif self.SIG_ALG_FALLBACK in self.sig_algs:
            self.sig_alg = self.SIG_ALG_FALLBACK
            logger.warning(f"Using fallback SIG: {self.SIG_ALG_FALLBACK}")
        else:
            raise RuntimeError(f"No supported SIG algorithm found. Available: {self.sig_algs}")
        
        logger.info(f"PQCManager initialized: KEM={self.kem_alg}, SIG={self.sig_alg}")
        
        # Get algorithm details
        kem_instance = oqs.KeyEncapsulation(self.kem_alg)
        sig_instance = oqs.Signature(self.sig_alg)
        
        self.KEM_PUBLIC_KEY_SIZE = kem_instance.details['length_public_key']
        self.KEM_SECRET_KEY_SIZE = kem_instance.details['length_secret_key']
        self.KEM_CIPHERTEXT_SIZE = kem_instance.details['length_ciphertext']
        self.KEM_SHARED_SECRET_SIZE = kem_instance.details['length_shared_secret']
        
        self.SIG_PUBLIC_KEY_SIZE = sig_instance.details['length_public_key']
        self.SIG_SECRET_KEY_SIZE = sig_instance.details['length_secret_key']
        self.SIG_SIGNATURE_SIZE = sig_instance.details['length_signature']
        
        logger.info(f"KEM sizes: pk={self.KEM_PUBLIC_KEY_SIZE}, sk={self.KEM_SECRET_KEY_SIZE}, "
                   f"ct={self.KEM_CIPHERTEXT_SIZE}, ss={self.KEM_SHARED_SECRET_SIZE}")
        logger.info(f"SIG sizes: pk={self.SIG_PUBLIC_KEY_SIZE}, sk={self.SIG_SECRET_KEY_SIZE}, "
                   f"sig={self.SIG_SIGNATURE_SIZE}")
    
    def generate_kem_keypair(self):
        """
        Generate ML-KEM-768 keypair.
        
        Returns:
            tuple: (public_key: bytes, secret_key: bytes)
        
        The keypair is generated using:
        1. Random seed generation (32 bytes)
        2. Matrix A generation from seed (using SHAKE-128)
        3. Secret vector s sampling (centered binomial distribution)
        4. Public key computation: t = As + e
        """
        kem = self.oqs.KeyEncapsulation(self.kem_alg)
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()
        return public_key, secret_key
    
    def encapsulate(self, public_key: bytes):
        """
        Encapsulate a shared secret using public key.
        
        Args:
            public_key: Recipient's public key (1184 bytes for ML-KEM-768)
        
        Returns:
            tuple: (ciphertext: bytes, shared_secret: bytes)
        
        Process:
        1. Generate random message m (32 bytes)
        2. Compute (K, r) = G(m || H(pk)) using SHAKE-256
        3. Compute ciphertext c = Enc(pk, m; r)
        4. Derive shared_secret K' = KDF(K || H(c))
        """
        kem = self.oqs.KeyEncapsulation(self.kem_alg)
        ciphertext, shared_secret = kem.encap_secret(public_key)
        return ciphertext, shared_secret
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes):
        """
        Decapsulate shared secret using secret key.
        
        Args:
            ciphertext: Ciphertext from encapsulation (1088 bytes for ML-KEM-768)
            secret_key: Recipient's secret key (2400 bytes for ML-KEM-768)
        
        Returns:
            bytes: Shared secret (32 bytes)
        
        Process:
        1. Decrypt ciphertext: m' = Dec(sk, c)
        2. Re-encapsulate: c' = Enc(pk, m'; r')
        3. If c == c': return KDF(K || H(c))
        4. Else: return KDF(z || H(c)) (implicit rejection)
        """
        kem = self.oqs.KeyEncapsulation(self.kem_alg, secret_key)
        shared_secret = kem.decap_secret(ciphertext)
        return shared_secret
    
    def generate_sig_keypair(self):
        """
        Generate ML-DSA-65 keypair.
        
        Returns:
            tuple: (public_key: bytes, secret_key: bytes)
        
        The keypair is generated using:
        1. Random seed ξ generation (32 bytes)
        2. Expand to (ρ, ρ', K) using SHAKE-256
        3. Generate matrix A from ρ
        4. Sample secret vectors (s1, s2) from ρ'
        5. Compute t = As1 + s2
        """
        sig = self.oqs.Signature(self.sig_alg)
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
        return public_key, secret_key
    
    def sign(self, message: bytes, secret_key: bytes):
        """
        Sign message using secret key.
        
        Args:
            message: Message to sign (arbitrary length)
            secret_key: Signer's secret key (4032 bytes for ML-DSA-65)
        
        Returns:
            bytes: Signature (3309 bytes average for ML-DSA-65)
        
        Process (Fiat-Shamir with Aborts):
        1. Compute µ = H(tr || M) where tr = H(pk)
        2. Sample masking vector y
        3. Compute w = Ay
        4. Compute challenge c = H(µ || w1)
        5. Compute z = y + cs1
        6. If ||z|| or ||hints|| too large, restart
        7. Return σ = (c, z, hints)
        """
        sig = self.oqs.Signature(self.sig_alg, secret_key)
        signature = sig.sign(message)
        return signature
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes):
        """
        Verify signature.
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's public key (1952 bytes for ML-DSA-65)
        
        Returns:
            bool: True if signature is valid
        
        Process:
        1. Parse σ = (c, z, hints)
        2. Compute w'1 = UseHint(Az - ct, hints)
        3. Compute c' = H(µ || w'1)
        4. Return c == c'
        """
        sig = self.oqs.Signature(self.sig_alg)
        return sig.verify(message, signature, public_key)


def _create_pqc_manager():
    """
    Create PQC manager (liboqs-python required).
    
    Raises:
        RuntimeError: If liboqs-python is not installed
    """
    return PQCManager()


# Initialize PQC manager (will fail if liboqs not installed)
_pqc_manager_instance = None

def _get_pqc_instance():
    """Get or create PQC manager instance."""
    global _pqc_manager_instance
    if _pqc_manager_instance is None:
        _pqc_manager_instance = _create_pqc_manager()
    return _pqc_manager_instance


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "2.1.2"
    components: Dict[str, str]
    ciphertext_count: int = 0


class CiphertextInfo(BaseModel):
    id: str
    created_at: str
    data_length: int
    operation: Optional[str] = None


class CiphertextListResponse(BaseModel):
    count: int
    ciphertexts: List[CiphertextInfo]


class KEMKeyPairResponse(BaseModel):
    public_key: str = Field(..., description="Hex-encoded public key")
    secret_key: str = Field(..., description="Hex-encoded secret key")
    algorithm: str = "ML-KEM-768"
    public_key_size: int
    secret_key_size: int


class EncapsulateRequest(BaseModel):
    public_key: str = Field(..., description="Hex-encoded public key")


class EncapsulateResponse(BaseModel):
    ciphertext: str
    shared_secret: str


class DecapsulateRequest(BaseModel):
    ciphertext: str
    secret_key: str


class DecapsulateResponse(BaseModel):
    shared_secret: str


class SignatureKeyPairResponse(BaseModel):
    public_key: str
    secret_key: str
    algorithm: str = "ML-DSA-65"
    public_key_size: int
    secret_key_size: int


class SignRequest(BaseModel):
    message: str
    secret_key: str


class SignResponse(BaseModel):
    signature: str
    signature_size: int


class VerifyRequest(BaseModel):
    message: str
    signature: str
    public_key: str


class VerifyResponse(BaseModel):
    valid: bool


class FHEEncryptRequest(BaseModel):
    data: List[float] = Field(
        ..., 
        description="List of floats to encrypt",
        json_schema_extra={"example": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [1.0, 2.0, 3.0, 4.0, 5.0]
                }
            ]
        }
    }


class FHEEncryptResponse(BaseModel):
    ciphertext_id: str = Field(
        ..., 
        description="Unique ID to reference this ciphertext in other operations",
        json_schema_extra={"example": "c40e1e56fda432fa80efdf0486959354"}
    )
    slot_count: int
    data_length: int
    created_at: str


class FHEDecryptRequest(BaseModel):
    ciphertext_id: str = Field(
        ..., 
        description="Ciphertext ID from /fhe/encrypt response. Use /fhe/ciphertexts to list available IDs.",
        json_schema_extra={"example": "c40e1e56fda432fa80efdf0486959354"}
    )
    length: Optional[int] = Field(
        None,
        description="Number of values to decrypt (auto-detected if not specified)"
    )


class FHEDecryptResponse(BaseModel):
    data: List[float] = Field(
        ...,
        description="Decrypted values",
        json_schema_extra={"example": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )


class FHEOperationRequest(BaseModel):
    ciphertext_id: str = Field(
        ..., 
        description="Primary ciphertext ID from /fhe/encrypt. Use /fhe/ciphertexts to list available IDs.",
        json_schema_extra={"example": "c40e1e56fda432fa80efdf0486959354"}
    )
    scalar: Optional[float] = Field(
        None,
        description="Scalar value for add_scalar or multiply_scalar operations",
        json_schema_extra={"example": 2.5}
    )
    other_ciphertext_id: Optional[str] = Field(
        None, 
        description="Second ciphertext ID for ciphertext-ciphertext operations",
        json_schema_extra={"example": "a1b2c3d4e5f6789012345678abcdef01"}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ciphertext_id": "c40e1e56fda432fa80efdf0486959354",
                    "scalar": 2.5,
                    "other_ciphertext_id": None
                }
            ]
        }
    }


class FHEOperationResponse(BaseModel):
    result_ciphertext_id: str = Field(
        ...,
        description="New ciphertext ID containing the operation result",
        json_schema_extra={"example": "result_abc123def456789012345678"}
    )
    operation: str = Field(
        ...,
        description="Operation performed",
        json_schema_extra={"example": "add_scalar(2.5)"}
    )
    created_at: str


# =============================================================================
# APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 60)
    logger.info("PQC-FHE Integration API Server v2.1.2")
    logger.info("=" * 60)
    
    liboqs_available = _check_liboqs_available()
    if liboqs_available:
        try:
            pqc = _get_pqc_instance()
            logger.info(f"PQC: {pqc.kem_alg} / {pqc.sig_alg}")
        except Exception as e:
            logger.error(f"PQC initialization failed: {e}")
    else:
        logger.error("liboqs-python is NOT installed - PQC operations will fail!")
        logger.error("Install: git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python && cd liboqs-python && pip install .")
    
    fhe = _get_fhe_engine()
    fhe_status = "available (bootstrap enabled)" if fhe else "unavailable"
    logger.info(f"FHE engine: {fhe_status}")
    
    logger.info("=" * 60)
    logger.info("Server ready at http://127.0.0.1:8000")
    logger.info("Web UI at http://127.0.0.1:8000/ui")
    logger.info("API docs at http://127.0.0.1:8000/docs")
    logger.info("=" * 60)
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="PQC-FHE Integration API",
    description="""
REST API for Post-Quantum Cryptography and Fully Homomorphic Encryption.

## Web UI

**Interactive Crypto Simulation available at: [/ui](/ui)**

The Web UI provides a user-friendly interface for:
- ML-KEM Key Exchange Simulation (Alice ↔ Bob)
- ML-DSA Digital Signature (Sign & Verify)
- FHE Encrypted Computation (Encrypt → Compute → Decrypt)

## Features

### Post-Quantum Cryptography (PQC)
- **ML-KEM-768**: Key Encapsulation Mechanism (NIST FIPS 203)
- **ML-DSA-65**: Digital Signature Algorithm (NIST FIPS 204)

### Fully Homomorphic Encryption (FHE)
- **CKKS Scheme**: Approximate arithmetic on encrypted data
- **Bootstrap**: Unlimited computation depth
- **GPU Acceleration**: CUDA-powered operations

## Smart Default Feature (v2.1.2+)

**No need to copy ciphertext IDs manually!**

When testing in Swagger UI:
1. First, call `POST /fhe/encrypt` with your data
2. Then, just click "Execute" on any FHE operation - the example ID will automatically use your latest ciphertext!

The API detects when example/placeholder IDs are used and automatically selects the most recent ciphertext.

## Manual Workflow (if preferred)

1. **Encrypt data:** `POST /fhe/encrypt` with `{"data": [1.0, 2.0, 3.0]}`
2. **Copy the ciphertext_id** from the response
3. **Use that ID** in other operations
4. **List available IDs:** `GET /fhe/ciphertexts`

## Requirements
- **liboqs-python**: Required for PQC operations (ML-KEM-768, ML-DSA-65)
- **desilofhe**: Required for FHE operations (CKKS scheme)
- Ciphertexts are stored in memory and cleared on server restart
""",
    version="2.1.2",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# WEB UI ROUTES
# =============================================================================

# Get the directory where server.py is located
_SERVER_DIR = Path(__file__).parent.parent
_WEB_UI_DIR = _SERVER_DIR / "web_ui"


@app.get("/ui", response_class=HTMLResponse, tags=["Web UI"])
async def web_ui():
    """
    Serve the interactive Web UI for crypto simulation.
    
    The Web UI provides visual, step-by-step demonstrations of:
    - **Key Exchange**: ML-KEM key encapsulation between Alice and Bob
    - **Digital Signature**: ML-DSA message signing and verification
    - **FHE Computation**: Encrypted data computation with CKKS
    """
    ui_path = _WEB_UI_DIR / "index.html"
    if not ui_path.exists():
        # Fallback: try current directory
        ui_path = Path("web_ui/index.html")
    if not ui_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web UI not found. Please ensure web_ui/index.html exists."
        )
    return FileResponse(ui_path, media_type="text/html")


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root endpoint."""
    return {
        "name": "PQC-FHE Integration API",
        "version": "2.1.2",
        "docs": "/docs",
        "health": "/health",
        "ciphertexts": "/fhe/ciphertexts"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and component status."""
    fhe = _get_fhe_engine()
    
    # Check PQC status
    liboqs_available = _check_liboqs_available()
    pqc_status = "unavailable"
    pqc_algorithms = ""
    
    if liboqs_available:
        try:
            pqc = _get_pqc_instance()
            pqc_status = "available"
            pqc_algorithms = f"{pqc.kem_alg} / {pqc.sig_alg}"
        except Exception as e:
            pqc_status = f"error: {str(e)}"
    
    components = {
        "api": "healthy",
        "liboqs": "available" if liboqs_available else "not installed",
        "pqc": pqc_status,
        "pqc_algorithms": pqc_algorithms if pqc_algorithms else "N/A",
        "fhe": "available" if fhe else "unavailable",
        "fhe_bootstrap": "enabled" if (fhe and fhe.config.use_bootstrap) else "disabled"
    }
    
    # Overall status
    status = "healthy" if (liboqs_available and fhe) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        components=components,
        ciphertext_count=len(_ciphertext_store)
    )


@app.get("/fhe/ciphertexts", response_model=CiphertextListResponse, tags=["FHE"])
async def list_ciphertexts():
    """
    List all stored ciphertexts.
    
    **Use this endpoint to find valid ciphertext IDs for other operations.**
    
    Each ciphertext entry contains:
    - `id`: The ciphertext ID to use in other endpoints
    - `created_at`: When the ciphertext was created
    - `data_length`: Number of values encrypted
    - `operation`: What operation created this ciphertext
    """
    ciphertexts = []
    for ct_id, ct_data in _ciphertext_store.items():
        ciphertexts.append(CiphertextInfo(
            id=ct_id,
            created_at=ct_data.get('created_at', 'unknown'),
            data_length=ct_data.get('data_length', 0),
            operation=ct_data.get('operation')
        ))
    
    return CiphertextListResponse(
        count=len(ciphertexts),
        ciphertexts=ciphertexts
    )


@app.delete("/fhe/ciphertexts", tags=["FHE"])
async def clear_ciphertexts():
    """Clear all stored ciphertexts (use with caution)."""
    count = len(_ciphertext_store)
    _ciphertext_store.clear()
    return {"message": f"Cleared {count} ciphertexts"}


@app.delete("/fhe/ciphertexts/{ciphertext_id}", tags=["FHE"])
async def delete_ciphertext(ciphertext_id: str):
    """Delete a specific ciphertext by ID."""
    if ciphertext_id not in _ciphertext_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ciphertext '{ciphertext_id}' not found. Use /fhe/ciphertexts to list available."
        )
    
    del _ciphertext_store[ciphertext_id]
    return {"message": f"Deleted ciphertext {ciphertext_id}"}


# =============================================================================
# PQC KEM ENDPOINTS
# =============================================================================

@app.post("/pqc/kem/keypair", response_model=KEMKeyPairResponse, tags=["PQC - KEM"])
async def generate_kem_keypair():
    """
    Generate ML-KEM-768 keypair.
    
    ML-KEM-768 (FIPS 203) provides IND-CCA2 security at NIST Level 3.
    
    Returns:
        - public_key: 1184 bytes (hex-encoded)
        - secret_key: 2400 bytes (hex-encoded)
    """
    try:
        pqc = _get_pqc_instance()
        public_key, secret_key = pqc.generate_kem_keypair()
        
        return KEMKeyPairResponse(
            public_key=public_key.hex(),
            secret_key=secret_key.hex(),
            algorithm=pqc.kem_alg,
            public_key_size=len(public_key),
            secret_key_size=len(secret_key)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PQC not available: {str(e)}. Please install liboqs-python."
        )


@app.post("/pqc/kem/encapsulate", response_model=EncapsulateResponse, tags=["PQC - KEM"])
async def encapsulate(request: EncapsulateRequest):
    """Encapsulate shared secret using public key."""
    # Validate public_key
    if not request.public_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="public_key is required. Generate a keypair first using POST /pqc/kem/keypair"
        )
    
    # Check if it's valid hex
    if not all(c in '0123456789abcdefABCDEF' for c in request.public_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="public_key contains invalid characters. Expected hex-encoded string"
        )
    
    try:
        public_key = bytes.fromhex(request.public_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid hex-encoded public_key: {str(e)}"
        )
    
    # Validate key size (ML-KEM-768 public key is 1184 bytes = 2368 hex chars)
    expected_size = 1184
    if len(public_key) != expected_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid public_key size: {len(public_key)} bytes. Expected {expected_size} bytes for ML-KEM-768"
        )
    
    try:
        pqc = _get_pqc_instance()
        ciphertext, shared_secret = pqc.encapsulate(public_key)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PQC not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Encapsulation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encapsulation failed: {str(e)}"
        )
    
    return EncapsulateResponse(
        ciphertext=ciphertext.hex(),
        shared_secret=shared_secret.hex()
    )


@app.post("/pqc/kem/decapsulate", response_model=DecapsulateResponse, tags=["PQC - KEM"])
async def decapsulate(request: DecapsulateRequest):
    """Decapsulate shared secret using secret key."""
    # Validate ciphertext
    if not request.ciphertext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ciphertext is required"
        )
    
    # Validate secret_key
    if not request.secret_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="secret_key is required. Generate a keypair first using POST /pqc/kem/keypair"
        )
    
    # Check hex format for ciphertext
    if not all(c in '0123456789abcdefABCDEF' for c in request.ciphertext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ciphertext contains invalid characters. Expected hex-encoded string"
        )
    
    # Check hex format for secret_key
    if not all(c in '0123456789abcdefABCDEF' for c in request.secret_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="secret_key contains invalid characters. Expected hex-encoded string"
        )
    
    try:
        ciphertext = bytes.fromhex(request.ciphertext)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid hex-encoded ciphertext: {str(e)}"
        )
    
    try:
        secret_key = bytes.fromhex(request.secret_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid hex-encoded secret_key: {str(e)}"
        )
    
    # Validate sizes (ML-KEM-768: ciphertext=1088, secret_key=2400)
    expected_ct_size = 1088
    expected_sk_size = 2400
    
    if len(ciphertext) != expected_ct_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ciphertext size: {len(ciphertext)} bytes. Expected {expected_ct_size} bytes for ML-KEM-768"
        )
    
    if len(secret_key) != expected_sk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid secret_key size: {len(secret_key)} bytes. Expected {expected_sk_size} bytes for ML-KEM-768"
        )
    
    try:
        pqc = _get_pqc_instance()
        shared_secret = pqc.decapsulate(ciphertext, secret_key)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PQC not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Decapsulation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decapsulation failed: {str(e)}"
        )
    
    return DecapsulateResponse(
        shared_secret=shared_secret.hex()
    )


# =============================================================================
# PQC SIGNATURE ENDPOINTS
# =============================================================================

def _store_sig_keypair(public_key: bytes, secret_key: bytes) -> str:
    """Store SIG keypair with metadata and return ID."""
    kp_id = secrets.token_hex(16)
    _sig_keypair_store[kp_id] = {
        'public_key': public_key,
        'secret_key': secret_key,
        'created_at': datetime.now().isoformat(),
    }
    return kp_id


def _store_signature(signature: bytes, message: str, keypair_id: str) -> str:
    """Store signature with metadata and return ID."""
    sig_id = secrets.token_hex(16)
    _signature_store[sig_id] = {
        'signature': signature,
        'message': message,
        'keypair_id': keypair_id,
        'created_at': datetime.now().isoformat(),
    }
    return sig_id


def _get_latest_sig_keypair() -> Optional[Dict[str, Any]]:
    """Get the most recently created SIG keypair."""
    if not _sig_keypair_store:
        return None
    latest_id = max(_sig_keypair_store.keys(),
                    key=lambda k: _sig_keypair_store[k].get('created_at', ''))
    return _sig_keypair_store[latest_id], latest_id


def _get_latest_signature() -> Optional[Dict[str, Any]]:
    """Get the most recently created signature."""
    if not _signature_store:
        return None
    latest_id = max(_signature_store.keys(),
                    key=lambda k: _signature_store[k].get('created_at', ''))
    return _signature_store[latest_id], latest_id


@app.post("/pqc/sig/keypair", response_model=SignatureKeyPairResponse, tags=["PQC - Signatures"])
async def generate_sig_keypair():
    """
    Generate ML-DSA-65 keypair.
    
    ML-DSA-65 (FIPS 204) provides EUF-CMA security at NIST Level 3.
    
    **Smart Default:** The generated keypair is stored and will be auto-used
    in subsequent /sign and /verify operations if placeholder values are provided.
    
    Returns:
        - public_key: 1952 bytes (hex-encoded)
        - secret_key: 4032 bytes (hex-encoded)
    """
    try:
        pqc = _get_pqc_instance()
        public_key, secret_key = pqc.generate_sig_keypair()
        
        # Store keypair for Smart Default
        kp_id = _store_sig_keypair(public_key, secret_key)
        logger.info(f"Stored SIG keypair: {kp_id}")
        
        return SignatureKeyPairResponse(
            public_key=public_key.hex(),
            secret_key=secret_key.hex(),
            algorithm=pqc.sig_alg,
            public_key_size=len(public_key),
            secret_key_size=len(secret_key)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PQC not available: {str(e)}. Please install liboqs-python."
        )


@app.post("/pqc/sig/sign", response_model=SignResponse, tags=["PQC - Signatures"])
async def sign_message(request: SignRequest):
    """
    Sign a message using ML-DSA-65.
    
    **Smart Default:** If secret_key is a placeholder (like "string"), 
    the most recently generated keypair will be used automatically.
    First call POST /pqc/sig/keypair to generate a keypair.
    """
    # Smart Default: Check if placeholder value is used
    if _is_placeholder_value(request.secret_key):
        keypair_data = _get_latest_sig_keypair()
        if keypair_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No keypairs available. Generate a keypair first using POST /pqc/sig/keypair"
            )
        keypair, kp_id = keypair_data
        secret_key = keypair['secret_key']
        logger.info(f"Smart Default: Using stored secret_key from keypair {kp_id}")
    else:
        # Normal validation
        if not request.secret_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="secret_key is required. Generate a keypair first using POST /pqc/sig/keypair"
            )
        
        # Check if it's valid hex
        if not all(c in '0123456789abcdefABCDEF' for c in request.secret_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"secret_key contains invalid characters. Expected hex-encoded string, got: '{request.secret_key[:20]}...'"
            )
        
        try:
            secret_key = bytes.fromhex(request.secret_key)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid hex-encoded secret_key: {str(e)}. Key length: {len(request.secret_key)} chars"
            )
        
        # Validate key size (ML-DSA-65 secret key is 4032 bytes = 8064 hex chars)
        expected_size = 4032
        if len(secret_key) != expected_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid secret_key size: {len(secret_key)} bytes. Expected {expected_size} bytes for ML-DSA-65"
            )
        kp_id = "manual"
    
    message = request.message.encode('utf-8')
    
    try:
        pqc = _get_pqc_instance()
        signature = pqc.sign(message, secret_key)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PQC not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Signing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signing operation failed: {str(e)}"
        )
    
    # Store signature for Smart Default
    sig_id = _store_signature(signature, request.message, kp_id)
    logger.info(f"Stored signature: {sig_id}")
    
    return SignResponse(
        signature=signature.hex(),
        signature_size=len(signature)
    )


@app.post("/pqc/sig/verify", response_model=VerifyResponse, tags=["PQC - Signatures"])
async def verify_signature(request: VerifyRequest):
    """
    Verify a signature using ML-DSA-65.
    
    **Smart Default:** If signature or public_key are placeholders (like "string"),
    the most recently created signature and keypair will be used automatically.
    First call POST /pqc/sig/keypair then POST /pqc/sig/sign.
    """
    use_smart_default_sig = _is_placeholder_value(request.signature)
    use_smart_default_key = _is_placeholder_value(request.public_key)
    
    # Smart Default for signature
    if use_smart_default_sig:
        sig_data = _get_latest_signature()
        if sig_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No signatures available. Sign a message first using POST /pqc/sig/sign"
            )
        sig_info, sig_id = sig_data
        signature = sig_info['signature']
        message_to_verify = sig_info['message']
        logger.info(f"Smart Default: Using stored signature {sig_id}")
    else:
        # Validate signature
        if not request.signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="signature is required"
            )
        
        # Validate hex format for signature
        if not all(c in '0123456789abcdefABCDEF' for c in request.signature):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"signature contains invalid characters. Expected hex-encoded string"
            )
        
        try:
            signature = bytes.fromhex(request.signature)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid hex-encoded signature: {str(e)}"
            )
        message_to_verify = request.message
    
    # Smart Default for public_key
    if use_smart_default_key:
        keypair_data = _get_latest_sig_keypair()
        if keypair_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No keypairs available. Generate a keypair first using POST /pqc/sig/keypair"
            )
        keypair, kp_id = keypair_data
        public_key = keypair['public_key']
        logger.info(f"Smart Default: Using stored public_key from keypair {kp_id}")
    else:
        if not request.public_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="public_key is required. Generate a keypair first using POST /pqc/sig/keypair"
            )
        
        # Validate hex format for public_key
        if not all(c in '0123456789abcdefABCDEF' for c in request.public_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"public_key contains invalid characters. Expected hex-encoded string"
            )
        
        try:
            public_key = bytes.fromhex(request.public_key)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid hex-encoded public_key: {str(e)}"
            )
    
    # Validate sizes (ML-DSA-65 FIPS 204: public_key=1952, signature=3309)
    # Note: NIST FIPS 204 specifies signature size of 3309 bytes for ML-DSA-65
    expected_pk_size = 1952
    min_sig_size = 3293  # Minimum expected (historical reference)
    max_sig_size = 3320  # Maximum with padding allowance
    
    # Skip size validation for Smart Default (keys from store are already valid)
    if not use_smart_default_key and len(public_key) != expected_pk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid public_key size: {len(public_key)} bytes. Expected {expected_pk_size} bytes for ML-DSA-65"
        )
    
    # Signature size validation with range (ML-DSA uses Fiat-Shamir with Aborts)
    if not use_smart_default_sig and not (min_sig_size <= len(signature) <= max_sig_size):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid signature size: {len(signature)} bytes. Expected {min_sig_size}-{max_sig_size} bytes for ML-DSA-65"
        )
    
    message = message_to_verify.encode('utf-8')
    
    try:
        pqc = _get_pqc_instance()
        is_valid = pqc.verify(message, signature, public_key)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PQC not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification operation failed: {str(e)}"
        )
    
    return VerifyResponse(
        valid=is_valid
    )


# =============================================================================
# FHE ENDPOINTS
# =============================================================================

# Example ciphertext IDs used in Swagger UI (for Smart Default feature)
_EXAMPLE_CIPHERTEXT_IDS = {
    "c40e1e56fda432fa80efdf0486959354",  # Primary example
    "a1b2c3d4e5f6789012345678abcdef01",  # other_ciphertext_id example
    "result_abc123def456789012345678",    # result example
    "string",                              # Default Swagger placeholder
}


def _store_ciphertext(ciphertext, data_length: int, operation: str = None) -> str:
    """Store ciphertext with metadata and return ID."""
    ct_id = secrets.token_hex(16)
    _ciphertext_store[ct_id] = {
        'ciphertext': ciphertext,
        'created_at': datetime.now().isoformat(),
        'data_length': data_length,
        'operation': operation
    }
    return ct_id


def _get_ciphertext(ct_id: str, auto_select: bool = True):
    """
    Get ciphertext by ID with Smart Default feature.
    
    If the provided ID is an example/placeholder value from Swagger UI
    and auto_select=True, automatically use the most recent ciphertext.
    This allows users to test the API without manually copying IDs.
    
    Args:
        ct_id: Ciphertext ID or example placeholder
        auto_select: If True, auto-select latest ciphertext for example IDs
        
    Returns:
        The ciphertext object
        
    Raises:
        HTTPException: If ciphertext not found and no auto-select available
    """
    # Smart Default: If example ID is used, auto-select the latest ciphertext
    if auto_select and ct_id in _EXAMPLE_CIPHERTEXT_IDS:
        if not _ciphertext_store:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No ciphertexts available. Please encrypt data first using POST /fhe/encrypt"
            )
        # Get the most recently created ciphertext
        latest_id = max(_ciphertext_store.keys(), 
                        key=lambda k: _ciphertext_store[k].get('created_at', ''))
        logger.info(f"Smart Default: Auto-selected ciphertext '{latest_id}' (example ID '{ct_id}' was provided)")
        return _ciphertext_store[latest_id]['ciphertext'], latest_id
    
    # Normal lookup
    if ct_id not in _ciphertext_store:
        available = list(_ciphertext_store.keys())[:5]
        hint = f" Available: {available}" if available else " No ciphertexts stored."
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ciphertext '{ct_id}' not found.{hint} Use /fhe/ciphertexts to list all."
        )
    return _ciphertext_store[ct_id]['ciphertext'], ct_id


def _get_ciphertext_simple(ct_id: str):
    """Get ciphertext by ID (simple version, no auto-select)."""
    result, _ = _get_ciphertext(ct_id, auto_select=True)
    return result


@app.post("/fhe/encrypt", response_model=FHEEncryptResponse, tags=["FHE"])
async def fhe_encrypt(request: FHEEncryptRequest):
    """
    Encrypt data using CKKS homomorphic encryption.
    
    **Workflow:**
    1. Call this endpoint to encrypt your data
    2. Copy the `ciphertext_id` from the response
    3. Use that ID in `/fhe/add`, `/fhe/multiply`, `/fhe/square`, or `/fhe/decrypt`
    
    **Example:**
    ```json
    {"data": [1.0, 2.0, 3.0, 4.0, 5.0]}
    ```
    """
    fhe = _get_fhe_engine()
    if fhe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FHE engine not available. Check server logs."
        )
    
    try:
        ciphertext = fhe.encrypt(request.data)
        ct_id = _store_ciphertext(ciphertext, len(request.data), "encrypt")
        
        return FHEEncryptResponse(
            ciphertext_id=ct_id,
            slot_count=fhe.engine.slot_count,
            data_length=len(request.data),
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encryption failed: {str(e)}"
        )


@app.post("/fhe/decrypt", response_model=FHEDecryptResponse, tags=["FHE"])
async def fhe_decrypt(request: FHEDecryptRequest):
    """
    Decrypt ciphertext.
    
    **Smart Default:** If you use the example ID, the latest ciphertext will be auto-selected.
    
    **To find available ciphertext IDs:** Call `GET /fhe/ciphertexts`
    
    **Example:**
    ```json
    {"ciphertext_id": "c40e1e56fda432fa80efdf0486959354"}
    ```
    """
    fhe = _get_fhe_engine()
    if fhe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FHE engine not available"
        )
    
    ciphertext, actual_id = _get_ciphertext(request.ciphertext_id)
    
    try:
        length = request.length
        if length is None:
            # Try to get original data length from metadata
            length = _ciphertext_store[actual_id].get('data_length')
        
        result = fhe.decrypt(ciphertext, length)
        
        return FHEDecryptResponse(
            data=result.tolist()
        )
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decryption failed: {str(e)}"
        )


@app.post("/fhe/add", response_model=FHEOperationResponse, tags=["FHE"])
async def fhe_add(request: FHEOperationRequest):
    """
    Add scalar to ciphertext or add two ciphertexts.
    
    **Smart Default:** If you use example IDs, the latest ciphertext will be auto-selected.
    
    **Two modes:**
    1. **Add scalar:** Provide `ciphertext_id` + `scalar`
    2. **Add ciphertexts:** Provide `ciphertext_id` + `other_ciphertext_id`
    
    **Example (add scalar):**
    ```json
    {"ciphertext_id": "c40e1e56fda432fa80efdf0486959354", "scalar": 10.0}
    ```
    """
    fhe = _get_fhe_engine()
    if fhe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FHE engine not available"
        )
    
    ct, actual_id = _get_ciphertext(request.ciphertext_id)
    data_length = _ciphertext_store[actual_id].get('data_length', 0)
    
    try:
        if request.scalar is not None:
            result = fhe.add_scalar(ct, request.scalar)
            op = f"add_scalar({request.scalar})"
        elif request.other_ciphertext_id:
            ct2, _ = _get_ciphertext(request.other_ciphertext_id)
            result = fhe.add(ct, ct2)
            op = "add_ciphertexts"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either scalar or other_ciphertext_id must be provided"
            )
        
        result_id = _store_ciphertext(result, data_length, op)
        
        return FHEOperationResponse(
            result_ciphertext_id=result_id,
            operation=op,
            created_at=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Operation failed: {str(e)}"
        )


@app.post("/fhe/multiply", response_model=FHEOperationResponse, tags=["FHE"])
async def fhe_multiply(request: FHEOperationRequest):
    """
    Multiply ciphertext by scalar or multiply two ciphertexts.
    
    **Smart Default:** If you use example IDs, the latest ciphertext will be auto-selected.
    
    **Two modes:**
    1. **Multiply by scalar:** Provide `ciphertext_id` + `scalar`
    2. **Multiply ciphertexts:** Provide `ciphertext_id` + `other_ciphertext_id`
    
    **Example (multiply by scalar):**
    ```json
    {"ciphertext_id": "c40e1e56fda432fa80efdf0486959354", "scalar": 2.0}
    ```
    """
    fhe = _get_fhe_engine()
    if fhe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FHE engine not available"
        )
    
    ct, actual_id = _get_ciphertext(request.ciphertext_id)
    data_length = _ciphertext_store[actual_id].get('data_length', 0)
    
    try:
        if request.scalar is not None:
            result = fhe.multiply_scalar(ct, request.scalar)
            op = f"multiply_scalar({request.scalar})"
        elif request.other_ciphertext_id:
            ct2, _ = _get_ciphertext(request.other_ciphertext_id)
            result = fhe.multiply(ct, ct2)
            op = "multiply_ciphertexts"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either scalar or other_ciphertext_id must be provided"
            )
        
        result_id = _store_ciphertext(result, data_length, op)
        
        return FHEOperationResponse(
            result_ciphertext_id=result_id,
            operation=op,
            created_at=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multiply operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Operation failed: {str(e)}"
        )


@app.post("/fhe/square", response_model=FHEOperationResponse, tags=["FHE"])
async def fhe_square(request: FHEOperationRequest):
    """
    Square a ciphertext (x^2).
    
    **Smart Default:** If you use the example ID, the latest ciphertext will be auto-selected.
    
    **Note:** For this operation, `scalar` and `other_ciphertext_id` are ignored.
    
    **Example:**
    ```json
    {"ciphertext_id": "c40e1e56fda432fa80efdf0486959354"}
    ```
    """
    fhe = _get_fhe_engine()
    if fhe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FHE engine not available"
        )
    
    ct, actual_id = _get_ciphertext(request.ciphertext_id)
    data_length = _ciphertext_store[actual_id].get('data_length', 0)
    
    try:
        result = fhe.square(ct)
        result_id = _store_ciphertext(result, data_length, "square")
        
        return FHEOperationResponse(
            result_ciphertext_id=result_id,
            operation="square",
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Square operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Operation failed: {str(e)}"
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the server using uvicorn."""
    import uvicorn
    
    print("\n" + "=" * 60)
    print("  PQC-FHE Integration API Server v2.1.2")
    print("=" * 60)
    print("\nStarting server...")
    print("API documentation: http://127.0.0.1:8000/docs")
    print("Health check: http://127.0.0.1:8000/health")
    print("List ciphertexts: http://127.0.0.1:8000/fhe/ciphertexts")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(
        "api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
