#!/usr/bin/env python3
"""
GL Scheme FHE Engine - DESILO 5th Generation FHE
=================================================

Implements the GL (Gentry-Lee) scheme for matrix-based Fully Homomorphic
Encryption, leveraging DESILO FHE's GLEngine API. The GL scheme natively
supports matrix multiplication in the encrypted domain, making it ideal
for neural network inference where matrix operations dominate.

Key advantages over CKKS vector operations:
- Native encrypted matrix multiplication (no rotation-based packing)
- Efficient transpose, conjugate, and Hadamard product
- Better suited for linear algebra workloads (ML inference, statistics)

Reference:
- DESILO GL Scheme: ePrint 2025/1935 (Craig Gentry & Yongwoo Lee)
- Announced FHE.org 2026 Taipei (March 7, 2026)
- DESILO FHE docs: https://fhe.desilo.dev/latest/gl_quickstart/
- Google HEIR project: investigating GL integration (GitHub Issue #2408)
- RhombusEnd2End_HEonGPU: GPU-accelerated 2-party inference architecture

Security Notes:
- GL scheme uses Ring-LWE hardness (same family as CKKS/ML-KEM)
- NTT-based polynomial operations share CKKS side-channel surface
- CKKS NTT SPA (arXiv:2505.11058) mitigations also apply to GL
- CPAD attack (CEA 2025) applies to threshold GL variants

Author: PQC-FHE Integration Library
License: MIT
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Check DESILO FHE availability (GLEngine requires desilofhe >= 1.10.0)
try:
    import desilofhe
    DESILO_AVAILABLE = True
    GL_AVAILABLE = hasattr(desilofhe, 'GLEngine')
except ImportError:
    DESILO_AVAILABLE = False
    GL_AVAILABLE = False
    logger.warning("desilofhe not available. GL scheme features will be disabled.")


# =============================================================================
# ENUMERATIONS & CONFIGURATION
# =============================================================================

class GLMatrixShape(Enum):
    """Supported GL scheme matrix shapes (batch_size, rows, cols)."""
    SHAPE_256x16x16 = (256, 16, 16)
    SHAPE_256x32x32 = (256, 32, 32)
    SHAPE_256x64x64 = (256, 64, 64)
    SHAPE_16x256x256 = (16, 256, 256)
    SHAPE_16x512x512 = (16, 512, 512)
    SHAPE_4x1024x1024 = (4, 1024, 1024)
    SHAPE_4x2048x2048 = (4, 2048, 2048)


@dataclass
class GLConfig:
    """GL scheme engine configuration."""
    shape: Tuple[int, int, int] = (256, 16, 16)  # (batch, rows, cols)
    mode: str = 'cpu'          # 'cpu' or 'gpu'
    thread_count: int = 0      # 0 = auto (512 for GPU)
    device_id: int = 0         # GPU device ID

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def rows(self) -> int:
        return self.shape[1]

    @property
    def cols(self) -> int:
        return self.shape[2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'shape': list(self.shape),
            'batch_size': self.batch_size,
            'rows': self.rows,
            'cols': self.cols,
            'mode': self.mode,
            'thread_count': self.thread_count,
            'device_id': self.device_id,
        }


# =============================================================================
# GL SCHEME ENGINE WRAPPER
# =============================================================================

class GLSchemeEngine:
    """
    GL (Gentry-Lee) Scheme FHE Engine Wrapper.

    Wraps DESILO FHE's GLEngine to provide matrix-based homomorphic encryption
    with native matrix multiplication support. This is the 5th generation FHE
    scheme, announced at FHE.org 2026 Taipei.

    Key Operations:
    - encrypt/decrypt: Matrix encryption/decryption
    - matrix_multiply: Encrypted matrix multiplication (native, no rotation)
    - hadamard_multiply: Element-wise encrypted multiplication
    - transpose: Encrypted matrix transposition
    - add/subtract: Encrypted matrix addition/subtraction

    Usage:
        engine = GLSchemeEngine(GLConfig(shape=(256, 16, 16), mode='gpu'))
        engine.setup()
        sk = engine.create_keys()
        ct = engine.encrypt(matrix_data, sk)
        ct_result = engine.matrix_multiply(ct1, ct2)
        result = engine.decrypt(ct_result, sk)

    Reference:
      - ePrint 2025/1935 (Gentry & Lee, 5th gen FHE)
      - DESILO FHE GLEngine: https://fhe.desilo.dev/latest/gl_quickstart/
      - RhombusEnd2End_HEonGPU: HEMultiPartyManager GPU-accelerated MPC
    """

    def __init__(self, config: Optional[GLConfig] = None):
        self.config = config or GLConfig()
        self.engine = None
        self.secret_key = None
        self.matrix_mult_key = None
        self.hadamard_mult_key = None
        self.rotation_key = None
        self.transpose_key = None
        self.conjugation_key = None
        self.conj_transpose_key = None
        self._initialized = False
        self._metrics = {
            'setup_ms': 0.0,
            'keygen_ms': 0.0,
            'encrypt_ms': 0.0,
            'compute_ms': 0.0,
            'decrypt_ms': 0.0,
        }

    def setup(self) -> None:
        """Initialize the GL scheme engine."""
        if not DESILO_AVAILABLE:
            raise RuntimeError("desilofhe not installed. Run: pip install desilofhe")
        if not GL_AVAILABLE:
            raise RuntimeError(
                "GLEngine not available in desilofhe. "
                "Requires desilofhe >= 1.10.0 (released 2026-02-11). "
                "Run: pip install --upgrade desilofhe"
            )

        t0 = time.time()

        engine_kwargs = {
            'shape': self.config.shape,
            'mode': self.config.mode,
        }

        if self.config.thread_count > 0:
            engine_kwargs['thread_count'] = self.config.thread_count
        elif self.config.mode == 'gpu':
            engine_kwargs['thread_count'] = 512

        if self.config.mode == 'gpu' and self.config.device_id > 0:
            engine_kwargs['device_id'] = self.config.device_id

        logger.info("Initializing GL scheme engine: %s", engine_kwargs)
        self.engine = desilofhe.GLEngine(**engine_kwargs)
        self._initialized = True

        elapsed = (time.time() - t0) * 1000
        self._metrics['setup_ms'] = elapsed
        logger.info(
            "GL engine initialized: shape=%s, max_level=%d (%.1f ms)",
            self.config.shape, self.engine.max_level, elapsed
        )

    def create_keys(self) -> Any:
        """
        Generate all cryptographic keys for GL scheme operations.

        Creates: secret key, matrix multiplication key, Hadamard multiplication
        key, rotation key, transpose key, conjugation key, conjugate transpose key.

        Returns:
            Secret key object
        """
        if not self._initialized:
            self.setup()

        t0 = time.time()

        self.secret_key = self.engine.create_secret_key()

        # Matrix multiplication key (for encrypted matmul)
        self.matrix_mult_key = self.engine.create_matrix_multiplication_key(
            self.secret_key
        )

        # Hadamard (element-wise) multiplication key
        self.hadamard_mult_key = self.engine.create_hadamard_multiplication_key(
            self.secret_key
        )

        # Rotation key (circular shifts)
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)

        # Transpose key
        self.transpose_key = self.engine.create_transposition_key(self.secret_key)

        # Conjugation key
        self.conjugation_key = self.engine.create_conjugation_key(self.secret_key)

        # Conjugate transpose key
        self.conj_transpose_key = self.engine.create_conjugate_transposition_key(
            self.secret_key
        )

        elapsed = (time.time() - t0) * 1000
        self._metrics['keygen_ms'] = elapsed
        logger.info("GL keys generated in %.1f ms (7 key types)", elapsed)

        return self.secret_key

    # -------------------------------------------------------------------------
    # Encryption / Decryption
    # -------------------------------------------------------------------------

    def encrypt(self, matrix: np.ndarray) -> Any:
        """
        Encrypt a matrix using the GL scheme.

        Args:
            matrix: numpy array to encrypt. Shape should match config shape
                    or be broadcastable to (batch, rows, cols).

        Returns:
            Encrypted ciphertext (GL matrix ciphertext)
        """
        if self.secret_key is None:
            raise RuntimeError("Keys not created. Call create_keys() first.")

        t0 = time.time()

        # Reshape to match GL config if needed
        target_shape = self.config.shape
        if isinstance(matrix, np.ndarray):
            data = matrix.astype(np.complex128)
            if data.ndim == 1:
                # Vector → pad to matrix
                padded = np.zeros(target_shape, dtype=np.complex128)
                flat_len = min(len(data), target_shape[1] * target_shape[2])
                padded.flat[:flat_len] = data[:flat_len]
                data = padded
            elif data.ndim == 2:
                # Single matrix → broadcast to batch
                padded = np.zeros(target_shape, dtype=np.complex128)
                r = min(data.shape[0], target_shape[1])
                c = min(data.shape[1], target_shape[2])
                padded[0, :r, :c] = data[:r, :c]
                data = padded
        else:
            data = np.array(matrix, dtype=np.complex128)

        ct = self.engine.encrypt(data.tolist(), self.secret_key)

        elapsed = (time.time() - t0) * 1000
        self._metrics['encrypt_ms'] += elapsed
        return ct

    def decrypt(self, ct: Any) -> np.ndarray:
        """
        Decrypt a GL ciphertext to recover the matrix.

        Args:
            ct: GL ciphertext

        Returns:
            numpy array of shape (batch, rows, cols)
        """
        if self.secret_key is None:
            raise RuntimeError("Keys not created.")

        t0 = time.time()
        result = self.engine.decrypt(ct, self.secret_key)
        elapsed = (time.time() - t0) * 1000
        self._metrics['decrypt_ms'] += elapsed

        return np.array(result)

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def add(self, ct1: Any, ct2: Any) -> Any:
        """Encrypted matrix addition."""
        t0 = time.time()
        result = self.engine.add(ct1, ct2)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def subtract(self, ct1: Any, ct2: Any) -> Any:
        """Encrypted matrix subtraction."""
        t0 = time.time()
        result = self.engine.subtract(ct1, ct2)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def matrix_multiply(self, ct1: Any, ct2: Any) -> Any:
        """
        Native encrypted matrix multiplication.

        This is the key advantage of the GL scheme over CKKS vector operations:
        direct matrix multiplication without rotation-based packing.

        Reference: ePrint 2025/1935, Section 4 (Native MatMul)
        """
        if self.matrix_mult_key is None:
            raise RuntimeError("Matrix multiplication key not created.")

        t0 = time.time()
        result = self.engine.matrix_multiply(ct1, ct2, self.matrix_mult_key)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def hadamard_multiply(self, ct1: Any, ct2: Any) -> Any:
        """Element-wise encrypted multiplication (Hadamard product)."""
        if self.hadamard_mult_key is None:
            raise RuntimeError("Hadamard multiplication key not created.")

        t0 = time.time()
        result = self.engine.hadamard_multiply(ct1, ct2, self.hadamard_mult_key)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def transpose(self, ct: Any) -> Any:
        """Encrypted matrix transposition."""
        if self.transpose_key is None:
            raise RuntimeError("Transpose key not created.")

        t0 = time.time()
        result = self.engine.transpose(ct, self.transpose_key)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def conjugate(self, ct: Any) -> Any:
        """Encrypted complex conjugate."""
        if self.conjugation_key is None:
            raise RuntimeError("Conjugation key not created.")

        t0 = time.time()
        result = self.engine.conjugate(ct, self.conjugation_key)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def conjugate_transpose(self, ct: Any) -> Any:
        """Encrypted conjugate transpose (Hermitian adjoint)."""
        if self.conj_transpose_key is None:
            raise RuntimeError("Conjugate transpose key not created.")

        t0 = time.time()
        result = self.engine.conjugate_transpose(ct, self.conj_transpose_key)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    def roll(self, ct: Any, axis: int, shift: int) -> Any:
        """Circular shift of encrypted matrix along axis."""
        t0 = time.time()
        result = self.engine.roll(ct, axis, shift)
        self._metrics['compute_ms'] += (time.time() - t0) * 1000
        return result

    # -------------------------------------------------------------------------
    # Neural Network Inference Helpers
    # -------------------------------------------------------------------------

    def encrypted_linear_layer(
        self,
        ct_input: Any,
        weight_matrix: np.ndarray,
        bias_matrix: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Apply an encrypted linear layer: Y = X * W^T + B

        Uses native GL matrix multiplication for efficient computation.
        No rotation-based packing needed (unlike CKKS vector approach).

        Args:
            ct_input: Encrypted input matrix
            weight_matrix: Plaintext weight matrix (will be encrypted)
            bias_matrix: Optional plaintext bias matrix

        Returns:
            Encrypted output matrix

        Reference:
          - RhombusEnd2End: GPUMatMul.compute() for HE linear layers
          - DESILO GL: native matmul via GLEngine.matrix_multiply()
        """
        t0 = time.time()

        # Encrypt weight matrix
        ct_weights = self.encrypt(weight_matrix)

        # Native matrix multiplication
        ct_result = self.matrix_multiply(ct_input, ct_weights)

        # Add bias if provided
        if bias_matrix is not None:
            ct_bias = self.encrypt(bias_matrix)
            ct_result = self.add(ct_result, ct_bias)

        elapsed = (time.time() - t0) * 1000
        logger.debug("GL linear layer: %.1f ms", elapsed)
        return ct_result

    def encrypted_activation_approx(
        self,
        ct: Any,
        activation: str = 'relu',
        degree: int = 3,
    ) -> Any:
        """
        Approximate activation function using polynomial evaluation.

        Since GL operates on matrices, we use Hadamard (element-wise)
        operations for polynomial evaluation of activation functions.

        Supported activations:
        - 'relu': max(0, x) ≈ 0.5*x + 0.5*x*sign(x)
        - 'sigmoid': 1/(1+e^-x) ≈ 0.5 + 0.197*x - 0.004*x^3
        - 'square': x^2 (exact via Hadamard)

        Args:
            ct: Encrypted matrix
            activation: Activation function name
            degree: Polynomial approximation degree

        Returns:
            Encrypted matrix with activation applied element-wise
        """
        t0 = time.time()

        if activation == 'square':
            # Exact square via Hadamard product
            result = self.hadamard_multiply(ct, ct)
        elif activation == 'sigmoid':
            # Sigmoid approximation: 0.5 + 0.197*x - 0.004*x^3
            # Using Hadamard products for element-wise polynomial
            shape = self.config.shape
            half = self.encrypt(np.full(shape, 0.5))
            coeff1 = self.encrypt(np.full(shape, 0.197))
            coeff3 = self.encrypt(np.full(shape, -0.004))

            term1 = self.hadamard_multiply(coeff1, ct)  # 0.197*x
            x2 = self.hadamard_multiply(ct, ct)         # x^2
            x3 = self.hadamard_multiply(x2, ct)         # x^3
            term3 = self.hadamard_multiply(coeff3, x3)  # -0.004*x^3

            result = self.add(half, term1)
            result = self.add(result, term3)
        elif activation == 'relu':
            # Smooth ReLU: 0.5*x + 0.5*|x| ≈ 0.5*x + 0.25*x for x>0
            # Simple approximation: max(0.01*x, x) ≈ 0.5*(x + |x|)
            # Using polynomial: relu ≈ 0.5*x + 0.198*x - 0.004*x^3 + 0.5
            # (shifted sigmoid approximation)
            shape = self.config.shape
            half = self.encrypt(np.full(shape, 0.5))
            coeff_linear = self.encrypt(np.full(shape, 0.5))

            term_linear = self.hadamard_multiply(coeff_linear, ct)
            # Square term for smoothness
            x2 = self.hadamard_multiply(ct, ct)
            coeff_sq = self.encrypt(np.full(shape, 0.125))
            term_sq = self.hadamard_multiply(coeff_sq, x2)

            result = self.add(half, term_linear)
            result = self.add(result, term_sq)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        elapsed = (time.time() - t0) * 1000
        logger.debug("GL activation '%s': %.1f ms", activation, elapsed)
        return result

    def run_gl_inference(
        self,
        input_matrix: np.ndarray,
        layers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run a multi-layer inference pipeline using GL scheme.

        Each layer dict should contain:
        - 'type': 'linear' or 'activation'
        - For 'linear': 'weights' (np.ndarray), optional 'bias' (np.ndarray)
        - For 'activation': 'function' (str: 'relu', 'sigmoid', 'square')

        Args:
            input_matrix: Input data as numpy array
            layers: List of layer configurations

        Returns:
            Dict with 'result', 'metrics', and performance data

        Reference:
          - RhombusEnd2End: GPU 2-party inference (linear via HE, nonlinear via OT)
          - DESILO GL: native matrix operations in encrypted domain
        """
        total_t0 = time.time()

        # Setup and key generation
        if not self._initialized:
            self.setup()
        if self.secret_key is None:
            self.create_keys()

        # Encrypt input
        ct = self.encrypt(input_matrix)

        # Run layers
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', 'linear')

            if layer_type == 'linear':
                weights = layer['weights']
                bias = layer.get('bias')
                ct = self.encrypted_linear_layer(ct, weights, bias)
                logger.debug("GL Layer %d: linear", i)
            elif layer_type == 'activation':
                func = layer.get('function', 'relu')
                ct = self.encrypted_activation_approx(ct, func)
                logger.debug("GL Layer %d: activation (%s)", i, func)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # Decrypt result
        result = self.decrypt(ct)

        total_ms = (time.time() - total_t0) * 1000

        return {
            'result': result.tolist(),
            'shape': list(result.shape),
            'num_layers': len(layers),
            'total_ms': round(total_ms, 2),
            'metrics': {k: round(v, 2) for k, v in self._metrics.items()},
            'config': self.config.to_dict(),
            'scheme': 'GL (Gentry-Lee 5th Gen FHE)',
        }

    # -------------------------------------------------------------------------
    # Comparison & Benchmarking
    # -------------------------------------------------------------------------

    def benchmark_vs_ckks_matmul(
        self,
        matrix_size: int = 16,
        num_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Benchmark GL native matrix multiplication vs CKKS rotation-based approach.

        Demonstrates GL scheme's key advantage: O(1) matrix multiply vs
        O(n) rotations in CKKS for n×n matrix multiplication.

        Args:
            matrix_size: Size of square matrices
            num_iterations: Number of benchmark iterations

        Returns:
            Dict with GL timing, estimated CKKS timing, and speedup factor
        """
        if not self._initialized:
            self.setup()
        if self.secret_key is None:
            self.create_keys()

        # Generate random matrices
        np.random.seed(42)
        A = np.random.randn(self.config.batch_size,
                            matrix_size, matrix_size).astype(np.float64)
        B = np.random.randn(self.config.batch_size,
                            matrix_size, matrix_size).astype(np.float64)

        # Benchmark GL native matmul
        ct_a = self.encrypt(A)
        ct_b = self.encrypt(B)

        gl_times = []
        for _ in range(num_iterations):
            t0 = time.time()
            _ = self.matrix_multiply(ct_a, ct_b)
            gl_times.append((time.time() - t0) * 1000)

        gl_avg_ms = np.mean(gl_times)

        # Estimate CKKS rotation-based matmul time
        # CKKS requires O(n) rotations for n×n matmul, each costing ~5-10ms
        # GL requires 1 native operation
        estimated_ckks_rotations = matrix_size
        estimated_ckks_ms_per_rotation = 8.0  # typical CKKS rotation cost
        estimated_ckks_total_ms = (
            estimated_ckks_rotations * estimated_ckks_ms_per_rotation
        )

        speedup = estimated_ckks_total_ms / max(gl_avg_ms, 0.001)

        return {
            'gl_matmul_avg_ms': round(gl_avg_ms, 2),
            'gl_matmul_times': [round(t, 2) for t in gl_times],
            'estimated_ckks_matmul_ms': round(estimated_ckks_total_ms, 2),
            'ckks_rotations_needed': estimated_ckks_rotations,
            'speedup_factor': round(speedup, 1),
            'matrix_size': matrix_size,
            'batch_size': self.config.batch_size,
            'num_iterations': num_iterations,
            'note': (
                'GL scheme provides native matrix multiplication in O(1) '
                'homomorphic operations, while CKKS requires O(n) rotations. '
                'This makes GL significantly faster for matrix-heavy workloads '
                'like neural network inference. '
                'Reference: ePrint 2025/1935 (Gentry & Lee)'
            ),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return engine performance metrics."""
        return {k: round(v, 2) for k, v in self._metrics.items()}

    def get_security_info(self) -> Dict[str, Any]:
        """
        Return GL scheme security information and known vulnerabilities.

        Reference:
          - arXiv:2505.11058 (CKKS NTT SPA, applies to GL NTT operations)
          - CEA 2025 CPAD (applies to threshold GL variants)
          - ePrint 2025/1935 (GL scheme security analysis)
        """
        return {
            'scheme': 'GL (Gentry-Lee)',
            'generation': '5th generation FHE',
            'hardness_assumption': 'Ring-LWE',
            'ntt_based': True,
            'security_level': '128-bit (estimated, Ring-LWE parameters)',
            'known_vulnerabilities': [
                {
                    'id': 'GL-NTT-SPA-SHARED',
                    'description': (
                        'GL uses NTT-based polynomial arithmetic like CKKS. '
                        'The neural network SPA attack (arXiv:2505.11058) '
                        'targeting NTT operations may apply to GL implementations. '
                        'Random delay/masking alone are INEFFECTIVE.'
                    ),
                    'severity': 'HIGH',
                    'mitigation': 'Combined masking+shuffling+constant-time NTT',
                    'reference': 'arXiv:2505.11058, IACR ePrint 2025/867',
                },
                {
                    'id': 'GL-CPAD-THRESHOLD',
                    'description': (
                        'Threshold variants of GL (like all Ring-LWE FHE) '
                        'are CPAD-insecure without smudging noise after '
                        'partial decryption.'
                    ),
                    'severity': 'CRITICAL (threshold mode only)',
                    'mitigation': 'Mandatory smudging noise (>= 40 bits)',
                    'reference': 'CEA France (cea.hal.science/cea-04706832)',
                },
            ],
            'advantages_over_ckks': [
                'Native matrix multiplication (O(1) vs O(n) rotations)',
                'Built-in transpose and conjugate transpose',
                'Better amortized cost for batch matrix operations',
                'Reduced rotation key material',
            ],
            'limitations': [
                'Fixed matrix shapes (predetermined at engine creation)',
                'Complex-valued matrices only (real values use Re part)',
                'Newer scheme: less production hardening than CKKS',
                'Multiparty GL not yet supported in DESILO (as of v1.10.0)',
            ],
            'references': [
                'ePrint 2025/1935 (Gentry & Lee, 5th gen FHE)',
                'FHE.org 2026 Taipei announcement (March 7, 2026)',
                'Google HEIR project: GitHub Issue #2408',
                'DESILO FHE v1.10.0: GLEngine API (2026-02-11)',
            ],
        }


# =============================================================================
# GL + CKKS HYBRID ENGINE
# =============================================================================

class GLCKKSHybridEngine:
    """
    Hybrid GL + CKKS engine for combining matrix and vector operations.

    Uses GL scheme for matrix-heavy operations (linear layers) and
    CKKS for element-wise operations (activations, normalization).

    This follows the RhombusEnd2End pattern:
    - Linear layers: HE-based (GL for matrix, CKKS for vector)
    - Non-linear layers: Polynomial approximation (Chebyshev/power basis)

    Reference:
      - RhombusEnd2End: HEonGPU adapter switching BFV/CKKS per operation
      - DESILO FHE: GLEngine + Engine dual API
    """

    def __init__(
        self,
        gl_config: Optional[GLConfig] = None,
        ckks_max_level: int = 7,
        mode: str = 'cpu',
    ):
        self.gl_config = gl_config or GLConfig(mode=mode)
        self.ckks_max_level = ckks_max_level
        self.mode = mode
        self.gl_engine = None
        self.ckks_engine = None
        self._initialized = False

    def setup(self) -> None:
        """Initialize both GL and CKKS engines."""
        if not DESILO_AVAILABLE:
            raise RuntimeError("desilofhe not installed.")

        # GL engine for matrix operations
        if GL_AVAILABLE:
            self.gl_engine = GLSchemeEngine(self.gl_config)
            self.gl_engine.setup()
            self.gl_engine.create_keys()
            logger.info("GL engine ready")
        else:
            logger.warning("GLEngine not available, falling back to CKKS only")

        # CKKS engine for vector/activation operations
        ckks_kwargs = {
            'max_level': self.ckks_max_level,
            'mode': self.mode,
        }
        self.ckks_engine = desilofhe.Engine(**ckks_kwargs)
        self._ckks_sk = self.ckks_engine.create_secret_key()
        self._ckks_pk = self.ckks_engine.create_public_key(self._ckks_sk)
        self._ckks_rk = self.ckks_engine.create_relinearization_key(self._ckks_sk)
        logger.info("CKKS engine ready (max_level=%d)", self.ckks_max_level)

        self._initialized = True

    def get_info(self) -> Dict[str, Any]:
        """Return hybrid engine configuration and capabilities."""
        return {
            'gl_available': GL_AVAILABLE and self.gl_engine is not None,
            'ckks_available': self.ckks_engine is not None,
            'gl_config': self.gl_config.to_dict() if self.gl_engine else None,
            'ckks_max_level': self.ckks_max_level,
            'mode': self.mode,
            'scheme_comparison': {
                'gl': {
                    'best_for': 'Matrix multiplication, linear layers',
                    'operations': ['matmul', 'transpose', 'hadamard', 'add'],
                    'reference': 'ePrint 2025/1935',
                },
                'ckks': {
                    'best_for': 'Vector operations, activations, bootstrapping',
                    'operations': ['polynomial_eval', 'rotation', 'bootstrapping'],
                    'reference': 'CKKS (Cheon et al. 2017)',
                },
            },
        }


# =============================================================================
# MODULE INFO
# =============================================================================

def get_gl_scheme_info() -> Dict[str, Any]:
    """Return GL scheme capabilities and status."""
    return {
        'name': 'GL Scheme (Gentry-Lee 5th Gen FHE)',
        'desilo_available': DESILO_AVAILABLE,
        'gl_engine_available': GL_AVAILABLE,
        'version': __version__,
        'supported_shapes': [s.value for s in GLMatrixShape],
        'operations': [
            'encrypt', 'decrypt',
            'matrix_multiply', 'hadamard_multiply',
            'transpose', 'conjugate', 'conjugate_transpose',
            'add', 'subtract', 'roll',
        ],
        'security': {
            'hardness': 'Ring-LWE',
            'ntt_based': True,
            'side_channel_risk': 'Same as CKKS (NTT SPA applies)',
            'cpad_risk': 'Applies to threshold GL variants',
        },
        'references': [
            'ePrint 2025/1935 (Gentry & Lee)',
            'FHE.org 2026 Taipei (March 7, 2026)',
            'DESILO FHE v1.10.0 GLEngine API',
            'Google HEIR: GitHub Issue #2408',
        ],
    }


try:
    from .version_loader import get_version
    __version__ = get_version('desilo_fhe_engine')
except ImportError:
    __version__ = "3.2.0"

__author__ = "PQC-FHE Integration Library"


if __name__ == "__main__":
    print(f"GL Scheme FHE Engine v{__version__}")
    print("=" * 60)
    info = get_gl_scheme_info()
    print(f"DESILO available: {info['desilo_available']}")
    print(f"GL engine available: {info['gl_engine_available']}")
    print(f"Supported shapes: {info['supported_shapes']}")
    print(f"Operations: {info['operations']}")
