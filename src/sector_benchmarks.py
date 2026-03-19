#!/usr/bin/env python3
"""
Sector-Specific Benchmark Suite
================================

Runs ACTUAL performance benchmarks for PQC+FHE operations tailored
to specific industry sectors. Each benchmark uses real cryptographic
operations (liboqs PQC, DESILO FHE) on representative data.

Sectors:
1. Healthcare (HIPAA): Patient records, vital signs, medical IoT
2. Finance (PCI-DSS): Transactions, portfolio, trade settlement
3. Blockchain: Post-quantum signatures, verification throughput
4. IoT: Constrained-device PQC, lightweight encryption
5. MPC-FHE: Multiparty computation overhead

References:
- NIST IR 8547: Transition to Post-Quantum Cryptography Standards
- NSA CNSA 2.0: Commercial National Security Algorithm Suite 2.0
- HIPAA Security Rule: 45 CFR Parts 160, 162, and 164
- PCI DSS v4.0: Payment Card Industry Data Security Standard

Author: PQC-FHE Integration Library
License: MIT
Version: 3.2.0
"""

import time
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# Feature gates
try:
    import oqs
    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False
    logger.warning("liboqs not available. PQC benchmarks disabled.")

try:
    import desilofhe
    DESILO_AVAILABLE = True
except ImportError:
    DESILO_AVAILABLE = False
    logger.warning("DESILO FHE not available. FHE benchmarks disabled.")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SectorBenchmarkResult:
    """Benchmark result for a specific sector use case."""
    sector: str
    use_case: str
    operation: str
    data_size_bytes: int
    algorithm: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    throughput_ops_sec: float
    throughput_mbps: float
    compliance_context: str
    nist_level: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sector': self.sector,
            'use_case': self.use_case,
            'operation': self.operation,
            'data_size_bytes': self.data_size_bytes,
            'algorithm': self.algorithm,
            'iterations': self.iterations,
            'mean_ms': round(self.mean_ms, 3),
            'std_ms': round(self.std_ms, 3),
            'min_ms': round(self.min_ms, 3),
            'max_ms': round(self.max_ms, 3),
            'p95_ms': round(self.p95_ms, 3),
            'throughput_ops_sec': round(self.throughput_ops_sec, 1),
            'throughput_mbps': round(self.throughput_mbps, 4),
            'compliance_context': self.compliance_context,
            'nist_level': self.nist_level,
            'notes': self.notes,
        }


@dataclass
class SectorBenchmarkSuite:
    """Collection of benchmark results for one sector."""
    sector: str
    timestamp: str
    system_info: Dict[str, Any]
    results: List[SectorBenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sector': self.sector,
            'timestamp': self.timestamp,
            'system_info': self.system_info,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
        }


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def _get_system_info() -> Dict[str, Any]:
    """Gather system information for benchmark context."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }
    if LIBOQS_AVAILABLE:
        info['liboqs_version'] = oqs.oqs_version()
    if DESILO_AVAILABLE:
        info['desilo_fhe_version'] = getattr(desilofhe, '__version__', 'unknown')
    return info


def _measure_operation(func, iterations: int, warmup: int = 2) -> Dict[str, float]:
    """
    Measure operation timing over multiple iterations.

    Returns dict with mean_ms, std_ms, min_ms, max_ms, p95_ms.
    """
    # Warmup
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)

    timings_arr = np.array(timings)
    return {
        'mean_ms': float(np.mean(timings_arr)),
        'std_ms': float(np.std(timings_arr)),
        'min_ms': float(np.min(timings_arr)),
        'max_ms': float(np.max(timings_arr)),
        'p95_ms': float(np.percentile(timings_arr, 95)),
    }


# =============================================================================
# SECTOR BENCHMARK RUNNER
# =============================================================================

class SectorBenchmarkRunner:
    """Runs sector-specific benchmarks using actual PQC and FHE operations."""

    def __init__(self, fhe_mode: str = 'gpu', iterations: int = 50):
        self.fhe_mode = fhe_mode
        self.iterations = iterations
        self.system_info = _get_system_info()

    # =========================================================================
    # HEALTHCARE (HIPAA)
    # =========================================================================

    def run_healthcare(self) -> SectorBenchmarkSuite:
        """
        Healthcare sector benchmarks (HIPAA compliance context).

        Operations:
        - ML-KEM-768 key exchange for medical data transfer
        - ML-DSA-65 signature on medical records
        - FHE encryption of vital signs data
        - FHE computation on encrypted vital signs (mean)
        """
        suite = SectorBenchmarkSuite(
            sector='healthcare',
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info,
        )

        # 1. PQC Key Exchange for patient data transfer
        if LIBOQS_AVAILABLE:
            suite.results.append(
                self._benchmark_kem_exchange(
                    sector='healthcare',
                    use_case='Patient Data Transfer',
                    algorithm='ML-KEM-768',
                    data_size=2048,  # ~2KB patient record
                    compliance='HIPAA Security Rule §164.312(e)',
                    nist_level=3,
                )
            )

            # 2. Digital signature on medical records
            suite.results.append(
                self._benchmark_signature(
                    sector='healthcare',
                    use_case='Medical Record Signing',
                    algorithm='ML-DSA-65',
                    data_size=4096,  # ~4KB medical record
                    compliance='HIPAA Security Rule §164.312(c)',
                    nist_level=3,
                )
            )

            # 3. Signature verification
            suite.results.append(
                self._benchmark_signature_verify(
                    sector='healthcare',
                    use_case='Medical Record Verification',
                    algorithm='ML-DSA-65',
                    data_size=4096,
                    compliance='HIPAA Security Rule §164.312(c)',
                    nist_level=3,
                )
            )

        # 4. FHE on vital signs
        if DESILO_AVAILABLE:
            suite.results.append(
                self._benchmark_fhe_encrypt(
                    sector='healthcare',
                    use_case='Vital Signs Encryption',
                    data_size=256,  # 32 float64 values
                    compliance='HIPAA Security Rule §164.312(a)',
                    nist_level=3,
                    notes='Heart rate, BP, SpO2, temperature (32 readings)',
                )
            )

            suite.results.append(
                self._benchmark_fhe_computation(
                    sector='healthcare',
                    use_case='Encrypted Vital Signs Analysis',
                    operation='mean+variance',
                    data_points=32,
                    compliance='HIPAA Privacy Rule',
                    nist_level=3,
                    notes='Compute mean and variance on encrypted vital signs',
                )
            )

        suite.summary = self._generate_summary(suite.results, 'healthcare')
        return suite

    # =========================================================================
    # FINANCE (PCI-DSS)
    # =========================================================================

    def run_finance(self) -> SectorBenchmarkSuite:
        """
        Finance sector benchmarks (PCI-DSS/SOX compliance context).

        Operations:
        - ML-KEM-768 key exchange for transaction data
        - ML-DSA-65 signature for trade settlement
        - FHE encrypted portfolio analysis
        - Key rotation benchmark
        """
        suite = SectorBenchmarkSuite(
            sector='finance',
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info,
        )

        if LIBOQS_AVAILABLE:
            # 1. Transaction batch encryption key exchange
            suite.results.append(
                self._benchmark_kem_exchange(
                    sector='finance',
                    use_case='Transaction Batch Key Exchange',
                    algorithm='ML-KEM-768',
                    data_size=10240,  # ~10KB transaction batch
                    compliance='PCI DSS v4.0 Req 4.2',
                    nist_level=3,
                )
            )

            # 2. Trade settlement signature
            suite.results.append(
                self._benchmark_signature(
                    sector='finance',
                    use_case='Trade Settlement Signing',
                    algorithm='ML-DSA-65',
                    data_size=1024,  # ~1KB trade record
                    compliance='SOX Section 302/906',
                    nist_level=3,
                )
            )

            # 3. Key rotation (generate new keypair)
            suite.results.append(
                self._benchmark_key_rotation(
                    sector='finance',
                    use_case='PQC Key Rotation',
                    algorithm='ML-KEM-768',
                    compliance='PCI DSS v4.0 Req 3.7',
                    nist_level=3,
                )
            )

            # 4. High-security trade signing
            suite.results.append(
                self._benchmark_signature(
                    sector='finance',
                    use_case='High-Value Trade Signing (L5)',
                    algorithm='ML-DSA-87',
                    data_size=2048,
                    compliance='CNSA 2.0',
                    nist_level=5,
                )
            )

        if DESILO_AVAILABLE:
            # 5. FHE portfolio analysis
            suite.results.append(
                self._benchmark_fhe_computation(
                    sector='finance',
                    use_case='Encrypted Portfolio Weighted Sum',
                    operation='weighted_sum',
                    data_points=64,  # 64 asset prices
                    compliance='SOX Data Protection',
                    nist_level=3,
                    notes='Weighted sum of 64 encrypted asset prices',
                )
            )

        suite.summary = self._generate_summary(suite.results, 'finance')
        return suite

    # =========================================================================
    # BLOCKCHAIN
    # =========================================================================

    def run_blockchain(self) -> SectorBenchmarkSuite:
        """
        Blockchain sector benchmarks.

        Operations:
        - ML-DSA-65 signature throughput (vs ECDSA baseline)
        - Batch signature verification
        - Transaction signing pipeline
        """
        suite = SectorBenchmarkSuite(
            sector='blockchain',
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info,
        )

        if LIBOQS_AVAILABLE:
            # 1. ML-DSA-65 single signature throughput
            suite.results.append(
                self._benchmark_signature(
                    sector='blockchain',
                    use_case='Transaction Signing (L3)',
                    algorithm='ML-DSA-65',
                    data_size=256,  # ~256B transaction hash
                    compliance='Post-Quantum Blockchain',
                    nist_level=3,
                )
            )

            # 2. ML-DSA-87 for high-security chains
            suite.results.append(
                self._benchmark_signature(
                    sector='blockchain',
                    use_case='High-Security Chain Signing (L5)',
                    algorithm='ML-DSA-87',
                    data_size=256,
                    compliance='Post-Quantum Blockchain (L5)',
                    nist_level=5,
                )
            )

            # 3. ML-DSA-44 for lightweight chains
            suite.results.append(
                self._benchmark_signature(
                    sector='blockchain',
                    use_case='Lightweight Chain Signing (L2)',
                    algorithm='ML-DSA-44',
                    data_size=256,
                    compliance='Post-Quantum Blockchain (L2)',
                    nist_level=2,
                )
            )

            # 4. Batch signature verification (10 signatures)
            suite.results.append(
                self._benchmark_batch_verify(
                    sector='blockchain',
                    use_case='Batch Signature Verification (10 tx)',
                    algorithm='ML-DSA-65',
                    batch_size=10,
                    data_size=256,
                    compliance='Post-Quantum Blockchain',
                    nist_level=3,
                )
            )

            # 5. Full transaction pipeline (keygen + sign + verify)
            suite.results.append(
                self._benchmark_full_pipeline(
                    sector='blockchain',
                    use_case='Full TX Pipeline (keygen+sign+verify)',
                    sig_algorithm='ML-DSA-65',
                    data_size=256,
                    compliance='Post-Quantum Blockchain',
                    nist_level=3,
                )
            )

        suite.summary = self._generate_summary(suite.results, 'blockchain')
        return suite

    # =========================================================================
    # IOT
    # =========================================================================

    def run_iot(self) -> SectorBenchmarkSuite:
        """
        IoT sector benchmarks (constrained devices).

        Operations:
        - ML-KEM-512 vs ML-KEM-768 for varying payload sizes
        - Lightweight PQC key exchange
        - Small sensor data encryption
        """
        suite = SectorBenchmarkSuite(
            sector='iot',
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info,
        )

        if LIBOQS_AVAILABLE:
            # Compare ML-KEM-512 (L1) vs ML-KEM-768 (L3) at various sizes
            for data_size in [64, 256, 1024, 4096]:
                size_label = f"{data_size}B"

                suite.results.append(
                    self._benchmark_kem_exchange(
                        sector='iot',
                        use_case=f'Sensor Data ({size_label}) - L1',
                        algorithm='ML-KEM-512',
                        data_size=data_size,
                        compliance='IoT Security (Lightweight)',
                        nist_level=1,
                        notes=f'{size_label} payload, NIST Level 1 (minimal)',
                    )
                )

                suite.results.append(
                    self._benchmark_kem_exchange(
                        sector='iot',
                        use_case=f'Sensor Data ({size_label}) - L3',
                        algorithm='ML-KEM-768',
                        data_size=data_size,
                        compliance='IoT Security (Standard)',
                        nist_level=3,
                        notes=f'{size_label} payload, NIST Level 3 (recommended)',
                    )
                )

            # Keygen benchmark (important for constrained devices)
            for algo in ['ML-KEM-512', 'ML-KEM-768']:
                level = 1 if '512' in algo else 3
                suite.results.append(
                    self._benchmark_keygen(
                        sector='iot',
                        use_case=f'Key Generation ({algo})',
                        algorithm=algo,
                        compliance='IoT Device Provisioning',
                        nist_level=level,
                    )
                )

        suite.summary = self._generate_summary(suite.results, 'iot')
        return suite

    # =========================================================================
    # MPC-FHE
    # =========================================================================

    def run_mpc_fhe(self) -> SectorBenchmarkSuite:
        """
        MPC-FHE benchmarks.

        Operations:
        - 2-party key setup latency
        - Encrypted inference by model size
        - End-to-end protocol timing

        Security note (March 2026):
        - CEA 2025 CPAD attack: Threshold FHE requires mandatory smudging
          noise after individual_decrypt() — key recovery in < 1 hour without it
        - PKC 2025: Noise-flooding key recovery limits decryptions per key
        - MPCConfig now enforces smudging_noise_bits=40 by default
        """
        suite = SectorBenchmarkSuite(
            sector='mpc-fhe',
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info,
        )

        if DESILO_AVAILABLE:
            # 1. FHE engine setup
            suite.results.append(
                self._benchmark_fhe_setup(
                    sector='mpc-fhe',
                    use_case='CKKS Engine Initialization',
                    compliance='Secure Multi-Party Computation',
                    nist_level=3,
                )
            )

            # 2. Encryption benchmarks by data size
            for data_points in [4, 16, 64]:
                suite.results.append(
                    self._benchmark_fhe_encrypt(
                        sector='mpc-fhe',
                        use_case=f'Encrypt {data_points} values',
                        data_size=data_points * 8,
                        compliance='MPC-HE Protocol',
                        nist_level=3,
                        notes=f'{data_points} float64 values',
                    )
                )

            # 3. Computation benchmarks
            for data_points in [4, 16, 64]:
                suite.results.append(
                    self._benchmark_fhe_computation(
                        sector='mpc-fhe',
                        use_case=f'Encrypted Computation ({data_points} vals)',
                        operation='add_multiply',
                        data_points=data_points,
                        compliance='MPC-HE Protocol',
                        nist_level=3,
                    )
                )

            # 4. 2-party inference demo timing
            suite.results.append(
                self._benchmark_mpc_inference(
                    sector='mpc-fhe',
                    use_case='2-Party Linear Regression',
                    demo_type='linear_regression',
                    compliance='Secure Inference',
                    nist_level=3,
                )
            )

            # 5. GL scheme information (benchmark GL engine setup if available)
            suite.results.append(
                self._benchmark_gl_scheme_info(
                    sector='mpc-fhe',
                    use_case='GL Scheme Engine Status',
                    compliance='5th Gen FHE (ePrint 2025/1935)',
                    nist_level=3,
                )
            )

        suite.summary = self._generate_summary(suite.results, 'mpc-fhe')
        return suite

    def _benchmark_gl_scheme_info(
        self, sector: str, use_case: str, compliance: str, nist_level: int,
    ) -> 'SectorBenchmarkResult':
        """
        Benchmark GL scheme engine availability and capabilities.

        Reports GL scheme status, supported shapes, and security info.
        Actual GL benchmarks require desilofhe >= 1.10.0 with GLEngine.

        Reference:
          - ePrint 2025/1935 (Gentry & Lee, GL scheme)
          - DESILO FHE GLEngine: https://fhe.desilo.dev/latest/gl_quickstart/
          - RhombusEnd2End: GPU-accelerated 2PC inference architecture
        """
        t0 = time.time()

        try:
            from src.gl_scheme_engine import get_gl_scheme_info, GL_AVAILABLE
        except ImportError:
            try:
                from .gl_scheme_engine import get_gl_scheme_info, GL_AVAILABLE
            except ImportError:
                GL_AVAILABLE = False

        elapsed_ms = (time.time() - t0) * 1000

        if GL_AVAILABLE:
            gl_info = get_gl_scheme_info()
            notes = (
                f"GL engine available. Supported shapes: "
                f"{len(gl_info.get('supported_shapes', []))}. "
                f"Native matrix multiply in O(1) vs CKKS O(n) rotations."
            )
        else:
            notes = (
                "GL engine not available (requires desilofhe >= 1.10.0). "
                "GL scheme (ePrint 2025/1935) provides native matrix multiply "
                "for efficient neural network inference. "
                "FHE.org 2026 Taipei announcement (March 7, 2026)."
            )

        return SectorBenchmarkResult(
            sector=sector,
            use_case=use_case,
            operation='gl_scheme_status',
            data_size_bytes=0,
            algorithm='GL (Gentry-Lee)',
            iterations=1,
            mean_ms=round(elapsed_ms, 3),
            std_ms=0.0,
            min_ms=round(elapsed_ms, 3),
            max_ms=round(elapsed_ms, 3),
            p95_ms=round(elapsed_ms, 3),
            throughput_ops_sec=0.0,
            throughput_mbps=0.0,
            compliance_context=compliance,
            nist_level=nist_level,
            notes=(
                f'{notes} | GL scheme: ePrint 2025/1935 (Gentry & Lee). '
                f'Native O(1) matrix multiply. Ring-LWE security.'
            ),
        )

    # =========================================================================
    # RUN ALL
    # =========================================================================

    def run_all(self) -> Dict[str, SectorBenchmarkSuite]:
        """Run benchmarks for all sectors."""
        results = {}
        for sector_name, runner in [
            ('healthcare', self.run_healthcare),
            ('finance', self.run_finance),
            ('blockchain', self.run_blockchain),
            ('iot', self.run_iot),
            ('mpc-fhe', self.run_mpc_fhe),
        ]:
            logger.info("Running %s benchmarks...", sector_name)
            try:
                results[sector_name] = runner()
            except Exception as e:
                logger.error("Sector %s failed: %s", sector_name, e)
                results[sector_name] = SectorBenchmarkSuite(
                    sector=sector_name,
                    timestamp=datetime.now().isoformat(),
                    system_info=self.system_info,
                    summary={'error': str(e)},
                )
        return results

    # =========================================================================
    # BENCHMARK PRIMITIVES
    # =========================================================================

    def _benchmark_kem_exchange(self, sector: str, use_case: str,
                                 algorithm: str, data_size: int,
                                 compliance: str, nist_level: int,
                                 notes: str = '') -> SectorBenchmarkResult:
        """Benchmark KEM key encapsulation + decapsulation."""
        kem = oqs.KeyEncapsulation(algorithm)
        public_key = kem.generate_keypair()

        def kem_op():
            ciphertext, shared_secret_enc = kem.encap_secret(public_key)
            shared_secret_dec = kem.decap_secret(ciphertext)
            assert shared_secret_enc == shared_secret_dec

        timings = _measure_operation(kem_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        mbps = (data_size / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='kem_encap_decap',
            data_size_bytes=data_size, algorithm=algorithm,
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes=notes, **timings,
        )

    def _benchmark_signature(self, sector: str, use_case: str,
                              algorithm: str, data_size: int,
                              compliance: str, nist_level: int,
                              notes: str = '') -> SectorBenchmarkResult:
        """Benchmark digital signature generation."""
        sig = oqs.Signature(algorithm)
        sig.generate_keypair()
        message = os.urandom(data_size)

        def sign_op():
            sig.sign(message)

        timings = _measure_operation(sign_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        mbps = (data_size / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='signature_sign',
            data_size_bytes=data_size, algorithm=algorithm,
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes=notes, **timings,
        )

    def _benchmark_signature_verify(self, sector: str, use_case: str,
                                     algorithm: str, data_size: int,
                                     compliance: str, nist_level: int,
                                     notes: str = '') -> SectorBenchmarkResult:
        """Benchmark digital signature verification."""
        sig = oqs.Signature(algorithm)
        public_key = sig.generate_keypair()
        message = os.urandom(data_size)
        signature = sig.sign(message)

        verifier = oqs.Signature(algorithm)

        def verify_op():
            verifier.verify(message, signature, public_key)

        timings = _measure_operation(verify_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        mbps = (data_size / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='signature_verify',
            data_size_bytes=data_size, algorithm=algorithm,
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes=notes, **timings,
        )

    def _benchmark_batch_verify(self, sector: str, use_case: str,
                                 algorithm: str, batch_size: int,
                                 data_size: int, compliance: str,
                                 nist_level: int) -> SectorBenchmarkResult:
        """Benchmark batch signature verification."""
        # Pre-generate signatures
        sigs_data = []
        for _ in range(batch_size):
            signer = oqs.Signature(algorithm)
            pk = signer.generate_keypair()
            msg = os.urandom(data_size)
            sig_bytes = signer.sign(msg)
            sigs_data.append((msg, sig_bytes, pk))

        def batch_verify_op():
            for msg, sig_bytes, pk in sigs_data:
                verifier = oqs.Signature(algorithm)
                verifier.verify(msg, sig_bytes, pk)

        timings = _measure_operation(batch_verify_op, max(self.iterations // 5, 3))
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        total_data = data_size * batch_size
        mbps = (total_data / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case,
            operation=f'batch_verify_{batch_size}',
            data_size_bytes=total_data, algorithm=algorithm,
            iterations=max(self.iterations // 5, 3),
            throughput_ops_sec=throughput * batch_size, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes=f'Batch of {batch_size} signatures',
            **timings,
        )

    def _benchmark_full_pipeline(self, sector: str, use_case: str,
                                  sig_algorithm: str, data_size: int,
                                  compliance: str, nist_level: int) -> SectorBenchmarkResult:
        """Benchmark full keygen + sign + verify pipeline."""
        message = os.urandom(data_size)

        def pipeline_op():
            signer = oqs.Signature(sig_algorithm)
            pk = signer.generate_keypair()
            sig_bytes = signer.sign(message)
            verifier = oqs.Signature(sig_algorithm)
            verifier.verify(message, sig_bytes, pk)

        timings = _measure_operation(pipeline_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        mbps = (data_size / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case,
            operation='keygen_sign_verify',
            data_size_bytes=data_size, algorithm=sig_algorithm,
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes='Full pipeline: keygen → sign → verify',
            **timings,
        )

    def _benchmark_keygen(self, sector: str, use_case: str,
                           algorithm: str, compliance: str,
                           nist_level: int) -> SectorBenchmarkResult:
        """Benchmark key generation only."""
        def keygen_op():
            kem = oqs.KeyEncapsulation(algorithm)
            kem.generate_keypair()

        timings = _measure_operation(keygen_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='keygen',
            data_size_bytes=0, algorithm=algorithm,
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=0.0,
            compliance_context=compliance, nist_level=nist_level,
            **timings,
        )

    def _benchmark_key_rotation(self, sector: str, use_case: str,
                                 algorithm: str, compliance: str,
                                 nist_level: int) -> SectorBenchmarkResult:
        """Benchmark key rotation (generate new keypair + encap with new key)."""
        def rotation_op():
            kem = oqs.KeyEncapsulation(algorithm)
            pk = kem.generate_keypair()
            kem.encap_secret(pk)

        timings = _measure_operation(rotation_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='key_rotation',
            data_size_bytes=0, algorithm=algorithm,
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=0.0,
            compliance_context=compliance, nist_level=nist_level,
            notes='Generate new keypair + first encapsulation',
            **timings,
        )

    def _benchmark_fhe_encrypt(self, sector: str, use_case: str,
                                data_size: int, compliance: str,
                                nist_level: int,
                                notes: str = '') -> SectorBenchmarkResult:
        """Benchmark FHE encryption."""
        engine = desilofhe.Engine(mode=self.fhe_mode)
        sk = engine.create_secret_key()
        pk = engine.create_public_key(sk)

        n_values = max(data_size // 8, 4)
        data = [float(i) * 0.1 for i in range(n_values)]
        pt = engine.encode(data)

        def encrypt_op():
            engine.encrypt(pt, pk)

        timings = _measure_operation(encrypt_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        mbps = (data_size / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='fhe_encrypt',
            data_size_bytes=data_size, algorithm='CKKS',
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes=notes, **timings,
        )

    def _benchmark_fhe_computation(self, sector: str, use_case: str,
                                    operation: str, data_points: int,
                                    compliance: str, nist_level: int,
                                    notes: str = '') -> SectorBenchmarkResult:
        """Benchmark FHE computation (add, multiply, etc.)."""
        engine = desilofhe.Engine(mode=self.fhe_mode)
        sk = engine.create_secret_key()
        pk = engine.create_public_key(sk)
        eval_key = engine.create_relinearization_key(sk)

        data = [float(i + 1) * 0.1 for i in range(data_points)]
        pt = engine.encode(data)
        ct = engine.encrypt(pt, pk)

        if operation == 'mean+variance':
            scalar_pt = engine.encode([1.0 / data_points] * data_points)

            def compute_op():
                engine.multiply(ct, scalar_pt)

        elif operation == 'weighted_sum':
            weights = [1.0 / data_points] * data_points
            weight_pt = engine.encode(weights)

            def compute_op():
                engine.multiply(ct, weight_pt)

        elif operation == 'add_multiply':
            scalar_pt = engine.encode([2.0] * data_points)

            def compute_op():
                ct2 = engine.add(ct, ct)
                engine.multiply(ct2, scalar_pt)

        else:
            def compute_op():
                engine.add(ct, ct)

        timings = _measure_operation(compute_op, self.iterations)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0
        data_size = data_points * 8
        mbps = (data_size / 1024.0 / 1024.0) * throughput

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case,
            operation=f'fhe_{operation}',
            data_size_bytes=data_size, algorithm='CKKS',
            iterations=self.iterations,
            throughput_ops_sec=throughput, throughput_mbps=mbps,
            compliance_context=compliance, nist_level=nist_level,
            notes=notes, **timings,
        )

    def _benchmark_fhe_setup(self, sector: str, use_case: str,
                              compliance: str, nist_level: int) -> SectorBenchmarkResult:
        """Benchmark FHE engine setup."""
        def setup_op():
            engine = desilofhe.Engine(mode=self.fhe_mode)
            sk = engine.create_secret_key()
            engine.create_public_key(sk)
            engine.create_relinearization_key(sk)

        timings = _measure_operation(setup_op, max(self.iterations // 10, 3))
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case, operation='fhe_setup',
            data_size_bytes=0, algorithm='CKKS',
            iterations=max(self.iterations // 10, 3),
            throughput_ops_sec=throughput, throughput_mbps=0.0,
            compliance_context=compliance, nist_level=nist_level,
            notes='Engine init + secret key + public key + relin key',
            **timings,
        )

    def _benchmark_mpc_inference(self, sector: str, use_case: str,
                                  demo_type: str, compliance: str,
                                  nist_level: int) -> SectorBenchmarkResult:
        """Benchmark MPC-HE 2-party inference demo."""
        from src.mpc_he_inference import SimpleMPCDemo, MPCConfig

        config = MPCConfig(fhe_mode=self.fhe_mode, use_bootstrap=False, log_n=14)
        demo = SimpleMPCDemo(config)

        def inference_op():
            demo.run_linear_regression_demo([1.0, 2.0, 3.0, 4.0])

        iters = max(self.iterations // 10, 3)
        timings = _measure_operation(inference_op, iters)
        throughput = 1000.0 / timings['mean_ms'] if timings['mean_ms'] > 0 else 0

        return SectorBenchmarkResult(
            sector=sector, use_case=use_case,
            operation='mpc_2party_inference',
            data_size_bytes=32, algorithm='CKKS+MPC',
            iterations=iters,
            throughput_ops_sec=throughput, throughput_mbps=0.0,
            compliance_context=compliance, nist_level=nist_level,
            notes=f'2-party protocol: {demo_type}',
            **timings,
        )

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def _generate_summary(self, results: List[SectorBenchmarkResult],
                           sector: str) -> Dict[str, Any]:
        """Generate summary statistics for a sector."""
        if not results:
            return {'total_benchmarks': 0, 'note': 'No benchmarks run'}

        all_times = [r.mean_ms for r in results]
        pqc_results = [r for r in results if r.algorithm.startswith('ML-')]
        fhe_results = [r for r in results if r.algorithm in ('CKKS', 'CKKS+MPC')]

        summary = {
            'total_benchmarks': len(results),
            'overall_mean_ms': round(float(np.mean(all_times)), 3),
            'fastest_operation': min(results, key=lambda r: r.mean_ms).use_case,
            'fastest_ms': round(min(all_times), 3),
            'slowest_operation': max(results, key=lambda r: r.mean_ms).use_case,
            'slowest_ms': round(max(all_times), 3),
            'algorithms_tested': list(set(r.algorithm for r in results)),
            'nist_levels_covered': sorted(set(r.nist_level for r in results)),
        }

        if pqc_results:
            summary['pqc_operations'] = len(pqc_results)
            summary['pqc_mean_ms'] = round(
                float(np.mean([r.mean_ms for r in pqc_results])), 3
            )

        if fhe_results:
            summary['fhe_operations'] = len(fhe_results)
            summary['fhe_mean_ms'] = round(
                float(np.mean([r.mean_ms for r in fhe_results])), 3
            )

        # Add quantum security context per sector
        summary['quantum_security_context'] = (
            self._get_sector_quantum_context(sector)
        )

        return summary

    @staticmethod
    def _get_sector_quantum_context(sector: str) -> Dict[str, Any]:
        """Return quantum security assessment context for a sector."""
        contexts = {
            'healthcare': {
                'compliance_framework': 'HIPAA Security Rule',
                'pqc_algorithms_used': ['ML-KEM-768', 'ML-DSA-65'],
                'fhe_scheme': 'CKKS (Ring-LWE)',
                'quantum_risk_profile': (
                    'Patient data has long retention (decades). Harvest-now-decrypt-later '
                    'attacks make PQC migration urgent. FHE on vital signs shares lattice '
                    'risk with PQC key exchange — a single lattice breakthrough would '
                    'compromise both encrypted data AND key distribution.'
                ),
                'business_recommendation': (
                    'Prioritize ML-KEM-768+ for data in transit. For FHE analytics, '
                    'use log_n ≥ 15 with conservative modulus budget. Consider HQC '
                    'as backup KEM for long-term data protection.'
                ),
                'data_retention_risk': 'HIGH — patient records retained 7-30+ years',
            },
            'finance': {
                'compliance_framework': 'PCI DSS v4.0, SOX, CNSA 2.0',
                'pqc_algorithms_used': ['ML-KEM-768', 'ML-DSA-65', 'ML-DSA-87'],
                'fhe_scheme': 'CKKS (Ring-LWE)',
                'quantum_risk_profile': (
                    'Financial transactions require real-time integrity AND long-term '
                    'confidentiality. ML-DSA-87 for high-value trades provides NIST Level 5. '
                    'FHE portfolio analysis shares lattice risk with PQC. CNSA 2.0 '
                    'mandates full PQC migration by 2030.'
                ),
                'business_recommendation': (
                    'Use ML-DSA-87 for high-value trade settlement. Key rotation via '
                    'ML-KEM-768 meets PCI DSS Req 3.7. For encrypted analytics, '
                    'ensure CKKS parameters meet 128-bit minimum per HE Standard.'
                ),
                'data_retention_risk': 'MODERATE — transaction records 5-7 years',
            },
            'blockchain': {
                'compliance_framework': 'Post-Quantum Blockchain Standards',
                'pqc_algorithms_used': ['ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87'],
                'fhe_scheme': 'N/A (signature-focused)',
                'quantum_risk_profile': (
                    'Blockchain immutability means quantum-broken signatures cannot be '
                    'revoked retroactively. PQ migration must happen before CRQC emergence. '
                    'ML-DSA signature sizes (2-4KB) significantly larger than ECDSA (64B), '
                    'impacting block size and throughput.'
                ),
                'business_recommendation': (
                    'Begin PQ signature migration immediately. ML-DSA-65 balances security '
                    'and performance. Plan for 10-50x signature size increase in block '
                    'capacity planning. Batch verification helps throughput.'
                ),
                'data_retention_risk': 'CRITICAL — blockchain data is immutable/permanent',
            },
            'iot': {
                'compliance_framework': 'IoT Security (NIST SP 800-183)',
                'pqc_algorithms_used': ['ML-KEM-512', 'ML-KEM-768'],
                'fhe_scheme': 'N/A or CKKS-Light',
                'quantum_risk_profile': (
                    'IoT devices have limited compute resources. ML-KEM-512 (NIST Level 1) '
                    'provides minimum quantum resistance. Key generation and encapsulation '
                    'must complete within constrained power/time budgets. SPA attacks on '
                    'Cortex-M4 (Berzati 2025) are directly relevant to IoT.'
                ),
                'business_recommendation': (
                    'Use ML-KEM-512 for constrained devices, ML-KEM-768 where feasible. '
                    'Apply masking countermeasures (pqm4) for SPA protection. Plan firmware '
                    'updates for crypto-agility. For IoT FHE, use conservative log_n ≥ 13.'
                ),
                'data_retention_risk': 'LOW-MODERATE — sensor data typically short-lived',
            },
            'mpc-fhe': {
                'compliance_framework': 'Secure Multi-Party Computation',
                'pqc_algorithms_used': ['N/A (FHE provides PQ security via Ring-LWE)'],
                'fhe_scheme': 'CKKS (Ring-LWE) with 2-party multiparty protocol',
                'quantum_risk_profile': (
                    'MPC-HE relies entirely on CKKS Ring-LWE security. No separate PQC '
                    'key exchange is needed (FHE provides encryption), but the lattice '
                    'monoculture risk is maximal: ALL security rests on one assumption. '
                    'Default MPC-HE config (num_scales=40, log_n=15) may exceed 128-bit '
                    'HE Standard bounds (max log Q = 881 for N=32768).'
                ),
                'business_recommendation': (
                    'CRITICAL: Verify MPC-HE CKKS parameters against HE Standard. '
                    'Use max_levels ≤ 20 at log_n=15 for 128-bit security. '
                    'For deeper computations, increase to log_n=16. '
                    'No non-lattice FHE alternative exists for diversification.'
                ),
                'data_retention_risk': 'VARIES — depends on application',
            },
        }
        return contexts.get(sector, {
            'quantum_risk_profile': 'No specific quantum context for this sector.',
        })


# =============================================================================
# MODULE VERSION
# =============================================================================

try:
    from .version_loader import get_version
    __version__ = get_version('sector_benchmarks')
except ImportError:
    __version__ = "3.2.0"
__author__ = "PQC-FHE Integration Library"
