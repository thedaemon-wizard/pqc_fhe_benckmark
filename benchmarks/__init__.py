"""
PQC-FHE Benchmark Suite
=======================

Performance benchmarks for Post-Quantum Cryptography and FHE operations.

Metrics:
- Key generation time
- Encapsulation/Decapsulation time
- Signing/Verification time
- FHE encryption/decryption time
- Bootstrap throughput
- Memory usage

References:
- NIST PQC Benchmarks: https://openquantumsafe.org/benchmarking/
- DESILO FHE Performance: https://fhe.desilo.dev/latest/
"""

import time
import json
import statistics
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    algorithm: str
    operation: str
    iterations: int
    times_ms: List[float] = field(default_factory=list)
    
    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0
    
    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
    
    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0
    
    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0
    
    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'algorithm': self.algorithm,
            'operation': self.operation,
            'iterations': self.iterations,
            'mean_ms': round(self.mean_ms, 4),
            'std_ms': round(self.std_ms, 4),
            'min_ms': round(self.min_ms, 4),
            'max_ms': round(self.max_ms, 4),
            'p95_ms': round(self.p95_ms, 4),
            'ops_per_sec': round(1000.0 / self.mean_ms, 2) if self.mean_ms > 0 else 0,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_info: Dict[str, Any] = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'system_info': self.system_info,
            'results': [r.to_dict() for r in self.results],
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"BENCHMARK RESULTS: {self.name}")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print("-" * 80)
        print(f"{'Operation':<30} {'Algorithm':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Ops/sec':<10}")
        print("-" * 80)
        
        for r in self.results:
            print(f"{r.operation:<30} {r.algorithm:<20} {r.mean_ms:<12.4f} {r.std_ms:<12.4f} {1000/r.mean_ms if r.mean_ms > 0 else 0:<10.2f}")
        
        print("=" * 80)


def get_system_info() -> Dict[str, Any]:
    """Collect system information for benchmark context"""
    import platform
    import sys
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # Check for GPU
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
    except ImportError:
        info['cuda_available'] = False
    
    # Check liboqs version
    try:
        import oqs
        info['liboqs_version'] = getattr(oqs, '__version__', 'unknown')
        info['liboqs_kem_count'] = len(oqs.get_enabled_kem_mechanisms())
        info['liboqs_sig_count'] = len(oqs.get_enabled_sig_mechanisms())
    except ImportError:
        info['liboqs_available'] = False
    
    return info


def benchmark_operation(
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    name: str = "operation",
    algorithm: str = "unknown",
    operation: str = "unknown"
) -> BenchmarkResult:
    """
    Benchmark a single operation
    
    Args:
        func: Function to benchmark (should take no arguments)
        iterations: Number of iterations
        warmup: Warmup iterations (not counted)
        name: Benchmark name
        algorithm: Algorithm name
        operation: Operation type
        
    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return BenchmarkResult(
        name=name,
        algorithm=algorithm,
        operation=operation,
        iterations=iterations,
        times_ms=times
    )


# =============================================================================
# PQC BENCHMARKS
# =============================================================================

def benchmark_pqc_kem(
    algorithms: List[str] = None,
    iterations: int = 100
) -> List[BenchmarkResult]:
    """
    Benchmark PQC KEM algorithms
    
    Args:
        algorithms: List of KEM algorithms to benchmark
        iterations: Number of iterations per operation
        
    Returns:
        List of BenchmarkResult
    """
    try:
        import oqs
    except ImportError:
        logger.error("liboqs not available")
        return []
    
    if algorithms is None:
        # Default: NIST standard algorithms
        algorithms = ['ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024']
    
    available = oqs.get_enabled_kem_mechanisms()
    algorithms = [a for a in algorithms if a in available]
    
    results = []
    
    for alg in algorithms:
        logger.info(f"Benchmarking {alg}...")
        
        # Key generation
        kem = oqs.KeyEncapsulation(alg)
        result_keygen = benchmark_operation(
            func=kem.generate_keypair,
            iterations=iterations,
            name=f"{alg}_keygen",
            algorithm=alg,
            operation="key_generation"
        )
        results.append(result_keygen)
        
        # Setup for encap/decap
        public_key = kem.generate_keypair()
        
        # Encapsulation
        result_encap = benchmark_operation(
            func=lambda: kem.encap_secret(public_key),
            iterations=iterations,
            name=f"{alg}_encap",
            algorithm=alg,
            operation="encapsulation"
        )
        results.append(result_encap)
        
        # Decapsulation
        ciphertext, _ = kem.encap_secret(public_key)
        result_decap = benchmark_operation(
            func=lambda: kem.decap_secret(ciphertext),
            iterations=iterations,
            name=f"{alg}_decap",
            algorithm=alg,
            operation="decapsulation"
        )
        results.append(result_decap)
        
        logger.info(f"  Keygen: {result_keygen.mean_ms:.4f} ms")
        logger.info(f"  Encap:  {result_encap.mean_ms:.4f} ms")
        logger.info(f"  Decap:  {result_decap.mean_ms:.4f} ms")
    
    return results


def benchmark_pqc_sig(
    algorithms: List[str] = None,
    iterations: int = 100,
    message_size: int = 1024
) -> List[BenchmarkResult]:
    """
    Benchmark PQC signature algorithms
    
    Args:
        algorithms: List of signature algorithms to benchmark
        iterations: Number of iterations per operation
        message_size: Size of message to sign (bytes)
        
    Returns:
        List of BenchmarkResult
    """
    try:
        import oqs
    except ImportError:
        logger.error("liboqs not available")
        return []
    
    if algorithms is None:
        # Default: NIST standard algorithms
        algorithms = ['ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87']
    
    available = oqs.get_enabled_sig_mechanisms()
    algorithms = [a for a in algorithms if a in available]
    
    message = bytes(range(256)) * (message_size // 256 + 1)
    message = message[:message_size]
    
    results = []
    
    for alg in algorithms:
        logger.info(f"Benchmarking {alg}...")
        
        # Key generation
        sig = oqs.Signature(alg)
        result_keygen = benchmark_operation(
            func=sig.generate_keypair,
            iterations=iterations,
            name=f"{alg}_keygen",
            algorithm=alg,
            operation="key_generation"
        )
        results.append(result_keygen)
        
        # Setup for sign/verify
        public_key = sig.generate_keypair()
        
        # Signing
        result_sign = benchmark_operation(
            func=lambda: sig.sign(message),
            iterations=iterations,
            name=f"{alg}_sign",
            algorithm=alg,
            operation="signing"
        )
        results.append(result_sign)
        
        # Verification
        signature = sig.sign(message)
        result_verify = benchmark_operation(
            func=lambda: sig.verify(message, signature, public_key),
            iterations=iterations,
            name=f"{alg}_verify",
            algorithm=alg,
            operation="verification"
        )
        results.append(result_verify)
        
        logger.info(f"  Keygen: {result_keygen.mean_ms:.4f} ms")
        logger.info(f"  Sign:   {result_sign.mean_ms:.4f} ms")
        logger.info(f"  Verify: {result_verify.mean_ms:.4f} ms")
    
    return results


# =============================================================================
# FHE BENCHMARKS
# =============================================================================

def benchmark_fhe_operations(
    iterations: int = 100,
    slot_counts: List[int] = None,
<<<<<<< HEAD
    mode: str = 'gpu'
=======
    mode: str = 'cpu'
>>>>>>> origin/main
) -> List[BenchmarkResult]:
    """
    Benchmark FHE operations
    
    Args:
        iterations: Number of iterations per operation
        slot_counts: List of slot counts to test
        mode: FHE mode ('cpu' or 'gpu')
        
    Returns:
        List of BenchmarkResult
    """
    try:
        import desilofhe
    except ImportError:
        logger.warning("DESILO FHE not available, skipping FHE benchmarks")
        return []
    
    if slot_counts is None:
        slot_counts = [2**14, 2**15]  # 16384, 32768 slots
    
    results = []
    
    for slot_count in slot_counts:
        logger.info(f"Benchmarking FHE with {slot_count} slots...")
        
        try:
            engine = desilofhe.Engine(mode=mode, slot_count=slot_count)
            
            # Key generation (single run, too expensive for many iterations)
            start = time.perf_counter()
            secret_key = engine.create_secret_key()
            public_key = engine.create_public_key(secret_key)
            relin_key = engine.create_relinearization_key(secret_key)
            keygen_time = (time.perf_counter() - start) * 1000
            
            results.append(BenchmarkResult(
                name=f"FHE_keygen_{slot_count}",
                algorithm=f"CKKS-{slot_count}",
                operation="key_generation",
                iterations=1,
                times_ms=[keygen_time]
            ))
            logger.info(f"  Keygen: {keygen_time:.2f} ms")
            
            # Test data
            data = list(np.random.randn(slot_count).astype(np.float64))
            
            # Encryption
            result_encrypt = benchmark_operation(
                func=lambda: engine.encrypt(data, public_key),
                iterations=min(iterations, 50),  # FHE is slow
                name=f"FHE_encrypt_{slot_count}",
                algorithm=f"CKKS-{slot_count}",
                operation="encryption"
            )
            results.append(result_encrypt)
            logger.info(f"  Encrypt: {result_encrypt.mean_ms:.2f} ms")
            
            # Decryption
            ct = engine.encrypt(data, public_key)
            result_decrypt = benchmark_operation(
                func=lambda: engine.decrypt(ct, secret_key),
                iterations=min(iterations, 50),
                name=f"FHE_decrypt_{slot_count}",
                algorithm=f"CKKS-{slot_count}",
                operation="decryption"
            )
            results.append(result_decrypt)
            logger.info(f"  Decrypt: {result_decrypt.mean_ms:.2f} ms")
            
            # Addition
            ct1 = engine.encrypt(data, public_key)
            ct2 = engine.encrypt(data, public_key)
            result_add = benchmark_operation(
                func=lambda: engine.add(ct1, ct2),
                iterations=iterations,
                name=f"FHE_add_{slot_count}",
                algorithm=f"CKKS-{slot_count}",
                operation="addition"
            )
            results.append(result_add)
            logger.info(f"  Add: {result_add.mean_ms:.4f} ms")
            
            # Multiplication
            result_mult = benchmark_operation(
                func=lambda: engine.multiply(ct1, ct2, relin_key),
                iterations=iterations,
                name=f"FHE_mult_{slot_count}",
                algorithm=f"CKKS-{slot_count}",
                operation="multiplication"
            )
            results.append(result_mult)
            logger.info(f"  Mult: {result_mult.mean_ms:.4f} ms")
            
        except Exception as e:
            logger.error(f"FHE benchmark failed for slot_count={slot_count}: {e}")
    
    return results


# =============================================================================
# FULL BENCHMARK SUITE
# =============================================================================

def run_full_benchmark(
    iterations: int = 100,
    include_fhe: bool = True,
<<<<<<< HEAD
    fhe_mode: str = 'gpu',
=======
    fhe_mode: str = 'cpu',
>>>>>>> origin/main
    output_file: str = None
) -> BenchmarkSuite:
    """
    Run complete benchmark suite
    
    Args:
        iterations: Number of iterations per operation
        include_fhe: Include FHE benchmarks (slower)
        fhe_mode: FHE execution mode
        output_file: Optional output file path
        
    Returns:
        BenchmarkSuite with all results
    """
    suite = BenchmarkSuite(
        name="PQC-FHE Integration Benchmark",
        system_info=get_system_info()
    )
    
    logger.info("=" * 60)
    logger.info("PQC-FHE BENCHMARK SUITE")
    logger.info("=" * 60)
    logger.info(f"Iterations per operation: {iterations}")
    logger.info(f"Include FHE: {include_fhe}")
    logger.info(f"FHE Mode: {fhe_mode}")
    logger.info("=" * 60)
    
    # PQC KEM benchmarks
    logger.info("\n[1/4] PQC KEM Benchmarks")
    logger.info("-" * 40)
    kem_results = benchmark_pqc_kem(iterations=iterations)
    for r in kem_results:
        suite.add_result(r)
    
    # PQC Signature benchmarks
    logger.info("\n[2/4] PQC Signature Benchmarks")
    logger.info("-" * 40)
    sig_results = benchmark_pqc_sig(iterations=iterations)
    for r in sig_results:
        suite.add_result(r)
    
    # FHE benchmarks (optional)
    if include_fhe:
        logger.info("\n[3/4] FHE Benchmarks")
        logger.info("-" * 40)
        fhe_results = benchmark_fhe_operations(
            iterations=min(iterations, 50),
            mode=fhe_mode
        )
        for r in fhe_results:
            suite.add_result(r)
    
    # Integration benchmarks
    logger.info("\n[4/4] Integration Benchmarks")
    logger.info("-" * 40)
    
    try:
        from .src.pqc_fhe_integration import PQCFHESystem, IntegrationConfig
        
        config = IntegrationConfig()
        config.fhe.mode = fhe_mode
        
        system = PQCFHESystem(config)
        
        # Full workflow benchmark
        test_data = list(np.random.randn(100).astype(np.float64))
        
        def full_workflow():
            return system.secure_computation_demo(test_data)
        
        result_workflow = benchmark_operation(
            func=full_workflow,
            iterations=min(iterations, 10),
            warmup=1,
            name="full_workflow",
            algorithm="PQC+FHE",
            operation="complete_workflow"
        )
        suite.add_result(result_workflow)
        logger.info(f"  Full workflow: {result_workflow.mean_ms:.2f} ms")
        
    except Exception as e:
        logger.warning(f"Integration benchmark skipped: {e}")
    
    # Print summary
    suite.print_summary()
    
    # Save results
    if output_file:
        suite.save(output_file)
        logger.info(f"\nResults saved to: {output_file}")
    
    return suite


# =============================================================================
# COMPARISON WITH CLASSICAL ALGORITHMS
# =============================================================================

def benchmark_classical_comparison(iterations: int = 100) -> BenchmarkSuite:
    """
    Benchmark classical vs post-quantum algorithms for comparison
    
    This demonstrates the performance trade-off between security levels.
    """
    suite = BenchmarkSuite(
        name="Classical vs Post-Quantum Comparison",
        system_info=get_system_info()
    )
    
    logger.info("=" * 60)
    logger.info("CLASSICAL vs POST-QUANTUM COMPARISON")
    logger.info("=" * 60)
    
    # Classical RSA/ECDH (using cryptography library)
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, ec, x25519
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        # RSA-2048
        logger.info("\n[Classical] RSA-2048")
        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        message = b"Test message for signing"
        
        result_rsa_sign = benchmark_operation(
            func=lambda: rsa_key.sign(
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            ),
            iterations=iterations,
            name="RSA-2048_sign",
            algorithm="RSA-2048",
            operation="signing"
        )
        suite.add_result(result_rsa_sign)
        logger.info(f"  Sign: {result_rsa_sign.mean_ms:.4f} ms")
        
        # X25519
        logger.info("\n[Classical] X25519")
        x25519_key = x25519.X25519PrivateKey.generate()
        peer_key = x25519.X25519PrivateKey.generate().public_key()
        
        result_x25519 = benchmark_operation(
            func=lambda: x25519_key.exchange(peer_key),
            iterations=iterations,
            name="X25519_exchange",
            algorithm="X25519",
            operation="key_exchange"
        )
        suite.add_result(result_x25519)
        logger.info(f"  Exchange: {result_x25519.mean_ms:.4f} ms")
        
    except ImportError:
        logger.warning("cryptography library not available for classical benchmarks")
    
    # Post-quantum comparison
    try:
        import oqs
        
        # ML-KEM-768 (comparable security to RSA-3072)
        logger.info("\n[Post-Quantum] ML-KEM-768")
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        pk = kem.generate_keypair()
        
        result_mlkem_encap = benchmark_operation(
            func=lambda: kem.encap_secret(pk),
            iterations=iterations,
            name="ML-KEM-768_encap",
            algorithm="ML-KEM-768",
            operation="encapsulation"
        )
        suite.add_result(result_mlkem_encap)
        logger.info(f"  Encap: {result_mlkem_encap.mean_ms:.4f} ms")
        
        # ML-DSA-65 (comparable security to RSA-3072)
        logger.info("\n[Post-Quantum] ML-DSA-65")
        sig = oqs.Signature("ML-DSA-65")
        sig.generate_keypair()
        
        result_mldsa_sign = benchmark_operation(
            func=lambda: sig.sign(message),
            iterations=iterations,
            name="ML-DSA-65_sign",
            algorithm="ML-DSA-65",
            operation="signing"
        )
        suite.add_result(result_mldsa_sign)
        logger.info(f"  Sign: {result_mldsa_sign.mean_ms:.4f} ms")
        
    except ImportError:
        logger.warning("liboqs not available for PQC benchmarks")
    
    suite.print_summary()
    return suite


<<<<<<< HEAD
# =============================================================================
# EXTENDED BENCHMARKS (v3.0.0)
# =============================================================================

@dataclass
class GPUBenchmarkResult(BenchmarkResult):
    """Extended benchmark result with GPU-specific metrics."""
    gpu_memory_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    speedup_vs_cpu: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['gpu_memory_mb'] = round(self.gpu_memory_mb, 2)
        d['gpu_utilization_pct'] = round(self.gpu_utilization_pct, 2)
        d['speedup_vs_cpu'] = round(self.speedup_vs_cpu, 2)
        return d


def _get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def benchmark_quantum_threat_estimation(
    iterations: int = 10,
) -> List[BenchmarkResult]:
    """Benchmark quantum threat resource estimation computations."""
    from src.quantum_threat_simulator import ShorSimulator, GroverSimulator, QuantumThreatTimeline

    results = []
    shor = ShorSimulator()
    grover = GroverSimulator()

    # Shor RSA estimation
    for key_size in [2048, 3072, 4096]:
        result = benchmark_operation(
            func=lambda ks=key_size: shor.estimate_rsa_resources(ks),
            iterations=iterations,
            name=f"shor_rsa_{key_size}",
            algorithm=f"Shor-RSA-{key_size}",
            operation="resource_estimation",
        )
        results.append(result)

    # Shor ECC estimation
    for curve in [256, 384]:
        result = benchmark_operation(
            func=lambda c=curve: shor.estimate_ecc_resources(c),
            iterations=iterations,
            name=f"shor_ecc_{curve}",
            algorithm=f"Shor-ECC-{curve}",
            operation="resource_estimation",
        )
        results.append(result)

    # Grover AES estimation
    for key_size in [128, 256]:
        result = benchmark_operation(
            func=lambda ks=key_size: grover.estimate_aes_resources(ks),
            iterations=iterations,
            name=f"grover_aes_{key_size}",
            algorithm=f"Grover-AES-{key_size}",
            operation="resource_estimation",
        )
        results.append(result)

    # Full timeline generation
    result = benchmark_operation(
        func=lambda: QuantumThreatTimeline('moderate').generate_full_timeline(),
        iterations=iterations,
        name="threat_timeline",
        algorithm="QuantumTimeline",
        operation="timeline_generation",
    )
    results.append(result)

    logger.info("Quantum threat benchmarks: %d results", len(results))
    return results


def benchmark_security_scoring(
    iterations: int = 50,
) -> List[BenchmarkResult]:
    """Benchmark security scoring engine performance."""
    from src.security_scoring import SecurityScoringEngine, ComplianceStandard

    results = []
    engine = SecurityScoringEngine()

    # Enterprise inventory scoring
    inventory = engine.create_sample_enterprise_inventory()
    org_profile = {
        'has_crypto_inventory': True,
        'has_migration_plan': False,
        'pqc_testing_started': True,
        'key_rotation_policy': True,
    }

    result = benchmark_operation(
        func=lambda: engine.calculate_overall_score(inventory, org_profile),
        iterations=iterations,
        name="scoring_enterprise",
        algorithm="SecurityScoring",
        operation="enterprise_assessment",
    )
    results.append(result)

    # Financial inventory scoring
    fin_inventory = engine.create_sample_financial_inventory()
    result = benchmark_operation(
        func=lambda: engine.calculate_overall_score(fin_inventory, org_profile),
        iterations=iterations,
        name="scoring_financial",
        algorithm="SecurityScoring",
        operation="financial_assessment",
    )
    results.append(result)

    # Compliance checking
    for standard in [ComplianceStandard.NIST_IR_8547, ComplianceStandard.CNSA_2_0]:
        result = benchmark_operation(
            func=lambda s=standard: engine.check_compliance(s, inventory, org_profile),
            iterations=iterations,
            name=f"compliance_{standard.value}",
            algorithm="SecurityScoring",
            operation=f"compliance_{standard.value}",
        )
        results.append(result)

    logger.info("Security scoring benchmarks: %d results", len(results))
    return results


def benchmark_mpc_he_protocol(
    data_sizes: Optional[List[int]] = None,
    fhe_mode: str = 'gpu',
    iterations: int = 3,
) -> List[BenchmarkResult]:
    """
    Benchmark MPC-HE 2-party protocol phases.

    Note: MPC-HE benchmarks use fewer iterations due to FHE key generation cost.
    """
    try:
        from src.mpc_he_inference import MPCHEProtocol, MPCConfig, MPCRole
    except ImportError:
        logger.warning("MPC-HE module not available")
        return []

    try:
        import desilofhe
    except ImportError:
        logger.warning("desilofhe not available for MPC-HE benchmarks")
        return []

    if data_sizes is None:
        data_sizes = [8, 32]

    results = []

    for size in data_sizes:
        data = np.random.randn(size).tolist()
        weights = np.random.randn(size).astype(np.float64)
        bias = np.random.randn(size).astype(np.float64) * 0.1

        model_layers = [
            {'type': 'linear', 'weights': weights, 'bias': bias},
        ]

        config = MPCConfig(
            fhe_mode=fhe_mode,
            use_bootstrap=False,
            log_n=14,
        )

        # Full protocol benchmark
        def run_protocol():
            proto = MPCHEProtocol(config)
            return proto.run_2party_inference(data, model_layers)

        result = benchmark_operation(
            func=run_protocol,
            iterations=iterations,
            warmup=1,
            name=f"mpc_he_full_{size}",
            algorithm="MPC-HE-CKKS",
            operation=f"2party_inference_n{size}",
        )
        results.append(result)

    logger.info("MPC-HE benchmarks: %d results", len(results))
    return results


def benchmark_gpu_vs_cpu_fhe(
    operations: Optional[List[str]] = None,
    iterations: int = 20,
) -> List[BenchmarkResult]:
    """Head-to-head GPU vs CPU FHE performance comparison."""
    try:
        import desilofhe
    except ImportError:
        logger.warning("desilofhe not available for GPU benchmarks")
        return []

    if operations is None:
        operations = ['encrypt', 'decrypt', 'add', 'multiply']

    results = []
    data = np.random.randn(100).tolist()

    for mode in ['cpu', 'gpu']:
        try:
            engine_kwargs = {'mode': mode}
            if mode == 'gpu':
                engine_kwargs['thread_count'] = 512
            engine = desilofhe.Engine(**engine_kwargs)
            sk = engine.create_secret_key()
            pk = engine.create_public_key(sk)
            rk = engine.create_relinearization_key(sk)

            ct1 = engine.encrypt(data, pk)
            ct2 = engine.encrypt(data, pk)

            for op in operations:
                if op == 'encrypt':
                    func = lambda: engine.encrypt(data, pk)
                elif op == 'decrypt':
                    func = lambda: engine.decrypt(ct1, sk)
                elif op == 'add':
                    func = lambda: engine.add(ct1, ct2, rk)
                elif op == 'multiply':
                    func = lambda: engine.multiply(ct1, ct2, rk)
                else:
                    continue

                gpu_mem_before = _get_gpu_memory_mb()
                r = benchmark_operation(
                    func=func,
                    iterations=iterations,
                    name=f"fhe_{op}_{mode}",
                    algorithm=f"CKKS-{mode.upper()}",
                    operation=op,
                )
                gpu_mem_after = _get_gpu_memory_mb()

                if mode == 'gpu':
                    gpu_result = GPUBenchmarkResult(
                        name=r.name,
                        algorithm=r.algorithm,
                        operation=r.operation,
                        iterations=r.iterations,
                        times_ms=r.times_ms,
                        gpu_memory_mb=gpu_mem_after - gpu_mem_before,
                    )
                    results.append(gpu_result)
                else:
                    results.append(r)

        except Exception as e:
            logger.warning("Failed to benchmark %s mode: %s", mode, e)

    # Calculate speedups
    cpu_times = {r.operation: r.mean_ms for r in results if 'CPU' in r.algorithm}
    for r in results:
        if isinstance(r, GPUBenchmarkResult) and r.operation in cpu_times:
            cpu_ms = cpu_times[r.operation]
            if r.mean_ms > 0:
                r.speedup_vs_cpu = cpu_ms / r.mean_ms

    logger.info("GPU vs CPU benchmarks: %d results", len(results))
    return results


def benchmark_security_performance_tradeoff(
    iterations: int = 50,
) -> List[BenchmarkResult]:
    """Measure performance at different NIST security levels."""
    results = []

    try:
        import oqs

        for algo, level in [
            ('ML-KEM-512', 1), ('ML-KEM-768', 3), ('ML-KEM-1024', 5),
        ]:
            kem = oqs.KeyEncapsulation(algo)
            pk = kem.generate_keypair()

            # Keygen
            result = benchmark_operation(
                func=lambda a=algo: oqs.KeyEncapsulation(a).generate_keypair(),
                iterations=iterations,
                name=f"{algo}_keygen",
                algorithm=algo,
                operation=f"keygen_level{level}",
            )
            results.append(result)

            # Encapsulation
            result = benchmark_operation(
                func=lambda: kem.encap_secret(pk),
                iterations=iterations,
                name=f"{algo}_encap",
                algorithm=algo,
                operation=f"encap_level{level}",
            )
            results.append(result)

        for algo, level in [
            ('ML-DSA-44', 2), ('ML-DSA-65', 3), ('ML-DSA-87', 5),
        ]:
            sig = oqs.Signature(algo)
            sig.generate_keypair()
            message = b"Benchmark message for security level comparison"

            result = benchmark_operation(
                func=lambda: sig.sign(message),
                iterations=iterations,
                name=f"{algo}_sign",
                algorithm=algo,
                operation=f"sign_level{level}",
            )
            results.append(result)

    except ImportError:
        logger.warning("liboqs not available for security-performance tradeoff benchmarks")

    logger.info("Security-performance tradeoff benchmarks: %d results", len(results))
    return results


def run_extended_benchmark(
    iterations: int = 50,
    include_quantum_threat: bool = True,
    include_security_scoring: bool = True,
    include_mpc_he: bool = True,
    include_gpu: bool = True,
    include_security_perf: bool = True,
    fhe_mode: str = 'gpu',
    output_file: Optional[str] = None,
) -> BenchmarkSuite:
    """Extended benchmark suite incorporating all new modules."""
    suite = BenchmarkSuite(
        name="PQC-FHE Extended Benchmark v3.0.0",
        system_info=get_system_info(),
    )

    if include_quantum_threat:
        logger.info("\n=== Quantum Threat Estimation Benchmarks ===")
        for r in benchmark_quantum_threat_estimation(iterations=min(iterations, 50)):
            suite.add_result(r)

    if include_security_scoring:
        logger.info("\n=== Security Scoring Benchmarks ===")
        for r in benchmark_security_scoring(iterations=min(iterations, 100)):
            suite.add_result(r)

    if include_mpc_he:
        logger.info("\n=== MPC-HE Protocol Benchmarks ===")
        for r in benchmark_mpc_he_protocol(fhe_mode=fhe_mode, iterations=min(iterations, 5)):
            suite.add_result(r)

    if include_gpu:
        logger.info("\n=== GPU vs CPU FHE Benchmarks ===")
        for r in benchmark_gpu_vs_cpu_fhe(iterations=min(iterations, 30)):
            suite.add_result(r)

    if include_security_perf:
        logger.info("\n=== Security-Performance Tradeoff ===")
        for r in benchmark_security_performance_tradeoff(iterations=min(iterations, 100)):
            suite.add_result(r)

    suite.print_summary()

    if output_file:
        suite.save(output_file)
        logger.info("Extended benchmark results saved to %s", output_file)

    return suite


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PQC-FHE Benchmarks')
    parser.add_argument('--iterations', '-n', type=int, default=100)
    parser.add_argument('--no-fhe', action='store_true', help='Skip FHE benchmarks')
    parser.add_argument('--fhe-mode', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--output', '-o', default='benchmark_results.json')
    parser.add_argument('--comparison', action='store_true', help='Run classical comparison')
    parser.add_argument('--extended', action='store_true', help='Run extended benchmarks (v3.0.0)')

    args = parser.parse_args()

    if args.extended:
        suite = run_extended_benchmark(
            iterations=args.iterations,
            fhe_mode=args.fhe_mode,
            output_file=args.output,
        )
    elif args.comparison:
=======
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PQC-FHE Benchmarks')
    parser.add_argument('--iterations', '-n', type=int, default=100)
    parser.add_argument('--no-fhe', action='store_true', help='Skip FHE benchmarks')
    parser.add_argument('--fhe-mode', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--output', '-o', default='benchmark_results.json')
    parser.add_argument('--comparison', action='store_true', help='Run classical comparison')
    
    args = parser.parse_args()
    
    if args.comparison:
>>>>>>> origin/main
        suite = benchmark_classical_comparison(iterations=args.iterations)
    else:
        suite = run_full_benchmark(
            iterations=args.iterations,
            include_fhe=not args.no_fhe,
            fhe_mode=args.fhe_mode,
            output_file=args.output
        )
