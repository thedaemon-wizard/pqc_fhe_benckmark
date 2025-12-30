# Performance Benchmarks Guide

This guide provides comprehensive benchmarking methodology for the PQC-FHE Integration Library, including test procedures, expected results, and optimization strategies.

## Overview

Performance benchmarking is critical for:
- Capacity planning
- Algorithm selection
- Hardware requirements
- SLA definitions

---

## Benchmarking Framework

### Core Benchmark Suite

```python
"""
PQC-FHE Performance Benchmark Suite

Measures:
1. PQC operations (key generation, encapsulation, signing)
2. FHE operations (encryption, computation, bootstrapping)
3. End-to-end workflows
4. Concurrent operation throughput

Reference: NIST PQC Performance Benchmarks (2024)
"""

import time
import statistics
import json
import psutil
import platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

from pqc_fhe_integration import (
    PQCKeyManager,
    FHEEngine,
    HybridCryptoManager,
    SecurityLevel
)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    operation: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    ops_per_second: float
    memory_mb: float


@dataclass
class SystemInfo:
    """System information for benchmark context"""
    platform: str
    processor: str
    cpu_count: int
    memory_gb: float
    python_version: str


class PQCFHEBenchmark:
    """
    Comprehensive benchmark suite for PQC-FHE operations
    
    Methodology:
    - Warmup runs (discarded)
    - Multiple iterations for statistical significance
    - Percentile calculations (p50, p95, p99)
    - Memory profiling
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH,
                 warmup_iterations: int = 5,
                 benchmark_iterations: int = 100):
        self.security_level = security_level
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        # Initialize managers
        self.pqc_manager = PQCKeyManager(security_level=security_level)
        self.fhe_engine = FHEEngine()
        self.hybrid_manager = HybridCryptoManager(security_level=security_level)
        
        self.results: List[BenchmarkResult] = []
    
    def get_system_info(self) -> SystemInfo:
        """Collect system information"""
        return SystemInfo(
            platform=platform.platform(),
            processor=platform.processor(),
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            python_version=platform.python_version()
        )
    
    def _measure_operation(self, operation_func, iterations: int) -> List[float]:
        """Measure operation timing over multiple iterations"""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation_func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        return times
    
    def _calculate_stats(self, operation: str, times: List[float]) -> BenchmarkResult:
        """Calculate statistics from timing data"""
        memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        return BenchmarkResult(
            operation=operation,
            iterations=len(times),
            total_time_ms=sum(times),
            mean_time_ms=statistics.mean(times),
            median_time_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            min_time_ms=min(times),
            max_time_ms=max(times),
            p95_time_ms=np.percentile(times, 95),
            p99_time_ms=np.percentile(times, 99),
            ops_per_second=1000 / statistics.mean(times),
            memory_mb=memory
        )
    
    def benchmark_kem_keygen(self) -> BenchmarkResult:
        """Benchmark ML-KEM key generation"""
        # Warmup
        for _ in range(self.warmup_iterations):
            self.pqc_manager.generate_kem_keypair()
        
        # Benchmark
        times = self._measure_operation(
            self.pqc_manager.generate_kem_keypair,
            self.benchmark_iterations
        )
        
        result = self._calculate_stats("ML-KEM KeyGen", times)
        self.results.append(result)
        return result
    
    def benchmark_kem_encapsulate(self) -> BenchmarkResult:
        """Benchmark ML-KEM encapsulation"""
        public_key, _ = self.pqc_manager.generate_kem_keypair()
        
        def encap():
            self.pqc_manager.encapsulate(public_key)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            encap()
        
        # Benchmark
        times = self._measure_operation(encap, self.benchmark_iterations)
        
        result = self._calculate_stats("ML-KEM Encapsulate", times)
        self.results.append(result)
        return result
    
    def benchmark_kem_decapsulate(self) -> BenchmarkResult:
        """Benchmark ML-KEM decapsulation"""
        public_key, secret_key = self.pqc_manager.generate_kem_keypair()
        ciphertext, _ = self.pqc_manager.encapsulate(public_key)
        
        def decap():
            self.pqc_manager.decapsulate(ciphertext, secret_key)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            decap()
        
        # Benchmark
        times = self._measure_operation(decap, self.benchmark_iterations)
        
        result = self._calculate_stats("ML-KEM Decapsulate", times)
        self.results.append(result)
        return result
    
    def benchmark_sig_keygen(self) -> BenchmarkResult:
        """Benchmark ML-DSA key generation"""
        # Warmup
        for _ in range(self.warmup_iterations):
            self.pqc_manager.generate_sig_keypair()
        
        # Benchmark
        times = self._measure_operation(
            self.pqc_manager.generate_sig_keypair,
            self.benchmark_iterations
        )
        
        result = self._calculate_stats("ML-DSA KeyGen", times)
        self.results.append(result)
        return result
    
    def benchmark_sig_sign(self, message_size: int = 1024) -> BenchmarkResult:
        """Benchmark ML-DSA signing"""
        _, secret_key = self.pqc_manager.generate_sig_keypair()
        message = b"x" * message_size
        
        def sign():
            self.pqc_manager.sign(message, secret_key)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            sign()
        
        # Benchmark
        times = self._measure_operation(sign, self.benchmark_iterations)
        
        result = self._calculate_stats(f"ML-DSA Sign ({message_size}B)", times)
        self.results.append(result)
        return result
    
    def benchmark_sig_verify(self, message_size: int = 1024) -> BenchmarkResult:
        """Benchmark ML-DSA verification"""
        public_key, secret_key = self.pqc_manager.generate_sig_keypair()
        message = b"x" * message_size
        signature = self.pqc_manager.sign(message, secret_key)
        
        def verify():
            self.pqc_manager.verify(message, signature, public_key)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            verify()
        
        # Benchmark
        times = self._measure_operation(verify, self.benchmark_iterations)
        
        result = self._calculate_stats(f"ML-DSA Verify ({message_size}B)", times)
        self.results.append(result)
        return result
    
    def benchmark_fhe_encrypt(self, vector_size: int = 1000) -> BenchmarkResult:
        """Benchmark FHE encryption"""
        data = [float(i) for i in range(vector_size)]
        
        def encrypt():
            self.fhe_engine.encrypt(data)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            encrypt()
        
        # Benchmark
        times = self._measure_operation(encrypt, self.benchmark_iterations)
        
        result = self._calculate_stats(f"FHE Encrypt ({vector_size} floats)", times)
        self.results.append(result)
        return result
    
    def benchmark_fhe_add(self, vector_size: int = 1000) -> BenchmarkResult:
        """Benchmark FHE addition"""
        data1 = [float(i) for i in range(vector_size)]
        data2 = [float(i * 2) for i in range(vector_size)]
        ct1 = self.fhe_engine.encrypt(data1)
        ct2 = self.fhe_engine.encrypt(data2)
        
        def add():
            self.fhe_engine.add(ct1, ct2)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            add()
        
        # Benchmark
        times = self._measure_operation(add, self.benchmark_iterations)
        
        result = self._calculate_stats(f"FHE Add ({vector_size} floats)", times)
        self.results.append(result)
        return result
    
    def benchmark_fhe_multiply(self, vector_size: int = 1000) -> BenchmarkResult:
        """Benchmark FHE multiplication"""
        data1 = [float(i) for i in range(vector_size)]
        data2 = [float(i * 2) for i in range(vector_size)]
        ct1 = self.fhe_engine.encrypt(data1)
        ct2 = self.fhe_engine.encrypt(data2)
        
        def multiply():
            self.fhe_engine.multiply(ct1, ct2)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            multiply()
        
        # Benchmark
        times = self._measure_operation(multiply, self.benchmark_iterations)
        
        result = self._calculate_stats(f"FHE Multiply ({vector_size} floats)", times)
        self.results.append(result)
        return result
    
    def benchmark_fhe_decrypt(self, vector_size: int = 1000) -> BenchmarkResult:
        """Benchmark FHE decryption"""
        data = [float(i) for i in range(vector_size)]
        ct = self.fhe_engine.encrypt(data)
        
        def decrypt():
            self.fhe_engine.decrypt(ct)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            decrypt()
        
        # Benchmark
        times = self._measure_operation(decrypt, self.benchmark_iterations)
        
        result = self._calculate_stats(f"FHE Decrypt ({vector_size} floats)", times)
        self.results.append(result)
        return result
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("=" * 60)
        print("PQC-FHE Performance Benchmark Suite")
        print("=" * 60)
        
        system_info = self.get_system_info()
        print(f"\nSystem: {system_info.platform}")
        print(f"CPU: {system_info.processor} ({system_info.cpu_count} cores)")
        print(f"Memory: {system_info.memory_gb:.1f} GB")
        print(f"Python: {system_info.python_version}")
        print(f"\nSecurity Level: {self.security_level}")
        print(f"Warmup iterations: {self.warmup_iterations}")
        print(f"Benchmark iterations: {self.benchmark_iterations}")
        print("-" * 60)
        
        # PQC benchmarks
        print("\n[PQC Benchmarks]")
        
        result = self.benchmark_kem_keygen()
        print(f"ML-KEM KeyGen: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_kem_encapsulate()
        print(f"ML-KEM Encapsulate: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_kem_decapsulate()
        print(f"ML-KEM Decapsulate: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_sig_keygen()
        print(f"ML-DSA KeyGen: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_sig_sign()
        print(f"ML-DSA Sign: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_sig_verify()
        print(f"ML-DSA Verify: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        # FHE benchmarks
        print("\n[FHE Benchmarks]")
        
        result = self.benchmark_fhe_encrypt()
        print(f"FHE Encrypt: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_fhe_add()
        print(f"FHE Add: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_fhe_multiply()
        print(f"FHE Multiply: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        result = self.benchmark_fhe_decrypt()
        print(f"FHE Decrypt: {result.mean_time_ms:.3f} ms "
              f"(p99: {result.p99_time_ms:.3f} ms)")
        
        print("-" * 60)
        
        return {
            "system_info": asdict(system_info),
            "security_level": str(self.security_level),
            "results": [asdict(r) for r in self.results]
        }
    
    def export_results(self, filepath: str):
        """Export results to JSON file"""
        report = {
            "system_info": asdict(self.get_system_info()),
            "benchmark_config": {
                "security_level": str(self.security_level),
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Results exported to: {filepath}")
```

---

## Expected Performance Results

### Reference Benchmarks

Based on NIST PQC standardization process and independent benchmarks:

#### ML-KEM (FIPS 203) Performance

| Operation | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|-----------|------------|------------|-------------|
| KeyGen | ~0.02 ms | ~0.03 ms | ~0.05 ms |
| Encapsulate | ~0.03 ms | ~0.04 ms | ~0.06 ms |
| Decapsulate | ~0.03 ms | ~0.04 ms | ~0.06 ms |

*Reference: liboqs benchmarks on Intel Core i7 @ 3.6GHz*

#### ML-DSA (FIPS 204) Performance

| Operation | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
|-----------|-----------|-----------|-----------|
| KeyGen | ~0.05 ms | ~0.08 ms | ~0.13 ms |
| Sign | ~0.15 ms | ~0.25 ms | ~0.40 ms |
| Verify | ~0.05 ms | ~0.08 ms | ~0.13 ms |

*Reference: liboqs benchmarks on Intel Core i7 @ 3.6GHz*

#### FHE (CKKS) Performance

| Operation | 1K Elements | 10K Elements | 100K Elements |
|-----------|-------------|--------------|---------------|
| Encrypt | ~10 ms | ~50 ms | ~500 ms |
| Add | ~1 ms | ~5 ms | ~50 ms |
| Multiply | ~5 ms | ~25 ms | ~250 ms |
| Decrypt | ~5 ms | ~25 ms | ~250 ms |

*Reference: SEAL library benchmarks with poly_modulus_degree=8192*

---

## Throughput Testing

### Concurrent Operations

```python
class ThroughputBenchmark:
    """
    Measure throughput under concurrent load
    
    Tests:
    - Single-threaded baseline
    - Multi-threaded scaling
    - Connection pooling efficiency
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
    
    def benchmark_concurrent_kem(self, num_workers: int = 4,
                                  operations: int = 1000) -> Dict:
        """
        Benchmark concurrent KEM operations
        
        Measures:
        - Total throughput (ops/sec)
        - Per-worker throughput
        - Latency distribution under load
        """
        from concurrent.futures import ThreadPoolExecutor
        import queue
        
        results_queue = queue.Queue()
        
        def worker(worker_id: int, num_ops: int):
            manager = PQCKeyManager(security_level=self.security_level)
            public_key, secret_key = manager.generate_kem_keypair()
            
            latencies = []
            for _ in range(num_ops):
                start = time.perf_counter()
                ct, ss = manager.encapsulate(public_key)
                manager.decapsulate(ct, secret_key)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            results_queue.put({
                "worker_id": worker_id,
                "operations": num_ops,
                "latencies": latencies
            })
        
        ops_per_worker = operations // num_workers
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker, i, ops_per_worker)
                for i in range(num_workers)
            ]
            for f in futures:
                f.result()
        
        end_time = time.perf_counter()
        
        # Collect results
        all_latencies = []
        while not results_queue.empty():
            result = results_queue.get()
            all_latencies.extend(result["latencies"])
        
        total_time = end_time - start_time
        total_ops = len(all_latencies)
        
        return {
            "num_workers": num_workers,
            "total_operations": total_ops,
            "total_time_sec": total_time,
            "throughput_ops_sec": total_ops / total_time,
            "mean_latency_ms": statistics.mean(all_latencies),
            "p50_latency_ms": np.percentile(all_latencies, 50),
            "p95_latency_ms": np.percentile(all_latencies, 95),
            "p99_latency_ms": np.percentile(all_latencies, 99)
        }
    
    def benchmark_scaling(self, max_workers: int = 16,
                          operations_per_test: int = 500) -> List[Dict]:
        """
        Test throughput scaling with worker count
        
        Identifies:
        - Optimal worker count
        - Scaling efficiency
        - Bottlenecks
        """
        results = []
        
        for num_workers in [1, 2, 4, 8, 12, 16]:
            if num_workers > max_workers:
                break
            
            result = self.benchmark_concurrent_kem(
                num_workers=num_workers,
                operations=operations_per_test
            )
            results.append(result)
            
            print(f"Workers: {num_workers:2d} | "
                  f"Throughput: {result['throughput_ops_sec']:,.0f} ops/sec | "
                  f"p99 Latency: {result['p99_latency_ms']:.2f} ms")
        
        return results
```

---

## Memory Profiling

### Memory Usage Analysis

```python
import tracemalloc
from typing import Callable

class MemoryProfiler:
    """
    Profile memory usage of PQC-FHE operations
    
    Measures:
    - Peak memory allocation
    - Memory per operation
    - Memory leak detection
    """
    
    @staticmethod
    def profile_operation(operation: Callable, iterations: int = 10) -> Dict:
        """
        Profile memory usage of an operation
        
        Returns:
            Peak memory, average allocation, potential leaks
        """
        tracemalloc.start()
        
        snapshots = []
        for _ in range(iterations):
            operation()
            snapshot = tracemalloc.take_snapshot()
            snapshots.append(snapshot)
        
        tracemalloc.stop()
        
        # Analyze snapshots
        stats = []
        for snapshot in snapshots:
            top_stats = snapshot.statistics("lineno")
            total = sum(stat.size for stat in top_stats)
            stats.append(total)
        
        return {
            "iterations": iterations,
            "peak_memory_mb": max(stats) / (1024 * 1024),
            "mean_memory_mb": statistics.mean(stats) / (1024 * 1024),
            "memory_growth_mb": (stats[-1] - stats[0]) / (1024 * 1024),
            "potential_leak": (stats[-1] - stats[0]) > stats[0] * 0.1
        }
    
    @staticmethod
    def profile_fhe_operation_sizes(sizes: List[int]) -> Dict:
        """
        Profile FHE memory usage across different vector sizes
        
        Helps determine optimal batch sizes for memory constraints
        """
        fhe = FHEEngine()
        results = {}
        
        for size in sizes:
            data = [float(i) for i in range(size)]
            
            tracemalloc.start()
            ct = fhe.encrypt(data)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[size] = {
                "current_mb": current / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024),
                "bytes_per_element": peak / size
            }
        
        return results


# Usage example
def run_memory_profile():
    """Run memory profiling suite"""
    profiler = MemoryProfiler()
    
    # Profile PQC operations
    manager = PQCKeyManager()
    
    print("\n[Memory Profile: PQC Operations]")
    
    result = profiler.profile_operation(manager.generate_kem_keypair)
    print(f"ML-KEM KeyGen: {result['peak_memory_mb']:.2f} MB peak")
    
    # Profile FHE across sizes
    print("\n[Memory Profile: FHE Vector Sizes]")
    
    sizes = [100, 1000, 10000, 50000]
    results = profiler.profile_fhe_operation_sizes(sizes)
    
    for size, result in results.items():
        print(f"Size {size:6d}: {result['peak_mb']:.2f} MB "
              f"({result['bytes_per_element']:.0f} bytes/element)")
```

---

## Optimization Guidelines

### PQC Optimization

```python
# 1. Key Caching
# Cache key pairs for repeated operations within a session

class OptimizedPQCManager:
    """PQC manager with key caching for improved performance"""
    
    def __init__(self, cache_size: int = 100):
        self.manager = PQCKeyManager()
        self._kem_cache = {}  # key_id -> (public_key, secret_key)
        self._sig_cache = {}
        self.cache_size = cache_size
    
    def get_or_create_kem_keypair(self, key_id: str):
        """Get cached key pair or create new one"""
        if key_id not in self._kem_cache:
            if len(self._kem_cache) >= self.cache_size:
                # Evict oldest entry
                oldest = next(iter(self._kem_cache))
                del self._kem_cache[oldest]
            self._kem_cache[key_id] = self.manager.generate_kem_keypair()
        return self._kem_cache[key_id]


# 2. Batch Operations
# Process multiple operations in batches for better throughput

def batch_encapsulate(manager, public_keys: List[bytes]) -> List[tuple]:
    """Batch encapsulation for multiple recipients"""
    return [manager.encapsulate(pk) for pk in public_keys]


# 3. Parallel Verification
# Verify multiple signatures in parallel

def parallel_verify(manager, verifications: List[tuple],
                    max_workers: int = 4) -> List[bool]:
    """
    Parallel signature verification
    
    verifications: List of (message, signature, public_key) tuples
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(manager.verify, msg, sig, pk)
            for msg, sig, pk in verifications
        ]
        return [f.result() for f in futures]
```

### FHE Optimization

```python
# 1. SIMD Batching
# Pack multiple values into single ciphertext

class SIMDOptimizedFHE:
    """FHE with SIMD optimization"""
    
    def __init__(self, poly_modulus_degree: int = 8192):
        self.engine = FHEEngine(poly_modulus_degree=poly_modulus_degree)
        self.slot_count = poly_modulus_degree // 2
    
    def batch_encrypt(self, data_batches: List[List[float]]):
        """
        Encrypt multiple small vectors in one ciphertext
        
        Packs up to slot_count values per ciphertext
        """
        packed_data = []
        current_batch = []
        
        for batch in data_batches:
            if len(current_batch) + len(batch) <= self.slot_count:
                current_batch.extend(batch)
            else:
                if current_batch:
                    packed_data.append(current_batch)
                current_batch = batch.copy()
        
        if current_batch:
            packed_data.append(current_batch)
        
        return [self.engine.encrypt(data) for data in packed_data]


# 2. Depth-Aware Computation
# Track multiplicative depth to avoid noise overflow

class DepthAwareFHE:
    """FHE with multiplicative depth tracking"""
    
    def __init__(self, max_depth: int = 5):
        self.engine = FHEEngine()
        self.max_depth = max_depth
    
    def safe_multiply(self, ct1, ct2, current_depth: int) -> tuple:
        """
        Multiply with depth checking
        
        Returns: (result, new_depth)
        Raises: ValueError if max depth exceeded
        """
        new_depth = current_depth + 1
        if new_depth > self.max_depth:
            raise ValueError(
                f"Multiplicative depth {new_depth} exceeds max {self.max_depth}"
            )
        
        result = self.engine.multiply(ct1, ct2)
        return result, new_depth


# 3. Lazy Evaluation
# Defer decryption until necessary

class LazyFHE:
    """FHE with lazy evaluation for computation chaining"""
    
    def __init__(self):
        self.engine = FHEEngine()
        self._operations = []
    
    def add(self, ct1, ct2):
        """Queue addition"""
        self._operations.append(("add", ct1, ct2))
        return self
    
    def multiply(self, ct1, ct2):
        """Queue multiplication"""
        self._operations.append(("multiply", ct1, ct2))
        return self
    
    def evaluate(self, initial_ct):
        """Execute all queued operations"""
        result = initial_ct
        for op, arg1, arg2 in self._operations:
            if op == "add":
                result = self.engine.add(result, arg2)
            elif op == "multiply":
                result = self.engine.multiply(result, arg2)
        self._operations = []
        return result
```

---

## Running Benchmarks

### Quick Start

```bash
# Run basic benchmark
python -c "
from benchmarks import PQCFHEBenchmark
benchmark = PQCFHEBenchmark(benchmark_iterations=50)
results = benchmark.run_full_benchmark()
benchmark.export_results('benchmark_results.json')
"
```

### Full Benchmark Suite

```bash
# Run comprehensive benchmarks
pqc-fhe benchmark --iterations 100 --output benchmark_report.json

# Run specific benchmark
pqc-fhe benchmark --type pqc --security-level high

# Run throughput test
pqc-fhe benchmark --type throughput --workers 8 --duration 60
```

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[benchmark]"
      
      - name: Run benchmarks
        run: |
          python -c "
          from benchmarks import PQCFHEBenchmark
          b = PQCFHEBenchmark(benchmark_iterations=100)
          results = b.run_full_benchmark()
          b.export_results('benchmark_results.json')
          "
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_results.json
```

---

## References

1. **NIST PQC Performance** - Post-Quantum Cryptography: Digital Signature Schemes (Round 3 Benchmarks)
2. **liboqs Benchmarks** - https://openquantumsafe.org/benchmarking/
3. **Microsoft SEAL Performance** - https://github.com/microsoft/SEAL/wiki/Benchmarks
4. **Lattigo Benchmarks** - https://github.com/tuneinsight/lattigo
5. **DESILO FHE Documentation** - https://fhe.desilo.dev/latest/
