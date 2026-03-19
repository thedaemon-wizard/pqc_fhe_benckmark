#!/usr/bin/env python3
"""
Quantum Threat Simulator - Shor & Grover Algorithm Resource Estimation
======================================================================

This module provides quantum resource estimation for assessing threats
to classical cryptographic systems from quantum computers:

1. Shor's Algorithm: Multi-era resource estimation (2021-2026 models)
2. Grover's Algorithm: Brute-force key search for symmetric crypto

Resource Estimation Models (Shor's Algorithm for RSA-2048):
  - Gidney-Ekera 2021: ~20M physical qubits (surface code, 8 hours)
  - Chevignard 2024: ~4M physical qubits (sublinear resources)
  - Gidney 2025: ~1M physical qubits (magic state cultivation)
  - Pinnacle 2026: ~100K physical qubits (QLDPC codes)

References:
- Gidney & Ekera (2021): "How to factor 2048 bit RSA integers in 8 hours
  using 20 million noisy qubits", Quantum 5, 433
- Chevignard et al. (2024): "Reducing the Number of Qubits in Quantum
  Factoring", ePrint 2024/222 (CRYPTO 2025)
- Gidney (May 2025): "How to factor 2048 bit RSA integers with less than
  a million noisy qubits", arXiv:2505.15917
- Pinnacle Architecture (Feb 2026): "QLDPC codes for efficient quantum
  factoring", Quantum Computing Report
- NIST IR 8547: "Transition to Post-Quantum Cryptography Standards" (Nov 2024, draft)
- NIST SP 800-227: "Recommendations for Key-Encapsulation Mechanisms" (Sep 2025, final)
- NSA CNSA 2.0 (updated May 2025): 2030 full PQC migration deadline
- NIST SP 800-57 Rev. 5: Recommendation for Key Management
- IBM Quantum Roadmap 2026: Kookaburra (4,158 qubits), Cockatoo (2027),
  Starling (2029, 200 logical qubits), Blue Jay (2029+, 2000 logical qubits)
- IBM Relay-BP decoder (Nov 2025): real-time qLDPC decoding <480ns, 10x accuracy
- Quantinuum Helios (Nov 2025): 98 trapped-ion qubits, 94 logical qubits (GHZ)
- BDGL sieve optimality proof (Jan 2026): NNS paradigm proven optimal for lattice sieving
- Engelberts et al. (Oct 2025): quantum 3-tuple sieve exponent 0.2846 (ePrint 2025/2189)
- Overlattice sieve lower bound (Jan 2026): confirms BDGL optimality (MDPI Cryptography)
- Roetteler et al. (2017): "Quantum Resource Estimates for Computing
  Elliptic Curve Discrete Logarithms", ASIACRYPT 2017

Author: PQC-FHE Integration Library
License: MIT
Version: 3.2.0
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# SHOR RESOURCE ESTIMATION ERAS
# =============================================================================

class ShorResourceEra(str, Enum):
    """
    Represents different generations of Shor's algorithm resource estimates.

    Each era reflects advances in quantum error correction, magic state
    distillation, and circuit compilation that dramatically reduce the
    physical qubit requirements for RSA factoring.
    """
    GIDNEY_EKERA_2021 = "gidney_ekera_2021"
    CHEVIGNARD_2024 = "chevignard_2024"
    GIDNEY_2025 = "gidney_2025"
    PINNACLE_2026 = "pinnacle_2026"


# Multi-era Shor resource models for RSA-2048 factoring.
# Each model provides: physical qubits, logical qubits, estimated hours,
# error correction method, and source reference.
SHOR_RESOURCE_MODELS = {
    ShorResourceEra.GIDNEY_EKERA_2021: {
        'name': 'Gidney-Ekera 2021',
        'physical_qubits_rsa2048': 20_000_000,
        'logical_qubits_rsa2048': 4097,
        'estimated_hours_rsa2048': 8.0,
        'ec_method': 'Surface code (distance 27)',
        'ec_overhead': 1000,
        'key_innovation': 'Windowed arithmetic, optimized modular exponentiation',
        'reference': 'Gidney & Ekera, Quantum 5, 433 (2021)',
    },
    ShorResourceEra.CHEVIGNARD_2024: {
        'name': 'Chevignard 2024',
        'physical_qubits_rsa2048': 4_000_000,
        'logical_qubits_rsa2048': 1730,
        'estimated_hours_rsa2048': 12.0,
        'ec_method': 'Optimized surface code',
        'ec_overhead': 500,
        'key_innovation': 'Sublinear-depth modular multiplication',
        'reference': 'Chevignard et al., ePrint 2024/222 (CRYPTO 2025)',
    },
    ShorResourceEra.GIDNEY_2025: {
        'name': 'Gidney 2025 (Magic State Cultivation)',
        'physical_qubits_rsa2048': 1_000_000,
        'logical_qubits_rsa2048': 2048,
        'estimated_hours_rsa2048': 3.5,
        'ec_method': 'Magic state cultivation + QLDPC distillation',
        'ec_overhead': 250,
        'key_innovation': 'Magic state cultivation eliminates distillation factories',
        'reference': 'Gidney, arXiv:2505.15917 (May 2025)',
    },
    ShorResourceEra.PINNACLE_2026: {
        'name': 'Pinnacle Architecture 2026',
        'physical_qubits_rsa2048': 100_000,
        'logical_qubits_rsa2048': 1500,
        'estimated_hours_rsa2048': 1.0,
        'ec_method': 'QLDPC codes (bivariate bicycle)',
        'ec_overhead': 50,
        'key_innovation': 'QLDPC codes: 10-50x fewer physical qubits than surface code',
        'reference': 'Pinnacle Architecture Team, QC Report (Feb 2026)',
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QuantumResourceEstimate:
    """Resources needed for a quantum attack on a specific cryptosystem."""
    algorithm: str
    key_size_bits: int
    attack_type: str               # "shor_factoring", "shor_dlog", "grover_search"
    logical_qubits: int
    physical_qubits: int           # With error correction overhead
    t_gates: int
    circuit_depth: int
    estimated_runtime_hours: float
    quantum_speedup_factor: float
    classical_bits_security: int
    post_quantum_security: int     # Effective security against quantum
    threat_level: str              # "critical", "high", "moderate", "low"
    estimated_threat_year: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'key_size_bits': self.key_size_bits,
            'attack_type': self.attack_type,
            'logical_qubits': self.logical_qubits,
            'physical_qubits': self.physical_qubits,
            't_gates': self.t_gates,
            'circuit_depth': self.circuit_depth,
            'estimated_runtime_hours': round(self.estimated_runtime_hours, 2),
            'quantum_speedup_factor': round(self.quantum_speedup_factor, 2),
            'classical_bits_security': self.classical_bits_security,
            'post_quantum_security': self.post_quantum_security,
            'threat_level': self.threat_level,
            'estimated_threat_year': self.estimated_threat_year,
            'notes': self.notes,
        }


@dataclass
class SimulationStep:
    """A single step in a quantum algorithm simulation."""
    step_number: int
    description: str
    quantum_state: Dict[str, Any]
    probability: float
    cumulative_progress: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_number': self.step_number,
            'description': self.description,
            'quantum_state': self.quantum_state,
            'probability': round(self.probability, 6),
            'cumulative_progress': round(self.cumulative_progress, 6),
        }


# =============================================================================
# SHOR'S ALGORITHM SIMULATOR
# =============================================================================

class ShorSimulator:
    """
    Shor's algorithm resource estimator for RSA/ECC/DH factoring.

    Resource models:
    - RSA: Gidney & Ekera (2021): 2n+1 logical qubits for n-bit integer,
      ~0.3n^3 Toffoli gates. Physical qubits = logical * code_distance^2.
    - ECC: Roetteler et al. (2017): 2448n + O(n^(2/3)) logical qubits for
      n-bit curves, ~448n^3 log(n) T-gates.
    - DH: Similar to RSA factoring for finite field DH.
    """

    # Error correction overhead: physical qubits per logical qubit
    # Updated 2025 moderate estimate: ~500 physical per logical
    # (down from 1000 in 2021, due to improvements in magic state
    #  cultivation and QLDPC codes)
    # Ref: Gidney (2025), Pinnacle Architecture (2026)
    DEFAULT_EC_OVERHEAD = 500

    # Quantum gate speed assumptions (nanoseconds per gate cycle)
    GATE_CYCLE_NS = 100  # ~100ns for superconducting qubits (IBM, Google)

    def __init__(self, error_correction_overhead: int = DEFAULT_EC_OVERHEAD):
        self.ec_overhead = error_correction_overhead
        logger.info(
            "ShorSimulator initialized (EC overhead: %d physical/logical)",
            self.ec_overhead,
        )

    def estimate_rsa_resources(self, key_size: int) -> QuantumResourceEstimate:
        """
        Estimate quantum resources to factor an RSA modulus.

        Based on Gidney & Ekera (2021):
        - Logical qubits: 2n + 1 (optimized) where n = key_size
        - Toffoli gates: ~0.3 * n^3
        - Circuit depth: ~8 * n^3 / logical_qubits
        """
        n = key_size

        # Gidney-Ekera optimized estimates
        logical_qubits = 2 * n + 1
        physical_qubits = logical_qubits * self.ec_overhead
        toffoli_gates = int(0.3 * n ** 3)
        t_gates = toffoli_gates * 4  # Each Toffoli ~ 4 T-gates
        circuit_depth = int(8 * n ** 3 / logical_qubits)

        # Runtime estimate: assume 1 MHz logical clock rate
        logical_clock_hz = 1e6
        runtime_seconds = circuit_depth / logical_clock_hz
        runtime_hours = runtime_seconds / 3600

        # Classical security: RSA-n provides ~n/3 bits of security
        # (from GNFS complexity)
        classical_security = self._rsa_security_bits(n)
        # Post-quantum: Shor's makes it 0 (polynomial time)
        post_quantum_security = 0

        speedup = 2 ** (classical_security / 2)  # Exponential -> polynomial
        threat_year = self._estimate_threat_year_rsa(n)
        threat_level = self._classify_threat(threat_year)

        return QuantumResourceEstimate(
            algorithm=f"RSA-{key_size}",
            key_size_bits=key_size,
            attack_type="shor_factoring",
            logical_qubits=logical_qubits,
            physical_qubits=physical_qubits,
            t_gates=t_gates,
            circuit_depth=circuit_depth,
            estimated_runtime_hours=runtime_hours,
            quantum_speedup_factor=speedup,
            classical_bits_security=classical_security,
            post_quantum_security=post_quantum_security,
            threat_level=threat_level,
            estimated_threat_year=threat_year,
            notes=f"Gidney-Ekera (2021) optimized. "
                  f"EC overhead: {self.ec_overhead}:1. "
                  f"Assumes {logical_clock_hz/1e6:.0f} MHz logical clock.",
        )

    def estimate_ecc_resources(self, curve_bits: int) -> QuantumResourceEstimate:
        """
        Estimate quantum resources to compute ECC discrete logarithm.

        Based on Roetteler et al. (2017):
        - Logical qubits: 2448*n + O(n^(2/3)) for n-bit curve
        - T-gates: ~448*n^3 * log2(n)
        """
        n = curve_bits

        logical_qubits = int(2448 * n + 2 * n ** (2 / 3))
        physical_qubits = logical_qubits * self.ec_overhead
        t_gates = int(448 * n ** 3 * math.log2(n))
        circuit_depth = int(t_gates / logical_qubits)

        logical_clock_hz = 1e6
        runtime_hours = circuit_depth / logical_clock_hz / 3600

        # ECC-n provides ~n/2 bits of classical security
        classical_security = n // 2
        post_quantum_security = 0

        speedup = 2 ** (classical_security / 2)
        threat_year = self._estimate_threat_year_ecc(n)
        threat_level = self._classify_threat(threat_year)

        curve_name = {256: "P-256", 384: "P-384", 521: "P-521"}.get(n, f"ECC-{n}")

        return QuantumResourceEstimate(
            algorithm=f"ECC {curve_name}",
            key_size_bits=curve_bits,
            attack_type="shor_dlog",
            logical_qubits=logical_qubits,
            physical_qubits=physical_qubits,
            t_gates=t_gates,
            circuit_depth=circuit_depth,
            estimated_runtime_hours=runtime_hours,
            quantum_speedup_factor=speedup,
            classical_bits_security=classical_security,
            post_quantum_security=post_quantum_security,
            threat_level=threat_level,
            estimated_threat_year=threat_year,
            notes=f"Roetteler et al. (2017). Curve: {curve_name}.",
        )

    def estimate_dh_resources(self, group_bits: int) -> QuantumResourceEstimate:
        """
        Estimate quantum resources for Diffie-Hellman discrete log.
        Similar to RSA factoring (finite field DH).
        """
        est = self.estimate_rsa_resources(group_bits)
        est.algorithm = f"DH-{group_bits}"
        est.attack_type = "shor_dlog"
        est.notes = "Finite field DH. Same complexity class as RSA factoring."
        return est

    def estimate_rsa_resources_multi_era(
        self, key_size: int = 2048
    ) -> Dict[str, Any]:
        """
        Estimate RSA factoring resources across four generations of
        Shor's algorithm implementations (2021-2026).

        Returns a comparison showing the dramatic reduction in physical
        qubit requirements as quantum error correction and compilation
        techniques have improved.

        Args:
            key_size: RSA key size in bits (default: 2048)

        Returns:
            Dict with per-era estimates and comparison metadata
        """
        # Scaling factor for non-2048 key sizes
        # Physical qubits scale roughly as O(n^2 * polylog(n))
        scale = (key_size / 2048) ** 2

        results = {}
        for era, model in SHOR_RESOURCE_MODELS.items():
            phys_2048 = model['physical_qubits_rsa2048']
            logical_2048 = model['logical_qubits_rsa2048']
            hours_2048 = model['estimated_hours_rsa2048']

            phys_scaled = int(phys_2048 * scale)
            logical_scaled = int(logical_2048 * (key_size / 2048))
            # Runtime scales as O(n^3)
            hours_scaled = hours_2048 * (key_size / 2048) ** 3

            results[era.value] = {
                'era_name': model['name'],
                'key_size': key_size,
                'physical_qubits': phys_scaled,
                'logical_qubits': logical_scaled,
                'estimated_hours': round(hours_scaled, 2),
                'ec_method': model['ec_method'],
                'ec_overhead': model['ec_overhead'],
                'key_innovation': model['key_innovation'],
                'reference': model['reference'],
            }

        # Compute reduction ratios relative to the earliest model
        base_phys = results[ShorResourceEra.GIDNEY_EKERA_2021.value]['physical_qubits']
        for era_key, era_data in results.items():
            era_data['reduction_from_2021'] = round(
                base_phys / max(era_data['physical_qubits'], 1), 1
            )

        # Threat year estimates for each era
        threat_years = self._estimate_threat_years_multi_scenario(key_size)

        return {
            'key_size': key_size,
            'eras': results,
            'threat_year_scenarios': threat_years,
            'summary': {
                'qubits_2021': results[ShorResourceEra.GIDNEY_EKERA_2021.value]['physical_qubits'],
                'qubits_2026': results[ShorResourceEra.PINNACLE_2026.value]['physical_qubits'],
                'reduction_factor': round(
                    results[ShorResourceEra.GIDNEY_EKERA_2021.value]['physical_qubits'] /
                    max(results[ShorResourceEra.PINNACLE_2026.value]['physical_qubits'], 1),
                    1
                ),
                'note': (
                    f'Physical qubit requirements for RSA-{key_size} have decreased '
                    f'~200x from 2021 to 2026 estimates due to advances in QLDPC codes '
                    f'and magic state cultivation.'
                ),
            },
            'generated_at': datetime.now().isoformat(),
        }

    def _estimate_threat_years_multi_scenario(
        self, key_size: int
    ) -> Dict[str, Any]:
        """
        Estimate when RSA of given key size becomes breakable under
        optimistic, moderate, and conservative scenarios.
        """
        scenarios = {}
        for scenario_name, model in QuantumThreatTimeline.QPU_GROWTH_MODELS.items():
            base_qubits = model['base_physical_qubits']
            doubling = model['doubling_years']

            # Use the Gidney 2025 model as the "realistic near-future" target
            required_qubits_2025 = int(
                SHOR_RESOURCE_MODELS[ShorResourceEra.GIDNEY_2025][
                    'physical_qubits_rsa2048'
                ] * (key_size / 2048) ** 2
            )
            # Use Pinnacle 2026 model as the "optimistic" target
            required_qubits_2026 = int(
                SHOR_RESOURCE_MODELS[ShorResourceEra.PINNACLE_2026][
                    'physical_qubits_rsa2048'
                ] * (key_size / 2048) ** 2
            )

            # Years to reach each target
            if required_qubits_2026 <= base_qubits:
                year_optimistic = 2026
            else:
                years_needed = math.log2(required_qubits_2026 / base_qubits) * doubling
                year_optimistic = int(2026 + years_needed)

            if required_qubits_2025 <= base_qubits:
                year_moderate = 2026
            else:
                years_needed = math.log2(required_qubits_2025 / base_qubits) * doubling
                year_moderate = int(2026 + years_needed)

            scenarios[scenario_name] = {
                'threat_year_optimistic': year_optimistic,
                'threat_year_moderate': year_moderate,
                'required_qubits_gidney2025': required_qubits_2025,
                'required_qubits_pinnacle2026': required_qubits_2026,
                'qpu_doubling_years': doubling,
            }

        return scenarios

    def simulate_factoring_progress(
        self, n_bits: int, steps: int = 50
    ) -> List[SimulationStep]:
        """
        Educational simulation of Shor's algorithm progress.

        Shows the conceptual phases:
        1. Superposition creation (Hadamard gates)
        2. Modular exponentiation
        3. Quantum Fourier Transform
        4. Period finding via measurement
        5. Classical post-processing (GCD)
        """
        simulation = []
        total_phases = 5
        steps_per_phase = steps // total_phases

        phase_descriptions = [
            ("Superposition creation", "Applying Hadamard gates to create equal "
             "superposition of all values 0 to 2^n-1"),
            ("Modular exponentiation", "Computing a^x mod N in superposition, "
             "entangling input and output registers"),
            ("Quantum Fourier Transform", "Applying QFT to extract periodicity "
             "from the quantum state"),
            ("Period measurement", "Measuring the QFT output register to obtain "
             "an estimate of the period r"),
            ("Classical post-processing", "Using continued fractions to extract "
             "period r, then computing gcd(a^(r/2) +/- 1, N)"),
        ]

        cumulative = 0.0
        step_num = 0

        for phase_idx, (phase_name, phase_desc) in enumerate(phase_descriptions):
            phase_steps = steps_per_phase if phase_idx < total_phases - 1 else (
                steps - step_num
            )

            for i in range(phase_steps):
                progress_in_phase = (i + 1) / phase_steps
                cumulative = (phase_idx + progress_in_phase) / total_phases

                # Simulate probability evolution
                if phase_idx <= 1:
                    # Superposition and modular exp: uniform probability
                    prob = 1.0 / (2 ** min(n_bits, 20))
                elif phase_idx == 2:
                    # QFT: probability peaks form
                    peak_height = progress_in_phase ** 2
                    prob = peak_height / math.ceil(n_bits / 2)
                elif phase_idx == 3:
                    # Measurement: probability collapses
                    prob = 0.5 + 0.5 * progress_in_phase
                else:
                    # Classical: deterministic
                    prob = 1.0 if progress_in_phase > 0.5 else 0.9

                simulation.append(SimulationStep(
                    step_number=step_num,
                    description=f"Phase {phase_idx + 1}: {phase_name} - {phase_desc}",
                    quantum_state={
                        'phase': phase_name,
                        'phase_index': phase_idx + 1,
                        'phase_progress': round(progress_in_phase, 4),
                        'n_bits': n_bits,
                        'qubits_active': min(2 * n_bits + 1,
                                             int((phase_idx + progress_in_phase) * n_bits)),
                    },
                    probability=prob,
                    cumulative_progress=cumulative,
                ))
                step_num += 1

        return simulation

    def _rsa_security_bits(self, key_size: int) -> int:
        """Estimate classical security bits for RSA key size (GNFS complexity)."""
        # Approximation from NIST SP 800-57
        mapping = {
            1024: 80,
            2048: 112,
            3072: 128,
            4096: 152,
            7680: 192,
            15360: 256,
        }
        if key_size in mapping:
            return mapping[key_size]
        # Interpolate using L-notation
        ln_n = key_size * math.log(2)
        ln_ln_n = math.log(ln_n)
        security = 1.923 * (ln_n ** (1 / 3)) * (ln_ln_n ** (2 / 3))
        return int(security)

    def _estimate_threat_year_rsa(self, key_size: int) -> int:
        """
        Estimate when RSA of given size becomes breakable.

        Uses the Gidney 2025 resource model (~1M physical qubits for RSA-2048)
        as the baseline, with moderate QPU growth (doubling every 2 years).
        """
        # Gidney 2025: ~1M physical qubits for RSA-2048
        scale = (key_size / 2048) ** 2
        required_physical = int(
            SHOR_RESOURCE_MODELS[ShorResourceEra.GIDNEY_2025][
                'physical_qubits_rsa2048'
            ] * scale
        )
        current_qubits = 1500  # 2026 estimate
        if required_physical <= current_qubits:
            return 2026
        years_from_2026 = max(0, math.log2(required_physical / current_qubits) * 2)
        return int(2026 + years_from_2026)

    def _estimate_threat_year_ecc(self, curve_bits: int) -> int:
        """Estimate when ECC of given curve size becomes breakable."""
        # ECC requires significantly more qubits per bit than RSA
        required_physical = int(2448 * curve_bits) * self.ec_overhead
        current_qubits = 1500
        if required_physical <= current_qubits:
            return 2026
        years_from_2026 = max(0, math.log2(required_physical / current_qubits) * 2)
        return int(2026 + years_from_2026)

    def _classify_threat(self, threat_year: int) -> str:
        """Classify threat level based on estimated year."""
        current_year = 2026
        years_away = threat_year - current_year
        if years_away <= 5:
            return "critical"
        elif years_away <= 10:
            return "high"
        elif years_away <= 20:
            return "moderate"
        else:
            return "low"


# =============================================================================
# GROVER'S ALGORITHM SIMULATOR
# =============================================================================

class GroverSimulator:
    """
    Grover's algorithm resource estimator for symmetric crypto.

    Key results:
    - Quadratic speedup: O(sqrt(N)) vs O(N) classical
    - AES-128: effectively 64-bit security post-quantum
    - AES-256: effectively 128-bit security post-quantum
    - SHA-256 collision: effectively 128-bit security post-quantum

    References:
    - Grover (1996): "A fast quantum mechanical algorithm for database search"
    - Grassl et al. (2016): "Applying Grover's algorithm to AES"
    - NIST SP 800-57 Rev. 5: Post-quantum key size recommendations
    """

    # AES circuit resource estimates from Grassl et al. (2016)
    AES_QUBIT_OVERHEAD = {
        128: {'qubits': 2953, 't_depth': 2.15e10},
        192: {'qubits': 4449, 't_depth': 3.22e10},
        256: {'qubits': 6681, 't_depth': 4.29e10},
    }

    def __init__(self):
        logger.info("GroverSimulator initialized")

    def estimate_aes_resources(self, key_size: int) -> QuantumResourceEstimate:
        """
        Estimate quantum resources for Grover's search on AES.

        Grover iterations: ~pi/4 * sqrt(2^n) for n-bit key
        """
        if key_size not in (128, 192, 256):
            raise ValueError(f"AES key size must be 128, 192, or 256, got {key_size}")

        n = key_size
        grover_iterations = int(math.pi / 4 * math.sqrt(2 ** n))

        aes_info = self.AES_QUBIT_OVERHEAD.get(key_size, {
            'qubits': n * 20, 't_depth': n ** 3 * 1e6,
        })

        logical_qubits = aes_info['qubits']
        physical_qubits = logical_qubits * 1000  # Surface code overhead
        t_gates = int(aes_info['t_depth'] * grover_iterations)
        circuit_depth = int(aes_info['t_depth'] * grover_iterations / logical_qubits)

        # Runtime: Grover iterations * AES circuit depth
        logical_clock_hz = 1e6
        runtime_hours = circuit_depth / logical_clock_hz / 3600

        classical_security = key_size
        post_quantum_security = key_size // 2  # Grover halves effective key length

        speedup = 2 ** (key_size // 2)  # Quadratic speedup factor

        threat_year = self._estimate_aes_threat_year(key_size)
        threat_level = self._classify_threat_symmetric(key_size)

        return QuantumResourceEstimate(
            algorithm=f"AES-{key_size}",
            key_size_bits=key_size,
            attack_type="grover_search",
            logical_qubits=logical_qubits,
            physical_qubits=physical_qubits,
            t_gates=t_gates,
            circuit_depth=circuit_depth,
            estimated_runtime_hours=runtime_hours,
            quantum_speedup_factor=speedup,
            classical_bits_security=classical_security,
            post_quantum_security=post_quantum_security,
            threat_level=threat_level,
            estimated_threat_year=threat_year,
            notes=f"Grassl et al. (2016). Grover iterations: ~sqrt(2^{n}). "
                  f"Post-quantum effective security: {post_quantum_security} bits.",
        )

    def estimate_sha_resources(self, hash_size: int) -> QuantumResourceEstimate:
        """
        Estimate quantum resources for Grover's search on hash functions.

        Preimage attack: sqrt(2^n) iterations -> n/2 bit security
        Collision attack: Already O(sqrt(2^n)) classically (birthday bound),
            Grover gives O(2^(n/3)) via BHT algorithm.
        """
        if hash_size not in (256, 384, 512):
            raise ValueError(f"SHA hash size must be 256, 384, or 512, got {hash_size}")

        n = hash_size

        # Preimage resistance under Grover
        grover_iterations = int(math.pi / 4 * math.sqrt(2 ** min(n, 128)))
        # SHA circuit needs roughly n qubits for state + ancilla
        logical_qubits = n * 8
        physical_qubits = logical_qubits * 1000
        t_gates = int(n ** 2 * grover_iterations)
        circuit_depth = int(t_gates / logical_qubits)

        logical_clock_hz = 1e6
        runtime_hours = circuit_depth / logical_clock_hz / 3600

        classical_security = n  # Preimage resistance
        post_quantum_security = n // 2

        speedup = 2 ** (n // 2)
        threat_level = "low"  # SHA-256+ remains secure post-quantum

        return QuantumResourceEstimate(
            algorithm=f"SHA-{hash_size}",
            key_size_bits=hash_size,
            attack_type="grover_search",
            logical_qubits=logical_qubits,
            physical_qubits=physical_qubits,
            t_gates=t_gates,
            circuit_depth=circuit_depth,
            estimated_runtime_hours=runtime_hours,
            quantum_speedup_factor=speedup,
            classical_bits_security=classical_security,
            post_quantum_security=post_quantum_security,
            threat_level=threat_level,
            estimated_threat_year=2060,  # SHA-256 remains secure well beyond 2050
            notes=f"Grover preimage: {n}→{n // 2} bit security. "
                  f"Collision (BHT): {n // 2}→{n // 3} bit security. "
                  f"NIST recommends SHA-256+ remains adequate.",
        )

    def simulate_key_search(
        self, key_bits: int, steps: int = 50
    ) -> List[SimulationStep]:
        """
        Educational simulation of Grover's amplitude amplification.

        Shows probability amplitude evolution over Grover iterations.
        Optimal iterations: ~pi/4 * sqrt(N) where N = 2^key_bits.
        """
        # For visualization, use a small search space
        vis_bits = min(key_bits, 10)
        N = 2 ** vis_bits
        optimal_iterations = int(math.pi / 4 * math.sqrt(N))
        if optimal_iterations < 1:
            optimal_iterations = 1

        simulation = []

        for step in range(steps):
            # Map step to iteration progress
            iteration_frac = step / (steps - 1) if steps > 1 else 1.0
            current_iteration = iteration_frac * optimal_iterations

            # Grover probability formula:
            # P(target) = sin^2((2k+1) * theta)
            # where theta = arcsin(1/sqrt(N)), k = iteration number
            theta = math.asin(1.0 / math.sqrt(N))
            prob_target = math.sin((2 * current_iteration + 1) * theta) ** 2
            prob_non_target = (1 - prob_target) / max(1, (N - 1))

            simulation.append(SimulationStep(
                step_number=step,
                description=(
                    f"Grover iteration {current_iteration:.1f}/{optimal_iterations}: "
                    f"Amplitude amplification for {key_bits}-bit key search"
                ),
                quantum_state={
                    'iteration': round(current_iteration, 2),
                    'optimal_iterations': optimal_iterations,
                    'search_space_bits': key_bits,
                    'visualization_bits': vis_bits,
                    'target_probability': round(prob_target, 6),
                    'non_target_probability': round(prob_non_target, 8),
                    'amplitude_target': round(math.sqrt(prob_target), 6),
                    'amplitude_non_target': round(
                        math.sqrt(prob_non_target) if prob_non_target > 0 else 0, 6
                    ),
                },
                probability=prob_target,
                cumulative_progress=iteration_frac,
            ))

        return simulation

    def _estimate_aes_threat_year(self, key_size: int) -> int:
        """Estimate when AES becomes breakable via Grover."""
        # AES-128: ~2^64 Grover iterations needed, practically infeasible
        # Even with future quantum computers, AES-256 stays safe
        if key_size == 128:
            return 2050  # Marginal - recommended to migrate to AES-256
        elif key_size == 192:
            return 2070
        else:  # 256
            return 2080  # Effectively never with known algorithms

    def _classify_threat_symmetric(self, key_size: int) -> str:
        """Classify threat for symmetric algorithms."""
        if key_size <= 64:
            return "critical"
        elif key_size <= 128:
            return "moderate"  # Grover reduces to 64-bit, borderline
        else:
            return "low"  # AES-256 -> 128-bit effective, still secure


# =============================================================================
# QUANTUM THREAT TIMELINE
# =============================================================================

class QuantumThreatTimeline:
    """
    Project when quantum computers will threaten specific algorithms.

    Based on:
    - Quantum computing roadmaps (IBM, Google, IonQ, Quantinuum)
    - NIST IR 8547 deprecation timeline
    - NSA CNSA 2.0 migration schedule
    """

    # QPU growth models: (base_year, base_qubits, doubling_years)
    # Hardware milestones (as of March 2026):
    #   IBM: Nighthawk (120 qubits, square lattice, 2025),
    #        Loon (~112 qubits, c-couplers + qLDPC PoC, 2025),
    #        Kookaburra (1,386 qubits × 3 = 4,158 with qLDPC memory + LPU, 2026)
    #   IBM Relay-BP: real-time qLDPC decoding in <480ns (Nov 2025, 1 year ahead)
    #     - [[144,12,12]] bivariate bicycle code: 12 logical qubits in 144 data qubits
    #     - ~10x accuracy improvement over BP+OSD, FPGA-implemented
    #   IBM Roadmap: Cockatoo (2027), Starling (2029, 200 logical qubits),
    #                Blue Jay (2029+, 2000 logical qubits)
    #   Google: Willow (105 qubits, first verifiable quantum advantage Oct 2025)
    #   Quantinuum: H2 (56 qubits, QV 2^25), Helios (98 qubits, Nov 2025,
    #     94 logical qubits in GHZ state, better-than-break-even, $10B valuation)
    #
    # Quantum sieving update (Oct 2025):
    #   Engelberts et al. (ePrint 2025/2189): quantum 3-tuple sieve exponent
    #   reduced from 0.3098 to 0.2846 (quantum random walk + LSF).
    #   Classical BDGL 0.292 remains gold standard; BDGL optimality confirmed
    #   by overlattice lower bound (MDPI Cryptography, Jan 2026).
    QPU_GROWTH_MODELS = {
        'conservative': {
            'description': '1000 logical qubits by 2040',
            'base_year': 2026,
            'base_physical_qubits': 4200,  # IBM Kookaburra 2026
            'doubling_years': 3.0,
            'error_rate_improvement_per_year': 0.8,
        },
        'moderate': {
            'description': '1000 logical qubits by 2035',
            'base_year': 2026,
            'base_physical_qubits': 4200,  # IBM Kookaburra 2026
            'doubling_years': 2.0,
            'error_rate_improvement_per_year': 0.75,
        },
        'aggressive': {
            'description': '1000 logical qubits by 2030',
            'base_year': 2026,
            'base_physical_qubits': 4200,  # IBM Kookaburra 2026
            'doubling_years': 1.5,
            'error_rate_improvement_per_year': 0.7,
        },
    }

    # NIST IR 8547 migration timeline
    NIST_TIMELINE = {
        'deprecate_rsa_ecc': 2030,
        'disallow_rsa_ecc': 2035,
        'require_pqc': 2035,
    }

    # NSA CNSA 2.0 requirements
    CNSA_2_0 = {
        'ML-KEM-1024': {'deadline': 2030, 'use': 'Key establishment'},
        'ML-DSA-87': {'deadline': 2030, 'use': 'Digital signatures'},
        'SLH-DSA-SHA2-256f': {'deadline': 2030, 'use': 'Software/firmware signing'},
        'AES-256': {'deadline': 'immediate', 'use': 'Symmetric encryption'},
        'SHA-384': {'deadline': 'immediate', 'use': 'Hashing'},
    }

    def __init__(self, growth_model: str = 'moderate'):
        if growth_model not in self.QPU_GROWTH_MODELS:
            raise ValueError(
                f"growth_model must be one of {list(self.QPU_GROWTH_MODELS.keys())}"
            )
        self.growth_model = growth_model
        self.model_params = self.QPU_GROWTH_MODELS[growth_model]
        self._shor = ShorSimulator()
        self._grover = GroverSimulator()
        logger.info("QuantumThreatTimeline initialized (model: %s)", growth_model)

    def estimate_threat_year(self, algorithm: str, key_size: int) -> int:
        """Estimate the year when a specific algorithm/key-size becomes breakable."""
        algo_lower = algorithm.lower()

        if 'rsa' in algo_lower:
            est = self._shor.estimate_rsa_resources(key_size)
        elif 'ecc' in algo_lower or 'ecdsa' in algo_lower or 'ecdh' in algo_lower:
            est = self._shor.estimate_ecc_resources(key_size)
        elif 'dh' in algo_lower:
            est = self._shor.estimate_dh_resources(key_size)
        elif 'aes' in algo_lower:
            est = self._grover.estimate_aes_resources(key_size)
        elif 'sha' in algo_lower:
            est = self._grover.estimate_sha_resources(key_size)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return est.estimated_threat_year

    def generate_full_timeline(self) -> Dict[str, Any]:
        """Generate a comprehensive threat timeline for all major algorithms."""
        algorithms = [
            ('RSA', [1024, 2048, 3072, 4096]),
            ('ECC', [256, 384, 521]),
            ('DH', [2048, 3072]),
            ('AES', [128, 192, 256]),
            ('SHA', [256, 384, 512]),
        ]

        timeline = {}
        for algo_family, key_sizes in algorithms:
            family_timeline = []
            for key_size in key_sizes:
                try:
                    if algo_family == 'RSA':
                        est = self._shor.estimate_rsa_resources(key_size)
                    elif algo_family == 'ECC':
                        est = self._shor.estimate_ecc_resources(key_size)
                    elif algo_family == 'DH':
                        est = self._shor.estimate_dh_resources(key_size)
                    elif algo_family == 'AES':
                        est = self._grover.estimate_aes_resources(key_size)
                    elif algo_family == 'SHA':
                        est = self._grover.estimate_sha_resources(key_size)
                    else:
                        continue

                    family_timeline.append(est.to_dict())
                except Exception as e:
                    logger.warning("Failed to estimate %s-%d: %s", algo_family, key_size, e)

            timeline[algo_family] = family_timeline

        return {
            'timeline': timeline,
            'growth_model': self.growth_model,
            'model_description': self.model_params['description'],
            'nist_timeline': self.NIST_TIMELINE,
            'cnsa_2_0': {k: v for k, v in self.CNSA_2_0.items()},
            'generated_at': datetime.now().isoformat(),
        }

    def compare_classical_vs_pqc(self) -> Dict[str, Any]:
        """
        Compare classical algorithm vulnerability with PQC resilience.

        Shows why ML-KEM, ML-DSA are quantum-resistant while RSA, ECC are not.
        """
        classical_vulnerable = [
            self._shor.estimate_rsa_resources(2048).to_dict(),
            self._shor.estimate_rsa_resources(3072).to_dict(),
            self._shor.estimate_ecc_resources(256).to_dict(),
            self._shor.estimate_ecc_resources(384).to_dict(),
            self._shor.estimate_dh_resources(2048).to_dict(),
        ]

        # PQC algorithms: lattice-based, resistant to both Shor and Grover
        pqc_resistant = [
            {
                'algorithm': 'ML-KEM-512',
                'nist_level': 1,
                'classical_security': 128,
                'post_quantum_security': 128,
                'attack_type': 'none_known_polynomial',
                'best_known_attack': 'Core-SVP (2^118)',
                'threat_level': 'low',
                'estimated_threat_year': 'N/A (lattice-based)',
                'notes': 'NIST FIPS 203. Security from Module-LWE hardness.',
            },
            {
                'algorithm': 'ML-KEM-768',
                'nist_level': 3,
                'classical_security': 192,
                'post_quantum_security': 192,
                'attack_type': 'none_known_polynomial',
                'best_known_attack': 'Core-SVP (2^174)',
                'threat_level': 'low',
                'estimated_threat_year': 'N/A (lattice-based)',
                'notes': 'NIST FIPS 203 RECOMMENDED. CNSA 2.0 approved.',
            },
            {
                'algorithm': 'ML-KEM-1024',
                'nist_level': 5,
                'classical_security': 256,
                'post_quantum_security': 256,
                'attack_type': 'none_known_polynomial',
                'best_known_attack': 'Core-SVP (2^232)',
                'threat_level': 'low',
                'estimated_threat_year': 'N/A (lattice-based)',
                'notes': 'NIST FIPS 203. NSA CNSA 2.0 required by 2030.',
            },
            {
                'algorithm': 'ML-DSA-65',
                'nist_level': 3,
                'classical_security': 192,
                'post_quantum_security': 192,
                'attack_type': 'none_known_polynomial',
                'best_known_attack': 'Core-SVP (2^163)',
                'threat_level': 'low',
                'estimated_threat_year': 'N/A (lattice-based)',
                'notes': 'NIST FIPS 204 RECOMMENDED. Digital signatures.',
            },
            {
                'algorithm': 'SLH-DSA-SHA2-256f',
                'nist_level': 5,
                'classical_security': 256,
                'post_quantum_security': 256,
                'attack_type': 'none_known_polynomial',
                'best_known_attack': 'Hash function security',
                'threat_level': 'low',
                'estimated_threat_year': 'N/A (hash-based)',
                'notes': 'NIST FIPS 205. Stateless hash-based signatures.',
            },
        ]

        # FHE security: lattice-based (same foundation as ML-KEM)
        fhe_security = [
            {
                'scheme': 'CKKS',
                'security_assumption': 'Ring-LWE',
                'post_quantum_secure': True,
                'notes': 'Lattice-based. Security parameters must be chosen '
                         'carefully for quantum resistance.',
                'recommended_params': {
                    'log_n_min': 14,
                    'security_bits': [128, 192, 256],
                },
                'side_channel_risks': [
                    'NTT SPA: 98.6% key extraction from single trace (arXiv:2505.11058)',
                    'CPAD: Key recovery <1hr without smudging noise (CEA 2025)',
                    'Noise-Flooding: Non-worst-case estimation key recovery (PKC 2025)',
                ],
            },
            {
                'scheme': 'GL (Gentry-Lee)',
                'security_assumption': 'Ring-LWE',
                'post_quantum_secure': True,
                'notes': '5th generation FHE (ePrint 2025/1935). Native matrix '
                         'multiplication. Same Ring-LWE hardness as CKKS. '
                         'Announced FHE.org 2026 Taipei (March 7, 2026). '
                         'Google HEIR project investigating (Issue #2408).',
                'recommended_params': {
                    'shapes': ['(256,16,16)', '(16,256,256)', '(4,1024,1024)'],
                    'security_bits': [128],
                },
                'side_channel_risks': [
                    'NTT SPA: Shares CKKS NTT operation surface (arXiv:2505.11058)',
                    'CPAD: Applies to threshold GL variants (CEA 2025)',
                ],
                'advantages': [
                    'Native O(1) matrix multiplication (vs CKKS O(n) rotations)',
                    'Built-in transpose and conjugate transpose',
                    'Efficient for ML inference linear layers',
                ],
            },
            {
                'scheme': 'BFV',
                'security_assumption': 'Ring-LWE',
                'post_quantum_secure': True,
                'notes': 'Lattice-based integer scheme. Same security '
                         'foundation as CKKS.',
            },
        ]

        return {
            'classical_vulnerable': classical_vulnerable,
            'pqc_resistant': pqc_resistant,
            'fhe_security': fhe_security,
            'summary': {
                'shor_breaks': ['RSA', 'ECC (ECDSA, ECDH)', 'DH', 'DSA'],
                'grover_weakens': ['AES (halves effective key length)',
                                   'SHA (halves preimage resistance)'],
                'quantum_resistant': ['ML-KEM (lattice)', 'ML-DSA (lattice)',
                                      'SLH-DSA (hash)', 'CKKS/BFV/GL (lattice-FHE)'],
                'quantum_sieving_update': {
                    'classical_bdgl': '2^{0.292n} (optimal, confirmed Jan 2026)',
                    'quantum_3tuple_2025': '2^{0.2846n} (Engelberts et al., Oct 2025)',
                    'quantum_k2_sieve': '2^{0.265n} (theoretical best, high memory)',
                    'implication': 'ML-KEM/ML-DSA parameters remain secure but '
                                   'security margins narrowed. Monitor closely.',
                    'reference': 'ePrint 2025/2189, MDPI Cryptography Jan 2026',
                },
                'recommendation': 'Migrate to ML-KEM-768 + ML-DSA-65 per NIST IR 8547. '
                                  'Use AES-256 for symmetric encryption. '
                                  'Target completion by 2030 (CNSA 2.0). '
                                  'Monitor quantum sieving improvements (0.2846 exponent).',
            },
            'generated_at': datetime.now().isoformat(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about available growth models."""
        return {
            'current_model': self.growth_model,
            'available_models': {
                name: params['description']
                for name, params in self.QPU_GROWTH_MODELS.items()
            },
            'nist_timeline': self.NIST_TIMELINE,
            'cnsa_2_0_requirements': self.CNSA_2_0,
        }


# =============================================================================
# MODULE INFO
# =============================================================================

try:
    from .version_loader import get_version
    __version__ = get_version('quantum_threat_simulator')
except ImportError:
    __version__ = "3.2.0"
__author__ = "PQC-FHE Integration Library"


if __name__ == "__main__":
    print(f"Quantum Threat Simulator v{__version__}")
    print("=" * 60)

    # Demo: Shor's algorithm resource estimation
    shor = ShorSimulator()
    print("\n--- Shor's Algorithm: RSA Threat Assessment ---")
    for key_size in [2048, 3072, 4096]:
        est = shor.estimate_rsa_resources(key_size)
        print(f"\nRSA-{key_size}:")
        print(f"  Logical qubits: {est.logical_qubits:,}")
        print(f"  Physical qubits: {est.physical_qubits:,}")
        print(f"  Estimated runtime: {est.estimated_runtime_hours:.1f} hours")
        print(f"  Threat level: {est.threat_level}")
        print(f"  Estimated threat year: {est.estimated_threat_year}")

    print("\n--- Shor's Algorithm: ECC Threat Assessment ---")
    for curve in [256, 384, 521]:
        est = shor.estimate_ecc_resources(curve)
        print(f"\n{est.algorithm}:")
        print(f"  Logical qubits: {est.logical_qubits:,}")
        print(f"  Physical qubits: {est.physical_qubits:,}")
        print(f"  Threat level: {est.threat_level}")
        print(f"  Estimated threat year: {est.estimated_threat_year}")

    # Demo: Grover's algorithm
    grover = GroverSimulator()
    print("\n--- Grover's Algorithm: Symmetric Crypto Assessment ---")
    for key_size in [128, 192, 256]:
        est = grover.estimate_aes_resources(key_size)
        print(f"\nAES-{key_size}:")
        print(f"  Post-quantum security: {est.post_quantum_security} bits")
        print(f"  Threat level: {est.threat_level}")

    # Demo: Full timeline
    timeline = QuantumThreatTimeline('moderate')
    print("\n--- Quantum Threat Timeline (Moderate Model) ---")
    full = timeline.generate_full_timeline()
    for family, entries in full['timeline'].items():
        print(f"\n{family}:")
        for e in entries:
            print(f"  {e['algorithm']}: threat year {e['estimated_threat_year']}, "
                  f"level={e['threat_level']}")
