#!/usr/bin/env python3
"""
Side-Channel Risk Assessment for PQC Algorithms
================================================

Evaluates implementation-level side-channel vulnerabilities in NIST
post-quantum cryptographic algorithms (ML-KEM, ML-DSA, SLH-DSA).

While PQC algorithms are designed to resist quantum attacks, their
implementations may be vulnerable to classical side-channel attacks
(power analysis, timing, cache). This module assesses these risks
based on the latest published research (2024-2026).

Key vulnerabilities tracked:
- ML-KEM: Power analysis key recovery in 30s (Berzati et al. 2025)
- ML-KEM: EM fault injection 89.5% success on ARM (2025)
- ML-KEM: Timing attacks via KyberSlash (2024, patched)
- ML-DSA: Signing leakage via profiling attacks (2025)
- SLH-DSA: Generally low risk (stateless, hash-based)
- CKKS-FHE: Neural network side-channel 98.6% key extraction (arXiv:2505.11058)

References:
- Berzati, A. et al. (2025): "Simple Power Analysis on ML-KEM: Key
  Recovery in 30 Seconds on Cortex-M4", CHES 2025
- KyberSlash (2024): "Timing-based attacks on ML-KEM implementations",
  IACR ePrint 2024
- ML-DSA Signing Leakage (2025): "Profiling attack on ML-DSA vector
  leakage during signing", IACR ePrint 2025
- arXiv:2505.11058 (May 2025): "Neural Network Classifier Achieves 98.6%
  Accuracy Extracting CKKS Secret Key Coefficients from Single Power
  Measurement During NTT Operations"

Author: PQC-FHE Integration Library
License: MIT
Version: 3.2.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class RiskLevel(str, Enum):
    """Side-channel risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class AttackVector(str, Enum):
    """Types of side-channel attack vectors."""
    POWER_ANALYSIS = "power_analysis"
    TIMING = "timing"
    CACHE = "cache"
    ELECTROMAGNETIC = "electromagnetic"
    FAULT_INJECTION = "fault_injection"
    SIGNING_LEAKAGE = "signing_leakage"


@dataclass
class Vulnerability:
    """A known side-channel vulnerability."""
    cve_or_id: str
    attack_vector: AttackVector
    risk_level: RiskLevel
    description: str
    affected_operations: List[str]
    reference: str
    year: int
    patched: bool = False
    patch_info: str = ""
    recovery_time: str = ""  # e.g., "30 seconds"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.cve_or_id,
            'attack_vector': self.attack_vector.value,
            'risk_level': self.risk_level.value,
            'description': self.description,
            'affected_operations': self.affected_operations,
            'reference': self.reference,
            'year': self.year,
            'patched': self.patched,
            'patch_info': self.patch_info,
            'recovery_time': self.recovery_time,
        }


@dataclass
class AlgorithmRiskProfile:
    """Side-channel risk profile for a specific algorithm."""
    algorithm: str
    overall_risk: RiskLevel
    vulnerabilities: List[Vulnerability]
    mitigations: List[str]
    implementation_hardening: Dict[str, Any]
    assessment_date: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'overall_risk': self.overall_risk.value,
            'vulnerability_count': len(self.vulnerabilities),
            'critical_count': sum(
                1 for v in self.vulnerabilities
                if v.risk_level == RiskLevel.CRITICAL
            ),
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'mitigations': self.mitigations,
            'implementation_hardening': self.implementation_hardening,
            'assessment_date': self.assessment_date,
        }


# =============================================================================
# KNOWN VULNERABILITIES DATABASE
# =============================================================================

KNOWN_VULNERABILITIES: Dict[str, List[Vulnerability]] = {
    'ML-KEM': [
        Vulnerability(
            cve_or_id='CHES-2025-MLKEM-SPA',
            attack_vector=AttackVector.POWER_ANALYSIS,
            risk_level=RiskLevel.CRITICAL,
            description=(
                'Simple Power Analysis (SPA) on ML-KEM decapsulation '
                'enables full secret key recovery in 30 seconds on '
                'Cortex-M4 microcontroller. Exploits power consumption '
                'patterns during NTT and polynomial multiplication.'
            ),
            affected_operations=['decapsulation', 'ntt', 'polynomial_multiply'],
            reference='Berzati et al., CHES 2025',
            year=2025,
            patched=False,
            recovery_time='30 seconds',
        ),
        Vulnerability(
            cve_or_id='KyberSlash-1',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.HIGH,
            description=(
                'Timing side-channel in ML-KEM reference implementation. '
                'Division operations in compression/decompression have '
                'data-dependent timing, enabling key recovery via '
                'remote timing measurements.'
            ),
            affected_operations=['compression', 'decompression', 'encapsulation'],
            reference='KyberSlash, IACR ePrint 2024',
            year=2024,
            patched=True,
            patch_info='Fixed in pqcrystals-kyber commit 9b8d306. Use constant-time Barrett reduction.',
        ),
        Vulnerability(
            cve_or_id='KyberSlash-2',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.HIGH,
            description=(
                'Second timing vulnerability in ML-KEM. Modular reduction '
                'in NTT butterfly operations leaks information about '
                'intermediate values.'
            ),
            affected_operations=['ntt_butterfly', 'inverse_ntt'],
            reference='KyberSlash-2, IACR ePrint 2024',
            year=2024,
            patched=True,
            patch_info='Fixed in liboqs v0.10.0+. Constant-time NTT implementation.',
        ),
        Vulnerability(
            cve_or_id='MLKEM-CACHE-2024',
            attack_vector=AttackVector.CACHE,
            risk_level=RiskLevel.MEDIUM,
            description=(
                'Cache-timing attack on ML-KEM NTT implementation. '
                'Table lookups during twiddle factor access leak '
                'information about secret polynomials. Requires '
                'co-located attacker (shared cache).'
            ),
            affected_operations=['ntt', 'twiddle_factor_lookup'],
            reference='Cache side-channels in lattice KEM, 2024',
            year=2024,
            patched=True,
            patch_info='Mitigated by avoiding table lookups in NTT. Use in-register computation.',
        ),
        Vulnerability(
            cve_or_id='MLKEM-EM-FAULT-2025',
            attack_vector=AttackVector.FAULT_INJECTION,
            risk_level=RiskLevel.HIGH,
            description=(
                'Electromagnetic fault injection on ML-KEM decapsulation '
                'achieves 89.5% success rate on ARM Cortex-M4. Induces '
                'faults during NTT computation to bypass FO transform '
                'and recover shared secret. Requires physical proximity.'
            ),
            affected_operations=['decapsulation', 'ntt', 'fo_transform'],
            reference='EM fault injection on ML-KEM, IACR ePrint 2025',
            year=2025,
            patched=False,
            recovery_time='minutes (with EM equipment)',
        ),
    ],
    'ML-DSA': [
        Vulnerability(
            cve_or_id='MLDSA-SIGN-LEAK-2025',
            attack_vector=AttackVector.SIGNING_LEAKAGE,
            risk_level=RiskLevel.HIGH,
            description=(
                'Profiling attack on ML-DSA signing operation. '
                'Leakage during rejection sampling and polynomial '
                'vector operations allows recovery of signing key '
                'components with ~10,000 observed signatures.'
            ),
            affected_operations=['signing', 'rejection_sampling', 'vector_operations'],
            reference='ML-DSA Signing Leakage, IACR ePrint 2025',
            year=2025,
            patched=False,
            recovery_time='10,000 signatures observation',
        ),
        Vulnerability(
            cve_or_id='MLDSA-TIMING-2024',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.MEDIUM,
            description=(
                'Timing variation in ML-DSA rejection sampling loop. '
                'Number of iterations before valid signature is found '
                'leaks information about secret key. Less severe than '
                'direct key recovery.'
            ),
            affected_operations=['signing', 'rejection_sampling'],
            reference='Dilithium timing analysis, 2024',
            year=2024,
            patched=True,
            patch_info='Constant-time rejection with dummy iterations in liboqs v0.11.0+.',
        ),
    ],
    'SLH-DSA': [
        Vulnerability(
            cve_or_id='SLHDSA-FAULT-2024',
            attack_vector=AttackVector.FAULT_INJECTION,
            risk_level=RiskLevel.LOW,
            description=(
                'Theoretical fault injection attack on SLH-DSA WOTS+ '
                'chains. Inducing faults during signature generation '
                'could leak one-time signing key material. Requires '
                'physical access and precise fault timing.'
            ),
            affected_operations=['wots_chain', 'signature_generation'],
            reference='Hash-based signature fault analysis, 2024',
            year=2024,
            patched=False,
            patch_info='Inherent to WOTS+ design. Use randomized signing to mitigate.',
        ),
    ],
    'CKKS-FHE': [
        Vulnerability(
            cve_or_id='CKKS-NOISE-FLOODING-2024',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.MEDIUM,
            description=(
                'CKKS approximate arithmetic can leak information through '
                'decryption noise patterns. Insufficient noise flooding in '
                'decrypted outputs may reveal input characteristics. '
                'Multiparty CKKS (MPC-HE) has additional leakage through '
                'individual decryption shares.'
            ),
            affected_operations=['decryption', 'individual_decrypt', 'noise_analysis'],
            reference='Li & Micciancio, CRYPTO 2021; FHE noise flooding analysis 2024',
            year=2024,
            patched=False,
            patch_info='Apply sufficient noise flooding before decryption output.',
        ),
        Vulnerability(
            cve_or_id='CKKS-NTT-SPA-2025',
            attack_vector=AttackVector.POWER_ANALYSIS,
            risk_level=RiskLevel.CRITICAL,
            description=(
                'Neural network classifier achieves 98.6% accuracy in '
                'extracting SEAL CKKS secret key coefficients from a '
                'single power measurement during NTT operations. '
                'Demonstrated on Microsoft SEAL v4.1 (ARM Cortex-M4F). '
                'CRITICAL: Random delay insertion is INEFFECTIVE against '
                'single-trace attack. Masking alone is also bypassed. '
                'At -O3 optimization, guard and mul_root operations expose '
                'NEW leakage points. Combined masking+shuffling+constant-time '
                'NTT or hardware isolation (TEE/SGX) required.'
            ),
            affected_operations=['ntt', 'key_operations', 'encryption'],
            reference='arXiv:2505.11058 (ePrint 2025/867), Ghaleb & Buchanan, May 2025',
            year=2025,
            patched=False,
            recovery_time='single power trace',
        ),
        Vulnerability(
            cve_or_id='CKKS-CPAD-THRESHOLD-2025',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.CRITICAL,
            description=(
                'Threshold variants of BFV/BGV/CKKS are CPAD-insecure '
                'without proper smudging noise addition after partial '
                'decryption. Full key recovery achievable in under 1 hour '
                'on a laptop with a few thousand ciphertexts. Directly '
                'impacts MPC-HE protocols using individual_decrypt().'
            ),
            affected_operations=['individual_decrypt', 'multiparty_decrypt', 'threshold_decrypt'],
            reference='CEA France, Threshold FHE CPAD Attack, 2025 (cea.hal.science/cea-04706832)',
            year=2025,
            patched=False,
            recovery_time='< 1 hour on laptop',
        ),
        Vulnerability(
            cve_or_id='CKKS-NOISE-FLOOD-PKC-2025',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.HIGH,
            description=(
                'Novel key-recovery attack on CKKS when noise-flooding '
                'countermeasures use non-worst-case noise estimation. '
                'Quantifies precise tradeoffs between allowed decryptions '
                'before key refreshing, noise-flooding levels, and concrete '
                'security. Standard noise-flooding parameters may be '
                'insufficient for long-running deployments.'
            ),
            affected_operations=['decryption', 'noise_flooding', 'key_refresh'],
            reference='PKC 2025, Roros Norway, May 2025 (dl.acm.org/doi/10.1007/978-3-031-91832-2_4)',
            year=2025,
            patched=False,
            patch_info='Increase noise-flooding level or limit decryption count before key refresh.',
        ),
        Vulnerability(
            cve_or_id='CKKS-KEYGEN-LEAK-2023',
            attack_vector=AttackVector.POWER_ANALYSIS,
            risk_level=RiskLevel.MEDIUM,
            description=(
                'FHE key generation involves sampling Gaussian/ternary '
                'secrets for Ring-LWE. Power analysis during secret key '
                'generation on embedded devices may leak key material. '
                'Partially mitigated by server-side deployments, but '
                'elevated risk given NTT-based neural network attacks.'
            ),
            affected_operations=['key_generation', 'secret_sampling'],
            reference='FHE implementation security analysis, 2023; elevated per arXiv:2505.11058',
            year=2023,
            patched=False,
            patch_info='Server-side key generation reduces but does not eliminate risk.',
        ),
    ],
    'HQC': [
        Vulnerability(
            cve_or_id='HQC-TIMING-2024',
            attack_vector=AttackVector.TIMING,
            risk_level=RiskLevel.MEDIUM,
            description=(
                'Timing side-channel in HQC decapsulation due to '
                'variable-time syndrome decoding. Reed-Muller/Reed-Solomon '
                'decoders may exhibit data-dependent timing. '
                'Constant-time implementations available.'
            ),
            affected_operations=['decapsulation', 'syndrome_decoding'],
            reference='Code-based KEM timing analysis, IACR ePrint 2024',
            year=2024,
            patched=True,
            patch_info='Constant-time decoders in liboqs v0.12.0+.',
        ),
    ],
}


# =============================================================================
# MITIGATION RECOMMENDATIONS
# =============================================================================

MITIGATION_RECOMMENDATIONS: Dict[str, List[str]] = {
    'ML-KEM': [
        'Apply first-order masking to NTT and polynomial operations '
        '(prevents SPA key recovery)',
        'Use constant-time Barrett/Montgomery reduction '
        '(prevents timing attacks)',
        'Ensure NTT twiddle factors are computed in-register, not via table lookup '
        '(prevents cache attacks)',
        'Update to liboqs >= v0.12.0 or pqcrystals-kyber with KyberSlash patches',
        'Consider higher-order masking (2nd order) for high-security deployments',
        'Implement power analysis countermeasures: shuffling, blinding, '
        'randomized execution order',
        'EM fault injection countermeasures: redundant computation in NTT, '
        'hardware-level EM shielding for ARM Cortex-M4 deployments',
        'For embedded/IoT: use hardware AES co-processor for PRF/XOF operations',
        'Enable Address Space Layout Randomization (ASLR) to mitigate cache attacks',
    ],
    'ML-DSA': [
        'Implement constant-time rejection sampling with dummy iterations',
        'Apply masking to signing operations (at least first-order)',
        'Limit signature observation capability in deployment '
        '(rate limiting, access control)',
        'Use deterministic signing mode where possible (reduces leakage surface)',
        'Update to liboqs >= v0.11.0 with timing patches',
        'Monitor signing performance for anomalous timing patterns',
    ],
    'SLH-DSA': [
        'Use randomized signing (SPHINCS+ variant with randomness) '
        'to mitigate fault injection',
        'Implement redundant computation in WOTS+ chains for fault detection',
        'SLH-DSA has inherently low side-channel risk due to hash-based design',
        'No NTT or polynomial operations = no lattice-specific side channels',
        'Consider SLH-DSA-SHA2-128f for fastest verification with acceptable signatures',
    ],
    'CKKS-FHE': [
        'CRITICAL: Single countermeasures are INSUFFICIENT against NTT SPA — '
        'random delay insertion and masking alone are both bypassed by neural '
        'network single-trace attack (arXiv:2505.11058). At -O3 optimization, '
        'guard and mul_root operations expose new leakage points.',
        'CRITICAL: Deploy COMBINED countermeasures: masking + shuffling + '
        'constant-time NTT implementation. For server-side deployments, '
        'hardware isolation via TEE/SGX is strongly recommended.',
        'CRITICAL: Threshold FHE (MPC-HE) requires MANDATORY smudging noise '
        'addition after individual_decrypt() — CPAD attack recovers full key '
        'in < 1 hour without it (CEA 2025). Enforce smudging_noise_bits >= 40.',
        'HIGH: Increase noise-flooding levels beyond standard parameters — '
        'PKC 2025 shows non-worst-case noise estimation enables key recovery. '
        'Limit decryptions before key refresh (recommended: < 1000 per key).',
        'Apply sufficient noise flooding before exposing decrypted outputs',
        'Use server-side key generation with OS-level CSPRNG '
        '(reduces but does not eliminate power analysis risk)',
        'In MPC-HE: add independent noise to each party\'s decryption share '
        'to prevent leakage through individual_decrypt()',
        'Monitor FHE parameter selection (log_n, log Q) against HE Standard bounds '
        'to ensure adequate lattice security margin',
        'Avoid revealing intermediate ciphertext noise levels to untrusted parties',
        'For GPU-accelerated FHE: ensure constant-time NTT on GPU '
        '(timing variation less critical than CPU but not zero)',
        'Consider DESILO GL scheme (5th gen FHE, ePrint 2025/1935, announced '
        'FHE.org 2026 Taipei, March 7 2026) for improved security properties. '
        'Google HEIR project (Issue #2408) investigating GL scheme integration.',
    ],
    'HQC': [
        'Use constant-time syndrome decoding (available in liboqs v0.12.0+)',
        'Avoid variable-time Reed-Muller/Reed-Solomon decoders',
        'HQC has generally lower side-channel risk than lattice-based schemes',
        'No NTT operations — side-channel surface is smaller than ML-KEM',
        'Monitor NIST standardization process for implementation guidance',
        'Code-based algorithms provide algorithm diversity independent of lattice hardness',
    ],
}


# =============================================================================
# IMPLEMENTATION HARDENING STATUS
# =============================================================================

IMPLEMENTATION_HARDENING: Dict[str, Dict[str, Any]] = {
    'ML-KEM': {
        'liboqs': {
            'version': '0.14.0',
            'constant_time_ntt': True,
            'kyberslash_patched': True,
            'masking_support': False,
            'spa_protection': False,
            'notes': 'KyberSlash patches applied. SPA protection requires additional masking.',
        },
        'pqcrystals': {
            'version': 'ref-20240812',
            'constant_time_ntt': True,
            'kyberslash_patched': True,
            'masking_support': False,
            'spa_protection': False,
            'notes': 'Reference implementation. Timing fixed, SPA unprotected.',
        },
        'pqm4': {
            'version': 'v2.0',
            'constant_time_ntt': True,
            'kyberslash_patched': True,
            'masking_support': True,
            'spa_protection': True,
            'notes': 'ARM Cortex-M4 optimized. First-order masking available.',
        },
    },
    'ML-DSA': {
        'liboqs': {
            'version': '0.14.0',
            'constant_time_rejection': True,
            'signing_leakage_protected': False,
            'masking_support': False,
            'notes': 'Timing patches applied. Signing leakage not yet addressed.',
        },
        'pqcrystals': {
            'version': 'ref-20240812',
            'constant_time_rejection': True,
            'signing_leakage_protected': False,
            'masking_support': False,
            'notes': 'Reference implementation with constant-time rejection.',
        },
    },
    'SLH-DSA': {
        'liboqs': {
            'version': '0.14.0',
            'randomized_signing': True,
            'fault_detection': False,
            'notes': 'Low side-channel risk. Hash-based design is inherently resistant.',
        },
        'sphincs_plus_ref': {
            'version': '3.1',
            'randomized_signing': True,
            'fault_detection': False,
            'notes': 'Reference SPHINCS+ implementation. Randomized signing available.',
        },
    },
    'HQC': {
        'liboqs': {
            'version': '0.14.0',
            'constant_time_decoder': True,
            'timing_patched': True,
            'masking_support': False,
            'notes': 'Constant-time Reed-Muller decoder. Code-based design avoids lattice side-channels.',
        },
    },
    'CKKS-FHE': {
        'desilofhe': {
            'version': '2024.x',
            'noise_flooding': False,
            'constant_time_ntt': True,
            'gpu_timing_isolation': True,
            'ntt_spa_protection': False,
            'notes': ('DESILO FHE GPU-accelerated CKKS. NTT on GPU has less timing '
                      'variation than CPU. Noise flooding must be application-managed. '
                      'No NTT SPA protection (arXiv:2505.11058). '
                      'GL scheme (5th gen FHE, ePrint 2025/1935) in development.'),
        },
        'openfhe': {
            'version': '1.5.0',
            'noise_flooding': True,
            'constant_time_ntt': True,
            'gpu_timing_isolation': False,
            'ntt_spa_protection': False,
            'notes': ('OpenFHE v1.5.0 (Feb 2026): BFV, BGV, CKKS with bootstrapping '
                      'and scheme switching. Noise flooding support built-in. '
                      'No NTT SPA protection against neural network attacks.'),
        },
        'seal': {
            'version': '4.1.x',
            'noise_flooding': False,
            'constant_time_ntt': True,
            'gpu_timing_isolation': False,
            'ntt_spa_protection': False,
            'notes': ('Microsoft SEAL: Directly vulnerable to arXiv:2505.11058 '
                      'attack (98.6% secret key extraction from single NTT trace). '
                      'No SPA countermeasures in NTT implementation.'),
        },
    },
}


# =============================================================================
# MAIN ASSESSMENT CLASS
# =============================================================================

class SideChannelRiskAssessment:
    """
    Comprehensive side-channel risk assessment for NIST PQC algorithms.

    Evaluates known vulnerabilities, provides risk ratings, and
    recommends mitigations based on the latest research (2024-2026).
    """

    def __init__(self):
        logger.info("SideChannelRiskAssessment initialized")

    def assess_algorithm(self, algorithm: str) -> AlgorithmRiskProfile:
        """
        Assess side-channel risks for a specific algorithm family.

        Args:
            algorithm: One of 'ML-KEM', 'ML-DSA', 'SLH-DSA'

        Returns:
            AlgorithmRiskProfile with vulnerabilities and mitigations
        """
        algo_key = self._normalize_algorithm(algorithm)
        if algo_key not in KNOWN_VULNERABILITIES:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: ML-KEM, ML-DSA, SLH-DSA"
            )

        vulns = KNOWN_VULNERABILITIES[algo_key]
        mitigations = MITIGATION_RECOMMENDATIONS.get(algo_key, [])
        hardening = IMPLEMENTATION_HARDENING.get(algo_key, {})

        # Determine overall risk from highest unpatched vulnerability
        unpatched = [v for v in vulns if not v.patched]
        if unpatched:
            risk_levels = [v.risk_level for v in unpatched]
            if RiskLevel.CRITICAL in risk_levels:
                overall_risk = RiskLevel.CRITICAL
            elif RiskLevel.HIGH in risk_levels:
                overall_risk = RiskLevel.HIGH
            elif RiskLevel.MEDIUM in risk_levels:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW
        else:
            # All patched
            overall_risk = RiskLevel.LOW

        return AlgorithmRiskProfile(
            algorithm=algo_key,
            overall_risk=overall_risk,
            vulnerabilities=vulns,
            mitigations=mitigations,
            implementation_hardening=hardening,
            assessment_date=datetime.now().strftime('%Y-%m-%d'),
        )

    def assess_all(self) -> Dict[str, Any]:
        """
        Assess side-channel risks for all supported PQC algorithms.

        Returns:
            Dict with per-algorithm profiles and summary
        """
        results = {}
        for algo in ['ML-KEM', 'ML-DSA', 'SLH-DSA', 'HQC', 'CKKS-FHE']:
            profile = self.assess_algorithm(algo)
            results[algo] = profile.to_dict()

        # GL scheme inherits CKKS-FHE NTT side-channel surface
        results['GL-FHE'] = {
            'algorithm': 'GL-FHE (Gentry-Lee 5th Gen)',
            'vulnerability_count': 2,
            'critical_count': 1,
            'overall_risk': RiskLevel.HIGH.value,
            'notes': (
                'GL scheme (ePrint 2025/1935) uses Ring-LWE with NTT-based '
                'polynomial arithmetic. Shares CKKS NTT side-channel surface. '
                'Threshold GL variants subject to CPAD (CEA 2025). '
                'Currently less production-hardened than CKKS. '
                'Multiparty GL not yet supported in DESILO v1.10.0.'
            ),
            'inherited_risks': [
                'NTT SPA (arXiv:2505.11058): applies to GL NTT operations',
                'CPAD (CEA 2025): applies to threshold GL variants',
            ],
            'advantages': [
                'Native matrix multiplication reduces operation count',
                'Fewer rotation operations may reduce some side-channel exposure',
                'Newer design may incorporate lessons from CKKS vulnerabilities',
            ],
            'mitigations': [
                'Combined masking+shuffling+constant-time NTT (same as CKKS)',
                'Hardware isolation (TEE/SGX) for NTT-sensitive operations',
                'Mandatory smudging noise for threshold GL variants',
            ],
            'reference': 'ePrint 2025/1935, FHE.org 2026 Taipei',
        }

        # Summary statistics
        total_vulns = sum(
            r['vulnerability_count'] for r in results.values()
        )
        total_critical = sum(
            r['critical_count'] for r in results.values()
        )
        unpatched_count = sum(
            sum(1 for v in KNOWN_VULNERABILITIES.get(algo, []) if not v.patched)
            for algo in results
        )

        return {
            'assessments': results,
            'summary': {
                'total_vulnerabilities': total_vulns,
                'critical_vulnerabilities': total_critical,
                'unpatched_vulnerabilities': unpatched_count,
                'highest_risk_algorithm': min(
                    results.items(),
                    key=lambda x: list(RiskLevel).index(
                        RiskLevel(x[1]['overall_risk'])
                    )
                )[0],
                'assessment_date': datetime.now().strftime('%Y-%m-%d'),
                'key_findings': [
                    'ML-KEM has a CRITICAL unpatched SPA vulnerability '
                    '(Berzati et al. 2025: key recovery in 30 seconds)',
                    'ML-KEM has HIGH risk from EM fault injection '
                    '(89.5% success rate on ARM Cortex-M4, bypasses FO transform)',
                    'ML-DSA has HIGH risk from signing leakage '
                    '(~10,000 signatures needed for key recovery)',
                    'SLH-DSA has LOW side-channel risk '
                    '(hash-based design avoids lattice-specific attacks)',
                    'HQC has MEDIUM risk from timing in syndrome decoding '
                    '(patched in liboqs v0.12.0+, code-based design avoids NTT attacks)',
                    'CKKS-FHE has CRITICAL risk from neural network NTT power analysis: '
                    '98.6% accuracy from single trace (arXiv:2505.11058). '
                    'Random delay insertion and masking alone are INEFFECTIVE.',
                    'CKKS-FHE has CRITICAL risk from Threshold CPAD attack: '
                    'full key recovery in < 1 hour without smudging noise '
                    '(CEA 2025). MPC-HE individual_decrypt() directly affected.',
                    'CKKS-FHE has HIGH risk from noise-flooding key recovery '
                    '(PKC 2025): non-worst-case noise estimation enables attack. '
                    'Limit decryptions per key, increase noise-flooding level.',
                    'KyberSlash timing attacks (2024) are patched in liboqs v0.10.0+',
                    'First-order masking recommended for all ML-KEM deployments',
                    'FHE NTT operations require COMBINED countermeasures: '
                    'masking + shuffling + constant-time NTT + hardware isolation',
                    'DESILO GL scheme (5th gen FHE, FHE.org 2026 Taipei) may '
                    'improve security; Google HEIR project investigating',
                    'GL-FHE (ePrint 2025/1935) inherits CKKS NTT side-channel '
                    'surface; native matrix multiply reduces operation count '
                    'but NTT SPA and CPAD risks remain for threshold variants',
                ],
                'overall_recommendation': (
                    'URGENT: CKKS-FHE NTT SPA and Threshold CPAD vulnerabilities '
                    'require immediate attention. Deploy combined countermeasures '
                    '(masking+shuffling+constant-time+TEE) for NTT SPA. Enforce '
                    'mandatory smudging noise in MPC-HE individual_decrypt(). '
                    'For signatures, prefer SLH-DSA (minimal side-channel surface). '
                    'For ML-KEM, apply first-order masking and update to latest '
                    'liboqs. All implementations should be regularly audited '
                    'against emerging side-channel research (March 2026 update).'
                ),
            },
            'generated_at': datetime.now().isoformat(),
        }

    def get_mitigations(self, algorithm: str) -> List[str]:
        """Get recommended mitigations for an algorithm."""
        algo_key = self._normalize_algorithm(algorithm)
        return MITIGATION_RECOMMENDATIONS.get(algo_key, [])

    def check_implementation_hardening(
        self, algorithm: str, impl: str = 'liboqs'
    ) -> Dict[str, Any]:
        """
        Check hardening status of a specific implementation.

        Args:
            algorithm: Algorithm family ('ML-KEM', 'ML-DSA', 'SLH-DSA')
            impl: Implementation name ('liboqs', 'pqcrystals', 'pqm4', etc.)

        Returns:
            Dict with hardening status and recommendations
        """
        algo_key = self._normalize_algorithm(algorithm)
        hardening = IMPLEMENTATION_HARDENING.get(algo_key, {})

        if impl not in hardening:
            available = list(hardening.keys()) if hardening else []
            return {
                'algorithm': algo_key,
                'implementation': impl,
                'status': 'unknown',
                'available_implementations': available,
                'note': f"No hardening data for {impl}. Available: {available}",
            }

        impl_info = hardening[impl]
        vulnerabilities = KNOWN_VULNERABILITIES.get(algo_key, [])
        unpatched = [v for v in vulnerabilities if not v.patched]

        return {
            'algorithm': algo_key,
            'implementation': impl,
            'hardening': impl_info,
            'unpatched_vulnerabilities': [v.to_dict() for v in unpatched],
            'mitigations': MITIGATION_RECOMMENDATIONS.get(algo_key, []),
            'recommendation': self._generate_impl_recommendation(
                algo_key, impl, impl_info, unpatched
            ),
        }

    def _normalize_algorithm(self, algorithm: str) -> str:
        """Normalize algorithm name to canonical form."""
        algo_upper = algorithm.upper().replace('-', '_').replace(' ', '_')
        mapping = {
            'ML_KEM': 'ML-KEM',
            'MLKEM': 'ML-KEM',
            'ML_KEM_512': 'ML-KEM',
            'ML_KEM_768': 'ML-KEM',
            'ML_KEM_1024': 'ML-KEM',
            'KYBER': 'ML-KEM',
            'CRYSTALS_KYBER': 'ML-KEM',
            'ML_DSA': 'ML-DSA',
            'MLDSA': 'ML-DSA',
            'ML_DSA_44': 'ML-DSA',
            'ML_DSA_65': 'ML-DSA',
            'ML_DSA_87': 'ML-DSA',
            'DILITHIUM': 'ML-DSA',
            'CRYSTALS_DILITHIUM': 'ML-DSA',
            'SLH_DSA': 'SLH-DSA',
            'SLHDSA': 'SLH-DSA',
            'SLH_DSA_128S': 'SLH-DSA',
            'SLH_DSA_192S': 'SLH-DSA',
            'SLH_DSA_256S': 'SLH-DSA',
            'SPHINCS': 'SLH-DSA',
            'SPHINCS_PLUS': 'SLH-DSA',
            'HQC': 'HQC',
            'HQC_128': 'HQC',
            'HQC_192': 'HQC',
            'HQC_256': 'HQC',
            'CKKS': 'CKKS-FHE',
            'CKKS_FHE': 'CKKS-FHE',
            'BGV': 'CKKS-FHE',
            'BFV': 'CKKS-FHE',
            'FHE': 'CKKS-FHE',
            'DESILO': 'CKKS-FHE',
            'DESILOFHE': 'CKKS-FHE',
        }
        return mapping.get(algo_upper, algorithm)

    def _generate_impl_recommendation(
        self,
        algorithm: str,
        impl: str,
        impl_info: Dict[str, Any],
        unpatched: List[Vulnerability],
    ) -> str:
        """Generate implementation-specific recommendation."""
        if not unpatched:
            return f'{impl} for {algorithm}: All known vulnerabilities patched. Maintain update schedule.'

        critical = [v for v in unpatched if v.risk_level == RiskLevel.CRITICAL]
        if critical:
            vuln_desc = critical[0].description[:100]
            return (
                f'URGENT: {impl} for {algorithm} has {len(critical)} critical '
                f'unpatched vulnerability: {vuln_desc}... '
                f'Apply masking countermeasures immediately.'
            )

        return (
            f'{impl} for {algorithm}: {len(unpatched)} unpatched vulnerabilities. '
            f'Review mitigations and apply available patches.'
        )


    def verify_masking_deployment(
        self, algorithms: Optional[List[str]] = None, impl: str = 'liboqs'
    ) -> Dict[str, Any]:
        """
        Verify masking countermeasure status for deployed algorithms.

        Checks whether the specified implementation provides masking
        protection against power analysis and other side-channel attacks.

        Args:
            algorithms: List of algorithms to check (default: all)
            impl: Implementation library ('liboqs', 'pqcrystals', 'pqm4')

        Returns:
            Dict with per-algorithm masking status and recommendations
        """
        if algorithms is None:
            algorithms = ['ML-KEM', 'ML-DSA', 'SLH-DSA', 'HQC', 'CKKS-FHE']

        results = {}
        overall_masked = True

        for algo in algorithms:
            algo_key = self._normalize_algorithm(algo)
            hardening = IMPLEMENTATION_HARDENING.get(algo_key, {})
            impl_info = hardening.get(impl, {})

            has_masking = impl_info.get('masking_support', False)
            has_spa = impl_info.get('spa_protection', False)

            if algo_key in ('ML-KEM', 'ML-DSA') and not has_masking:
                overall_masked = False

            risk_if_unmasked = 'critical' if algo_key == 'ML-KEM' else (
                'high' if algo_key == 'ML-DSA' else 'low'
            )

            results[algo_key] = {
                'implementation': impl,
                'masking_support': has_masking,
                'spa_protection': has_spa,
                'risk_without_masking': risk_if_unmasked if not has_masking else 'mitigated',
                'recommendation': (
                    f'URGENT: Apply first-order masking to {algo_key}. '
                    f'Use pqm4 for ARM or implement custom masking for x86.'
                    if not has_masking and algo_key == 'ML-KEM' else
                    f'Apply masking to {algo_key} signing operations.'
                    if not has_masking and algo_key == 'ML-DSA' else
                    f'{algo_key} masking adequate for {impl}.'
                ),
            }

        return {
            'implementation': impl,
            'overall_masking_adequate': overall_masked,
            'algorithm_status': results,
            'critical_finding': (
                'ML-KEM on liboqs/pqcrystals lacks masking protection. '
                'SPA key recovery in 30 seconds is possible (Berzati et al. 2025). '
                'For production deployments requiring SPA resistance, '
                'use pqm4 (ARM Cortex-M4) or implement custom first-order masking.'
            ) if not overall_masked else None,
            'assessed_at': datetime.now().isoformat(),
        }


# =============================================================================
# MODULE VERSION
# =============================================================================

try:
    from .version_loader import get_version
    __version__ = get_version('side_channel_assessment')
except ImportError:
    __version__ = "3.2.0"
__author__ = "PQC-FHE Integration Library"
