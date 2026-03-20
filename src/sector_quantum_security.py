#!/usr/bin/env python3
"""
Sector-Specific PQC Security Countermeasure Simulator
======================================================

Integrates Shor's algorithm (RSA/ECC factoring) and Grover's algorithm
(symmetric key search) simulations with sector-specific parameters to
provide comprehensive quantum security assessments per industry sector.

Simulations provided per sector:
1. Shor vs RSA         — 4-generation Shor resource estimates vs current RSA/ECC keys
2. Shor vs Hybrid      — RSA+PQC hybrid migration strategy quantum resistance
3. Shor vs PQC Primary — ML-KEM/ML-DSA main deployment post-quantum security
4. Shor vs PQC Only    — Full PQC migration residual risk (lattice sieve attacks)
5. Grover vs AES-128   — Grover search impact on AES-128 effective security
6. Grover vs AES-256   — AES-256 quantum resistance confirmation
7. HNDL Simulation     — Harvest-Now-Decrypt-Later threat window per sector

References:
- NIST IR 8547: Transition to Post-Quantum Cryptography Standards
- NSA CNSA 2.0: Commercial National Security Algorithm Suite 2.0
- Gidney (2025): Magic state cultivation → RSA-2048 ~1M physical qubits
- Pinnacle Architecture (Feb 2026): QLDPC codes → ~100K physical qubits
- Dutch team (Oct 2025): Quantum sieve exponent 0.3098 → 0.2846
- Zhao & Ding (2025): BKZ improvements → 3-4 bit security reduction
- Berzati et al. (2025): ML-KEM SPA key recovery in 30 seconds
- CPAD impossibility (ePrint 2026/203): No BFV/BGV/CKKS IND-CPA^D security
- Q-Day median estimate: 2029-2032 (ECC falls before RSA)

Author: PQC-FHE Integration Library
License: MIT
Version: 3.2.0
"""

import logging
import math
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS FROM EXISTING MODULES
# =============================================================================

try:
    from src.quantum_threat_simulator import (
        ShorSimulator,
        GroverSimulator,
        QuantumThreatTimeline,
        ShorResourceEra,
        SHOR_RESOURCE_MODELS,
    )
except ImportError:
    from quantum_threat_simulator import (
        ShorSimulator,
        GroverSimulator,
        QuantumThreatTimeline,
        ShorResourceEra,
        SHOR_RESOURCE_MODELS,
    )

try:
    from src.side_channel_assessment import SideChannelRiskAssessment
except ImportError:
    try:
        from side_channel_assessment import SideChannelRiskAssessment
    except ImportError:
        SideChannelRiskAssessment = None
        logger.warning("SideChannelRiskAssessment not available")


# =============================================================================
# SECTOR QUANTUM PROFILES
# =============================================================================

VALID_SECTORS = ['healthcare', 'finance', 'blockchain', 'iot', 'mpc-fhe']

SECTOR_QUANTUM_PROFILES: Dict[str, Dict[str, Any]] = {
    'healthcare': {
        'name': 'Healthcare (HIPAA)',
        'data_retention_years': 50,
        'compliance_framework': 'HIPAA Security Rule',
        'compliance_deadline_year': 2030,
        'current_algorithms': {
            'key_exchange': ['RSA-2048'],
            'signatures': ['RSA-2048', 'ECDSA-256'],
            'symmetric': ['AES-128', 'AES-256'],
            'hashing': ['SHA-256'],
        },
        'pqc_targets': {
            'key_exchange': ['ML-KEM-768'],
            'signatures': ['ML-DSA-65'],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-384'],
        },
        'fhe_dependency': True,
        'fhe_scheme': 'CKKS (Ring-LWE)',
        'sndl_risk_base': 'CRITICAL',
        'side_channel_exposure': 'HIGH',
        'key_concern': (
            'Patient records harvested today decryptable in 3-6 years. '
            'FHE analytics on vital signs shares lattice risk with PQC key exchange.'
        ),
    },
    'finance': {
        'name': 'Finance (PCI-DSS / CNSA 2.0)',
        'data_retention_years': 7,
        'compliance_framework': 'PCI DSS v4.0, SOX, CNSA 2.0',
        'compliance_deadline_year': 2030,
        'current_algorithms': {
            'key_exchange': ['RSA-2048', 'RSA-4096'],
            'signatures': ['RSA-2048', 'ECDSA-256', 'ECDSA-384'],
            'symmetric': ['AES-128', 'AES-256'],
            'hashing': ['SHA-256', 'SHA-384'],
        },
        'pqc_targets': {
            'key_exchange': ['ML-KEM-768', 'ML-KEM-1024'],
            'signatures': ['ML-DSA-65', 'ML-DSA-87'],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-384'],
        },
        'fhe_dependency': True,
        'fhe_scheme': 'CKKS (Ring-LWE)',
        'sndl_risk_base': 'HIGH',
        'side_channel_exposure': 'MODERATE',
        'key_concern': (
            'CNSA 2.0 mandates full PQC migration by 2030 — 4 years remaining. '
            'High-value trade settlement requires NIST Level 5 (ML-DSA-87).'
        ),
    },
    'blockchain': {
        'name': 'Blockchain',
        'data_retention_years': 999,
        'compliance_framework': 'Post-Quantum Blockchain Standards',
        'compliance_deadline_year': 2028,
        'current_algorithms': {
            'key_exchange': [],
            'signatures': ['ECDSA-256', 'Ed25519'],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-256', 'SHA-3-256'],
        },
        'pqc_targets': {
            'key_exchange': [],
            'signatures': ['ML-DSA-65', 'ML-DSA-87'],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-384'],
        },
        'fhe_dependency': False,
        'fhe_scheme': 'N/A (signature-focused)',
        'sndl_risk_base': 'CRITICAL',
        'side_channel_exposure': 'LOW',
        'key_concern': (
            'Blockchain immutability means quantum-broken signatures cannot be revoked '
            'retroactively. ML-DSA-65 signatures (2-4KB) vs ECDSA (64B) impact throughput.'
        ),
    },
    'iot': {
        'name': 'IoT / Edge',
        'data_retention_years': 10,
        'compliance_framework': 'IoT Security (NIST SP 800-183)',
        'compliance_deadline_year': 2032,
        'current_algorithms': {
            'key_exchange': ['RSA-2048', 'ECDH-256'],
            'signatures': ['ECDSA-256'],
            'symmetric': ['AES-128'],
            'hashing': ['SHA-256'],
        },
        'pqc_targets': {
            'key_exchange': ['ML-KEM-512', 'ML-KEM-768'],
            'signatures': ['ML-DSA-44'],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-256'],
        },
        'fhe_dependency': False,
        'fhe_scheme': 'N/A or CKKS-Light',
        'sndl_risk_base': 'MODERATE',
        'side_channel_exposure': 'CRITICAL',
        'key_concern': (
            'SPA key recovery in 30 seconds on Cortex-M4 (Berzati 2025). '
            'Constrained devices cannot support large PQC key/signature sizes.'
        ),
    },
    'mpc-fhe': {
        'name': 'MPC-FHE Multi-Party Private Inference',
        'data_retention_years': 1,
        'compliance_framework': 'Secure Multi-Party Computation',
        'compliance_deadline_year': 2030,
        'current_algorithms': {
            'key_exchange': [],
            'signatures': [],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-256'],
        },
        'pqc_targets': {
            'key_exchange': ['CKKS Ring-LWE (log_n >= 16)'],
            'signatures': [],
            'symmetric': ['AES-256'],
            'hashing': ['SHA-384'],
        },
        'fhe_dependency': True,
        'fhe_scheme': 'CKKS (Ring-LWE) + GL Scheme (5th Gen)',
        'sndl_risk_base': 'LOW',
        'side_channel_exposure': 'HIGH',
        'key_concern': (
            'Lattice monoculture: ALL security rests on Ring-LWE assumption. '
            'CPAD impossibility (ePrint 2026/203) proves no BFV/BGV/CKKS achieves IND-CPA^D. '
            'NTT SPA achieves 98.6% key extraction (arXiv:2505.11058).'
        ),
    },
}


# =============================================================================
# MIGRATION STRATEGIES
# =============================================================================

MIGRATION_STRATEGIES: Dict[str, Dict[str, Any]] = {
    'rsa_only': {
        'name': 'Current RSA/ECC (No Migration)',
        'description': 'Continue using classical RSA/ECC without any PQC migration',
        'shor_vulnerable': True,
        'grover_impact': 'Applies to AES component',
        'security_model': 'Classical security only — fully broken by Shor',
        'deployment_complexity': 'None (status quo)',
        'nist_compliant_2035': False,
        'cnsa_2_0_compliant': False,
    },
    'hybrid': {
        'name': 'Hybrid: RSA+PQC (Transitional)',
        'description': 'Combine RSA/ECC with ML-KEM/ML-DSA for defense-in-depth',
        'shor_vulnerable': False,
        'grover_impact': 'Applies to AES component',
        'security_model': (
            'Security = max(classical, PQC). If either survives, communication is secure. '
            'RSA provides backward compatibility during transition.'
        ),
        'deployment_complexity': 'MODERATE — dual key management, larger handshakes',
        'nist_compliant_2035': True,
        'cnsa_2_0_compliant': True,
    },
    'pqc_primary': {
        'name': 'PQC Primary (ML-KEM/ML-DSA)',
        'description': 'Primary PQC with optional classical fallback',
        'shor_vulnerable': False,
        'grover_impact': 'Applies to AES component',
        'security_model': (
            'Lattice-based security (Module-LWE). Shor ineffective. '
            'Residual risk: quantum sieve improvements (0.257*beta, Dutch 2025) '
            'and BKZ advances (Zhao & Ding 2025, 3-4 bit reduction).'
        ),
        'deployment_complexity': 'HIGH — new key sizes, certificate updates, protocol changes',
        'nist_compliant_2035': True,
        'cnsa_2_0_compliant': True,
    },
    'pqc_only': {
        'name': 'PQC Only (Full Migration)',
        'description': 'Complete removal of all classical RSA/ECC cryptography',
        'shor_vulnerable': False,
        'grover_impact': 'Applies to AES component',
        'security_model': (
            'Fully quantum-resistant (lattice+hash-based). Shor completely neutralized. '
            'Residual risks: (1) Lattice monoculture if only ML-KEM/ML-DSA used, '
            '(2) Quantum sieve margin erosion, (3) CPAD for FHE applications.'
        ),
        'deployment_complexity': 'VERY HIGH — no backward compatibility, full ecosystem change',
        'nist_compliant_2035': True,
        'cnsa_2_0_compliant': True,
    },
}


# =============================================================================
# QUANTUM SIEVING CONSTANTS (2026 State of Art)
# =============================================================================

# Core-SVP exponents for lattice security estimation
QUANTUM_SIEVE_EXPONENT = 0.2846      # Dutch team (Oct 2025): quantum 3-tuple sieve
CLASSICAL_SIEVE_EXPONENT = 0.292      # BDGL (confirmed optimal, Jan 2026)
THEORETICAL_BEST_EXPONENT = 0.265     # K2-sieve (high memory, Laarhoven 2015)
BKZ_SECURITY_REDUCTION = 3.5          # Zhao & Ding (2025): practical BKZ improvement

# Q-Day estimates (2026 consensus)
Q_DAY_OPTIMISTIC = 2029    # Aggressive QPU growth
Q_DAY_MODERATE = 2031      # Moderate growth
Q_DAY_CONSERVATIVE = 2035  # Conservative estimate

# PQC security parameters (NIST post-quantum security levels)
PQC_SECURITY_BITS = {
    'ML-KEM-512': {'classical': 128, 'quantum': 128, 'nist_level': 1},
    'ML-KEM-768': {'classical': 192, 'quantum': 192, 'nist_level': 3},
    'ML-KEM-1024': {'classical': 256, 'quantum': 256, 'nist_level': 5},
    'ML-DSA-44': {'classical': 128, 'quantum': 128, 'nist_level': 2},
    'ML-DSA-65': {'classical': 192, 'quantum': 192, 'nist_level': 3},
    'ML-DSA-87': {'classical': 256, 'quantum': 256, 'nist_level': 5},
    'SLH-DSA-128f': {'classical': 128, 'quantum': 128, 'nist_level': 1},
    'SLH-DSA-192f': {'classical': 192, 'quantum': 192, 'nist_level': 3},
    'SLH-DSA-256f': {'classical': 256, 'quantum': 256, 'nist_level': 5},
}


# =============================================================================
# SECTOR QUANTUM SECURITY ASSESSOR
# =============================================================================

class SectorQuantumSecurityAssessor:
    """
    Comprehensive per-sector quantum security simulator.

    Combines Shor's algorithm (RSA/ECC factoring), Grover's algorithm
    (symmetric key search), and HNDL threat modeling with sector-specific
    parameters to produce actionable quantum security assessments.
    """

    def __init__(self, growth_model: str = 'moderate'):
        """
        Initialize with QPU growth model for threat timeline estimation.

        Args:
            growth_model: 'conservative', 'moderate', or 'aggressive'
        """
        self._shor = ShorSimulator()
        self._grover = GroverSimulator()
        self._timeline = QuantumThreatTimeline(growth_model)
        self._growth_model = growth_model
        self._side_channel = (
            SideChannelRiskAssessment() if SideChannelRiskAssessment else None
        )
        logger.info(
            "SectorQuantumSecurityAssessor initialized (growth_model=%s)",
            growth_model,
        )

    # ------------------------------------------------------------------
    # PUBLIC: Single-Sector Assessment
    # ------------------------------------------------------------------

    def assess_sector(self, sector: str) -> Dict[str, Any]:
        """
        Run all 7 simulations for a single sector.

        Returns dict with keys:
            sector_profile, shor_vs_rsa, shor_vs_hybrid,
            shor_vs_pqc_primary, shor_vs_pqc_only,
            grover_vs_aes128, grover_vs_aes256,
            hndl_analysis, migration_urgency, side_channel_risk,
            strategy_comparison, recommendations, generated_at
        """
        if sector not in SECTOR_QUANTUM_PROFILES:
            raise ValueError(
                f"Unknown sector: {sector}. Valid: {VALID_SECTORS}"
            )

        profile = SECTOR_QUANTUM_PROFILES[sector]

        shor_rsa = self._simulate_shor_vs_rsa(sector)
        shor_hybrid = self._simulate_shor_vs_hybrid(sector)
        shor_pqc_primary = self._simulate_shor_vs_pqc_primary(sector)
        shor_pqc_only = self._simulate_shor_vs_pqc_only(sector)
        grover_128 = self._simulate_grover_vs_aes(sector, 128)
        grover_256 = self._simulate_grover_vs_aes(sector, 256)
        hndl = self._simulate_hndl(sector)
        urgency = self._calculate_migration_urgency(sector)
        side_ch = self._assess_side_channel(sector)
        strategy_cmp = self._compare_strategies(
            sector, shor_rsa, shor_hybrid, shor_pqc_primary, shor_pqc_only
        )
        recs = self._generate_recommendations(sector, urgency, hndl, side_ch)

        return {
            'sector': sector,
            'sector_profile': {
                'name': profile['name'],
                'data_retention_years': profile['data_retention_years'],
                'compliance_framework': profile['compliance_framework'],
                'compliance_deadline_year': profile['compliance_deadline_year'],
                'fhe_dependency': profile['fhe_dependency'],
                'fhe_scheme': profile['fhe_scheme'],
                'key_concern': profile['key_concern'],
            },
            'shor_vs_rsa': shor_rsa,
            'shor_vs_hybrid': shor_hybrid,
            'shor_vs_pqc_primary': shor_pqc_primary,
            'shor_vs_pqc_only': shor_pqc_only,
            'grover_vs_aes128': grover_128,
            'grover_vs_aes256': grover_256,
            'hndl_analysis': hndl,
            'migration_urgency': urgency,
            'side_channel_risk': side_ch,
            'strategy_comparison': strategy_cmp,
            'recommendations': recs,
            'growth_model': self._growth_model,
            'generated_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # PUBLIC: All-Sector Comprehensive Assessment
    # ------------------------------------------------------------------

    def run_comprehensive_all_sectors(self) -> Dict[str, Any]:
        """
        Run assessments for all 5 sectors with cross-sector comparison.
        """
        sector_results = {}
        for sector in VALID_SECTORS:
            try:
                sector_results[sector] = self.assess_sector(sector)
            except Exception as e:
                logger.error("Sector %s assessment failed: %s", sector, e)
                sector_results[sector] = {'error': str(e)}

        # Cross-sector comparison: rank by migration urgency
        urgency_ranking = sorted(
            [
                (s, r.get('migration_urgency', {}).get('overall_score', 0))
                for s, r in sector_results.items()
                if 'error' not in r
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            'sectors': sector_results,
            'cross_sector_comparison': {
                'migration_urgency_ranking': [
                    {'sector': s, 'urgency_score': sc}
                    for s, sc in urgency_ranking
                ],
                'highest_urgency': urgency_ranking[0] if urgency_ranking else None,
                'hndl_critical_sectors': [
                    s for s, r in sector_results.items()
                    if r.get('hndl_analysis', {}).get('risk_level') == 'CRITICAL'
                    and 'error' not in r
                ],
                'shor_immediately_vulnerable': [
                    s for s, r in sector_results.items()
                    if r.get('shor_vs_rsa', {}).get('overall_verdict') == 'VULNERABLE'
                    and 'error' not in r
                ],
            },
            'summary': {
                'total_sectors': len(VALID_SECTORS),
                'assessed': len([r for r in sector_results.values() if 'error' not in r]),
                'growth_model': self._growth_model,
                'q_day_estimate': {
                    'optimistic': Q_DAY_OPTIMISTIC,
                    'moderate': Q_DAY_MODERATE,
                    'conservative': Q_DAY_CONSERVATIVE,
                },
            },
            'generated_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # PRIVATE: Shor vs RSA (Current Classical Algorithms)
    # ------------------------------------------------------------------

    def _simulate_shor_vs_rsa(self, sector: str) -> Dict[str, Any]:
        """
        Shor 4-generation resource estimates vs sector's current RSA/ECC keys.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]
        current_algos = profile['current_algorithms']
        results = []

        # RSA key exchange
        for algo in current_algos.get('key_exchange', []):
            if 'RSA' in algo.upper():
                key_size = int(algo.split('-')[1])
                multi_era = self._shor.estimate_rsa_resources_multi_era(key_size)
                est = self._shor.estimate_rsa_resources(key_size)
                results.append({
                    'algorithm': algo,
                    'type': 'key_exchange',
                    'shor_breaks': True,
                    'classical_security_bits': est.classical_bits_security,
                    'post_quantum_security_bits': 0,
                    'threat_year': est.estimated_threat_year,
                    'threat_level': est.threat_level,
                    'multi_era_qubits': {
                        era: data['physical_qubits']
                        for era, data in multi_era['eras'].items()
                    },
                    'multi_era_hours': {
                        era: data['estimated_hours']
                        for era, data in multi_era['eras'].items()
                    },
                    'threat_year_scenarios': multi_era.get('threat_year_scenarios', {}),
                })
            elif 'ECDH' in algo.upper() or 'ECC' in algo.upper():
                curve_bits = int(algo.split('-')[1])
                est = self._shor.estimate_ecc_resources(curve_bits)
                results.append({
                    'algorithm': algo,
                    'type': 'key_exchange',
                    'shor_breaks': True,
                    'classical_security_bits': est.classical_bits_security,
                    'post_quantum_security_bits': 0,
                    'threat_year': est.estimated_threat_year,
                    'threat_level': est.threat_level,
                })

        # ECC/RSA signatures
        for algo in current_algos.get('signatures', []):
            if 'RSA' in algo.upper():
                key_size = int(algo.split('-')[1])
                est = self._shor.estimate_rsa_resources(key_size)
                results.append({
                    'algorithm': algo,
                    'type': 'signature',
                    'shor_breaks': True,
                    'classical_security_bits': est.classical_bits_security,
                    'post_quantum_security_bits': 0,
                    'threat_year': est.estimated_threat_year,
                    'threat_level': est.threat_level,
                })
            elif 'ECDSA' in algo.upper() or 'Ed25519' in algo.lower():
                curve_bits = 256 if 'Ed25519' in algo else int(algo.split('-')[1])
                est = self._shor.estimate_ecc_resources(curve_bits)
                results.append({
                    'algorithm': algo,
                    'type': 'signature',
                    'shor_breaks': True,
                    'classical_security_bits': est.classical_bits_security,
                    'post_quantum_security_bits': 0,
                    'threat_year': est.estimated_threat_year,
                    'threat_level': est.threat_level,
                })

        # Determine earliest threat year
        threat_years = [r['threat_year'] for r in results if r.get('shor_breaks')]
        earliest_threat = min(threat_years) if threat_years else 9999

        return {
            'strategy': 'rsa_only',
            'strategy_name': MIGRATION_STRATEGIES['rsa_only']['name'],
            'algorithm_assessments': results,
            'earliest_threat_year': earliest_threat,
            'years_until_threat': earliest_threat - 2026,
            'overall_verdict': 'VULNERABLE' if results else 'N/A',
            'data_at_risk': (
                earliest_threat - 2026 < profile['data_retention_years']
            ) if results else False,
            'note': (
                f"All RSA/ECC algorithms broken by Shor's algorithm. "
                f"Earliest threat: {earliest_threat} ({earliest_threat - 2026} years). "
                f"Data retention: {profile['data_retention_years']} years."
            ),
        }

    # ------------------------------------------------------------------
    # PRIVATE: Shor vs Hybrid (RSA + PQC)
    # ------------------------------------------------------------------

    def _simulate_shor_vs_hybrid(self, sector: str) -> Dict[str, Any]:
        """
        Evaluate hybrid RSA+PQC migration strategy.
        Security = max(classical, PQC) — if either survives, data is protected.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]
        pqc_targets = profile['pqc_targets']

        # Hybrid KEM: RSA-2048 + ML-KEM-768
        hybrid_kem_assessments = []
        for ke_algo in profile['current_algorithms'].get('key_exchange', []):
            if 'RSA' in ke_algo.upper():
                key_size = int(ke_algo.split('-')[1])
                rsa_est = self._shor.estimate_rsa_resources(key_size)
                # Pair with first PQC KEM target
                pqc_kem = pqc_targets.get('key_exchange', ['ML-KEM-768'])[0]
                pqc_sec = PQC_SECURITY_BITS.get(pqc_kem, {})
                hybrid_kem_assessments.append({
                    'classical_component': ke_algo,
                    'pqc_component': pqc_kem,
                    'classical_broken_by_shor': True,
                    'pqc_shor_resistant': True,
                    'hybrid_security_bits': pqc_sec.get('quantum', 192),
                    'nist_level': pqc_sec.get('nist_level', 3),
                    'rsa_threat_year': rsa_est.estimated_threat_year,
                    'hybrid_survives_shor': True,
                    'note': (
                        f'RSA-{key_size} broken by Shor (year ~{rsa_est.estimated_threat_year}), '
                        f'but {pqc_kem} provides {pqc_sec.get("quantum", "?")}‑bit '
                        f'post-quantum security (NIST Level {pqc_sec.get("nist_level", "?")}).'
                    ),
                })

        # Hybrid signatures: ECDSA + ML-DSA
        hybrid_sig_assessments = []
        for sig_algo in profile['current_algorithms'].get('signatures', []):
            if 'ECDSA' in sig_algo.upper() or 'Ed25519' in sig_algo.lower():
                curve = 256 if 'Ed25519' in sig_algo else int(sig_algo.split('-')[1])
                ecc_est = self._shor.estimate_ecc_resources(curve)
                pqc_sig = pqc_targets.get('signatures', ['ML-DSA-65'])[0]
                pqc_sec = PQC_SECURITY_BITS.get(pqc_sig, {})
                hybrid_sig_assessments.append({
                    'classical_component': sig_algo,
                    'pqc_component': pqc_sig,
                    'classical_broken_by_shor': True,
                    'pqc_shor_resistant': True,
                    'hybrid_security_bits': pqc_sec.get('quantum', 192),
                    'nist_level': pqc_sec.get('nist_level', 3),
                    'ecc_threat_year': ecc_est.estimated_threat_year,
                    'hybrid_survives_shor': True,
                })

        combined_security = min(
            [a['hybrid_security_bits'] for a in hybrid_kem_assessments + hybrid_sig_assessments]
        ) if (hybrid_kem_assessments or hybrid_sig_assessments) else 0

        return {
            'strategy': 'hybrid',
            'strategy_name': MIGRATION_STRATEGIES['hybrid']['name'],
            'hybrid_kem': hybrid_kem_assessments,
            'hybrid_signatures': hybrid_sig_assessments,
            'deployment_complexity': 'MODERATE',
            'overall_shor_resistant': True,
            'effective_security_bits': combined_security,
            'transition_benefit': (
                'Backward compatibility maintained while gaining PQC protection. '
                'Recommended as first migration step per NIST CSWP 39.'
            ),
            'residual_risks': [
                'Larger TLS handshake sizes (hybrid key exchange ~1.5-2x)',
                'Increased computation for dual key operations',
                'Lattice-based PQC component subject to sieve improvements',
            ],
        }

    # ------------------------------------------------------------------
    # PRIVATE: Shor vs PQC Primary
    # ------------------------------------------------------------------

    def _simulate_shor_vs_pqc_primary(self, sector: str) -> Dict[str, Any]:
        """
        ML-KEM/ML-DSA as primary algorithms — Shor is ineffective.
        Assess residual risks from quantum sieving improvements.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]
        pqc_targets = profile['pqc_targets']

        assessments = []
        for category, algos in pqc_targets.items():
            for algo in algos:
                pqc_sec = PQC_SECURITY_BITS.get(algo)
                if pqc_sec:
                    # Quantum sieve impact: reduces effective security by BKZ_SECURITY_REDUCTION
                    effective_quantum_bits = pqc_sec['quantum'] - BKZ_SECURITY_REDUCTION
                    assessments.append({
                        'algorithm': algo,
                        'category': category,
                        'nominal_security_bits': pqc_sec['quantum'],
                        'effective_security_after_sieve': round(effective_quantum_bits, 1),
                        'nist_level': pqc_sec['nist_level'],
                        'shor_effective': False,
                        'quantum_sieve_impact': f'-{BKZ_SECURITY_REDUCTION} bits (Zhao & Ding 2025)',
                        'sieve_exponent': QUANTUM_SIEVE_EXPONENT,
                        'still_secure': effective_quantum_bits >= 100,
                    })

        min_security = min(
            [a['effective_security_after_sieve'] for a in assessments]
        ) if assessments else 0

        return {
            'strategy': 'pqc_primary',
            'strategy_name': MIGRATION_STRATEGIES['pqc_primary']['name'],
            'pqc_assessments': assessments,
            'shor_completely_neutralized': True,
            'effective_minimum_security_bits': round(min_security, 1),
            'quantum_sieve_impact': {
                'exponent': QUANTUM_SIEVE_EXPONENT,
                'source': 'Dutch team (van Hoof et al.), ePrint Oct 2025',
                'bkz_reduction': f'{BKZ_SECURITY_REDUCTION} bits (Zhao & Ding 2025)',
                'impact_summary': (
                    f'Quantum sieve improvements reduce lattice security by '
                    f'~{BKZ_SECURITY_REDUCTION} bits, but all NIST-approved PQC '
                    f'algorithms retain ≥{round(min_security, 1)}-bit effective security.'
                ),
            },
            'residual_risks': [
                f'Quantum sieve margin erosion: -{BKZ_SECURITY_REDUCTION} bits (ongoing research)',
                'All PQC KEM/signature algorithms share lattice assumption (Module-LWE)',
                'ML-DSA: 6+ independent attack papers in 2025-2026 (expanding attack surface)',
            ],
        }

    # ------------------------------------------------------------------
    # PRIVATE: Shor vs PQC Only (Full Migration)
    # ------------------------------------------------------------------

    def _simulate_shor_vs_pqc_only(self, sector: str) -> Dict[str, Any]:
        """
        Full PQC migration: no classical RSA/ECC remains.
        Focus on lattice monoculture risk and FHE-specific concerns.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]

        # Lattice monoculture analysis
        pqc_families_used = set()
        for algos in profile['pqc_targets'].values():
            for algo in algos:
                if 'ML-KEM' in algo or 'ML-DSA' in algo:
                    pqc_families_used.add('lattice')
                elif 'SLH-DSA' in algo:
                    pqc_families_used.add('hash')
                elif 'HQC' in algo:
                    pqc_families_used.add('code')
                elif 'CKKS' in algo or 'Ring-LWE' in algo:
                    pqc_families_used.add('lattice')

        monoculture_risk = 'HIGH' if len(pqc_families_used) <= 1 else (
            'MODERATE' if len(pqc_families_used) == 2 else 'LOW'
        )

        # FHE-specific risks
        fhe_risks = []
        if profile['fhe_dependency']:
            fhe_risks = [
                {
                    'risk': 'CPAD Impossibility',
                    'severity': 'CRITICAL',
                    'detail': (
                        'ePrint 2026/203 proves NO BFV/BGV/CKKS variant achieves '
                        'IND-CPA^D security. Noise-flooding mitigation adds 40-60% overhead.'
                    ),
                },
                {
                    'risk': 'Shared Lattice Assumption',
                    'severity': 'HIGH',
                    'detail': (
                        'FHE (Ring-LWE) and PQC (Module-LWE) share the same lattice hardness '
                        'assumption. A single lattice breakthrough compromises BOTH.'
                    ),
                },
                {
                    'risk': 'NTT Side-Channel (SPA)',
                    'severity': 'CRITICAL',
                    'detail': (
                        'arXiv:2505.11058: 98.6% key extraction via NTT power analysis. '
                        'Applies to all NTT-based schemes including CKKS and GL.'
                    ),
                },
            ]

        return {
            'strategy': 'pqc_only',
            'strategy_name': MIGRATION_STRATEGIES['pqc_only']['name'],
            'shor_completely_neutralized': True,
            'classical_algorithms_removed': True,
            'pqc_families_deployed': sorted(list(pqc_families_used)),
            'family_count': len(pqc_families_used),
            'lattice_monoculture_risk': monoculture_risk,
            'fhe_specific_risks': fhe_risks,
            'residual_risks': [
                f'Lattice monoculture: {len(pqc_families_used)} PQC family(ies) deployed',
                f'Quantum sieve margin: -{BKZ_SECURITY_REDUCTION} bits from 2025 advances',
                'No backward compatibility — interop requires all parties to support PQC',
            ] + ([
                'FHE shares lattice assumption with PQC — single point of failure'
            ] if profile['fhe_dependency'] else []),
            'diversification_recommendation': (
                'Deploy SLH-DSA (hash-based) for software signing and '
                'HQC (code-based, NIST Round 4) for backup KEM to reduce lattice monoculture.'
            ),
        }

    # ------------------------------------------------------------------
    # PRIVATE: Grover vs AES
    # ------------------------------------------------------------------

    def _simulate_grover_vs_aes(
        self, sector: str, key_size: int
    ) -> Dict[str, Any]:
        """
        Grover's algorithm impact on AES-128 or AES-256 per sector.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]
        est = self._grover.estimate_aes_resources(key_size)

        effective_security = key_size // 2  # Grover halves effective key length
        meets_sector_requirement = True

        # Sector-specific AES adequacy check
        if sector in ('healthcare', 'finance') and key_size == 128:
            meets_sector_requirement = False
            sector_note = (
                f'{profile["name"]}: AES-128 provides only {effective_security}-bit '
                f'post-quantum security — insufficient for {profile["compliance_framework"]}. '
                f'CNSA 2.0 requires AES-256.'
            )
        elif key_size == 256:
            sector_note = (
                f'{profile["name"]}: AES-256 provides {effective_security}-bit '
                f'post-quantum security — adequate for all compliance frameworks.'
            )
        else:
            sector_note = (
                f'{profile["name"]}: AES-{key_size} provides {effective_security}-bit '
                f'post-quantum security.'
            )

        # Check if sector currently uses this AES variant
        current_symmetric = profile['current_algorithms'].get('symmetric', [])
        currently_deployed = f'AES-{key_size}' in current_symmetric

        return {
            'algorithm': f'AES-{key_size}',
            'classical_security_bits': key_size,
            'post_quantum_security_bits': effective_security,
            'grover_speedup': f'2^{key_size // 2} (quadratic)',
            'physical_qubits_required': est.physical_qubits,
            'logical_qubits_required': est.logical_qubits,
            'estimated_runtime_hours': est.estimated_runtime_hours,
            'threat_year': est.estimated_threat_year,
            'threat_level': est.threat_level,
            'currently_deployed': currently_deployed,
            'meets_sector_requirement': meets_sector_requirement,
            'cnsa_2_0_compliant': key_size >= 256,
            'sector_note': sector_note,
            'recommendation': (
                'Upgrade to AES-256' if key_size < 256
                else 'AES-256 is quantum-safe — no action needed'
            ),
        }

    # ------------------------------------------------------------------
    # PRIVATE: HNDL (Harvest-Now-Decrypt-Later) Simulation
    # ------------------------------------------------------------------

    def _simulate_hndl(self, sector: str) -> Dict[str, Any]:
        """
        Calculate HNDL threat window per sector.

        HNDL window = data_retention_years - years_until_CRQC
        If positive, data harvested TODAY will still be sensitive when
        quantum computers can decrypt it.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]
        retention = profile['data_retention_years']

        # Years until CRQC for each scenario
        scenarios = {
            'optimistic': Q_DAY_OPTIMISTIC - 2026,
            'moderate': Q_DAY_MODERATE - 2026,
            'conservative': Q_DAY_CONSERVATIVE - 2026,
        }

        hndl_windows = {}
        for scenario, years_to_crqc in scenarios.items():
            window = retention - years_to_crqc
            hndl_windows[scenario] = {
                'years_until_crqc': years_to_crqc,
                'data_retention_years': retention,
                'hndl_exposure_years': max(window, 0),
                'data_at_risk': window > 0,
            }

        # Overall risk assessment based on moderate scenario
        moderate_window = hndl_windows['moderate']['hndl_exposure_years']
        if moderate_window > 20:
            risk_level = 'CRITICAL'
        elif moderate_window > 10:
            risk_level = 'HIGH'
        elif moderate_window > 0:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'

        # Determine urgency message
        if risk_level == 'CRITICAL':
            urgency_msg = (
                f'IMMEDIATE ACTION REQUIRED: {retention}-year data retention means '
                f'{moderate_window} years of HNDL exposure. Data harvested today '
                f'will be decryptable while still sensitive.'
            )
        elif risk_level == 'HIGH':
            urgency_msg = (
                f'URGENT: {retention}-year retention with {moderate_window} years of '
                f'HNDL exposure. Begin PQC migration immediately.'
            )
        elif risk_level == 'MODERATE':
            urgency_msg = (
                f'MODERATE: {retention}-year retention with {moderate_window} years of '
                f'potential HNDL exposure. Plan PQC migration within 1-2 years.'
            )
        else:
            urgency_msg = (
                f'LOW: {retention}-year retention with minimal HNDL exposure. '
                f'Monitor quantum computing progress and plan migration.'
            )

        return {
            'sector': sector,
            'data_retention_years': retention,
            'scenarios': hndl_windows,
            'risk_level': risk_level,
            'moderate_hndl_window_years': moderate_window,
            'urgency_message': urgency_msg,
            'actively_harvested': risk_level in ('CRITICAL', 'HIGH'),
            'note': (
                f'HNDL (Harvest-Now-Decrypt-Later): Adversaries collect encrypted data today '
                f'to decrypt when quantum computers mature. Sectors with long data retention '
                f'face the greatest risk.'
            ),
        }

    # ------------------------------------------------------------------
    # PRIVATE: Migration Urgency Score
    # ------------------------------------------------------------------

    def _calculate_migration_urgency(self, sector: str) -> Dict[str, Any]:
        """
        Calculate 0-100 migration urgency score.

        Weights:
            HNDL/SNDL risk:        30%
            Compliance proximity:   25%
            Side-channel exposure:  20%
            FHE lattice risk:       15%
            Data retention:         10%
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]

        # 1. HNDL/SNDL risk (0-100)
        sndl_map = {'CRITICAL': 100, 'HIGH': 75, 'MODERATE': 50, 'LOW': 25}
        sndl_score = sndl_map.get(profile['sndl_risk_base'], 50)

        # 2. Compliance deadline proximity (0-100)
        years_to_deadline = profile['compliance_deadline_year'] - 2026
        if years_to_deadline <= 2:
            compliance_score = 100
        elif years_to_deadline <= 4:
            compliance_score = 80
        elif years_to_deadline <= 6:
            compliance_score = 60
        else:
            compliance_score = 40

        # 3. Side-channel exposure (0-100)
        sc_map = {'CRITICAL': 100, 'HIGH': 75, 'MODERATE': 50, 'LOW': 25}
        sc_score = sc_map.get(profile['side_channel_exposure'], 50)

        # 4. FHE lattice risk (0-100)
        fhe_score = 80 if profile['fhe_dependency'] else 20

        # 5. Data retention (0-100, scaled logarithmically)
        retention = profile['data_retention_years']
        if retention >= 50:
            retention_score = 100
        elif retention >= 20:
            retention_score = 80
        elif retention >= 7:
            retention_score = 60
        elif retention >= 3:
            retention_score = 40
        else:
            retention_score = 20

        # Weighted overall
        overall = (
            sndl_score * 0.30
            + compliance_score * 0.25
            + sc_score * 0.20
            + fhe_score * 0.15
            + retention_score * 0.10
        )
        overall = round(overall, 1)

        # Classify
        if overall >= 80:
            urgency_level = 'CRITICAL'
        elif overall >= 60:
            urgency_level = 'HIGH'
        elif overall >= 40:
            urgency_level = 'MODERATE'
        else:
            urgency_level = 'LOW'

        return {
            'overall_score': overall,
            'urgency_level': urgency_level,
            'component_scores': {
                'sndl_risk': {'score': sndl_score, 'weight': 0.30},
                'compliance_proximity': {'score': compliance_score, 'weight': 0.25},
                'side_channel_exposure': {'score': sc_score, 'weight': 0.20},
                'fhe_lattice_risk': {'score': fhe_score, 'weight': 0.15},
                'data_retention': {'score': retention_score, 'weight': 0.10},
            },
        }

    # ------------------------------------------------------------------
    # PRIVATE: Side-Channel Risk Assessment
    # ------------------------------------------------------------------

    def _assess_side_channel(self, sector: str) -> Dict[str, Any]:
        """
        Side-channel vulnerability assessment contextualized for sector.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]

        if not self._side_channel:
            return {
                'available': False,
                'note': 'SideChannelRiskAssessment module not available',
            }

        # Assess algorithms used by this sector
        algorithms_to_check = set()
        for algos in profile['pqc_targets'].values():
            for algo in algos:
                if 'ML-KEM' in algo:
                    algorithms_to_check.add('ML-KEM')
                elif 'ML-DSA' in algo:
                    algorithms_to_check.add('ML-DSA')
                elif 'SLH-DSA' in algo:
                    algorithms_to_check.add('SLH-DSA')
        if profile['fhe_dependency']:
            algorithms_to_check.add('CKKS-FHE')

        assessments = {}
        for algo in sorted(algorithms_to_check):
            try:
                risk_profile = self._side_channel.assess_algorithm(algo)
                assessments[algo] = risk_profile.to_dict()
            except Exception as e:
                assessments[algo] = {'error': str(e)}

        # Sector-specific context
        sector_context = {
            'healthcare': (
                'Medical IoT devices (Cortex-M4) are directly vulnerable to '
                'SPA attacks (Berzati 2025). Masking countermeasures REQUIRED.'
            ),
            'finance': (
                'HSM-based deployments provide physical protection. '
                'Timing attacks on TLS implementations remain a concern.'
            ),
            'blockchain': (
                'Server-side signing reduces physical attack surface. '
                'ML-DSA signing leakage is the primary concern.'
            ),
            'iot': (
                'CRITICAL: Cortex-M4 SPA recovers ML-KEM keys in 30 seconds. '
                'All IoT PQC deployments MUST include masking (pqm4 v2.0+).'
            ),
            'mpc-fhe': (
                'NTT SPA achieves 98.6% key extraction on CKKS implementations. '
                'CPAD impossibility means NO BFV/BGV/CKKS is IND-CPA^D secure.'
            ),
        }

        return {
            'available': True,
            'sector_exposure': profile['side_channel_exposure'],
            'algorithms_assessed': list(algorithms_to_check),
            'assessments': assessments,
            'sector_context': sector_context.get(sector, ''),
        }

    # ------------------------------------------------------------------
    # PRIVATE: Strategy Comparison
    # ------------------------------------------------------------------

    def _compare_strategies(
        self,
        sector: str,
        shor_rsa: Dict,
        shor_hybrid: Dict,
        shor_pqc_primary: Dict,
        shor_pqc_only: Dict,
    ) -> Dict[str, Any]:
        """
        Side-by-side comparison of all 4 migration strategies.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]

        # For lattice-native sectors (e.g. MPC-FHE) that have no RSA/ECC,
        # derive PQC security from FHE scheme parameters instead.
        is_lattice_native = (
            not profile['current_algorithms'].get('key_exchange')
            and not profile['current_algorithms'].get('signatures')
        )
        if is_lattice_native and profile.get('fhe_dependency'):
            # CKKS Ring-LWE with log_n >= 16: ~128-bit classical security
            # After quantum sieve reduction: 128 - BKZ_SECURITY_REDUCTION
            lattice_security = 128 - BKZ_SECURITY_REDUCTION  # 124.5 bits
            hybrid_security = lattice_security
            pqc_security = lattice_security
        else:
            hybrid_security = shor_hybrid.get('effective_security_bits', 0)
            pqc_security = shor_pqc_primary.get(
                'effective_minimum_security_bits', 0
            )
            lattice_security = pqc_security

        strategies = [
            {
                'strategy': 'rsa_only',
                'name': MIGRATION_STRATEGIES['rsa_only']['name'],
                'shor_resistant': False,
                'effective_security_bits': 0,
                'threat_year': shor_rsa.get('earliest_threat_year', 'N/A'),
                'nist_compliant_2035': False,
                'cnsa_2_0_compliant': False,
                'hndl_protected': False,
                'deployment_complexity': 'None',
                'verdict': 'UNACCEPTABLE — Fully broken by Shor',
            },
            {
                'strategy': 'hybrid',
                'name': MIGRATION_STRATEGIES['hybrid']['name'],
                'shor_resistant': True,
                'effective_security_bits': hybrid_security,
                'threat_year': 'Post-2060+ (lattice)',
                'nist_compliant_2035': True,
                'cnsa_2_0_compliant': True,
                'hndl_protected': True,
                'deployment_complexity': 'Moderate',
                'verdict': 'RECOMMENDED — Best transition strategy',
            },
            {
                'strategy': 'pqc_primary',
                'name': MIGRATION_STRATEGIES['pqc_primary']['name'],
                'shor_resistant': True,
                'effective_security_bits': pqc_security,
                'threat_year': 'Post-2060+ (lattice sieve only)',
                'nist_compliant_2035': True,
                'cnsa_2_0_compliant': True,
                'hndl_protected': True,
                'deployment_complexity': 'High',
                'verdict': 'GOOD — Target state for most sectors',
            },
            {
                'strategy': 'pqc_only',
                'name': MIGRATION_STRATEGIES['pqc_only']['name'],
                'shor_resistant': True,
                'effective_security_bits': pqc_security,
                'threat_year': 'Post-2060+ (lattice sieve only)',
                'nist_compliant_2035': True,
                'cnsa_2_0_compliant': True,
                'hndl_protected': True,
                'deployment_complexity': 'Very High',
                'verdict': (
                    'IDEAL (long-term) — Eliminates all classical vulnerabilities'
                ),
            },
        ]

        # Recommend strategy based on sector
        if profile['compliance_deadline_year'] <= 2030:
            recommended = 'hybrid'
            reasoning = (
                f'{profile["name"]} compliance deadline {profile["compliance_deadline_year"]} '
                f'requires immediate action. Hybrid provides fastest path to quantum resistance.'
            )
        elif profile['data_retention_years'] >= 50:
            recommended = 'pqc_primary'
            reasoning = (
                f'{profile["name"]} long data retention ({profile["data_retention_years"]} years) '
                f'demands strong PQC protection. Hybrid as interim, PQC primary as target.'
            )
        else:
            recommended = 'hybrid'
            reasoning = f'{profile["name"]}: Hybrid migration recommended as first step.'

        return {
            'strategies': strategies,
            'recommended_strategy': recommended,
            'recommendation_reasoning': reasoning,
            'sector_deadline': profile['compliance_deadline_year'],
        }

    # ------------------------------------------------------------------
    # PRIVATE: Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        sector: str,
        urgency: Dict,
        hndl: Dict,
        side_ch: Dict,
    ) -> List[Dict[str, Any]]:
        """
        Generate prioritized recommendations based on assessment results.
        """
        profile = SECTOR_QUANTUM_PROFILES[sector]
        recs = []
        priority = 1

        # HNDL-driven recommendations
        if hndl.get('risk_level') in ('CRITICAL', 'HIGH'):
            recs.append({
                'priority': priority,
                'category': 'HNDL Mitigation',
                'action': (
                    f'Begin hybrid PQC migration IMMEDIATELY for {profile["name"]}. '
                    f'HNDL exposure: {hndl.get("moderate_hndl_window_years", 0)} years.'
                ),
                'deadline': '2026 Q4',
            })
            priority += 1

        # Compliance-driven
        years_left = profile['compliance_deadline_year'] - 2026
        if years_left <= 4:
            recs.append({
                'priority': priority,
                'category': 'Compliance',
                'action': (
                    f'{profile["compliance_framework"]} PQC deadline: '
                    f'{profile["compliance_deadline_year"]} ({years_left} years). '
                    f'Deploy ML-KEM-768+ for key exchange, ML-DSA-65+ for signatures.'
                ),
                'deadline': f'{profile["compliance_deadline_year"] - 1}',
            })
            priority += 1

        # AES upgrade
        current_sym = profile['current_algorithms'].get('symmetric', [])
        if 'AES-128' in current_sym:
            recs.append({
                'priority': priority,
                'category': 'Symmetric Upgrade',
                'action': (
                    'Upgrade AES-128 to AES-256 for CNSA 2.0 compliance and '
                    '128-bit post-quantum security (vs 64-bit with AES-128).'
                ),
                'deadline': '2027',
            })
            priority += 1

        # Side-channel mitigation
        if profile['side_channel_exposure'] in ('CRITICAL', 'HIGH'):
            recs.append({
                'priority': priority,
                'category': 'Side-Channel Mitigation',
                'action': (
                    f'Deploy masking countermeasures for {profile["name"]} PQC implementations. '
                    f'Use pqm4 v2.0+ for IoT, constant-time NTT for server deployments.'
                ),
                'deadline': '2027 Q2',
            })
            priority += 1

        # FHE-specific
        if profile['fhe_dependency']:
            recs.append({
                'priority': priority,
                'category': 'FHE Security',
                'action': (
                    'Verify CKKS parameters against HE Standard bounds. Use log_n >= 16 '
                    'for deep computations. Apply noise flooding for CPAD mitigation.'
                ),
                'deadline': '2027',
            })
            priority += 1

        # Diversification
        recs.append({
            'priority': priority,
            'category': 'Algorithm Diversification',
            'action': (
                'Deploy SLH-DSA (hash-based) for software signing to reduce lattice '
                'monoculture risk. Track HQC (code-based) standardization for backup KEM.'
            ),
            'deadline': '2028',
        })

        return recs


# =============================================================================
# MODULE VERSION
# =============================================================================

try:
    from .version_loader import get_version
    __version__ = get_version('sector_quantum_security')
except ImportError:
    __version__ = "3.2.0"
__author__ = "PQC-FHE Integration Library"
