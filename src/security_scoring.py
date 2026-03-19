#!/usr/bin/env python3
"""
Security Scoring Framework - Enterprise PQC Readiness Assessment
================================================================

This module provides a comprehensive security scoring engine for evaluating
an organization's readiness to transition to post-quantum cryptography:

1. Cryptographic inventory analysis
2. PQC readiness assessment (per NIST IR 8547)
3. Compliance checking (CNSA 2.0, FIPS 140-3)
4. Risk scoring (0-100) with categories
5. Migration priority recommendations

References:
- NIST IR 8547: "Transition to Post-Quantum Cryptography Standards" (Nov 2024)
- NSA CNSA 2.0: Commercial National Security Algorithm Suite 2.0
- NIST SP 800-57 Rev. 5: Recommendation for Key Management
- CISA Post-Quantum Cryptography Initiative
- NIST SP 800-131A Rev. 2: Transitioning Cryptographic Algorithms

Author: PQC-FHE Integration Library
License: MIT
Version: 3.2.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ComplianceStandard(Enum):
    """Supported compliance standards for evaluation."""
    NIST_IR_8547 = "nist_ir_8547"
    CNSA_2_0 = "cnsa_2_0"
    FIPS_140_3 = "fips_140_3"
    NIST_SP_800_57 = "sp_800_57"


class RiskCategory(Enum):
    """Risk categories based on overall score."""
    CRITICAL = "critical"    # 0-25: Immediate action required
    HIGH = "high"            # 26-50: Action within 1 year
    MODERATE = "moderate"    # 51-75: Action within 2-3 years
    LOW = "low"              # 76-100: Monitoring sufficient


class AssetType(Enum):
    """Types of cryptographic assets."""
    KEY_EXCHANGE = "key_exchange"
    DIGITAL_SIGNATURE = "digital_signature"
    SYMMETRIC_ENCRYPTION = "symmetric_encryption"
    HASH_FUNCTION = "hash_function"
    TLS_PROTOCOL = "tls_protocol"
    CERTIFICATE = "certificate"


class DataSensitivity(Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    TOP_SECRET = "top_secret"


class MigrationComplexity(Enum):
    """Migration complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CryptoAsset:
    """Represents a cryptographic algorithm deployment in the organization."""
    name: str
    asset_type: str
    key_size: int
    usage_count: int = 1
    data_sensitivity: str = "internal"
    replacement_algorithm: str = ""
    migration_complexity: str = "medium"
    estimated_migration_months: int = 6
    is_quantum_vulnerable: bool = True
    is_pqc_ready: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'asset_type': self.asset_type,
            'key_size': self.key_size,
            'usage_count': self.usage_count,
            'data_sensitivity': self.data_sensitivity,
            'replacement_algorithm': self.replacement_algorithm,
            'migration_complexity': self.migration_complexity,
            'estimated_migration_months': self.estimated_migration_months,
            'is_quantum_vulnerable': self.is_quantum_vulnerable,
            'is_pqc_ready': self.is_pqc_ready,
            'notes': self.notes,
        }


@dataclass
class ComplianceCheck:
    """Result of a single compliance check."""
    standard: str
    requirement: str
    status: str          # "pass", "fail", "partial", "not_applicable"
    details: str
    priority: str = "medium"  # "critical", "high", "medium", "low"
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'standard': self.standard,
            'requirement': self.requirement,
            'status': self.status,
            'details': self.details,
            'priority': self.priority,
            'remediation': self.remediation,
        }


@dataclass
class SecurityScore:
    """Comprehensive security assessment result."""
    overall_score: float
    category: str
    timestamp: str

    # Sub-scores (0-100 each)
    algorithm_strength_score: float
    pqc_readiness_score: float
    compliance_score: float
    key_management_score: float
    crypto_agility_score: float

    # Inventory analysis
    crypto_inventory: List[Dict[str, Any]] = field(default_factory=list)
    total_assets: int = 0
    vulnerable_assets: int = 0
    pqc_ready_assets: int = 0

    # Issues and recommendations
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Compliance
    compliance_checks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Migration
    migration_timeline: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': round(self.overall_score, 1),
            'category': self.category,
            'timestamp': self.timestamp,
            'sub_scores': {
                'algorithm_strength': round(self.algorithm_strength_score, 1),
                'pqc_readiness': round(self.pqc_readiness_score, 1),
                'compliance': round(self.compliance_score, 1),
                'key_management': round(self.key_management_score, 1),
                'crypto_agility': round(self.crypto_agility_score, 1),
            },
            'inventory_summary': {
                'total_assets': self.total_assets,
                'vulnerable_assets': self.vulnerable_assets,
                'pqc_ready_assets': self.pqc_ready_assets,
                'vulnerability_ratio': round(
                    self.vulnerable_assets / max(1, self.total_assets), 3
                ),
            },
            'vulnerabilities': self.vulnerabilities,
            'recommendations': self.recommendations,
            'compliance_checks': self.compliance_checks,
            'migration_timeline': self.migration_timeline,
        }


# =============================================================================
# ALGORITHM KNOWLEDGE BASE
# =============================================================================

# Quantum vulnerability classification
ALGORITHM_QUANTUM_STATUS = {
    # Public-key algorithms (vulnerable to Shor's)
    'RSA-1024': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-512'},
    'RSA-2048': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-768'},
    'RSA-3072': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-768'},
    'RSA-4096': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-1024'},
    'ECDSA-256': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-DSA-44'},
    'ECDSA-384': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-DSA-65'},
    'ECDH-256': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-512'},
    'ECDH-384': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-768'},
    'DH-2048': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-768'},
    'DH-3072': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-768'},
    'DSA-2048': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-DSA-65'},
    'Ed25519': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-DSA-44'},
    'X25519': {'vulnerable': True, 'severity': 'critical', 'replacement': 'ML-KEM-768'},

    # Symmetric algorithms (weakened by Grover's but generally safe)
    'AES-128': {'vulnerable': False, 'severity': 'moderate', 'replacement': 'AES-256',
                'note': 'Grover halves to 64-bit, upgrade to AES-256'},
    'AES-192': {'vulnerable': False, 'severity': 'low', 'replacement': 'AES-256'},
    'AES-256': {'vulnerable': False, 'severity': 'low', 'replacement': 'AES-256 (adequate)'},
    'ChaCha20': {'vulnerable': False, 'severity': 'low', 'replacement': 'ChaCha20 (adequate)',
                 'note': '256-bit key, 128-bit post-quantum security'},

    # Hash functions (weakened by Grover's but generally safe)
    'SHA-256': {'vulnerable': False, 'severity': 'low', 'replacement': 'SHA-256 (adequate)'},
    'SHA-384': {'vulnerable': False, 'severity': 'low', 'replacement': 'SHA-384 (adequate)'},
    'SHA-512': {'vulnerable': False, 'severity': 'low', 'replacement': 'SHA-512 (adequate)'},
    'SHA-1': {'vulnerable': False, 'severity': 'critical', 'replacement': 'SHA-256',
              'note': 'Already broken classically'},
    'MD5': {'vulnerable': False, 'severity': 'critical', 'replacement': 'SHA-256',
            'note': 'Already broken classically'},

    # PQC algorithms (quantum resistant) — Lattice-based
    'ML-KEM-512': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                   'family': 'lattice'},
    'ML-KEM-768': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                   'family': 'lattice'},
    'ML-KEM-1024': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                    'family': 'lattice'},
    'ML-DSA-44': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                  'family': 'lattice'},
    'ML-DSA-65': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                  'family': 'lattice'},
    'ML-DSA-87': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                  'family': 'lattice'},

    # PQC algorithms — Hash-based
    'SLH-DSA-SHA2-128f': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                          'family': 'hash'},
    'SLH-DSA-SHA2-192f': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                          'family': 'hash'},
    'SLH-DSA-SHA2-256f': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                          'family': 'hash'},

    # PQC algorithms — Code-based (NIST IR 8545, March 2025)
    'HQC-128': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                'family': 'code', 'note': 'NIST 4th-round selection, code-based KEM'},
    'HQC-192': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                'family': 'code'},
    'HQC-256': {'vulnerable': False, 'severity': 'low', 'replacement': 'N/A (PQC)',
                'family': 'code'},

    # FHE schemes (Ring-LWE based — shares lattice hardness with ML-KEM)
    'CKKS': {'vulnerable': False, 'severity': 'moderate', 'replacement': 'N/A (FHE)',
             'family': 'lattice',
             'note': 'Ring-LWE based FHE. Shares lattice monoculture risk with ML-KEM/ML-DSA. '
                     'Security depends on (N, log Q) parameters per HE Standard.'},
    'BGV': {'vulnerable': False, 'severity': 'moderate', 'replacement': 'N/A (FHE)',
            'family': 'lattice',
            'note': 'Ring-LWE based exact integer FHE. Same lattice risk as CKKS.'},
    'BFV': {'vulnerable': False, 'severity': 'moderate', 'replacement': 'N/A (FHE)',
            'family': 'lattice',
            'note': 'Ring-LWE based exact integer FHE. Same lattice risk as CKKS.'},

    # Hybrid
    'X25519+ML-KEM-768': {'vulnerable': False, 'severity': 'low',
                          'replacement': 'N/A (Hybrid PQC)', 'family': 'hybrid'},
}


# PQC algorithm family classification for diversity analysis
PQC_FAMILY_INFO = {
    'lattice': {
        'name': 'Lattice-based',
        'algorithms': ['ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024',
                       'ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87'],
        'fhe_schemes': ['CKKS', 'BGV', 'BFV'],
        'hard_problem': 'Module-LWE / Ring-LWE / Module-SIS',
        'risk_note': ('All lattice-based schemes share Module-LWE/Ring-LWE hardness assumption. '
                      'A breakthrough in lattice reduction (BKZ improvements, quantum sieving) '
                      'would compromise ML-KEM, ML-DSA, AND lattice-based FHE (CKKS/BGV/BFV) '
                      'simultaneously. This is the critical lattice monoculture risk: '
                      'PQC key exchange AND homomorphic encryption share one failure point.'),
        'nist_standard': 'FIPS 203, FIPS 204',
        'fhe_note': ('CKKS/BGV/BFV FHE schemes use Ring-LWE, a specialized case of '
                     'the same lattice problem. FHE security depends on (N, log Q) '
                     'parameters per the HE Standard (homomorphicencryption.org). '
                     'No non-lattice FHE alternative exists in practical deployment.'),
    },
    'hash': {
        'name': 'Hash-based',
        'algorithms': ['SLH-DSA-SHA2-128f', 'SLH-DSA-SHA2-192f', 'SLH-DSA-SHA2-256f'],
        'hard_problem': 'Hash function preimage/collision resistance',
        'risk_note': ('Hash-based schemes rely on well-studied hash function security. '
                      'Minimal risk from lattice-specific attacks. Recommended for '
                      'algorithm diversity in signature applications.'),
        'nist_standard': 'FIPS 205',
    },
    'code': {
        'name': 'Code-based',
        'algorithms': ['HQC-128', 'HQC-192', 'HQC-256'],
        'hard_problem': 'Quasi-Cyclic Syndrome Decoding',
        'risk_note': ('HQC selected by NIST IR 8545 (March 2025) as 4th-round code-based KEM. '
                      'Provides algorithmic diversity for key encapsulation independent of '
                      'lattice hardness assumptions.'),
        'nist_standard': 'NIST IR 8545 (pending FIPS)',
    },
}


# CNSA 2.0 migration phase gates (NSA, updated May 2025)
CNSA_2_0_PHASE_GATES = {
    '2025': {
        'phase': 'Phase 1: Inventory & Planning',
        'requirements': [
            'Complete cryptographic inventory of all systems',
            'Identify quantum-vulnerable algorithms in production',
            'Begin PQC testing and evaluation',
        ],
        'status_check': lambda profile: profile.get('has_crypto_inventory', False),
    },
    '2027': {
        'phase': 'Phase 2: Hybrid Deployment',
        'requirements': [
            'Deploy hybrid classical+PQC for high-sensitivity channels',
            'X25519+ML-KEM-768 for TLS 1.3 key exchange',
            'Begin ML-DSA pilot for digital signatures',
        ],
        'status_check': lambda profile: profile.get('has_hybrid_deployment', False),
    },
    '2029': {
        'phase': 'Phase 3: Full PQC Migration',
        'requirements': [
            'ML-KEM-1024 for all key establishment',
            'ML-DSA-87 for all digital signatures',
            'AES-256 for all symmetric encryption',
            'SHA-384 minimum for all hash functions',
            'Deprecate all classical public-key algorithms',
        ],
        'status_check': lambda profile: profile.get('pqc_migration_complete', False),
    },
    '2030': {
        'phase': 'Phase 4: CNSA 2.0 Compliance',
        'requirements': [
            'Full CNSA 2.0 compliance achieved',
            'No RSA, ECC, or DH in production',
            'All certificates PQC-only or hybrid',
            'Continuous monitoring for lattice security margin',
        ],
        'status_check': lambda profile: profile.get('cnsa_2_0_compliant', False),
    },
    '2035': {
        'phase': 'Phase 5: NIST Disallow Date',
        'requirements': [
            'NIST IR 8547 disallow date for RSA/ECC',
            'All legacy classical algorithms removed',
            'HQC standardized and available as lattice alternative',
        ],
        'status_check': lambda profile: False,  # Future target
    },
}


# =============================================================================
# SECURITY SCORING ENGINE
# =============================================================================

class SecurityScoringEngine:
    """
    Enterprise PQC readiness scoring engine.

    Scoring methodology (weighted average):
    1. Algorithm Strength (25%): Current algorithm quantum resistance
    2. PQC Readiness (25%): PQC alternative deployment progress
    3. Compliance (20%): NIST IR 8547, CNSA 2.0, FIPS 140-3
    4. Key Management (15%): Key rotation, lifecycle practices
    5. Crypto Agility (15%): Ability to switch algorithms rapidly

    References:
    - NIST IR 8547: Section 4 (Migration Planning)
    - CISA PQC Assessment methodology
    """

    WEIGHT_ALGORITHM_STRENGTH = 0.25
    WEIGHT_PQC_READINESS = 0.25
    WEIGHT_COMPLIANCE = 0.20
    WEIGHT_KEY_MANAGEMENT = 0.15
    WEIGHT_CRYPTO_AGILITY = 0.15

    def __init__(self):
        logger.info("SecurityScoringEngine initialized")

    def calculate_overall_score(
        self,
        inventory: List[CryptoAsset],
        org_profile: Optional[Dict[str, Any]] = None,
    ) -> SecurityScore:
        """
        Calculate comprehensive security score for the organization.

        Args:
            inventory: List of cryptographic assets in use
            org_profile: Optional organizational profile with:
                - has_crypto_inventory: bool
                - has_migration_plan: bool
                - has_hybrid_deployment: bool
                - key_rotation_policy: bool
                - crypto_agility_framework: bool
                - pqc_testing_started: bool
        """
        if org_profile is None:
            org_profile = {}

        # Calculate sub-scores
        algo_score = self._score_algorithm_strength(inventory)
        pqc_score = self._score_pqc_readiness(inventory, org_profile)
        compliance_score = self._score_compliance(inventory, org_profile)
        key_mgmt_score = self._score_key_management(org_profile)
        agility_score = self._score_crypto_agility(org_profile)

        # Weighted overall score
        overall = (
            self.WEIGHT_ALGORITHM_STRENGTH * algo_score
            + self.WEIGHT_PQC_READINESS * pqc_score
            + self.WEIGHT_COMPLIANCE * compliance_score
            + self.WEIGHT_KEY_MANAGEMENT * key_mgmt_score
            + self.WEIGHT_CRYPTO_AGILITY * agility_score
        )

        category = self._categorize_score(overall)

        # Analyze inventory
        total = len(inventory)
        vulnerable = sum(1 for a in inventory if a.is_quantum_vulnerable)
        pqc_ready = sum(1 for a in inventory if a.is_pqc_ready)

        # Generate findings
        vulnerabilities = self._identify_vulnerabilities(inventory)
        recommendations = self._generate_recommendations(
            inventory, org_profile, overall
        )
        compliance_checks = self._run_all_compliance_checks(inventory, org_profile)
        migration = self._generate_migration_timeline(inventory, org_profile)

        return SecurityScore(
            overall_score=overall,
            category=category,
            timestamp=datetime.now().isoformat(),
            algorithm_strength_score=algo_score,
            pqc_readiness_score=pqc_score,
            compliance_score=compliance_score,
            key_management_score=key_mgmt_score,
            crypto_agility_score=agility_score,
            crypto_inventory=[a.to_dict() for a in inventory],
            total_assets=total,
            vulnerable_assets=vulnerable,
            pqc_ready_assets=pqc_ready,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            compliance_checks=compliance_checks,
            migration_timeline=migration,
        )

    def check_compliance(
        self,
        standard: ComplianceStandard,
        inventory: List[CryptoAsset],
        org_profile: Optional[Dict[str, Any]] = None,
    ) -> List[ComplianceCheck]:
        """Run compliance checks for a specific standard."""
        if org_profile is None:
            org_profile = {}

        if standard == ComplianceStandard.NIST_IR_8547:
            return self._check_nist_ir_8547(inventory, org_profile)
        elif standard == ComplianceStandard.CNSA_2_0:
            return self._check_cnsa_2_0(inventory, org_profile)
        elif standard == ComplianceStandard.FIPS_140_3:
            return self._check_fips_140_3(inventory, org_profile)
        elif standard == ComplianceStandard.NIST_SP_800_57:
            return self._check_sp_800_57(inventory, org_profile)
        else:
            raise ValueError(f"Unknown compliance standard: {standard}")

    def generate_migration_plan(
        self, score: SecurityScore
    ) -> Dict[str, Any]:
        """Generate a detailed migration plan based on assessment results."""
        phases = []

        # Phase 1: Discovery & Inventory (0-3 months)
        phases.append({
            'phase': 1,
            'name': 'Discovery & Inventory',
            'duration_months': 3,
            'description': 'Complete cryptographic inventory and risk assessment',
            'tasks': [
                'Enumerate all cryptographic algorithm deployments',
                'Map data flows and identify quantum-vulnerable channels',
                'Assess data sensitivity and retention requirements',
                'Identify "harvest now, decrypt later" risks',
            ],
            'status': 'completed' if score.total_assets > 0 else 'pending',
        })

        # Phase 2: Planning & Testing (3-9 months)
        phases.append({
            'phase': 2,
            'name': 'Planning & Testing',
            'duration_months': 6,
            'description': 'Evaluate PQC alternatives and develop migration strategy',
            'tasks': [
                'Test ML-KEM-768/1024 for key exchange replacement',
                'Test ML-DSA-65/87 for signature replacement',
                'Evaluate hybrid (classical + PQC) deployment options',
                'Benchmark PQC performance in target environment',
                'Develop phased migration roadmap',
            ],
            'status': 'in_progress' if score.pqc_readiness_score > 20 else 'pending',
        })

        # Phase 3: Hybrid Deployment (9-18 months)
        phases.append({
            'phase': 3,
            'name': 'Hybrid Deployment',
            'duration_months': 9,
            'description': 'Deploy hybrid classical+PQC algorithms',
            'tasks': [
                'Deploy X25519+ML-KEM-768 hybrid key exchange',
                'Update TLS configurations for PQC cipher suites',
                'Migrate high-sensitivity data channels first',
                'Monitor performance and compatibility',
                'Update certificate infrastructure',
            ],
            'priority_assets': [
                v for v in score.vulnerabilities
                if v.get('priority') == 'critical'
            ],
        })

        # Phase 4: Full PQC Migration (18-36 months)
        phases.append({
            'phase': 4,
            'name': 'Full PQC Migration',
            'duration_months': 18,
            'description': 'Complete transition to PQC-only algorithms',
            'tasks': [
                'Deprecate all quantum-vulnerable algorithms',
                'Complete certificate migration to PQC',
                'Update all key management systems',
                'Achieve CNSA 2.0 compliance',
                'Final security audit and validation',
            ],
            'target_year': 2030,
        })

        total_months = sum(p['duration_months'] for p in phases)

        return {
            'phases': phases,
            'total_duration_months': total_months,
            'estimated_completion_year': 2026 + total_months // 12,
            'nist_deadline': 2035,
            'cnsa_deadline': 2030,
            'current_score': round(score.overall_score, 1),
            'target_score': 85.0,
        }

    # -------------------------------------------------------------------------
    # Sub-score calculations
    # -------------------------------------------------------------------------

    def _score_algorithm_strength(self, inventory: List[CryptoAsset]) -> float:
        """Score based on quantum resistance of deployed algorithms (0-100)."""
        if not inventory:
            return 50.0  # No data -> neutral score

        total_weight = 0
        weighted_score = 0

        sensitivity_weight = {
            'public': 1, 'internal': 2, 'confidential': 4, 'top_secret': 8,
        }

        for asset in inventory:
            weight = sensitivity_weight.get(asset.data_sensitivity, 2) * asset.usage_count
            total_weight += weight

            status = ALGORITHM_QUANTUM_STATUS.get(asset.name, {})
            if status.get('vulnerable', True):
                severity = status.get('severity', 'critical')
                if severity == 'critical':
                    weighted_score += 0
                elif severity == 'high':
                    weighted_score += weight * 0.25
                elif severity == 'moderate':
                    weighted_score += weight * 0.50
                else:
                    weighted_score += weight * 0.75
            else:
                weighted_score += weight * 1.0

        return (weighted_score / max(1, total_weight)) * 100

    def _score_pqc_readiness(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> float:
        """Score based on PQC migration progress (0-100)."""
        score = 0.0

        # PQC algorithms already deployed (40 points max)
        pqc_count = sum(1 for a in inventory if a.is_pqc_ready)
        total = max(1, len(inventory))
        score += min(40, (pqc_count / total) * 40)

        # Organizational readiness (60 points max)
        if org_profile.get('has_crypto_inventory'):
            score += 15
        if org_profile.get('has_migration_plan'):
            score += 15
        if org_profile.get('has_hybrid_deployment'):
            score += 15
        if org_profile.get('pqc_testing_started'):
            score += 15

        return min(100, score)

    def _score_compliance(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> float:
        """Score based on compliance with PQC standards (0-100)."""
        checks_nist = self._check_nist_ir_8547(inventory, org_profile)
        checks_cnsa = self._check_cnsa_2_0(inventory, org_profile)

        all_checks = checks_nist + checks_cnsa
        if not all_checks:
            return 50.0

        pass_count = sum(1 for c in all_checks if c.status == 'pass')
        partial_count = sum(1 for c in all_checks if c.status == 'partial')
        total = len(all_checks)

        return ((pass_count + 0.5 * partial_count) / max(1, total)) * 100

    def _score_key_management(self, org_profile: Dict) -> float:
        """Score based on key management practices (0-100)."""
        score = 30.0  # Base score

        if org_profile.get('key_rotation_policy'):
            score += 25
        if org_profile.get('key_lifecycle_management'):
            score += 20
        if org_profile.get('hardware_security_module'):
            score += 15
        if org_profile.get('automated_key_rotation'):
            score += 10

        return min(100, score)

    def _score_crypto_agility(self, org_profile: Dict) -> float:
        """Score based on ability to switch algorithms rapidly (0-100)."""
        score = 20.0  # Base score

        if org_profile.get('crypto_agility_framework'):
            score += 30
        if org_profile.get('abstraction_layer'):
            score += 20
        if org_profile.get('algorithm_negotiation'):
            score += 15
        if org_profile.get('automated_migration_tools'):
            score += 15

        return min(100, score)

    # -------------------------------------------------------------------------
    # Compliance checks
    # -------------------------------------------------------------------------

    def _check_nist_ir_8547(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> List[ComplianceCheck]:
        """NIST IR 8547: Transition to Post-Quantum Cryptography Standards."""
        checks = []

        # Requirement 1: Cryptographic inventory exists
        checks.append(ComplianceCheck(
            standard='NIST IR 8547',
            requirement='Maintain a cryptographic inventory',
            status='pass' if org_profile.get('has_crypto_inventory') or len(inventory) > 0 else 'fail',
            details='Organization must maintain inventory of all cryptographic algorithms in use',
            priority='high',
            remediation='Create and maintain a comprehensive cryptographic inventory',
        ))

        # Requirement 2: Migration plan exists
        checks.append(ComplianceCheck(
            standard='NIST IR 8547',
            requirement='Develop PQC migration plan',
            status='pass' if org_profile.get('has_migration_plan') else 'fail',
            details='Plan for migrating from quantum-vulnerable to PQC algorithms',
            priority='high',
            remediation='Develop a phased migration plan targeting 2030 completion',
        ))

        # Requirement 3: No RSA/ECC after 2035
        vulnerable_public_key = [
            a for a in inventory
            if a.asset_type in ('key_exchange', 'digital_signature')
            and a.is_quantum_vulnerable
        ]
        if vulnerable_public_key:
            checks.append(ComplianceCheck(
                standard='NIST IR 8547',
                requirement='Deprecate quantum-vulnerable public-key algorithms by 2030',
                status='partial',
                details=f'{len(vulnerable_public_key)} quantum-vulnerable public-key '
                        f'algorithms still in use',
                priority='critical',
                remediation='Replace with ML-KEM for key exchange, ML-DSA for signatures',
            ))
        else:
            checks.append(ComplianceCheck(
                standard='NIST IR 8547',
                requirement='Deprecate quantum-vulnerable public-key algorithms by 2030',
                status='pass',
                details='No quantum-vulnerable public-key algorithms detected',
                priority='critical',
            ))

        # Requirement 4: Use approved PQC algorithms
        pqc_algos = {'ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024',
                      'ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87',
                      'SLH-DSA-SHA2-128f', 'SLH-DSA-SHA2-256f'}
        has_pqc = any(a.name in pqc_algos for a in inventory)
        checks.append(ComplianceCheck(
            standard='NIST IR 8547',
            requirement='Deploy NIST-approved PQC algorithms',
            status='pass' if has_pqc else 'fail',
            details='Use ML-KEM (FIPS 203), ML-DSA (FIPS 204), or SLH-DSA (FIPS 205)',
            priority='high',
            remediation='Begin deploying ML-KEM-768 and ML-DSA-65 as recommended defaults',
        ))

        # Requirement 5: Symmetric algorithms adequate
        weak_symmetric = [
            a for a in inventory
            if a.asset_type == 'symmetric_encryption'
            and a.key_size < 256
        ]
        checks.append(ComplianceCheck(
            standard='NIST IR 8547',
            requirement='Use AES-256 for post-quantum symmetric security',
            status='pass' if not weak_symmetric else 'partial',
            details=f'{len(weak_symmetric)} symmetric algorithms below 256-bit' if weak_symmetric
                    else 'All symmetric algorithms at 256-bit or above',
            priority='medium',
            remediation='Upgrade AES-128 to AES-256 for post-quantum security margin',
        ))

        return checks

    def _check_cnsa_2_0(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> List[ComplianceCheck]:
        """NSA CNSA 2.0: Commercial National Security Algorithm Suite."""
        checks = []

        # CNSA 2.0 requirements
        cnsa_requirements = {
            'key_establishment': ('ML-KEM-1024', 'key_exchange'),
            'digital_signature': ('ML-DSA-87', 'digital_signature'),
            'symmetric': ('AES-256', 'symmetric_encryption'),
            'hash': ('SHA-384', 'hash_function'),
        }

        for purpose, (required_algo, asset_type) in cnsa_requirements.items():
            matching = [
                a for a in inventory
                if a.name == required_algo or (
                    a.asset_type == asset_type
                    and not a.is_quantum_vulnerable
                    and a.key_size >= 256
                )
            ]
            has_exact = any(a.name == required_algo for a in inventory)

            checks.append(ComplianceCheck(
                standard='CNSA 2.0',
                requirement=f'{purpose}: {required_algo} required by 2030',
                status='pass' if has_exact else ('partial' if matching else 'fail'),
                details=f'CNSA 2.0 requires {required_algo} for {purpose}',
                priority='critical' if purpose in ('key_establishment', 'digital_signature') else 'high',
                remediation=f'Deploy {required_algo} for {purpose}',
            ))

        return checks

    def _check_fips_140_3(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> List[ComplianceCheck]:
        """FIPS 140-3: Cryptographic module validation."""
        checks = []

        checks.append(ComplianceCheck(
            standard='FIPS 140-3',
            requirement='Use FIPS-validated cryptographic modules',
            status='partial',
            details='Ensure all PQC implementations are from FIPS-validated modules',
            priority='high',
            remediation='Use liboqs with FIPS-validated backend or await NIST CMVP certification',
        ))

        # Check for deprecated algorithms
        deprecated = [a for a in inventory if a.name in ('SHA-1', 'MD5', 'DES', '3DES')]
        checks.append(ComplianceCheck(
            standard='FIPS 140-3',
            requirement='No deprecated algorithms in use',
            status='fail' if deprecated else 'pass',
            details=f'{len(deprecated)} deprecated algorithms found' if deprecated
                    else 'No deprecated algorithms detected',
            priority='critical' if deprecated else 'low',
            remediation='Remove SHA-1, MD5, DES, 3DES from all deployments',
        ))

        return checks

    def _check_sp_800_57(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> List[ComplianceCheck]:
        """NIST SP 800-57: Key Management recommendations."""
        checks = []

        # Key size adequacy
        short_keys = [
            a for a in inventory
            if a.asset_type == 'key_exchange' and a.key_size < 2048
        ]
        checks.append(ComplianceCheck(
            standard='SP 800-57',
            requirement='Minimum key sizes for target security level',
            status='fail' if short_keys else 'pass',
            details=f'{len(short_keys)} assets with inadequate key sizes' if short_keys
                    else 'All key sizes meet minimum requirements',
            priority='high',
            remediation='Increase key sizes or migrate to PQC algorithms',
        ))

        # Key rotation
        checks.append(ComplianceCheck(
            standard='SP 800-57',
            requirement='Key rotation policy in place',
            status='pass' if org_profile.get('key_rotation_policy') else 'fail',
            details='Regular key rotation reduces exposure to compromise',
            priority='medium',
            remediation='Implement automated key rotation with maximum 1-year lifecycle',
        ))

        return checks

    # -------------------------------------------------------------------------
    # Analysis helpers
    # -------------------------------------------------------------------------

    def _identify_vulnerabilities(self, inventory: List[CryptoAsset]) -> List[Dict]:
        """Identify specific vulnerabilities in the crypto inventory."""
        vulns = []

        for asset in inventory:
            status = ALGORITHM_QUANTUM_STATUS.get(asset.name, {})
            if status.get('vulnerable', True) and asset.asset_type in (
                'key_exchange', 'digital_signature'
            ):
                severity = status.get('severity', 'critical')
                vulns.append({
                    'asset': asset.name,
                    'type': asset.asset_type,
                    'severity': severity,
                    'priority': 'critical' if asset.data_sensitivity in (
                        'confidential', 'top_secret'
                    ) else 'high',
                    'description': f'{asset.name} is vulnerable to Shor\'s algorithm',
                    'impact': f'{asset.usage_count} deployment(s) at risk',
                    'replacement': status.get('replacement', 'ML-KEM-768 / ML-DSA-65'),
                    'harvest_risk': asset.data_sensitivity in ('confidential', 'top_secret'),
                })

            # Check for classically broken algorithms
            if asset.name in ('SHA-1', 'MD5', 'DES', '3DES', 'RC4'):
                vulns.append({
                    'asset': asset.name,
                    'type': asset.asset_type,
                    'severity': 'critical',
                    'priority': 'critical',
                    'description': f'{asset.name} is classically broken',
                    'impact': 'Immediate security risk, not just quantum threat',
                    'replacement': status.get('replacement', 'SHA-256 / AES-256'),
                    'harvest_risk': False,
                })

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'moderate': 2, 'low': 3}
        vulns.sort(key=lambda v: severity_order.get(v['severity'], 4))

        return vulns

    def _generate_recommendations(
        self,
        inventory: List[CryptoAsset],
        org_profile: Dict,
        overall_score: float,
    ) -> List[Dict]:
        """Generate actionable recommendations."""
        recs = []
        priority_num = 1

        # Critical: quantum-vulnerable public-key algorithms
        vulnerable_pk = [
            a for a in inventory
            if a.is_quantum_vulnerable
            and a.asset_type in ('key_exchange', 'digital_signature')
        ]
        if vulnerable_pk:
            recs.append({
                'priority': priority_num,
                'category': 'critical',
                'title': 'Migrate quantum-vulnerable public-key algorithms',
                'description': (
                    f'{len(vulnerable_pk)} public-key algorithm(s) are vulnerable to '
                    f'Shor\'s algorithm. Migrate to ML-KEM-768 for key exchange '
                    f'and ML-DSA-65 for digital signatures.'
                ),
                'effort': 'high',
                'timeline': '6-18 months',
            })
            priority_num += 1

        # High: no migration plan
        if not org_profile.get('has_migration_plan'):
            recs.append({
                'priority': priority_num,
                'category': 'high',
                'title': 'Develop PQC migration plan',
                'description': (
                    'Create a phased migration plan per NIST IR 8547 guidance. '
                    'Target hybrid deployment by 2028, full PQC by 2030.'
                ),
                'effort': 'medium',
                'timeline': '1-3 months',
            })
            priority_num += 1

        # High: deploy hybrid cryptography
        if not org_profile.get('has_hybrid_deployment'):
            recs.append({
                'priority': priority_num,
                'category': 'high',
                'title': 'Deploy hybrid classical+PQC cryptography',
                'description': (
                    'Use X25519+ML-KEM-768 hybrid key exchange as a transition step. '
                    'This provides defense-in-depth during migration.'
                ),
                'effort': 'medium',
                'timeline': '3-6 months',
            })
            priority_num += 1

        # Medium: crypto agility
        if not org_profile.get('crypto_agility_framework'):
            recs.append({
                'priority': priority_num,
                'category': 'medium',
                'title': 'Implement crypto-agility framework',
                'description': (
                    'Build abstraction layers that allow rapid algorithm switching. '
                    'This future-proofs against further cryptographic transitions.'
                ),
                'effort': 'high',
                'timeline': '6-12 months',
            })
            priority_num += 1

        # Medium: upgrade AES-128 to AES-256
        weak_sym = [a for a in inventory if a.name == 'AES-128']
        if weak_sym:
            recs.append({
                'priority': priority_num,
                'category': 'medium',
                'title': 'Upgrade AES-128 to AES-256',
                'description': (
                    f'{len(weak_sym)} deployment(s) using AES-128. '
                    f'Grover\'s algorithm halves effective key length to 64 bits. '
                    f'Upgrade to AES-256 for 128-bit post-quantum security.'
                ),
                'effort': 'low',
                'timeline': '1-3 months',
            })
            priority_num += 1

        return recs

    def _generate_migration_timeline(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> Dict[str, Any]:
        """Generate migration timeline estimates."""
        vulnerable_count = sum(1 for a in inventory if a.is_quantum_vulnerable)
        total_months = sum(
            a.estimated_migration_months
            for a in inventory
            if a.is_quantum_vulnerable
        )
        # Assume 30% parallel execution
        effective_months = int(total_months * 0.3) if total_months > 0 else 0

        return {
            'vulnerable_assets_to_migrate': vulnerable_count,
            'total_effort_months': total_months,
            'effective_duration_months': effective_months,
            'nist_deadline_2030_feasible': effective_months <= 48,
            'recommended_start': 'immediate',
            'milestones': {
                '2026': 'Complete cryptographic inventory and risk assessment',
                '2027': 'Begin hybrid deployment (X25519+ML-KEM)',
                '2028': 'Complete hybrid migration for high-sensitivity channels',
                '2029': 'Begin full PQC migration, deprecate classical algorithms',
                '2030': 'Achieve CNSA 2.0 compliance',
                '2035': 'NIST IR 8547 disallow date for RSA/ECC',
            },
        }

    def _run_all_compliance_checks(
        self, inventory: List[CryptoAsset], org_profile: Dict
    ) -> Dict[str, List[Dict]]:
        """Run all compliance checks and return organized results."""
        results = {}
        for standard in ComplianceStandard:
            checks = self.check_compliance(standard, inventory, org_profile)
            results[standard.value] = [c.to_dict() for c in checks]
        return results

    def _categorize_score(self, score: float) -> str:
        """Categorize overall score into risk category."""
        if score <= 25:
            return RiskCategory.CRITICAL.value
        elif score <= 50:
            return RiskCategory.HIGH.value
        elif score <= 75:
            return RiskCategory.MODERATE.value
        else:
            return RiskCategory.LOW.value

    # -------------------------------------------------------------------------
    # Sample inventories
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Algorithm diversity and CNSA 2.0 assessments (v3.3.0)
    # -------------------------------------------------------------------------

    def assess_algorithm_diversity(
        self, inventory: List[CryptoAsset]
    ) -> Dict[str, Any]:
        """
        Assess PQC algorithm family diversity to detect lattice monoculture risk.

        A deployment relying solely on lattice-based PQC (ML-KEM + ML-DSA) has a
        single point of failure: a breakthrough in Module-LWE would compromise all
        key exchange AND digital signatures simultaneously.

        Scoring:
        - 100: 3+ PQC families deployed (lattice + hash + code)
        - 70:  2 PQC families (e.g., lattice + hash)
        - 40:  1 PQC family only (lattice monoculture)
        - 0:   No PQC deployed

        Returns:
            Dict with diversity score, family breakdown, and recommendations
        """
        pqc_assets = [
            a for a in inventory
            if not a.is_quantum_vulnerable and a.asset_type in (
                'key_exchange', 'digital_signature'
            )
        ]

        families_deployed = set()
        family_counts = {}
        for asset in pqc_assets:
            algo_info = ALGORITHM_QUANTUM_STATUS.get(asset.name, {})
            family = algo_info.get('family', 'unknown')
            if family not in ('unknown', 'hybrid'):
                families_deployed.add(family)
                family_counts[family] = family_counts.get(family, 0) + asset.usage_count

        num_families = len(families_deployed)
        if num_families >= 3:
            diversity_score = 100
            risk_level = 'low'
        elif num_families == 2:
            diversity_score = 70
            risk_level = 'moderate'
        elif num_families == 1:
            diversity_score = 40
            risk_level = 'high'
        else:
            diversity_score = 0
            risk_level = 'critical'

        lattice_only = families_deployed == {'lattice'}
        recommendations = []
        if lattice_only or num_families <= 1:
            recommendations.append({
                'priority': 'high',
                'action': 'Add SLH-DSA (hash-based) for critical signature operations',
                'rationale': ('SLH-DSA provides algorithm diversity independent of '
                              'lattice hardness assumptions. NIST FIPS 205.'),
            })
            recommendations.append({
                'priority': 'medium',
                'action': 'Plan HQC deployment when NIST finalizes standardization',
                'rationale': ('HQC (code-based KEM, NIST IR 8545) provides key exchange '
                              'diversity independent of Module-LWE.'),
            })
        if num_families < 3:
            recommendations.append({
                'priority': 'medium',
                'action': 'Implement crypto-agility to enable rapid algorithm switching',
                'rationale': ('If a lattice breakthrough occurs, organizations need to '
                              'switch to hash-based or code-based alternatives within weeks.'),
            })

        family_details = {}
        for fam_key, fam_info in PQC_FAMILY_INFO.items():
            family_details[fam_key] = {
                'name': fam_info['name'],
                'deployed': fam_key in families_deployed,
                'usage_count': family_counts.get(fam_key, 0),
                'algorithms': fam_info['algorithms'],
                'hard_problem': fam_info['hard_problem'],
                'nist_standard': fam_info['nist_standard'],
            }

        return {
            'diversity_score': diversity_score,
            'risk_level': risk_level,
            'families_deployed': sorted(families_deployed),
            'num_families': num_families,
            'lattice_monoculture': lattice_only,
            'family_details': family_details,
            'pqc_asset_count': len(pqc_assets),
            'recommendations': recommendations,
            'concern': (
                'LATTICE MONOCULTURE WARNING: All deployed PQC algorithms (ML-KEM, ML-DSA) '
                'rely on Module-LWE hardness. A single lattice reduction breakthrough would '
                'compromise key exchange AND signatures simultaneously. '
                'Recent research (Dutch team Oct 2025: 8% quantum sieve improvement, '
                'Zhao & Ding 2025: 3-4 bit BKZ reduction) shows lattice security margins '
                'are shrinking. Deploy SLH-DSA and plan for HQC.'
            ) if lattice_only else None,
            'assessed_at': datetime.now().isoformat(),
        }

    def assess_cnsa_2_0_readiness(
        self,
        inventory: List[CryptoAsset],
        org_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detailed CNSA 2.0 readiness assessment with phase gates.

        Evaluates current progress against NSA CNSA 2.0 milestones
        (updated May 2025) with 2030 full compliance deadline.

        Returns:
            Dict with phase-by-phase compliance status, gaps, and timeline
        """
        if org_profile is None:
            org_profile = {}

        current_year = datetime.now().year

        # Check CNSA 2.0 required algorithms in inventory
        has_ml_kem_1024 = any(a.name == 'ML-KEM-1024' for a in inventory)
        has_ml_dsa_87 = any(a.name == 'ML-DSA-87' for a in inventory)
        has_aes_256 = any(
            a.name == 'AES-256' and a.asset_type == 'symmetric_encryption'
            for a in inventory
        )
        has_sha_384_plus = any(
            a.name in ('SHA-384', 'SHA-512') and a.asset_type == 'hash_function'
            for a in inventory
        )
        has_slh_dsa = any('SLH-DSA' in a.name for a in inventory)
        has_weak_sym = any(
            a.name == 'AES-128' and a.asset_type == 'symmetric_encryption'
            for a in inventory
        )
        has_weak_hash = any(
            a.name in ('SHA-1', 'MD5') for a in inventory
        )

        # Phase gate assessment
        phases = {}
        for year_str, gate in CNSA_2_0_PHASE_GATES.items():
            year = int(year_str)
            met = gate['status_check'](org_profile)
            overdue = year <= current_year and not met
            phases[year_str] = {
                'phase': gate['phase'],
                'requirements': gate['requirements'],
                'met': met,
                'overdue': overdue,
                'target_year': year,
            }

        # Algorithm compliance
        algo_compliance = {
            'ML-KEM-1024': {
                'required': True,
                'deployed': has_ml_kem_1024,
                'purpose': 'Key Establishment',
                'cnsa_requirement': 'Required by 2030',
            },
            'ML-DSA-87': {
                'required': True,
                'deployed': has_ml_dsa_87,
                'purpose': 'Digital Signature',
                'cnsa_requirement': 'Required by 2030',
            },
            'AES-256': {
                'required': True,
                'deployed': has_aes_256,
                'purpose': 'Symmetric Encryption',
                'cnsa_requirement': 'Required (replaces AES-128/192)',
            },
            'SHA-384': {
                'required': True,
                'deployed': has_sha_384_plus,
                'purpose': 'Hash Function',
                'cnsa_requirement': 'SHA-384 minimum required',
            },
            'SLH-DSA': {
                'required': False,
                'deployed': has_slh_dsa,
                'purpose': 'Firmware/Long-term Signatures',
                'cnsa_requirement': 'Recommended for algorithm diversity',
            },
        }

        algo_score = sum(
            1 for a in algo_compliance.values()
            if a['required'] and a['deployed']
        )
        algo_total = sum(1 for a in algo_compliance.values() if a['required'])

        # Overall readiness
        readiness_pct = (algo_score / algo_total * 100) if algo_total > 0 else 0
        years_until_deadline = 2030 - current_year

        gaps = []
        if not has_ml_kem_1024:
            gaps.append('Deploy ML-KEM-1024 for key establishment')
        if not has_ml_dsa_87:
            gaps.append('Deploy ML-DSA-87 for digital signatures')
        if not has_aes_256:
            gaps.append('Upgrade to AES-256 for symmetric encryption')
        if not has_sha_384_plus:
            gaps.append('Upgrade to SHA-384 or SHA-512 for hash functions')
        if has_weak_sym:
            gaps.append('Remove AES-128 deployments (Grover reduces to 64-bit security)')
        if has_weak_hash:
            gaps.append('Remove SHA-1/MD5 (classically broken)')
        if not has_slh_dsa:
            gaps.append('Consider SLH-DSA for algorithm diversity (recommended)')

        return {
            'cnsa_2_0_readiness_pct': readiness_pct,
            'algorithm_compliance': algo_compliance,
            'phase_gates': phases,
            'compliance_gaps': gaps,
            'years_until_deadline': years_until_deadline,
            'deadline_year': 2030,
            'on_track': readiness_pct >= 50 or years_until_deadline > 3,
            'symmetric_pq_ready': has_aes_256 and not has_weak_sym,
            'hash_pq_ready': has_sha_384_plus and not has_weak_hash,
            'references': [
                'NSA CNSA 2.0 (updated May 2025): 2030 full PQC migration',
                'NIST IR 8547: Transition to Post-Quantum Cryptography Standards',
                'NIST SP 800-227: Recommendations for Key-Encapsulation Mechanisms',
            ],
            'assessed_at': datetime.now().isoformat(),
        }

    def assess_fhe_quantum_security(
        self,
        log_n: int = 15,
        max_levels: int = 40,
        scale_bits: int = 40,
    ) -> Dict[str, Any]:
        """
        Assess quantum security of FHE (CKKS) deployment in the context
        of the overall PQC security posture.

        This evaluates how FHE's reliance on Ring-LWE hardness interacts
        with the organization's PQC deployment (which uses Module-LWE).

        Args:
            log_n: CKKS ring dimension power (N = 2^log_n)
            max_levels: Maximum multiplicative depth
            scale_bits: Bits per rescaling level

        Returns:
            Dict with FHE quantum security assessment and business impact
        """
        from src.quantum_verification import CKKSSecurityVerifier
        verifier = CKKSSecurityVerifier()
        ckks_result = verifier.verify_ckks_config(
            log_n=log_n, max_levels=max_levels, scale_bits=scale_bits,
            config_name=f'Custom(log_n={log_n}, levels={max_levels})',
        )

        sec = ckks_result['security_assessment']
        nist_level = sec['nist_level_classical']
        within_128 = sec['within_128bit_bound']

        # Correlate with PQC deployment risk
        pqc_lattice_algorithms = [
            'ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024',
            'ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87',
        ]

        # Risk assessment
        risk_score = 100  # Start at 100 (safe)
        risk_factors = []

        if not within_128:
            risk_score -= 40
            risk_factors.append({
                'factor': 'FHE parameters exceed 128-bit security bound',
                'impact': 'critical',
                'detail': (
                    f'log Q = {sec.get("core_svp_classical", "?")} bits estimated '
                    f'Core-SVP. HE Standard max log Q = {sec["he_standard_max_log_q_128"]}.'
                ),
            })

        if nist_level < 1:
            risk_score -= 30
            risk_factors.append({
                'factor': 'FHE below NIST Level 1',
                'impact': 'critical',
                'detail': f'Classical Core-SVP = {sec["core_svp_classical"]} bits < 118 required.',
            })
        elif nist_level < 3:
            risk_score -= 10
            risk_factors.append({
                'factor': 'FHE below NIST Level 3',
                'impact': 'moderate',
                'detail': 'NIST Level 1 is minimum; Level 3 recommended for sensitive data.',
            })

        # Lattice monoculture penalty
        risk_score -= 15
        risk_factors.append({
            'factor': 'Lattice monoculture: FHE + PQC share Ring-LWE/Module-LWE',
            'impact': 'high',
            'detail': (
                'CKKS FHE and ML-KEM/ML-DSA all rely on lattice hardness. '
                'A lattice breakthrough compromises the ENTIRE encryption stack.'
            ),
        })

        risk_score = max(0, min(100, risk_score))

        return {
            'fhe_security': ckks_result,
            'risk_score': risk_score,
            'risk_category': (
                'critical' if risk_score < 25 else
                'high' if risk_score < 50 else
                'moderate' if risk_score < 75 else
                'low'
            ),
            'risk_factors': risk_factors,
            'shared_lattice_risk': {
                'pqc_algorithms_affected': pqc_lattice_algorithms,
                'fhe_schemes_affected': ['CKKS', 'BGV', 'BFV'],
                'non_lattice_alternatives': {
                    'pqc': 'HQC (code-based KEM), SLH-DSA (hash-based signatures)',
                    'fhe': 'None in practical deployment (all major FHE schemes are lattice-based)',
                },
                'diversification_strategy': (
                    'Use HQC for key exchange alongside ML-KEM to hedge against lattice '
                    'failure. For FHE, no alternative exists — mitigation is to use '
                    'conservative parameters (log_n ≥ 15, log Q within HE Standard bounds).'
                ),
            },
            'recommendations': [
                f'Ensure log Q ≤ {sec["he_standard_max_log_q_128"]} for 128-bit security at log_n={log_n}',
                'Monitor lattice security margin: quantum sieve improvements of 8% already observed',
                'Deploy HQC as backup KEM to hedge against lattice monoculture risk',
                'For CNSA 2.0 compliance, use AES-256 for symmetric layer alongside FHE',
                'Review FHE parameter selection when quantum sieve constants are updated',
            ],
            'assessed_at': datetime.now().isoformat(),
        }

    @staticmethod
    def create_sample_enterprise_inventory() -> List[CryptoAsset]:
        """Typical enterprise IT environment (pre-PQC migration)."""
        return [
            CryptoAsset('RSA-2048', 'key_exchange', 2048, 500, 'confidential',
                        'ML-KEM-768', 'high', 12, True, False,
                        'TLS certificates and VPN connections'),
            CryptoAsset('RSA-2048', 'digital_signature', 2048, 200, 'confidential',
                        'ML-DSA-65', 'high', 12, True, False,
                        'Code signing and document signatures'),
            CryptoAsset('ECDSA-256', 'digital_signature', 256, 100, 'internal',
                        'ML-DSA-44', 'medium', 6, True, False,
                        'API authentication tokens'),
            CryptoAsset('ECDH-256', 'key_exchange', 256, 150, 'internal',
                        'ML-KEM-512', 'medium', 6, True, False,
                        'Internal service-to-service TLS'),
            CryptoAsset('AES-256', 'symmetric_encryption', 256, 1000, 'confidential',
                        'AES-256 (adequate)', 'low', 0, False, True,
                        'Data at rest and in transit encryption'),
            CryptoAsset('AES-128', 'symmetric_encryption', 128, 300, 'internal',
                        'AES-256', 'low', 3, False, False,
                        'Legacy application encryption'),
            CryptoAsset('SHA-256', 'hash_function', 256, 800, 'internal',
                        'SHA-256 (adequate)', 'low', 0, False, True,
                        'Data integrity and checksums'),
            CryptoAsset('SHA-1', 'hash_function', 160, 50, 'public',
                        'SHA-256', 'low', 3, False, False,
                        'Legacy system checksums (deprecated)'),
        ]

    @staticmethod
    def create_sample_financial_inventory() -> List[CryptoAsset]:
        """Financial services environment with high-sensitivity data."""
        return [
            CryptoAsset('RSA-4096', 'key_exchange', 4096, 100, 'top_secret',
                        'ML-KEM-1024', 'high', 18, True, False,
                        'Payment gateway connections'),
            CryptoAsset('RSA-2048', 'digital_signature', 2048, 500, 'confidential',
                        'ML-DSA-65', 'high', 12, True, False,
                        'Transaction signing'),
            CryptoAsset('ECDSA-384', 'digital_signature', 384, 200, 'top_secret',
                        'ML-DSA-87', 'high', 12, True, False,
                        'Regulatory compliance signatures'),
            CryptoAsset('X25519', 'key_exchange', 256, 300, 'confidential',
                        'ML-KEM-768', 'medium', 6, True, False,
                        'TLS 1.3 key exchange'),
            CryptoAsset('AES-256', 'symmetric_encryption', 256, 2000, 'top_secret',
                        'AES-256 (adequate)', 'low', 0, False, True,
                        'Financial data encryption'),
            CryptoAsset('SHA-384', 'hash_function', 384, 500, 'confidential',
                        'SHA-384 (adequate)', 'low', 0, False, True,
                        'Transaction integrity'),
            CryptoAsset('ML-KEM-768', 'key_exchange', 768, 10, 'confidential',
                        'N/A (PQC)', 'low', 0, False, True,
                        'PQC pilot deployment'),
        ]

    @staticmethod
    def create_sample_government_inventory() -> List[CryptoAsset]:
        """Government/defense environment targeting CNSA 2.0 compliance."""
        return [
            CryptoAsset('RSA-3072', 'key_exchange', 3072, 200, 'top_secret',
                        'ML-KEM-1024', 'high', 18, True, False,
                        'Classified network connections'),
            CryptoAsset('ECDSA-384', 'digital_signature', 384, 300, 'top_secret',
                        'ML-DSA-87', 'high', 18, True, False,
                        'Classified document signing'),
            CryptoAsset('DH-3072', 'key_exchange', 3072, 100, 'confidential',
                        'ML-KEM-1024', 'high', 12, True, False,
                        'VPN key exchange'),
            CryptoAsset('AES-256', 'symmetric_encryption', 256, 1500, 'top_secret',
                        'AES-256 (adequate)', 'low', 0, False, True,
                        'NSA Type 1 encryption'),
            CryptoAsset('SHA-384', 'hash_function', 384, 1000, 'top_secret',
                        'SHA-384 (adequate)', 'low', 0, False, True,
                        'CNSA 2.0 compliant hashing'),
            CryptoAsset('ML-KEM-1024', 'key_exchange', 1024, 50, 'top_secret',
                        'N/A (PQC)', 'low', 0, False, True,
                        'CNSA 2.0 pilot program'),
            CryptoAsset('ML-DSA-87', 'digital_signature', 87, 30, 'top_secret',
                        'N/A (PQC)', 'low', 0, False, True,
                        'CNSA 2.0 pilot program'),
            CryptoAsset('SLH-DSA-SHA2-256f', 'digital_signature', 256, 10, 'top_secret',
                        'N/A (PQC)', 'low', 0, False, True,
                        'Firmware signing (CNSA 2.0)'),
        ]


# =============================================================================
# MODULE INFO
# =============================================================================

try:
    from .version_loader import get_version
    __version__ = get_version('security_scoring')
except ImportError:
    __version__ = "3.2.0"
__author__ = "PQC-FHE Integration Library"


if __name__ == "__main__":
    print(f"Security Scoring Framework v{__version__}")
    print("=" * 60)

    engine = SecurityScoringEngine()

    # Demo: Enterprise assessment
    print("\n--- Enterprise Security Assessment ---")
    inventory = engine.create_sample_enterprise_inventory()
    score = engine.calculate_overall_score(inventory, {
        'has_crypto_inventory': True,
        'has_migration_plan': False,
        'has_hybrid_deployment': False,
        'pqc_testing_started': True,
        'key_rotation_policy': True,
        'crypto_agility_framework': False,
    })

    print(f"Overall Score: {score.overall_score:.1f}/100 ({score.category})")
    print(f"  Algorithm Strength: {score.algorithm_strength_score:.1f}")
    print(f"  PQC Readiness:      {score.pqc_readiness_score:.1f}")
    print(f"  Compliance:         {score.compliance_score:.1f}")
    print(f"  Key Management:     {score.key_management_score:.1f}")
    print(f"  Crypto Agility:     {score.crypto_agility_score:.1f}")
    print(f"\nAssets: {score.total_assets} total, "
          f"{score.vulnerable_assets} vulnerable, "
          f"{score.pqc_ready_assets} PQC-ready")

    print(f"\nVulnerabilities ({len(score.vulnerabilities)}):")
    for v in score.vulnerabilities[:5]:
        print(f"  [{v['severity'].upper()}] {v['asset']}: {v['description']}")

    print(f"\nRecommendations ({len(score.recommendations)}):")
    for r in score.recommendations[:5]:
        print(f"  #{r['priority']} [{r['category'].upper()}] {r['title']}")
