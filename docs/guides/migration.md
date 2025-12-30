# Migration Guide

This guide provides comprehensive strategies for migrating to PQC-FHE systems from classical cryptography, including hybrid approaches, data migration, and rollback procedures.

## Overview

Migration to post-quantum cryptography is mandated by NIST timeline (IR 8547):

| Milestone | Date | Requirement |
|-----------|------|-------------|
| Standards Published | August 2024 | FIPS 203, 204, 205 finalized |
| High-Risk Systems | 2025-2027 | Begin PQC migration |
| Deprecation | 2030 | RSA/ECC at 112-bit security deprecated |
| Full Migration | 2035 | Complete PQC transition |

**Key Drivers for Migration:**

- **Harvest Now, Decrypt Later (HNDL)**: Adversaries may store encrypted data for future quantum decryption
- **Regulatory Compliance**: Government and financial sector mandates
- **Long-lived Data**: Data requiring decades of protection needs immediate action
- **Supply Chain**: Dependencies on external systems transitioning to PQC

## Migration Strategies

### Strategy Comparison

| Strategy | Risk Level | Complexity | Timeline | Best For |
|----------|------------|------------|----------|----------|
| Big Bang | High | Low | Short | Small systems, greenfield |
| Phased | Medium | Medium | Medium | Most organizations |
| Hybrid | Low | High | Long | Risk-averse, critical systems |
| Parallel Run | Low | High | Long | High-availability requirements |

### Strategy 1: Hybrid Migration (Recommended)

Run classical and post-quantum algorithms in parallel during transition.

```python
"""
Hybrid migration implementation.

References:
- NIST SP 800-56C Rev. 2: Key Derivation Methods
- ETSI TS 103 744: Hybrid Key Exchange
- RFC 9180: Hybrid Public Key Encryption
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List
import hashlib
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class MigrationPhase(Enum):
    """Migration phases."""
    PREPARATION = "preparation"
    HYBRID_RECEIVE = "hybrid_receive"      # Accept both, prefer classical
    HYBRID_BALANCED = "hybrid_balanced"    # Accept both, no preference
    HYBRID_PREFER_PQC = "hybrid_prefer_pqc"  # Accept both, prefer PQC
    PQC_ONLY = "pqc_only"                  # PQC only, reject classical


@dataclass
class MigrationConfig:
    """Migration configuration."""
    current_phase: MigrationPhase = MigrationPhase.PREPARATION
    
    # Algorithm selection
    classical_kem: str = "X25519"
    pqc_kem: str = "ML-KEM-768"
    classical_dsa: str = "Ed25519"
    pqc_dsa: str = "ML-DSA-65"
    
    # Hybrid key derivation
    kdf_algorithm: str = "HKDF-SHA384"
    key_combiner: str = "concatenate"  # or "xor"
    
    # Timeline
    hybrid_start_date: Optional[datetime] = None
    pqc_only_date: Optional[datetime] = None
    
    # Monitoring
    log_classical_usage: bool = True
    alert_on_classical_only: bool = True


@dataclass
class HybridKeyPair:
    """Combined classical + PQC key pair."""
    classical_public: bytes
    classical_private: bytes
    pqc_public: bytes
    pqc_private: bytes
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_combined_public(self) -> bytes:
        """Get combined public key for distribution."""
        # Format: [4-byte classical length][classical key][pqc key]
        classical_len = len(self.classical_public).to_bytes(4, 'big')
        return classical_len + self.classical_public + self.pqc_public


@dataclass
class HybridSharedSecret:
    """Combined shared secret from hybrid key exchange."""
    classical_secret: bytes
    pqc_secret: bytes
    combined_secret: bytes
    used_classical: bool
    used_pqc: bool


class HybridKeyExchange:
    """
    Hybrid key exchange combining classical and PQC algorithms.
    
    Security guarantee: Combined scheme is secure if EITHER
    classical OR PQC scheme remains secure.
    
    Reference: ETSI TS 103 744
    """
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self._init_algorithms()
    
    def _init_algorithms(self):
        """Initialize cryptographic algorithms."""
        # In production, use actual cryptographic libraries
        # This is a structural example
        self.classical_kem = None  # Initialize X25519
        self.pqc_kem = None        # Initialize ML-KEM-768
    
    def generate_keypair(self) -> HybridKeyPair:
        """Generate hybrid key pair."""
        logger.info("Generating hybrid key pair")
        
        # Generate classical key pair (X25519)
        classical_private = self._generate_classical_private()
        classical_public = self._derive_classical_public(classical_private)
        
        # Generate PQC key pair (ML-KEM-768)
        pqc_private, pqc_public = self._generate_pqc_keypair()
        
        return HybridKeyPair(
            classical_public=classical_public,
            classical_private=classical_private,
            pqc_public=pqc_public,
            pqc_private=pqc_private
        )
    
    def encapsulate(self, peer_public: bytes) -> Tuple[bytes, HybridSharedSecret]:
        """
        Encapsulate shared secret using hybrid scheme.
        
        Returns:
            Tuple of (combined_ciphertext, shared_secret)
        """
        phase = self.config.current_phase
        
        # Parse peer's combined public key
        classical_pub, pqc_pub = self._parse_combined_public(peer_public)
        
        # Perform encapsulation based on phase
        if phase == MigrationPhase.PQC_ONLY:
            # PQC only
            pqc_ct, pqc_ss = self._pqc_encapsulate(pqc_pub)
            combined_ss = pqc_ss
            classical_ct = b''
            classical_ss = b''
            used_classical = False
            used_pqc = True
            
        elif phase in [MigrationPhase.HYBRID_RECEIVE, 
                       MigrationPhase.HYBRID_BALANCED,
                       MigrationPhase.HYBRID_PREFER_PQC]:
            # Hybrid mode: use both
            classical_ct, classical_ss = self._classical_encapsulate(classical_pub)
            pqc_ct, pqc_ss = self._pqc_encapsulate(pqc_pub)
            combined_ss = self._combine_secrets(classical_ss, pqc_ss)
            used_classical = True
            used_pqc = True
            
        else:
            raise ValueError(f"Invalid phase for encapsulation: {phase}")
        
        # Combine ciphertexts
        combined_ct = self._combine_ciphertexts(classical_ct, pqc_ct)
        
        shared_secret = HybridSharedSecret(
            classical_secret=classical_ss,
            pqc_secret=pqc_ss,
            combined_secret=combined_ss,
            used_classical=used_classical,
            used_pqc=used_pqc
        )
        
        if self.config.log_classical_usage and used_classical:
            logger.info("Classical algorithm used in hybrid exchange")
        
        return combined_ct, shared_secret
    
    def decapsulate(self, ciphertext: bytes, 
                    keypair: HybridKeyPair) -> HybridSharedSecret:
        """
        Decapsulate shared secret using hybrid scheme.
        
        Returns:
            HybridSharedSecret with combined secret
        """
        phase = self.config.current_phase
        
        # Parse combined ciphertext
        classical_ct, pqc_ct = self._parse_combined_ciphertext(ciphertext)
        
        classical_ss = b''
        pqc_ss = b''
        used_classical = False
        used_pqc = False
        
        # Decapsulate based on available ciphertexts and phase
        if pqc_ct:
            pqc_ss = self._pqc_decapsulate(pqc_ct, keypair.pqc_private)
            used_pqc = True
        
        if classical_ct and phase != MigrationPhase.PQC_ONLY:
            classical_ss = self._classical_decapsulate(
                classical_ct, keypair.classical_private
            )
            used_classical = True
        
        # Combine secrets
        if used_classical and used_pqc:
            combined_ss = self._combine_secrets(classical_ss, pqc_ss)
        elif used_pqc:
            combined_ss = pqc_ss
        elif used_classical:
            combined_ss = classical_ss
            if self.config.alert_on_classical_only:
                logger.warning("Classical-only decapsulation detected!")
        else:
            raise ValueError("No valid ciphertext components")
        
        return HybridSharedSecret(
            classical_secret=classical_ss,
            pqc_secret=pqc_ss,
            combined_secret=combined_ss,
            used_classical=used_classical,
            used_pqc=used_pqc
        )
    
    def _combine_secrets(self, classical: bytes, pqc: bytes) -> bytes:
        """
        Combine classical and PQC shared secrets.
        
        Uses HKDF for secure key derivation from both secrets.
        
        Reference: NIST SP 800-56C Rev. 2
        """
        if self.config.key_combiner == "concatenate":
            # HKDF(classical || pqc)
            combined_input = classical + pqc
        elif self.config.key_combiner == "xor":
            # XOR followed by HKDF (requires equal length)
            if len(classical) != len(pqc):
                # Pad shorter to match
                max_len = max(len(classical), len(pqc))
                classical = classical.ljust(max_len, b'\x00')
                pqc = pqc.ljust(max_len, b'\x00')
            combined_input = bytes(a ^ b for a, b in zip(classical, pqc))
        else:
            raise ValueError(f"Unknown combiner: {self.config.key_combiner}")
        
        # Derive final key using HKDF
        return self._hkdf_derive(combined_input, length=32)
    
    def _hkdf_derive(self, ikm: bytes, length: int = 32) -> bytes:
        """Derive key using HKDF-SHA384."""
        # Simplified - use actual HKDF in production
        h = hashlib.sha384(ikm).digest()
        return h[:length]
    
    # Placeholder methods - implement with actual crypto libraries
    def _generate_classical_private(self) -> bytes:
        """Generate X25519 private key."""
        import secrets
        return secrets.token_bytes(32)
    
    def _derive_classical_public(self, private: bytes) -> bytes:
        """Derive X25519 public key."""
        # Use actual X25519 implementation
        return hashlib.sha256(private).digest()
    
    def _generate_pqc_keypair(self) -> Tuple[bytes, bytes]:
        """Generate ML-KEM-768 key pair."""
        import secrets
        private = secrets.token_bytes(2400)  # Approximate ML-KEM-768 private key
        public = secrets.token_bytes(1184)   # ML-KEM-768 public key size
        return private, public
    
    def _classical_encapsulate(self, public: bytes) -> Tuple[bytes, bytes]:
        """X25519 key exchange."""
        import secrets
        ciphertext = secrets.token_bytes(32)
        shared_secret = secrets.token_bytes(32)
        return ciphertext, shared_secret
    
    def _pqc_encapsulate(self, public: bytes) -> Tuple[bytes, bytes]:
        """ML-KEM-768 encapsulation."""
        import secrets
        ciphertext = secrets.token_bytes(1088)  # ML-KEM-768 ciphertext size
        shared_secret = secrets.token_bytes(32)
        return ciphertext, shared_secret
    
    def _classical_decapsulate(self, ciphertext: bytes, 
                               private: bytes) -> bytes:
        """X25519 decapsulation."""
        return hashlib.sha256(ciphertext + private).digest()
    
    def _pqc_decapsulate(self, ciphertext: bytes, private: bytes) -> bytes:
        """ML-KEM-768 decapsulation."""
        return hashlib.sha256(ciphertext + private).digest()
    
    def _parse_combined_public(self, combined: bytes) -> Tuple[bytes, bytes]:
        """Parse combined public key."""
        classical_len = int.from_bytes(combined[:4], 'big')
        classical = combined[4:4+classical_len]
        pqc = combined[4+classical_len:]
        return classical, pqc
    
    def _combine_ciphertexts(self, classical: bytes, pqc: bytes) -> bytes:
        """Combine ciphertexts for transmission."""
        classical_len = len(classical).to_bytes(4, 'big')
        return classical_len + classical + pqc
    
    def _parse_combined_ciphertext(self, combined: bytes) -> Tuple[bytes, bytes]:
        """Parse combined ciphertext."""
        classical_len = int.from_bytes(combined[:4], 'big')
        classical = combined[4:4+classical_len] if classical_len > 0 else b''
        pqc = combined[4+classical_len:]
        return classical, pqc
```

### Strategy 2: Phased Migration

Migrate components incrementally with validation gates.

```python
"""
Phased migration implementation with validation gates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """System component types."""
    KEY_EXCHANGE = "key_exchange"
    DIGITAL_SIGNATURE = "digital_signature"
    DATA_ENCRYPTION = "data_encryption"
    TLS_HANDSHAKE = "tls_handshake"
    API_AUTHENTICATION = "api_authentication"
    DATABASE_ENCRYPTION = "database_encryption"


class MigrationStatus(Enum):
    """Migration status for a component."""
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    check_name: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationGate:
    """Validation gate between migration phases."""
    name: str
    description: str
    validators: List[Callable[[], ValidationResult]]
    required_pass_rate: float = 1.0  # 100% by default
    
    def evaluate(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validators and check pass rate."""
        results = [v() for v in self.validators]
        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / len(results) if results else 0
        
        gate_passed = pass_rate >= self.required_pass_rate
        
        logger.info(
            f"Gate '{self.name}': {passed}/{len(results)} passed "
            f"({pass_rate:.1%}), required: {self.required_pass_rate:.1%}"
        )
        
        return gate_passed, results


@dataclass
class ComponentMigration:
    """Migration state for a single component."""
    component_type: ComponentType
    name: str
    status: MigrationStatus = MigrationStatus.NOT_STARTED
    
    # Algorithm configuration
    current_algorithm: Optional[str] = None
    target_algorithm: Optional[str] = None
    
    # Timeline
    planned_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Validation gates
    pre_migration_gate: Optional[MigrationGate] = None
    post_migration_gate: Optional[MigrationGate] = None
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)


class PhasedMigrationManager:
    """
    Manages phased migration with validation gates.
    
    Ensures safe, incremental transition to PQC.
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentMigration] = {}
        self.migration_log: List[Dict[str, Any]] = []
    
    def register_component(self, component: ComponentMigration):
        """Register a component for migration."""
        self.components[component.name] = component
        logger.info(f"Registered component: {component.name}")
    
    def check_dependencies(self, component_name: str) -> bool:
        """Check if component dependencies are migrated."""
        component = self.components.get(component_name)
        if not component:
            raise ValueError(f"Unknown component: {component_name}")
        
        for dep_name in component.depends_on:
            dep = self.components.get(dep_name)
            if not dep or dep.status != MigrationStatus.COMPLETED:
                logger.warning(
                    f"Dependency '{dep_name}' not completed for '{component_name}'"
                )
                return False
        
        return True
    
    def start_migration(self, component_name: str) -> bool:
        """Start migration for a component."""
        component = self.components.get(component_name)
        if not component:
            raise ValueError(f"Unknown component: {component_name}")
        
        # Check dependencies
        if not self.check_dependencies(component_name):
            logger.error(f"Cannot start migration: dependencies not met")
            return False
        
        # Run pre-migration gate
        if component.pre_migration_gate:
            passed, results = component.pre_migration_gate.evaluate()
            if not passed:
                logger.error(f"Pre-migration gate failed for '{component_name}'")
                self._log_event(component_name, "pre_gate_failed", results)
                return False
        
        # Update status
        component.status = MigrationStatus.IN_PROGRESS
        component.actual_start = datetime.utcnow()
        
        self._log_event(component_name, "migration_started", {
            "current_algorithm": component.current_algorithm,
            "target_algorithm": component.target_algorithm
        })
        
        logger.info(f"Migration started for '{component_name}'")
        return True
    
    def complete_migration(self, component_name: str) -> bool:
        """Complete migration for a component."""
        component = self.components.get(component_name)
        if not component:
            raise ValueError(f"Unknown component: {component_name}")
        
        if component.status != MigrationStatus.IN_PROGRESS:
            logger.error(f"Component '{component_name}' not in progress")
            return False
        
        # Update to validating
        component.status = MigrationStatus.VALIDATING
        
        # Run post-migration gate
        if component.post_migration_gate:
            passed, results = component.post_migration_gate.evaluate()
            if not passed:
                logger.error(f"Post-migration gate failed for '{component_name}'")
                self._log_event(component_name, "post_gate_failed", results)
                return False
        
        # Mark completed
        component.status = MigrationStatus.COMPLETED
        component.completed_at = datetime.utcnow()
        component.current_algorithm = component.target_algorithm
        
        self._log_event(component_name, "migration_completed", {
            "duration_seconds": (
                component.completed_at - component.actual_start
            ).total_seconds() if component.actual_start else None
        })
        
        logger.info(f"Migration completed for '{component_name}'")
        return True
    
    def rollback(self, component_name: str, reason: str) -> bool:
        """Rollback migration for a component."""
        component = self.components.get(component_name)
        if not component:
            raise ValueError(f"Unknown component: {component_name}")
        
        if component.status not in [MigrationStatus.IN_PROGRESS, 
                                     MigrationStatus.VALIDATING]:
            logger.error(f"Cannot rollback component '{component_name}'")
            return False
        
        component.status = MigrationStatus.ROLLED_BACK
        
        self._log_event(component_name, "rollback", {"reason": reason})
        
        logger.warning(f"Rolled back '{component_name}': {reason}")
        return True
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get overall migration status."""
        status_counts = {}
        for component in self.components.values():
            status = component.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total = len(self.components)
        completed = status_counts.get("completed", 0)
        
        return {
            "total_components": total,
            "completed": completed,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "status_breakdown": status_counts,
            "components": {
                name: {
                    "status": c.status.value,
                    "current": c.current_algorithm,
                    "target": c.target_algorithm
                }
                for name, c in self.components.items()
            }
        }
    
    def _log_event(self, component: str, event: str, details: Any):
        """Log migration event."""
        self.migration_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "event": event,
            "details": details
        })


# Example validation functions
def validate_performance() -> ValidationResult:
    """Validate performance meets requirements."""
    # Simulated check
    latency_ms = 45  # Example measured latency
    threshold_ms = 100
    
    passed = latency_ms < threshold_ms
    
    return ValidationResult(
        passed=passed,
        check_name="performance",
        message=f"Latency: {latency_ms}ms (threshold: {threshold_ms}ms)",
        details={"latency_ms": latency_ms, "threshold_ms": threshold_ms}
    )


def validate_interoperability() -> ValidationResult:
    """Validate interoperability with existing systems."""
    # Simulated check
    tests_passed = 45
    tests_total = 50
    pass_rate = tests_passed / tests_total
    
    passed = pass_rate >= 0.95
    
    return ValidationResult(
        passed=passed,
        check_name="interoperability",
        message=f"Passed {tests_passed}/{tests_total} tests ({pass_rate:.1%})",
        details={"passed": tests_passed, "total": tests_total}
    )


def validate_security_audit() -> ValidationResult:
    """Validate security audit completion."""
    # Simulated check
    audit_complete = True
    findings_resolved = True
    
    passed = audit_complete and findings_resolved
    
    return ValidationResult(
        passed=passed,
        check_name="security_audit",
        message="Security audit completed, all findings resolved",
        details={"audit_complete": audit_complete, 
                 "findings_resolved": findings_resolved}
    )


# Example usage
def setup_migration_plan():
    """Set up a phased migration plan."""
    manager = PhasedMigrationManager()
    
    # Define validation gates
    pre_gate = MigrationGate(
        name="pre-migration-checks",
        description="Validate readiness for migration",
        validators=[validate_performance, validate_interoperability]
    )
    
    post_gate = MigrationGate(
        name="post-migration-validation",
        description="Validate successful migration",
        validators=[validate_performance, validate_interoperability, 
                    validate_security_audit]
    )
    
    # Register components in dependency order
    manager.register_component(ComponentMigration(
        component_type=ComponentType.KEY_EXCHANGE,
        name="internal-key-exchange",
        current_algorithm="X25519",
        target_algorithm="ML-KEM-768 + X25519",
        pre_migration_gate=pre_gate,
        post_migration_gate=post_gate
    ))
    
    manager.register_component(ComponentMigration(
        component_type=ComponentType.TLS_HANDSHAKE,
        name="api-tls",
        current_algorithm="TLS 1.3 + X25519",
        target_algorithm="TLS 1.3 + ML-KEM-768",
        depends_on=["internal-key-exchange"],
        pre_migration_gate=pre_gate,
        post_migration_gate=post_gate
    ))
    
    manager.register_component(ComponentMigration(
        component_type=ComponentType.DIGITAL_SIGNATURE,
        name="api-signatures",
        current_algorithm="Ed25519",
        target_algorithm="ML-DSA-65 + Ed25519",
        depends_on=["internal-key-exchange"],
        pre_migration_gate=pre_gate,
        post_migration_gate=post_gate
    ))
    
    return manager
```

## Data Migration

### Encrypted Data Re-encryption

```python
"""
Data migration utilities for transitioning encrypted data to PQC.

Handles re-encryption of existing data with new algorithms.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Callable, Any
from datetime import datetime
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


logger = logging.getLogger(__name__)


@dataclass
class DataRecord:
    """Encrypted data record."""
    id: str
    ciphertext: bytes
    key_id: str
    algorithm: str
    created_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class MigrationProgress:
    """Migration progress tracking."""
    total_records: int = 0
    processed_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100
    
    @property
    def success_rate(self) -> float:
        if self.processed_records == 0:
            return 0.0
        return (self.successful_records / self.processed_records) * 100


class DataMigrationManager:
    """
    Manages encrypted data migration to PQC.
    
    Features:
    - Batch processing with configurable size
    - Progress tracking and resumption
    - Parallel processing
    - Validation and rollback support
    - Audit logging
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        max_workers: int = 4,
        validate_after_migration: bool = True,
        keep_original: bool = True,
        dry_run: bool = False
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.validate_after_migration = validate_after_migration
        self.keep_original = keep_original
        self.dry_run = dry_run
        
        self.progress = MigrationProgress()
        self._lock = threading.Lock()
        self._stop_requested = False
        
        # Callbacks
        self.on_record_migrated: Optional[Callable[[str], None]] = None
        self.on_record_failed: Optional[Callable[[str, Exception], None]] = None
        self.on_batch_complete: Optional[Callable[[int], None]] = None
    
    def migrate_data(
        self,
        source_reader: Callable[[], Iterator[DataRecord]],
        target_writer: Callable[[DataRecord], bool],
        old_decryptor: Callable[[bytes, str], bytes],
        new_encryptor: Callable[[bytes], tuple],
        checkpoint_writer: Optional[Callable[[str], None]] = None,
        resume_from: Optional[str] = None
    ) -> MigrationProgress:
        """
        Migrate encrypted data from old encryption to PQC.
        
        Args:
            source_reader: Function returning iterator of source records
            target_writer: Function to write migrated records
            old_decryptor: Function to decrypt with old algorithm
            new_encryptor: Function to encrypt with PQC (returns ciphertext, key_id)
            checkpoint_writer: Optional function to save progress checkpoints
            resume_from: Optional record ID to resume from
            
        Returns:
            MigrationProgress with final statistics
        """
        logger.info(f"Starting data migration (dry_run={self.dry_run})")
        
        self.progress = MigrationProgress(start_time=datetime.utcnow())
        self._stop_requested = False
        
        # Get source records
        records = source_reader()
        
        # Skip to resume point if specified
        if resume_from:
            records = self._skip_to_resume_point(records, resume_from)
        
        # Process in batches
        batch = []
        batch_num = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for record in records:
                if self._stop_requested:
                    logger.warning("Migration stopped by request")
                    break
                
                batch.append(record)
                
                if len(batch) >= self.batch_size:
                    batch_num += 1
                    self._process_batch(
                        batch, executor, old_decryptor, new_encryptor,
                        target_writer, checkpoint_writer
                    )
                    batch = []
            
            # Process remaining records
            if batch:
                self._process_batch(
                    batch, executor, old_decryptor, new_encryptor,
                    target_writer, checkpoint_writer
                )
        
        self.progress.end_time = datetime.utcnow()
        
        logger.info(
            f"Migration complete: {self.progress.successful_records}/"
            f"{self.progress.processed_records} successful "
            f"({self.progress.success_rate:.1f}%)"
        )
        
        return self.progress
    
    def _process_batch(
        self,
        batch: List[DataRecord],
        executor: ThreadPoolExecutor,
        old_decryptor: Callable,
        new_encryptor: Callable,
        target_writer: Callable,
        checkpoint_writer: Optional[Callable]
    ):
        """Process a batch of records."""
        futures = {}
        
        for record in batch:
            future = executor.submit(
                self._migrate_single_record,
                record, old_decryptor, new_encryptor, target_writer
            )
            futures[future] = record.id
        
        for future in as_completed(futures):
            record_id = futures[future]
            try:
                success = future.result()
                with self._lock:
                    self.progress.processed_records += 1
                    if success:
                        self.progress.successful_records += 1
                        if self.on_record_migrated:
                            self.on_record_migrated(record_id)
                    else:
                        self.progress.skipped_records += 1
            except Exception as e:
                with self._lock:
                    self.progress.processed_records += 1
                    self.progress.failed_records += 1
                logger.error(f"Failed to migrate record {record_id}: {e}")
                if self.on_record_failed:
                    self.on_record_failed(record_id, e)
        
        # Save checkpoint
        if checkpoint_writer and batch:
            checkpoint_writer(batch[-1].id)
        
        if self.on_batch_complete:
            self.on_batch_complete(len(batch))
        
        logger.info(
            f"Batch complete: {self.progress.progress_percent:.1f}% "
            f"({self.progress.processed_records} records)"
        )
    
    def _migrate_single_record(
        self,
        record: DataRecord,
        old_decryptor: Callable,
        new_encryptor: Callable,
        target_writer: Callable
    ) -> bool:
        """Migrate a single record."""
        try:
            # Decrypt with old algorithm
            plaintext = old_decryptor(record.ciphertext, record.key_id)
            
            # Encrypt with new PQC algorithm
            new_ciphertext, new_key_id = new_encryptor(plaintext)
            
            # Validate if required
            if self.validate_after_migration:
                # Re-decrypt and compare
                # This is a placeholder - implement actual validation
                pass
            
            if self.dry_run:
                logger.debug(f"Dry run: would migrate record {record.id}")
                return True
            
            # Create migrated record
            migrated_record = DataRecord(
                id=record.id,
                ciphertext=new_ciphertext,
                key_id=new_key_id,
                algorithm="ML-KEM-768",
                created_at=datetime.utcnow(),
                metadata={
                    **record.metadata,
                    "migrated_from": record.algorithm,
                    "original_key_id": record.key_id,
                    "migration_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Write to target
            success = target_writer(migrated_record)
            
            return success
            
        except Exception as e:
            logger.error(f"Error migrating record {record.id}: {e}")
            raise
    
    def _skip_to_resume_point(
        self,
        records: Iterator[DataRecord],
        resume_from: str
    ) -> Iterator[DataRecord]:
        """Skip records until resume point."""
        found_resume_point = False
        
        for record in records:
            if found_resume_point:
                yield record
            elif record.id == resume_from:
                found_resume_point = True
                logger.info(f"Resuming from record {resume_from}")
        
        if not found_resume_point:
            logger.warning(f"Resume point {resume_from} not found")
    
    def stop(self):
        """Request migration to stop."""
        self._stop_requested = True
        logger.info("Migration stop requested")


@dataclass
class KeyMigrationRecord:
    """Key migration tracking record."""
    old_key_id: str
    new_key_id: str
    old_algorithm: str
    new_algorithm: str
    migrated_at: datetime
    data_records_count: int


class KeyMigrationManager:
    """
    Manages cryptographic key migration.
    
    Handles the lifecycle of key transitions including:
    - Key generation with new algorithms
    - Key mapping (old -> new)
    - Grace period management
    - Key retirement
    """
    
    def __init__(
        self,
        key_store,
        grace_period_days: int = 90,
        parallel_key_usage: bool = True
    ):
        self.key_store = key_store
        self.grace_period_days = grace_period_days
        self.parallel_key_usage = parallel_key_usage
        
        self.key_mappings: Dict[str, KeyMigrationRecord] = {}
    
    def migrate_key(
        self,
        old_key_id: str,
        new_algorithm: str = "ML-KEM-768"
    ) -> KeyMigrationRecord:
        """
        Migrate a key to new algorithm.
        
        Args:
            old_key_id: ID of existing key
            new_algorithm: Target algorithm
            
        Returns:
            KeyMigrationRecord with mapping information
        """
        # Get old key info
        old_key_info = self.key_store.get_key_info(old_key_id)
        if not old_key_info:
            raise ValueError(f"Key {old_key_id} not found")
        
        # Generate new key
        new_key_id = self.key_store.generate_key(
            algorithm=new_algorithm,
            metadata={
                "migrated_from": old_key_id,
                "original_algorithm": old_key_info.get("algorithm")
            }
        )
        
        # Create mapping record
        record = KeyMigrationRecord(
            old_key_id=old_key_id,
            new_key_id=new_key_id,
            old_algorithm=old_key_info.get("algorithm", "unknown"),
            new_algorithm=new_algorithm,
            migrated_at=datetime.utcnow(),
            data_records_count=0
        )
        
        self.key_mappings[old_key_id] = record
        
        logger.info(
            f"Key migrated: {old_key_id} ({record.old_algorithm}) -> "
            f"{new_key_id} ({new_algorithm})"
        )
        
        return record
    
    def get_active_key_id(self, old_key_id: str) -> str:
        """
        Get the active key ID for operations.
        
        During migration, returns new key ID if available.
        """
        if old_key_id in self.key_mappings:
            return self.key_mappings[old_key_id].new_key_id
        return old_key_id
    
    def can_retire_key(self, old_key_id: str) -> bool:
        """Check if old key can be safely retired."""
        if old_key_id not in self.key_mappings:
            return False
        
        record = self.key_mappings[old_key_id]
        grace_period_end = record.migrated_at + timedelta(
            days=self.grace_period_days
        )
        
        if datetime.utcnow() < grace_period_end:
            return False
        
        # Check if any data still uses old key
        # This is a placeholder - implement actual check
        return True
    
    def retire_key(self, old_key_id: str) -> bool:
        """Retire old key after migration."""
        if not self.can_retire_key(old_key_id):
            logger.warning(f"Cannot retire key {old_key_id} yet")
            return False
        
        # Mark key for deletion
        self.key_store.schedule_deletion(old_key_id)
        
        logger.info(f"Key {old_key_id} scheduled for retirement")
        return True
```

## FHE Migration

### Transitioning to Homomorphic Encryption

```python
"""
FHE migration utilities.

Handles transition from plaintext or classical encryption to FHE.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class FHEMigrationPhase(Enum):
    """FHE migration phases."""
    ASSESSMENT = "assessment"      # Evaluate FHE suitability
    PROTOTYPE = "prototype"        # Develop FHE implementation
    PARALLEL = "parallel"          # Run FHE alongside existing
    VALIDATION = "validation"      # Validate FHE results
    CUTOVER = "cutover"           # Switch to FHE only
    OPTIMIZATION = "optimization"  # Optimize FHE performance


@dataclass
class ComputationProfile:
    """Profile of a computation for FHE migration assessment."""
    name: str
    description: str
    
    # Operation characteristics
    input_size_bytes: int
    output_size_bytes: int
    operations: List[str]  # add, multiply, compare, etc.
    multiplicative_depth: int
    
    # Performance requirements
    max_latency_ms: Optional[float] = None
    throughput_required: Optional[float] = None
    
    # Data sensitivity
    data_sensitivity: str = "high"  # low, medium, high, critical
    
    # Current implementation
    current_encryption: Optional[str] = None  # None = plaintext
    
    def __post_init__(self):
        self.fhe_suitable = self._assess_fhe_suitability()
    
    def _assess_fhe_suitability(self) -> Dict[str, Any]:
        """Assess suitability for FHE migration."""
        assessment = {
            "suitable": True,
            "concerns": [],
            "recommendations": []
        }
        
        # Check multiplicative depth
        if self.multiplicative_depth > 20:
            assessment["concerns"].append(
                f"High multiplicative depth ({self.multiplicative_depth}) "
                "may require frequent bootstrapping"
            )
            assessment["recommendations"].append(
                "Consider circuit optimization to reduce depth"
            )
        
        # Check latency requirements
        if self.max_latency_ms and self.max_latency_ms < 100:
            assessment["concerns"].append(
                f"Strict latency requirement ({self.max_latency_ms}ms) "
                "may be challenging for FHE"
            )
            assessment["recommendations"].append(
                "Consider GPU acceleration or hybrid approach"
            )
        
        # Check for unsupported operations
        unsupported = {"division", "sqrt", "log", "exp"}
        problematic_ops = set(self.operations) & unsupported
        if problematic_ops:
            assessment["concerns"].append(
                f"Operations {problematic_ops} require polynomial approximation"
            )
            assessment["recommendations"].append(
                "Use polynomial approximations or pre-compute lookup tables"
            )
        
        # Determine overall suitability
        assessment["suitable"] = len(assessment["concerns"]) < 3
        
        return assessment


@dataclass 
class FHEMigrationPlan:
    """Migration plan for transitioning to FHE."""
    computation: ComputationProfile
    current_phase: FHEMigrationPhase = FHEMigrationPhase.ASSESSMENT
    
    # FHE parameters
    target_scheme: str = "CKKS"  # or BFV, BGV
    poly_modulus_degree: int = 8192
    security_level: int = 128
    
    # Validation
    validation_strategy: str = "parallel_compare"
    acceptable_error: float = 1e-6  # For CKKS approximate results
    
    # Timeline
    phases_completed: List[FHEMigrationPhase] = field(default_factory=list)
    
    def advance_phase(self) -> bool:
        """Advance to next migration phase."""
        phase_order = list(FHEMigrationPhase)
        current_idx = phase_order.index(self.current_phase)
        
        if current_idx < len(phase_order) - 1:
            self.phases_completed.append(self.current_phase)
            self.current_phase = phase_order[current_idx + 1]
            logger.info(f"Advanced to phase: {self.current_phase.value}")
            return True
        
        return False


class FHEMigrationManager:
    """
    Manages FHE migration process.
    """
    
    def __init__(self):
        self.plans: Dict[str, FHEMigrationPlan] = {}
    
    def assess_computation(
        self,
        profile: ComputationProfile
    ) -> Dict[str, Any]:
        """
        Assess a computation for FHE migration.
        
        Returns detailed assessment with recommendations.
        """
        assessment = profile.fhe_suitable.copy()
        
        # Estimate FHE parameters
        assessment["recommended_params"] = self._estimate_parameters(profile)
        
        # Estimate performance impact
        assessment["performance_estimate"] = self._estimate_performance(profile)
        
        # Estimate implementation effort
        assessment["effort_estimate"] = self._estimate_effort(profile)
        
        return assessment
    
    def _estimate_parameters(
        self,
        profile: ComputationProfile
    ) -> Dict[str, Any]:
        """Estimate optimal FHE parameters."""
        # Based on multiplicative depth
        depth = profile.multiplicative_depth
        
        if depth <= 5:
            poly_degree = 4096
            scheme = "CKKS"
        elif depth <= 15:
            poly_degree = 8192
            scheme = "CKKS"
        elif depth <= 30:
            poly_degree = 16384
            scheme = "CKKS"
        else:
            poly_degree = 32768
            scheme = "CKKS"
        
        return {
            "scheme": scheme,
            "poly_modulus_degree": poly_degree,
            "security_level": 128,
            "estimated_ciphertext_size_kb": poly_degree * 8 // 1024,
            "bootstrapping_required": depth > 20
        }
    
    def _estimate_performance(
        self,
        profile: ComputationProfile
    ) -> Dict[str, Any]:
        """Estimate FHE performance characteristics."""
        params = self._estimate_parameters(profile)
        
        # Very rough estimates (actual depends on many factors)
        base_encrypt_ms = params["poly_modulus_degree"] / 1000
        base_mult_ms = params["poly_modulus_degree"] / 500
        base_add_ms = params["poly_modulus_degree"] / 10000
        
        # Estimate total latency
        mult_count = profile.operations.count("multiply")
        add_count = profile.operations.count("add")
        
        estimated_latency_ms = (
            base_encrypt_ms * 2 +  # encrypt + decrypt
            base_mult_ms * mult_count +
            base_add_ms * add_count
        )
        
        # Add bootstrapping time if needed
        if params["bootstrapping_required"]:
            bootstrap_count = profile.multiplicative_depth // 15
            estimated_latency_ms += bootstrap_count * 500  # ~500ms per bootstrap
        
        return {
            "estimated_latency_ms": estimated_latency_ms,
            "slowdown_factor": estimated_latency_ms / (profile.max_latency_ms or 1000),
            "cpu_recommended": estimated_latency_ms < 5000,
            "gpu_recommended": estimated_latency_ms >= 5000
        }
    
    def _estimate_effort(
        self,
        profile: ComputationProfile
    ) -> Dict[str, Any]:
        """Estimate implementation effort."""
        complexity = len(profile.operations)
        
        if complexity <= 5:
            effort = "low"
            days = 5
        elif complexity <= 15:
            effort = "medium"
            days = 15
        else:
            effort = "high"
            days = 30
        
        # Add time for problematic operations
        problematic = {"division", "comparison", "branching"}
        if set(profile.operations) & problematic:
            days += 10
            effort = "high" if effort != "high" else effort
        
        return {
            "effort_level": effort,
            "estimated_days": days,
            "requires_approximations": bool(
                set(profile.operations) & {"division", "sqrt", "log", "exp"}
            ),
            "requires_custom_circuits": complexity > 20
        }
    
    def create_migration_plan(
        self,
        profile: ComputationProfile
    ) -> FHEMigrationPlan:
        """Create a migration plan for a computation."""
        params = self._estimate_parameters(profile)
        
        plan = FHEMigrationPlan(
            computation=profile,
            target_scheme=params["scheme"],
            poly_modulus_degree=params["poly_modulus_degree"],
            security_level=params["security_level"]
        )
        
        self.plans[profile.name] = plan
        
        logger.info(f"Created migration plan for '{profile.name}'")
        return plan
    
    def validate_fhe_implementation(
        self,
        computation_name: str,
        plaintext_func,
        fhe_func,
        test_inputs: List[Any],
        acceptable_error: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Validate FHE implementation against plaintext version.
        
        Args:
            computation_name: Name of computation
            plaintext_func: Original plaintext function
            fhe_func: FHE implementation
            test_inputs: List of test inputs
            acceptable_error: Maximum acceptable error (for CKKS)
            
        Returns:
            Validation results
        """
        results = {
            "total_tests": len(test_inputs),
            "passed": 0,
            "failed": 0,
            "errors": [],
            "max_error": 0.0
        }
        
        for i, test_input in enumerate(test_inputs):
            try:
                # Run plaintext version
                expected = plaintext_func(test_input)
                
                # Run FHE version
                actual = fhe_func(test_input)
                
                # Compare results
                if hasattr(expected, '__iter__'):
                    errors = [abs(e - a) for e, a in zip(expected, actual)]
                    max_error = max(errors)
                else:
                    max_error = abs(expected - actual)
                
                results["max_error"] = max(results["max_error"], max_error)
                
                if max_error <= acceptable_error:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "test_index": i,
                        "error": max_error,
                        "expected": expected,
                        "actual": actual
                    })
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "test_index": i,
                    "exception": str(e)
                })
        
        results["success_rate"] = results["passed"] / results["total_tests"]
        results["validation_passed"] = results["success_rate"] >= 0.99
        
        return results
```

## Rollback Procedures

### Rollback Implementation

```python
"""
Rollback procedures for PQC-FHE migration.

Provides safe rollback capabilities for all migration types.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import logging
import json
import copy


logger = logging.getLogger(__name__)


class RollbackType(Enum):
    """Types of rollback operations."""
    ALGORITHM = "algorithm"      # Rollback algorithm change
    KEY = "key"                  # Rollback key migration
    DATA = "data"                # Rollback data migration
    CONFIGURATION = "config"     # Rollback configuration
    FULL = "full"                # Full system rollback


@dataclass
class RollbackPoint:
    """Snapshot for rollback."""
    id: str
    created_at: datetime
    rollback_type: RollbackType
    description: str
    
    # State snapshot
    state_snapshot: Dict[str, Any]
    
    # Metadata
    created_by: str
    tags: List[str] = field(default_factory=list)
    
    # Validation
    validated: bool = False
    validation_result: Optional[Dict[str, Any]] = None


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    rollback_point_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Details
    components_rolled_back: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Verification
    verified: bool = False
    verification_result: Optional[Dict[str, Any]] = None


class RollbackManager:
    """
    Manages rollback operations for PQC-FHE migrations.
    
    Features:
    - Automatic rollback point creation
    - Staged rollback with verification
    - Audit logging
    - Partial rollback support
    """
    
    def __init__(
        self,
        max_rollback_points: int = 10,
        auto_create_points: bool = True,
        verify_after_rollback: bool = True
    ):
        self.max_rollback_points = max_rollback_points
        self.auto_create_points = auto_create_points
        self.verify_after_rollback = verify_after_rollback
        
        self.rollback_points: Dict[str, RollbackPoint] = {}
        self.rollback_history: List[RollbackResult] = []
        
        # Component handlers
        self.state_captors: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.state_restorers: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self.state_verifiers: Dict[str, Callable[[], bool]] = {}
    
    def register_component(
        self,
        component_name: str,
        captor: Callable[[], Dict[str, Any]],
        restorer: Callable[[Dict[str, Any]], bool],
        verifier: Optional[Callable[[], bool]] = None
    ):
        """
        Register a component for rollback management.
        
        Args:
            component_name: Unique component identifier
            captor: Function to capture current state
            restorer: Function to restore state from snapshot
            verifier: Optional function to verify state after restore
        """
        self.state_captors[component_name] = captor
        self.state_restorers[component_name] = restorer
        if verifier:
            self.state_verifiers[component_name] = verifier
        
        logger.info(f"Registered rollback component: {component_name}")
    
    def create_rollback_point(
        self,
        rollback_type: RollbackType,
        description: str,
        created_by: str = "system",
        components: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> RollbackPoint:
        """
        Create a rollback point with current state.
        
        Args:
            rollback_type: Type of rollback this point supports
            description: Human-readable description
            created_by: Creator identifier
            components: Specific components to capture (None = all)
            tags: Optional tags for categorization
            
        Returns:
            Created RollbackPoint
        """
        # Capture state
        state_snapshot = {}
        target_components = components or list(self.state_captors.keys())
        
        for component in target_components:
            if component in self.state_captors:
                try:
                    state_snapshot[component] = self.state_captors[component]()
                except Exception as e:
                    logger.error(f"Failed to capture state for {component}: {e}")
                    raise
        
        # Generate ID
        point_id = f"rp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{rollback_type.value}"
        
        # Create rollback point
        point = RollbackPoint(
            id=point_id,
            created_at=datetime.utcnow(),
            rollback_type=rollback_type,
            description=description,
            state_snapshot=state_snapshot,
            created_by=created_by,
            tags=tags or []
        )
        
        # Store point
        self.rollback_points[point_id] = point
        
        # Cleanup old points
        self._cleanup_old_points()
        
        logger.info(
            f"Created rollback point {point_id}: {description} "
            f"({len(state_snapshot)} components)"
        )
        
        return point
    
    def validate_rollback_point(self, point_id: str) -> bool:
        """
        Validate that a rollback point can be used.
        
        Checks:
        - Point exists
        - State snapshot is complete
        - All components can be restored
        """
        if point_id not in self.rollback_points:
            logger.error(f"Rollback point {point_id} not found")
            return False
        
        point = self.rollback_points[point_id]
        
        # Check all components have restorers
        for component in point.state_snapshot.keys():
            if component not in self.state_restorers:
                logger.error(f"No restorer for component {component}")
                return False
        
        # Additional validation could include:
        # - Schema validation of state snapshots
        # - Dependency checking
        # - Resource availability
        
        point.validated = True
        point.validation_result = {
            "validated_at": datetime.utcnow().isoformat(),
            "components_validated": list(point.state_snapshot.keys())
        }
        
        logger.info(f"Rollback point {point_id} validated successfully")
        return True
    
    def execute_rollback(
        self,
        point_id: str,
        components: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> RollbackResult:
        """
        Execute rollback to a specific point.
        
        Args:
            point_id: ID of rollback point
            components: Specific components to rollback (None = all)
            dry_run: If True, simulate without actual changes
            
        Returns:
            RollbackResult with operation details
        """
        result = RollbackResult(
            success=False,
            rollback_point_id=point_id,
            started_at=datetime.utcnow()
        )
        
        # Validate point
        if not self.validate_rollback_point(point_id):
            result.errors.append(f"Invalid rollback point: {point_id}")
            return result
        
        point = self.rollback_points[point_id]
        target_components = components or list(point.state_snapshot.keys())
        
        logger.info(
            f"{'Simulating' if dry_run else 'Executing'} rollback to {point_id} "
            f"({len(target_components)} components)"
        )
        
        # Execute rollback for each component
        for component in target_components:
            if component not in point.state_snapshot:
                result.warnings.append(f"No snapshot for component {component}")
                continue
            
            if component not in self.state_restorers:
                result.errors.append(f"No restorer for component {component}")
                continue
            
            try:
                if dry_run:
                    logger.info(f"  [DRY RUN] Would restore {component}")
                else:
                    state = point.state_snapshot[component]
                    success = self.state_restorers[component](state)
                    
                    if success:
                        result.components_rolled_back.append(component)
                        logger.info(f"  Restored {component}")
                    else:
                        result.errors.append(f"Failed to restore {component}")
                        
            except Exception as e:
                result.errors.append(f"Error restoring {component}: {e}")
                logger.error(f"  Error restoring {component}: {e}")
        
        # Verify if requested
        if self.verify_after_rollback and not dry_run:
            result.verification_result = self._verify_rollback(target_components)
            result.verified = all(
                result.verification_result.get(c, False)
                for c in result.components_rolled_back
            )
        
        # Determine success
        result.success = (
            len(result.errors) == 0 and
            len(result.components_rolled_back) == len(target_components)
        )
        result.completed_at = datetime.utcnow()
        
        # Log result
        self.rollback_history.append(result)
        
        status = "SUCCESS" if result.success else "FAILED"
        logger.info(
            f"Rollback {status}: {len(result.components_rolled_back)}/"
            f"{len(target_components)} components restored"
        )
        
        return result
    
    def _verify_rollback(
        self,
        components: List[str]
    ) -> Dict[str, bool]:
        """Verify rollback was successful."""
        results = {}
        
        for component in components:
            if component in self.state_verifiers:
                try:
                    results[component] = self.state_verifiers[component]()
                except Exception as e:
                    logger.error(f"Verification failed for {component}: {e}")
                    results[component] = False
            else:
                # Assume success if no verifier
                results[component] = True
        
        return results
    
    def _cleanup_old_points(self):
        """Remove old rollback points beyond limit."""
        if len(self.rollback_points) <= self.max_rollback_points:
            return
        
        # Sort by creation time
        sorted_points = sorted(
            self.rollback_points.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest
        to_remove = len(sorted_points) - self.max_rollback_points
        for point_id, _ in sorted_points[:to_remove]:
            del self.rollback_points[point_id]
            logger.info(f"Removed old rollback point: {point_id}")
    
    def list_rollback_points(
        self,
        rollback_type: Optional[RollbackType] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List available rollback points."""
        points = []
        
        for point in self.rollback_points.values():
            # Filter by type
            if rollback_type and point.rollback_type != rollback_type:
                continue
            
            # Filter by tags
            if tags and not any(t in point.tags for t in tags):
                continue
            
            points.append({
                "id": point.id,
                "type": point.rollback_type.value,
                "description": point.description,
                "created_at": point.created_at.isoformat(),
                "created_by": point.created_by,
                "components": list(point.state_snapshot.keys()),
                "validated": point.validated,
                "tags": point.tags
            })
        
        return sorted(points, key=lambda x: x["created_at"], reverse=True)
    
    def export_rollback_point(self, point_id: str) -> str:
        """Export rollback point to JSON."""
        if point_id not in self.rollback_points:
            raise ValueError(f"Rollback point {point_id} not found")
        
        point = self.rollback_points[point_id]
        
        export_data = {
            "id": point.id,
            "created_at": point.created_at.isoformat(),
            "rollback_type": point.rollback_type.value,
            "description": point.description,
            "created_by": point.created_by,
            "tags": point.tags,
            "state_snapshot": point.state_snapshot,
            "validated": point.validated
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def import_rollback_point(self, json_data: str) -> RollbackPoint:
        """Import rollback point from JSON."""
        data = json.loads(json_data)
        
        point = RollbackPoint(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            rollback_type=RollbackType(data["rollback_type"]),
            description=data["description"],
            state_snapshot=data["state_snapshot"],
            created_by=data["created_by"],
            tags=data.get("tags", []),
            validated=data.get("validated", False)
        )
        
        self.rollback_points[point.id] = point
        return point
```

## Testing Migration

### Migration Testing Framework

```python
"""
Testing framework for PQC-FHE migrations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


@dataclass
class MigrationTest:
    """Definition of a migration test."""
    name: str
    description: str
    category: str  # functional, performance, security, compatibility
    
    # Test function
    test_func: Callable[[], bool]
    
    # Expected behavior
    expected_result: bool = True
    timeout_seconds: float = 60.0
    
    # Criticality
    critical: bool = False  # If True, migration stops on failure
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a migration test."""
    test_name: str
    passed: bool
    execution_time_ms: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class MigrationTestSuite:
    """
    Test suite for migration validation.
    """
    
    def __init__(self):
        self.tests: Dict[str, MigrationTest] = {}
        self.results: List[TestResult] = []
    
    def add_test(self, test: MigrationTest):
        """Add a test to the suite."""
        self.tests[test.name] = test
    
    def run_tests(
        self,
        categories: Optional[List[str]] = None,
        stop_on_critical_failure: bool = True
    ) -> Dict[str, Any]:
        """
        Run all tests in the suite.
        
        Args:
            categories: Optional list of categories to run
            stop_on_critical_failure: Stop if critical test fails
            
        Returns:
            Summary of test results
        """
        self.results = []
        
        # Filter tests by category
        tests_to_run = []
        for test in self.tests.values():
            if categories is None or test.category in categories:
                tests_to_run.append(test)
        
        # Sort by dependencies
        tests_to_run = self._sort_by_dependencies(tests_to_run)
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test in tests_to_run:
            # Check dependencies
            deps_passed = all(
                any(r.test_name == d and r.passed for r in self.results)
                for d in test.depends_on
            )
            
            if not deps_passed:
                logger.warning(f"Skipping {test.name}: dependencies not met")
                skipped += 1
                continue
            
            # Run test
            result = self._run_single_test(test)
            self.results.append(result)
            
            if result.passed:
                passed += 1
            else:
                failed += 1
                if test.critical and stop_on_critical_failure:
                    logger.error(
                        f"Critical test {test.name} failed, stopping suite"
                    )
                    break
        
        return {
            "total": len(tests_to_run),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
            "results": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "time_ms": r.execution_time_ms,
                    "message": r.message
                }
                for r in self.results
            ]
        }
    
    def _run_single_test(self, test: MigrationTest) -> TestResult:
        """Run a single test."""
        import time
        
        start = time.perf_counter()
        
        try:
            result = test.test_func()
            passed = result == test.expected_result
            message = "Passed" if passed else f"Expected {test.expected_result}, got {result}"
            
        except Exception as e:
            passed = False
            message = f"Exception: {e}"
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return TestResult(
            test_name=test.name,
            passed=passed,
            execution_time_ms=elapsed_ms,
            message=message
        )
    
    def _sort_by_dependencies(
        self,
        tests: List[MigrationTest]
    ) -> List[MigrationTest]:
        """Sort tests by dependencies (topological sort)."""
        # Simple implementation - production would use proper graph sort
        sorted_tests = []
        remaining = tests.copy()
        
        while remaining:
            # Find tests with no unmet dependencies
            ready = [
                t for t in remaining
                if all(
                    d in [st.name for st in sorted_tests]
                    for d in t.depends_on
                )
            ]
            
            if not ready:
                # Circular dependency or missing dependency
                sorted_tests.extend(remaining)
                break
            
            sorted_tests.extend(ready)
            for t in ready:
                remaining.remove(t)
        
        return sorted_tests


# Example test definitions
def create_standard_migration_tests() -> MigrationTestSuite:
    """Create standard test suite for PQC migration."""
    suite = MigrationTestSuite()
    
    # Functional tests
    suite.add_test(MigrationTest(
        name="pqc_key_generation",
        description="Verify PQC key generation works",
        category="functional",
        test_func=lambda: True,  # Replace with actual test
        critical=True
    ))
    
    suite.add_test(MigrationTest(
        name="pqc_encapsulation",
        description="Verify PQC encapsulation/decapsulation",
        category="functional",
        test_func=lambda: True,
        critical=True,
        depends_on=["pqc_key_generation"]
    ))
    
    suite.add_test(MigrationTest(
        name="hybrid_key_exchange",
        description="Verify hybrid key exchange",
        category="functional",
        test_func=lambda: True,
        critical=True,
        depends_on=["pqc_encapsulation"]
    ))
    
    # Performance tests
    suite.add_test(MigrationTest(
        name="keygen_performance",
        description="Key generation within latency budget",
        category="performance",
        test_func=lambda: True,
        depends_on=["pqc_key_generation"]
    ))
    
    suite.add_test(MigrationTest(
        name="throughput_test",
        description="System meets throughput requirements",
        category="performance",
        test_func=lambda: True
    ))
    
    # Security tests
    suite.add_test(MigrationTest(
        name="key_isolation",
        description="Keys properly isolated in memory",
        category="security",
        test_func=lambda: True,
        critical=True
    ))
    
    suite.add_test(MigrationTest(
        name="side_channel_basic",
        description="Basic side-channel resistance",
        category="security",
        test_func=lambda: True
    ))
    
    # Compatibility tests
    suite.add_test(MigrationTest(
        name="backward_compatibility",
        description="Can decrypt pre-migration data",
        category="compatibility",
        test_func=lambda: True,
        critical=True
    ))
    
    suite.add_test(MigrationTest(
        name="api_compatibility",
        description="API contract unchanged",
        category="compatibility",
        test_func=lambda: True
    ))
    
    return suite
```

## Summary

This migration guide covers:

1. **Hybrid Migration Strategy**: Combining classical and PQC algorithms for safe transition
2. **Phased Migration**: Incremental component migration with validation gates
3. **Data Migration**: Re-encrypting existing data with PQC algorithms
4. **FHE Migration**: Transitioning computations to homomorphic encryption
5. **Rollback Procedures**: Safe rollback capabilities with state snapshots
6. **Testing Framework**: Comprehensive testing for migration validation

## Key Recommendations

1. **Start with Hybrid**: Use hybrid cryptography during transition to maintain security if either scheme is broken
2. **Prioritize by Risk**: Migrate systems handling long-lived sensitive data first (HNDL threat)
3. **Validate Continuously**: Use validation gates between migration phases
4. **Maintain Rollback Capability**: Always have tested rollback procedures ready
5. **Test Extensively**: Run comprehensive test suites before each migration phase
6. **Document Everything**: Maintain detailed audit logs for compliance

## References

- NIST IR 8547: Transition to Post-Quantum Cryptography Standards
- NIST SP 800-56C Rev. 2: Recommendation for Key-Derivation Methods
- ETSI TS 103 744: Quantum-Safe Hybrid Key Exchanges
- RFC 9180: Hybrid Public Key Encryption
- IETF draft-ietf-tls-hybrid-design: Hybrid Key Exchange in TLS 1.3
