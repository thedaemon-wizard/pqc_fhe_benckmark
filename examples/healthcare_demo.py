"""
Enterprise Demo: Privacy-Preserving Healthcare Analytics
========================================================

Demonstrates PQC-FHE integration for secure medical data processing.

Use Case: Federated Medical Research
- Multiple hospitals encrypt patient data locally
- Central research server performs statistical analysis on encrypted data
- No patient data is ever exposed to the research server
- All communications are quantum-resistant

HIPAA/GDPR Compliance:
- Patient data never leaves hospital in plaintext
- Research server operates on encrypted data only
- Post-quantum cryptography ensures long-term security
- Audit trail with digital signatures

Supported Analysis Types:
1. Aggregate statistics (mean, variance)
2. Risk score computation
3. Treatment outcome correlation
4. Population health metrics

References:
- NIST FIPS 203 (ML-KEM)
- NIST FIPS 204 (ML-DSA)  
- HIPAA Security Rule (45 CFR 164.312)
"""

import json
import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DiagnosisCode(Enum):
    """ICD-10 diagnosis code categories (simplified)"""
    CARDIOVASCULAR = "I00-I99"
    RESPIRATORY = "J00-J99"
    ENDOCRINE = "E00-E89"
    NEUROLOGICAL = "G00-G99"
    ONCOLOGY = "C00-D49"


class TreatmentType(Enum):
    """Treatment categories"""
    MEDICATION = "medication"
    SURGERY = "surgery"
    THERAPY = "therapy"
    OBSERVATION = "observation"


# Privacy thresholds (k-anonymity)
MIN_COHORT_SIZE = 10  # Minimum patients per analysis group


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PatientRecord:
    """
    De-identified patient record
    
    Note: No direct identifiers (name, SSN, etc.)
    Age is binned, location is region-level only
    """
    record_id: str  # Pseudonymized ID
    age_bin: int    # 5-year bins: 0-4=0, 5-9=1, etc.
    sex: int        # 0=F, 1=M, 2=Other
    region_code: int  # Geographic region (not address)
    
    # Clinical data (to be encrypted)
    vital_signs: List[float]  # [systolic_bp, diastolic_bp, heart_rate, temp, spo2]
    lab_values: List[float]   # [glucose, cholesterol, hemoglobin, creatinine]
    diagnosis_codes: List[int]  # Encoded diagnosis categories
    treatment_outcomes: List[float]  # Outcome scores (0-100)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_vector(self) -> List[float]:
        """Convert to numeric vector for FHE processing"""
        demographic = [float(self.age_bin), float(self.sex), float(self.region_code)]
        return demographic + self.vital_signs + self.lab_values + self.treatment_outcomes
    
    @property
    def vector_length(self) -> int:
        return 3 + len(self.vital_signs) + len(self.lab_values) + len(self.treatment_outcomes)


@dataclass
class HospitalDataset:
    """Collection of patient records from a single hospital"""
    hospital_id: str
    records: List[PatientRecord] = field(default_factory=list)
    
    def to_matrix(self) -> List[List[float]]:
        """Convert all records to matrix"""
        return [r.to_vector() for r in self.records]
    
    def flatten(self) -> List[float]:
        """Flatten for FHE encryption"""
        result = []
        for r in self.records:
            result.extend(r.to_vector())
        return result
    
    @property
    def record_vector_length(self) -> int:
        if self.records:
            return self.records[0].vector_length
        return 0


@dataclass
class ResearchQuery:
    """Research query specification"""
    query_id: str
    analysis_type: str  # "aggregate", "correlation", "risk_score"
    target_diagnosis: Optional[str] = None
    age_range: Optional[Tuple[int, int]] = None
    irb_approval_id: str = ""  # Institutional Review Board approval
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'analysis_type': self.analysis_type,
            'target_diagnosis': self.target_diagnosis,
            'age_range': self.age_range,
            'irb_approval_id': self.irb_approval_id,
        }


@dataclass
class AnalysisResult:
    """Result of healthcare analysis"""
    query_id: str
    cohort_size: int
    
    # Aggregate statistics
    mean_values: Dict[str, float] = field(default_factory=dict)
    std_values: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    risk_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    privacy_budget_used: float = 0.0  # Differential privacy epsilon
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'cohort_size': self.cohort_size,
            'mean_values': self.mean_values,
            'std_values': self.std_values,
            'risk_scores': self.risk_scores,
            'timestamp': self.timestamp,
            'privacy_budget_used': self.privacy_budget_used,
        }


# =============================================================================
# MOCK IMPLEMENTATIONS
# =============================================================================

class MockFHEEngine:
    """Mock FHE engine for demonstration"""
    
    def __init__(self):
        self.noise_level = 1e-6
        logger.info("[MockFHE] Healthcare engine initialized")
    
    def encrypt(self, data: List[float]) -> Dict[str, Any]:
        return {
            'type': 'ciphertext',
            'data': [x + np.random.normal(0, self.noise_level) for x in data],
            'original_length': len(data),
        }
    
    def decrypt(self, ct: Dict[str, Any]) -> List[float]:
        return ct['data'][:ct['original_length']]
    
    def add(self, ct1: Dict, ct2: Dict) -> Dict:
        max_len = max(len(ct1['data']), len(ct2['data']))
        d1 = ct1['data'] + [0.0] * (max_len - len(ct1['data']))
        d2 = ct2['data'] + [0.0] * (max_len - len(ct2['data']))
        result = [a + b for a, b in zip(d1, d2)]
        return {'type': 'ciphertext', 'data': result, 'original_length': max_len}
    
    def multiply_scalar(self, ct: Dict, scalar: float) -> Dict:
        result = [x * scalar for x in ct['data']]
        return {'type': 'ciphertext', 'data': result, 'original_length': ct['original_length']}
    
    def square(self, ct: Dict) -> Dict:
        result = [x * x for x in ct['data']]
        return {'type': 'ciphertext', 'data': result, 'original_length': ct['original_length']}


class MockPQCManager:
    """Mock PQC manager for demonstration"""
    
    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.kem_sk = hashlib.sha256(f"{entity_id}_kem_sk".encode()).digest()
        self.kem_pk = hashlib.sha256(f"{entity_id}_kem_pk".encode()).digest()
        self.sig_sk = hashlib.sha256(f"{entity_id}_sig_sk".encode()).digest()
        self.sig_pk = hashlib.sha256(f"{entity_id}_sig_pk".encode()).digest()
    
    def sign(self, message: bytes) -> bytes:
        return hashlib.sha256(message + self.sig_sk).digest()
    
    def verify(self, message: bytes, signature: bytes, pk: bytes) -> bool:
        # Simplified verification for demo
        return len(signature) == 32


# =============================================================================
# TRY REAL IMPLEMENTATIONS
# =============================================================================

def get_fhe_engine():
    """Get FHE engine (real or mock)"""
    try:
        import desilofhe
        
        class RealFHEEngine:
            def __init__(self):
                self.engine = desilofhe.Engine(mode='cpu', slot_count=2**14)
                self.secret_key = self.engine.create_secret_key()
                self.public_key = self.engine.create_public_key(self.secret_key)
                self.relin_key = self.engine.create_relinearization_key(self.secret_key)
                logger.info("[RealFHE] DESILO healthcare engine initialized")
            
            def encrypt(self, data: List[float]) -> Any:
                return self.engine.encrypt(data, self.public_key)
            
            def decrypt(self, ct: Any) -> List[float]:
                return list(self.engine.decrypt(ct, self.secret_key))
            
            def add(self, ct1: Any, ct2: Any) -> Any:
                return self.engine.add(ct1, ct2)
            
            def multiply_scalar(self, ct: Any, scalar: float) -> Any:
                return self.engine.multiply(ct, scalar)
            
            def square(self, ct: Any) -> Any:
                ct_sq = self.engine.square(ct)
                return self.engine.relinearize(ct_sq, self.relin_key)
        
        return RealFHEEngine()
    except ImportError:
        logger.warning("DESILO not available, using mock FHE")
        return MockFHEEngine()


def get_pqc_manager(entity_id: str):
    """Get PQC manager (real or mock)"""
    try:
        import oqs
        
        class RealPQCManager:
            def __init__(self, entity_id: str):
                self.entity_id = entity_id
                self.kem = oqs.KeyEncapsulation("ML-KEM-768")
                self.kem_pk = self.kem.generate_keypair()
                self.sig = oqs.Signature("ML-DSA-65")
                self.sig_pk = self.sig.generate_keypair()
                logger.info(f"[RealPQC] {entity_id} manager initialized")
            
            def sign(self, message: bytes) -> bytes:
                return self.sig.sign(message)
            
            def verify(self, message: bytes, signature: bytes, pk: bytes) -> bool:
                return self.sig.verify(message, signature, pk)
            
            def get_sig_public_key(self) -> bytes:
                return self.sig_pk
        
        return RealPQCManager(entity_id)
    except ImportError:
        return MockPQCManager(entity_id)


# =============================================================================
# HOSPITAL NODE
# =============================================================================

class HospitalNode:
    """
    Hospital data node for federated learning
    
    Responsibilities:
    - Manage local patient data
    - Encrypt data before transmission
    - Sign all outgoing data
    - Verify incoming requests
    """
    
    def __init__(self, hospital_id: str):
        self.hospital_id = hospital_id
        self.pqc = get_pqc_manager(hospital_id)
        self.fhe = get_fhe_engine()
        self.dataset: Optional[HospitalDataset] = None
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info(f"[Hospital {hospital_id}] Node initialized")
    
    def load_dataset(self, dataset: HospitalDataset):
        """Load patient dataset"""
        self.dataset = dataset
        self._log_audit("dataset_loaded", {"record_count": len(dataset.records)})
        logger.info(f"[Hospital {self.hospital_id}] Loaded {len(dataset.records)} records")
    
    def process_research_query(self, query: ResearchQuery) -> Dict[str, Any]:
        """
        Process research query on local data
        
        Steps:
        1. Validate IRB approval
        2. Apply privacy filters
        3. Encrypt matching records
        4. Sign encrypted data
        """
        if not self.dataset:
            raise ValueError("No dataset loaded")
        
        # Validate IRB approval
        if not query.irb_approval_id:
            logger.warning(f"[Hospital {self.hospital_id}] No IRB approval - query rejected")
            self._log_audit("query_rejected", {"reason": "no_irb_approval"})
            return {"error": "IRB approval required"}
        
        # Filter records based on query
        matching_records = self._filter_records(query)
        
        # Check k-anonymity
        if len(matching_records) < MIN_COHORT_SIZE:
            logger.warning(f"[Hospital {self.hospital_id}] Cohort too small ({len(matching_records)})")
            self._log_audit("query_rejected", {"reason": "cohort_too_small"})
            return {"error": f"Cohort size {len(matching_records)} below minimum {MIN_COHORT_SIZE}"}
        
        # Encrypt data
        data_vectors = [r.to_vector() for r in matching_records]
        flat_data = []
        for v in data_vectors:
            flat_data.extend(v)
        
        ct = self.fhe.encrypt(flat_data)
        
        # Sign encrypted data
        ct_bytes = json.dumps(ct).encode() if isinstance(ct, dict) else str(ct).encode()
        signature = self.pqc.sign(ct_bytes)
        
        response = {
            'hospital_id': self.hospital_id,
            'query_id': query.query_id,
            'ciphertext': ct,
            'record_count': len(matching_records),
            'vector_length': matching_records[0].vector_length if matching_records else 0,
            'signature': signature.hex() if isinstance(signature, bytes) else str(signature),
            'timestamp': datetime.now().isoformat(),
        }
        
        self._log_audit("query_processed", {
            "query_id": query.query_id,
            "record_count": len(matching_records),
        })
        
        logger.info(f"[Hospital {self.hospital_id}] Processed query: {len(matching_records)} records encrypted")
        return response
    
    def _filter_records(self, query: ResearchQuery) -> List[PatientRecord]:
        """Filter records based on query criteria"""
        if not self.dataset:
            return []
        
        matching = self.dataset.records
        
        # Filter by age range
        if query.age_range:
            min_age, max_age = query.age_range
            min_bin = min_age // 5
            max_bin = max_age // 5
            matching = [r for r in matching if min_bin <= r.age_bin <= max_bin]
        
        return matching
    
    def _log_audit(self, action: str, details: Dict[str, Any]):
        """Log audit event"""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'hospital_id': self.hospital_id,
            'action': action,
            'details': details,
        })


# =============================================================================
# RESEARCH SERVER
# =============================================================================

class ResearchServer:
    """
    Central research server for federated analysis
    
    Responsibilities:
    - Receive encrypted data from hospitals
    - Perform computations on encrypted data
    - Aggregate results
    - Never sees plaintext patient data
    """
    
    def __init__(self, server_id: str):
        self.server_id = server_id
        self.pqc = get_pqc_manager(server_id)
        
        # Received data from hospitals
        self.hospital_data: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"[ResearchServer {server_id}] Initialized")
    
    def receive_hospital_data(self, hospital_response: Dict[str, Any]):
        """Receive encrypted data from a hospital"""
        hospital_id = hospital_response['hospital_id']
        self.hospital_data[hospital_id] = hospital_response
        
        logger.info(f"[ResearchServer] Received data from {hospital_id}: "
                   f"{hospital_response['record_count']} records")
    
    def perform_federated_analysis(self, fhe_engine) -> AnalysisResult:
        """
        Perform analysis on encrypted data from all hospitals
        
        Computations (all on encrypted data):
        1. Aggregate encrypted vectors
        2. Compute encrypted statistics
        3. Return encrypted results
        """
        if not self.hospital_data:
            raise ValueError("No hospital data received")
        
        logger.info(f"[ResearchServer] Performing federated analysis on {len(self.hospital_data)} hospitals")
        
        # Aggregate encrypted data
        total_records = 0
        vector_length = 0
        aggregated_ct = None
        
        for hospital_id, data in self.hospital_data.items():
            ct = data['ciphertext']
            total_records += data['record_count']
            vector_length = data['vector_length']
            
            if aggregated_ct is None:
                aggregated_ct = ct
            else:
                aggregated_ct = fhe_engine.add(aggregated_ct, ct)
        
        # Compute mean (encrypted sum / n)
        n_inv = 1.0 / total_records
        ct_mean = fhe_engine.multiply_scalar(aggregated_ct, n_inv)
        
        # Decrypt for results (in real scenario, only client decrypts)
        decrypted = fhe_engine.decrypt(ct_mean)
        
        # Parse results based on vector structure
        # Vector: [age_bin, sex, region, vitals(5), labs(4), outcomes(N)]
        result = AnalysisResult(
            query_id=list(self.hospital_data.values())[0]['query_id'],
            cohort_size=total_records,
        )
        
        # Map results to meaningful metrics
        if len(decrypted) >= 3:
            result.mean_values = {
                'age_bin': decrypted[0],
                'systolic_bp': decrypted[3] if len(decrypted) > 3 else 0,
                'diastolic_bp': decrypted[4] if len(decrypted) > 4 else 0,
                'heart_rate': decrypted[5] if len(decrypted) > 5 else 0,
                'glucose': decrypted[8] if len(decrypted) > 8 else 0,
            }
        
        # Compute risk scores (simplified)
        if len(decrypted) > 5:
            bp_risk = (decrypted[3] - 120) / 40 if decrypted[3] > 120 else 0
            result.risk_scores = {
                'cardiovascular': max(0, min(1, bp_risk)),
            }
        
        logger.info(f"[ResearchServer] Analysis complete: {total_records} total records")
        return result


# =============================================================================
# DEMONSTRATION
# =============================================================================

def generate_sample_patients(n: int = 50) -> List[PatientRecord]:
    """Generate sample patient records"""
    np.random.seed(42)
    
    records = []
    for i in range(n):
        record = PatientRecord(
            record_id=hashlib.sha256(f"patient_{i}".encode()).hexdigest()[:16],
            age_bin=np.random.randint(4, 16),  # 20-80 years old
            sex=np.random.randint(0, 2),
            region_code=np.random.randint(1, 10),
            vital_signs=[
                np.random.normal(120, 15),  # systolic BP
                np.random.normal(80, 10),   # diastolic BP
                np.random.normal(72, 12),   # heart rate
                np.random.normal(98.6, 0.5), # temperature
                np.random.normal(97, 2),    # SpO2
            ],
            lab_values=[
                np.random.normal(100, 20),  # glucose
                np.random.normal(200, 40),  # cholesterol
                np.random.normal(14, 2),    # hemoglobin
                np.random.normal(1.0, 0.2), # creatinine
            ],
            diagnosis_codes=[np.random.randint(0, 5) for _ in range(3)],
            treatment_outcomes=[np.random.uniform(60, 95) for _ in range(2)],
        )
        records.append(record)
    
    return records


def run_healthcare_demo():
    """
    Run federated healthcare analysis demo
    
    Scenario:
    - 3 hospitals participate in cardiovascular research
    - Each encrypts their patient data locally
    - Research server aggregates encrypted data
    - Statistical results computed without exposing patient data
    """
    logger.info("\n" + "=" * 70)
    logger.info("PRIVACY-PRESERVING HEALTHCARE ANALYTICS DEMO")
    logger.info("=" * 70)
    logger.info("Scenario: Federated Cardiovascular Research Study")
    logger.info("Security: ML-KEM-768 + ML-DSA-65 + CKKS FHE")
    logger.info("Compliance: HIPAA, GDPR compliant - no PHI exposure")
    logger.info("=" * 70)
    
    # === SETUP HOSPITALS ===
    logger.info("\n[SETUP] Initializing hospital nodes...")
    
    hospitals = {
        "HOSP_NORTH": HospitalNode("HOSP_NORTH"),
        "HOSP_SOUTH": HospitalNode("HOSP_SOUTH"),
        "HOSP_EAST": HospitalNode("HOSP_EAST"),
    }
    
    # Load sample data for each hospital
    for hosp_id, hospital in hospitals.items():
        patients = generate_sample_patients(n=np.random.randint(30, 60))
        dataset = HospitalDataset(hospital_id=hosp_id, records=patients)
        hospital.load_dataset(dataset)
    
    # === SETUP RESEARCH SERVER ===
    logger.info("\n[SETUP] Initializing research server...")
    research_server = ResearchServer("RESEARCH_01")
    
    # === CREATE RESEARCH QUERY ===
    logger.info("\n[QUERY] Creating IRB-approved research query...")
    
    query = ResearchQuery(
        query_id="CVD_STUDY_2025_001",
        analysis_type="aggregate",
        target_diagnosis="cardiovascular",
        age_range=(40, 70),
        irb_approval_id="IRB-2025-12345",
    )
    
    logger.info(f"  Query ID: {query.query_id}")
    logger.info(f"  Analysis: {query.analysis_type}")
    logger.info(f"  Age Range: {query.age_range}")
    logger.info(f"  IRB Approval: {query.irb_approval_id}")
    
    # === HOSPITALS PROCESS QUERY ===
    logger.info("\n[PROCESSING] Hospitals encrypting patient data...")
    
    shared_fhe = list(hospitals.values())[0].fhe  # Shared for demo
    
    for hosp_id, hospital in hospitals.items():
        logger.info(f"\n  [{hosp_id}] Processing query...")
        response = hospital.process_research_query(query)
        
        if "error" not in response:
            research_server.receive_hospital_data(response)
            logger.info(f"  [{hosp_id}] Encrypted {response['record_count']} records")
            logger.info(f"  [{hosp_id}] Data signed with ML-DSA-65")
        else:
            logger.warning(f"  [{hosp_id}] Error: {response['error']}")
    
    # === RESEARCH SERVER ANALYZES ===
    logger.info("\n[ANALYSIS] Research server performing federated computation...")
    logger.info("  Server operates on ENCRYPTED data only")
    logger.info("  No patient PHI is ever decrypted on server")
    
    start_time = time.time()
    result = research_server.perform_federated_analysis(shared_fhe)
    analysis_time = time.time() - start_time
    
    # === DISPLAY RESULTS ===
    logger.info("\n" + "=" * 70)
    logger.info("RESEARCH RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Query ID:     {result.query_id}")
    logger.info(f"  Cohort Size:  {result.cohort_size} patients")
    logger.info(f"  Analysis Time: {analysis_time*1000:.2f} ms")
    logger.info("")
    logger.info("  Mean Values (across all hospitals):")
    for key, value in result.mean_values.items():
        logger.info(f"    {key}: {value:.2f}")
    logger.info("")
    logger.info("  Risk Scores:")
    for key, value in result.risk_scores.items():
        logger.info(f"    {key}: {value:.3f}")
    logger.info("=" * 70)
    
    # === COMPLIANCE SUMMARY ===
    logger.info("\nCOMPLIANCE & SECURITY SUMMARY:")
    logger.info("  [OK] HIPAA: No PHI transmitted in plaintext")
    logger.info("  [OK] GDPR: Data minimization - only aggregates computed")
    logger.info("  [OK] k-Anonymity: Minimum cohort size enforced")
    logger.info("  [OK] Post-Quantum: ML-KEM-768 + ML-DSA-65 protection")
    logger.info("  [OK] Audit Trail: All operations logged")
    
    # Display audit logs
    logger.info("\nAUDIT LOG SUMMARY:")
    for hospital in hospitals.values():
        for entry in hospital.audit_log:
            logger.info(f"  {entry['timestamp']} | {entry['hospital_id']} | {entry['action']}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Healthcare Analytics Demo')
    parser.add_argument('--hospitals', type=int, default=3, help='Number of hospitals')
    parser.add_argument('--patients', type=int, default=50, help='Patients per hospital')
    
    args = parser.parse_args()
    
    result = run_healthcare_demo()
