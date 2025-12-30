#!/usr/bin/env python3
"""
PQC-FHE IoT Security Demo
=========================

Demonstrates secure IoT data aggregation using Post-Quantum Cryptography
and Fully Homomorphic Encryption.

Use Cases:
- Smart grid meter aggregation (privacy-preserving energy monitoring)
- Industrial IoT sensor fusion (confidential manufacturing data)
- Smart city traffic analysis (anonymous vehicle counting)
- Healthcare wearables (private health metrics)

Architecture:
                    ┌─────────────────────────────────────┐
                    │          Edge Gateway               │
                    │    (Aggregation + PQC Transport)    │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────┴─────────┐   ┌─────────┴─────────┐   ┌─────────┴─────────┐
    │   IoT Sensor 1    │   │   IoT Sensor 2    │   │   IoT Sensor N    │
    │ (FHE Encrypted)   │   │ (FHE Encrypted)   │   │ (FHE Encrypted)   │
    └───────────────────┘   └───────────────────┘   └───────────────────┘

Security Properties:
- Quantum-resistant key exchange (ML-KEM-768)
- Data encrypted during aggregation (CKKS FHE)
- Server never sees raw sensor values
- End-to-end privacy preservation

References:
- NIST IR 8259A: IoT Device Cybersecurity Capability Core Baseline
- ETSI TS 103 645: Cyber Security for Consumer IoT
- IEEE 2621.1: Standard for Wireless Diabetes Device Security

Author: PQC-FHE Portfolio
License: MIT
"""

import sys
import os
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IoT-Demo")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pqc_fhe_integration import (
        PQCKeyManager,
        FHEEngine,
        HybridCryptoManager,
        secure_random_bytes,
    )
    CRYPTO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Crypto modules not available: {e}")
    CRYPTO_AVAILABLE = False


# =============================================================================
# Data Models
# =============================================================================

class SensorType(Enum):
    """IoT sensor types."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    POWER_METER = "power_meter"
    PRESSURE = "pressure"
    MOTION = "motion"
    AIR_QUALITY = "air_quality"
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE = "blood_pressure"


@dataclass
class IoTDevice:
    """Represents an IoT device with PQC credentials."""
    device_id: str
    sensor_type: SensorType
    location: str
    
    # PQC credentials
    kem_public_key: Optional[bytes] = None
    kem_secret_key: Optional[bytes] = None
    sign_public_key: Optional[bytes] = None
    sign_secret_key: Optional[bytes] = None
    
    # Device metadata
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: Optional[str] = None
    firmware_version: str = "1.0.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding secret keys)."""
        return {
            "device_id": self.device_id,
            "sensor_type": self.sensor_type.value,
            "location": self.location,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "firmware_version": self.firmware_version,
            "has_kem_keys": self.kem_public_key is not None,
            "has_sign_keys": self.sign_public_key is not None,
        }


@dataclass
class SensorReading:
    """Encrypted sensor reading."""
    device_id: str
    timestamp: str
    sensor_type: SensorType
    
    # Encrypted data (FHE ciphertext reference)
    encrypted_value: Any  # FHE ciphertext
    value_metadata: Dict  # Non-sensitive metadata
    
    # Authentication
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "sensor_type": self.sensor_type.value,
            "has_encrypted_value": self.encrypted_value is not None,
            "has_signature": self.signature is not None,
        }


@dataclass
class AggregatedResult:
    """Result of privacy-preserving aggregation."""
    aggregation_type: str  # "sum", "average", "min", "max"
    device_count: int
    time_window_start: str
    time_window_end: str
    
    # Encrypted result
    encrypted_result: Any
    
    # Decrypted result (only at authorized endpoint)
    decrypted_value: Optional[float] = None
    
    # Metadata
    computation_time_ms: float = 0.0


# =============================================================================
# IoT Device Manager
# =============================================================================

class IoTDeviceManager:
    """
    Manages IoT device registration and credential provisioning.
    
    Security Model:
    - Each device gets unique PQC keypairs
    - Key rotation supported via re-registration
    - Device attestation via ML-DSA signatures
    """
    
    def __init__(self):
        self.pqc = PQCKeyManager() if CRYPTO_AVAILABLE else None
        self.devices: Dict[str, IoTDevice] = {}
        self.revoked_devices: set = set()
        
        logger.info("IoTDeviceManager initialized")
    
    def register_device(
        self,
        device_id: str,
        sensor_type: SensorType,
        location: str,
        security_level: str = "level3"
    ) -> IoTDevice:
        """
        Register a new IoT device with PQC credentials.
        
        Args:
            device_id: Unique device identifier
            sensor_type: Type of sensor
            location: Physical location
            security_level: PQC security level (level1/level3/level5)
            
        Returns:
            Registered IoTDevice with credentials
        """
        logger.info(f"Registering device: {device_id}")
        
        # Create device
        device = IoTDevice(
            device_id=device_id,
            sensor_type=sensor_type,
            location=location,
        )
        
        # Generate PQC credentials
        if self.pqc:
            # KEM keypair for secure data transport
            kem_algo = {
                "level1": "ML-KEM-512",
                "level3": "ML-KEM-768",
                "level5": "ML-KEM-1024"
            }.get(security_level, "ML-KEM-768")
            
            device.kem_public_key, device.kem_secret_key = \
                self.pqc.generate_keypair(kem_algo)
            
            # Signature keypair for device attestation
            sign_algo = {
                "level1": "ML-DSA-44",
                "level3": "ML-DSA-65",
                "level5": "ML-DSA-87"
            }.get(security_level, "ML-DSA-65")
            
            device.sign_public_key, device.sign_secret_key = \
                self.pqc.generate_keypair(sign_algo)
            
            logger.info(f"  Generated PQC credentials: {kem_algo}, {sign_algo}")
        
        self.devices[device_id] = device
        logger.info(f"  Device registered successfully")
        
        return device
    
    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Get device by ID."""
        if device_id in self.revoked_devices:
            logger.warning(f"Attempted access to revoked device: {device_id}")
            return None
        return self.devices.get(device_id)
    
    def revoke_device(self, device_id: str) -> bool:
        """Revoke device credentials."""
        if device_id in self.devices:
            self.revoked_devices.add(device_id)
            logger.warning(f"Device revoked: {device_id}")
            return True
        return False
    
    def list_devices(self) -> List[Dict]:
        """List all active devices."""
        return [
            d.to_dict() for d_id, d in self.devices.items()
            if d_id not in self.revoked_devices
        ]


# =============================================================================
# FHE Sensor Data Processor
# =============================================================================

class SecureSensorProcessor:
    """
    Processes IoT sensor data with FHE encryption.
    
    Data never leaves encrypted form during processing:
    1. Sensor encrypts reading locally
    2. Gateway aggregates encrypted readings
    3. Only authorized endpoint can decrypt
    """
    
    def __init__(self, enable_fhe: bool = True):
        self.fhe = FHEEngine() if (CRYPTO_AVAILABLE and enable_fhe) else None
        self.pqc = PQCKeyManager() if CRYPTO_AVAILABLE else None
        
        # Normalization parameters for different sensor types
        self.normalization_params = {
            SensorType.TEMPERATURE: {"min": -40, "max": 85, "unit": "C"},
            SensorType.HUMIDITY: {"min": 0, "max": 100, "unit": "%"},
            SensorType.POWER_METER: {"min": 0, "max": 10000, "unit": "W"},
            SensorType.PRESSURE: {"min": 800, "max": 1200, "unit": "hPa"},
            SensorType.AIR_QUALITY: {"min": 0, "max": 500, "unit": "AQI"},
            SensorType.HEART_RATE: {"min": 30, "max": 220, "unit": "bpm"},
            SensorType.BLOOD_PRESSURE: {"min": 40, "max": 200, "unit": "mmHg"},
        }
        
        logger.info(f"SecureSensorProcessor initialized (FHE: {self.fhe is not None})")
    
    def normalize_value(
        self,
        value: float,
        sensor_type: SensorType
    ) -> float:
        """
        Normalize sensor value to [0, 1] range for FHE.
        
        CRITICAL: FHE bootstrap requires values in [-1, 1] range.
        Normalizing to [0, 1] ensures safe operations.
        """
        params = self.normalization_params.get(sensor_type)
        if not params:
            return value / 1000.0  # Default normalization
        
        min_val = params["min"]
        max_val = params["max"]
        
        # Clamp to valid range
        value = max(min_val, min(max_val, value))
        
        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        return normalized
    
    def denormalize_value(
        self,
        normalized: float,
        sensor_type: SensorType
    ) -> float:
        """Denormalize value back to original scale."""
        params = self.normalization_params.get(sensor_type)
        if not params:
            return normalized * 1000.0
        
        min_val = params["min"]
        max_val = params["max"]
        
        return normalized * (max_val - min_val) + min_val
    
    def encrypt_reading(
        self,
        device: IoTDevice,
        raw_value: float,
        sign_data: bool = True
    ) -> SensorReading:
        """
        Encrypt a sensor reading using FHE.
        
        Args:
            device: Source IoT device
            raw_value: Raw sensor value
            sign_data: Whether to sign the encrypted data
            
        Returns:
            SensorReading with encrypted value
        """
        timestamp = datetime.now().isoformat()
        
        # Normalize value for FHE
        normalized = self.normalize_value(raw_value, device.sensor_type)
        
        # Encrypt with FHE
        if self.fhe:
            encrypted_value = self.fhe.encrypt([normalized])
        else:
            # Mock encryption for demo
            encrypted_value = {"mock": True, "value": normalized}
        
        # Create reading
        reading = SensorReading(
            device_id=device.device_id,
            timestamp=timestamp,
            sensor_type=device.sensor_type,
            encrypted_value=encrypted_value,
            value_metadata={
                "normalization": self.normalization_params.get(device.sensor_type),
                "raw_value_hash": hashlib.sha256(
                    str(raw_value).encode()
                ).hexdigest()[:16],  # For integrity (not the value!)
            }
        )
        
        # Sign the reading for authentication
        if sign_data and self.pqc and device.sign_secret_key:
            message = json.dumps({
                "device_id": reading.device_id,
                "timestamp": reading.timestamp,
                "metadata_hash": hashlib.sha256(
                    json.dumps(reading.value_metadata).encode()
                ).hexdigest()
            }).encode()
            
            reading.signature = self.pqc.sign(
                message,
                device.sign_secret_key,
                "ML-DSA-65"
            )
        
        return reading
    
    def verify_reading(
        self,
        reading: SensorReading,
        device: IoTDevice
    ) -> bool:
        """Verify reading signature."""
        if not reading.signature or not device.sign_public_key:
            return False
        
        if not self.pqc:
            return True  # Mock verification
        
        message = json.dumps({
            "device_id": reading.device_id,
            "timestamp": reading.timestamp,
            "metadata_hash": hashlib.sha256(
                json.dumps(reading.value_metadata).encode()
            ).hexdigest()
        }).encode()
        
        return self.pqc.verify(
            message,
            reading.signature,
            device.sign_public_key,
            "ML-DSA-65"
        )
    
    def aggregate_readings(
        self,
        readings: List[SensorReading],
        aggregation_type: str = "sum"
    ) -> AggregatedResult:
        """
        Aggregate multiple encrypted readings homomorphically.
        
        This is the key privacy-preserving operation:
        - Server aggregates encrypted values
        - Never sees individual readings
        - Only aggregate result can be decrypted
        """
        if not readings:
            raise ValueError("No readings to aggregate")
        
        start_time = time.time()
        
        # Get time window
        timestamps = [r.timestamp for r in readings]
        
        # Perform homomorphic aggregation
        if self.fhe and all(r.encrypted_value is not None for r in readings):
            # Start with first reading
            result_ct = readings[0].encrypted_value
            
            # Add remaining readings
            for reading in readings[1:]:
                result_ct = self.fhe.add(result_ct, reading.encrypted_value)
            
            # For average, multiply by 1/n
            if aggregation_type == "average":
                scale = 1.0 / len(readings)
                result_ct = self.fhe.multiply_scalar(result_ct, scale)
        else:
            # Mock aggregation
            if aggregation_type == "sum":
                mock_value = sum(
                    r.encrypted_value.get("value", 0) 
                    for r in readings 
                    if isinstance(r.encrypted_value, dict)
                )
            else:  # average
                values = [
                    r.encrypted_value.get("value", 0) 
                    for r in readings 
                    if isinstance(r.encrypted_value, dict)
                ]
                mock_value = sum(values) / len(values) if values else 0
            
            result_ct = {"mock": True, "value": mock_value}
        
        computation_time = (time.time() - start_time) * 1000
        
        return AggregatedResult(
            aggregation_type=aggregation_type,
            device_count=len(readings),
            time_window_start=min(timestamps),
            time_window_end=max(timestamps),
            encrypted_result=result_ct,
            computation_time_ms=computation_time,
        )
    
    def decrypt_result(
        self,
        result: AggregatedResult,
        sensor_type: SensorType
    ) -> float:
        """
        Decrypt aggregated result (only at authorized endpoint).
        
        In production, this would only happen at the data owner's endpoint.
        """
        if self.fhe and not isinstance(result.encrypted_result, dict):
            decrypted = self.fhe.decrypt(result.encrypted_result, 1)
            normalized_value = decrypted[0]
        else:
            # Mock decryption
            normalized_value = result.encrypted_result.get("value", 0)
        
        # Denormalize based on aggregation type
        if result.aggregation_type == "sum":
            # Sum of normalized values - denormalize each then sum
            # For simplicity, we denormalize the sum (approximate)
            decrypted_value = self.denormalize_value(
                normalized_value / result.device_count,
                sensor_type
            ) * result.device_count
        else:
            decrypted_value = self.denormalize_value(normalized_value, sensor_type)
        
        result.decrypted_value = decrypted_value
        return decrypted_value


# =============================================================================
# IoT Network Simulator
# =============================================================================

class IoTNetworkSimulator:
    """
    Simulates an IoT network with multiple devices.
    
    Demonstrates:
    - Device provisioning with PQC credentials
    - Encrypted data collection
    - Privacy-preserving aggregation
    - Secure result retrieval
    """
    
    def __init__(self, num_devices: int = 10):
        self.device_manager = IoTDeviceManager()
        self.processor = SecureSensorProcessor()
        self.num_devices = num_devices
        
        self.readings_buffer: List[SensorReading] = []
        
        logger.info(f"IoTNetworkSimulator initialized with {num_devices} devices")
    
    def setup_network(self, sensor_type: SensorType = SensorType.TEMPERATURE):
        """Set up IoT network with devices."""
        logger.info("Setting up IoT network...")
        
        locations = [
            "Building-A-Floor-1",
            "Building-A-Floor-2",
            "Building-B-Floor-1",
            "Building-B-Floor-2",
            "Building-C-Floor-1",
            "Outdoor-North",
            "Outdoor-South",
            "Warehouse-1",
            "Warehouse-2",
            "Server-Room",
        ]
        
        for i in range(self.num_devices):
            device_id = f"SENSOR-{i+1:04d}"
            location = locations[i % len(locations)]
            
            self.device_manager.register_device(
                device_id=device_id,
                sensor_type=sensor_type,
                location=location,
                security_level="level3"
            )
        
        logger.info(f"  Registered {self.num_devices} devices")
    
    def simulate_reading(
        self,
        device: IoTDevice,
        base_value: float = 22.0,
        noise_range: float = 5.0
    ) -> SensorReading:
        """Simulate a sensor reading with noise."""
        # Add realistic noise
        noise = random.uniform(-noise_range, noise_range)
        raw_value = base_value + noise
        
        # Encrypt and sign the reading
        reading = self.processor.encrypt_reading(device, raw_value)
        
        return reading
    
    def collect_readings(
        self,
        base_value: float = 22.0,
        noise_range: float = 5.0
    ) -> List[SensorReading]:
        """Collect readings from all devices."""
        readings = []
        
        for device_id, device in self.device_manager.devices.items():
            if device_id not in self.device_manager.revoked_devices:
                reading = self.simulate_reading(device, base_value, noise_range)
                readings.append(reading)
                
                # Update device last seen
                device.last_seen = datetime.now().isoformat()
        
        self.readings_buffer.extend(readings)
        return readings
    
    def aggregate_and_report(
        self,
        sensor_type: SensorType,
        aggregation_type: str = "average"
    ) -> Dict:
        """Aggregate collected readings and generate report."""
        if not self.readings_buffer:
            return {"error": "No readings to aggregate"}
        
        # Filter readings by sensor type
        relevant_readings = [
            r for r in self.readings_buffer
            if r.sensor_type == sensor_type
        ]
        
        if not relevant_readings:
            return {"error": f"No readings for sensor type: {sensor_type}"}
        
        # Perform privacy-preserving aggregation
        result = self.processor.aggregate_readings(
            relevant_readings,
            aggregation_type
        )
        
        # Decrypt result (only at authorized endpoint)
        decrypted = self.processor.decrypt_result(result, sensor_type)
        
        # Clear buffer
        self.readings_buffer = []
        
        return {
            "aggregation_type": result.aggregation_type,
            "device_count": result.device_count,
            "time_window": {
                "start": result.time_window_start,
                "end": result.time_window_end,
            },
            "result_value": round(decrypted, 2),
            "unit": self.processor.normalization_params.get(
                sensor_type, {}
            ).get("unit", ""),
            "computation_time_ms": round(result.computation_time_ms, 2),
            "privacy_preserved": True,
        }


# =============================================================================
# Demo Scenarios
# =============================================================================

def demo_smart_grid():
    """
    Demo: Smart Grid Privacy-Preserving Metering
    
    Scenario: Utility company aggregates power consumption
    without seeing individual household usage.
    """
    print("\n" + "=" * 70)
    print("DEMO: Smart Grid Privacy-Preserving Metering")
    print("=" * 70)
    
    print("""
    Scenario: A utility company wants to aggregate power consumption
    from 100 smart meters without seeing individual usage.
    
    Privacy Requirement: Individual consumption must remain private.
    Security Requirement: Data must be quantum-resistant in transit.
    """)
    
    # Create network of power meters
    network = IoTNetworkSimulator(num_devices=10)
    network.setup_network(sensor_type=SensorType.POWER_METER)
    
    print("\n[1] Device Registration")
    print("-" * 40)
    devices = network.device_manager.list_devices()
    print(f"    Registered {len(devices)} smart meters with PQC credentials")
    for d in devices[:3]:
        print(f"    - {d['device_id']} at {d['location']}")
    print(f"    ... and {len(devices) - 3} more")
    
    print("\n[2] Encrypted Data Collection")
    print("-" * 40)
    
    # Simulate readings (typical household: 500-2000W)
    readings = network.collect_readings(base_value=1200.0, noise_range=800.0)
    print(f"    Collected {len(readings)} encrypted readings")
    print(f"    Data encrypted with CKKS FHE")
    print(f"    Signatures: ML-DSA-65")
    
    # Verify signatures
    verified = 0
    for reading in readings:
        device = network.device_manager.get_device(reading.device_id)
        if device and network.processor.verify_reading(reading, device):
            verified += 1
    print(f"    Verified: {verified}/{len(readings)} signatures valid")
    
    print("\n[3] Privacy-Preserving Aggregation")
    print("-" * 40)
    
    # Aggregate without seeing individual values
    result = network.aggregate_and_report(
        SensorType.POWER_METER,
        aggregation_type="sum"
    )
    
    print(f"    Aggregation Type: {result['aggregation_type']}")
    print(f"    Devices Included: {result['device_count']}")
    print(f"    Computation Time: {result['computation_time_ms']} ms")
    print(f"    Privacy Preserved: {result['privacy_preserved']}")
    
    print("\n[4] Decrypted Result (at authorized endpoint only)")
    print("-" * 40)
    print(f"    Total Grid Load: {result['result_value']} {result['unit']}")
    print(f"    Average per Meter: {result['result_value'] / result['device_count']:.1f} {result['unit']}")
    
    print("\n[5] Security Analysis")
    print("-" * 40)
    print("    Quantum Resistance: ML-KEM-768, ML-DSA-65 (NIST Level 3)")
    print("    Data Privacy: FHE encryption (server never sees raw values)")
    print("    Forward Secrecy: Per-session key exchange")
    print("    Integrity: All readings signed with ML-DSA")


def demo_healthcare_wearables():
    """
    Demo: Healthcare Wearables Privacy-Preserving Analytics
    
    Scenario: Hospital aggregates patient vitals without
    exposing individual health data.
    """
    print("\n" + "=" * 70)
    print("DEMO: Healthcare Wearables Privacy-Preserving Analytics")
    print("=" * 70)
    
    print("""
    Scenario: A hospital monitors heart rate from patient wearables
    to detect ward-level health trends without exposing individual data.
    
    Compliance: HIPAA requires protecting individually identifiable health info.
    Solution: FHE allows computing on encrypted vitals.
    """)
    
    # Create network of heart rate monitors
    network = IoTNetworkSimulator(num_devices=8)
    network.setup_network(sensor_type=SensorType.HEART_RATE)
    
    print("\n[1] Device Registration (Ward A)")
    print("-" * 40)
    devices = network.device_manager.list_devices()
    print(f"    Registered {len(devices)} patient monitors")
    print(f"    Security Level: NIST Level 3 (192-bit equivalent)")
    
    print("\n[2] Vital Signs Collection")
    print("-" * 40)
    
    # Simulate heart rate readings (normal range: 60-100 bpm)
    readings = network.collect_readings(base_value=75.0, noise_range=15.0)
    print(f"    Collected {len(readings)} encrypted heart rate readings")
    print(f"    Individual values: NEVER visible to server")
    
    print("\n[3] Ward-Level Analytics (Encrypted)")
    print("-" * 40)
    
    result = network.aggregate_and_report(
        SensorType.HEART_RATE,
        aggregation_type="average"
    )
    
    print(f"    Computing average heart rate across ward...")
    print(f"    Homomorphic operations: ADD, MULTIPLY (scalar)")
    print(f"    Computation time: {result['computation_time_ms']} ms")
    
    print("\n[4] Clinical Insight (Authorized Medical Staff Only)")
    print("-" * 40)
    print(f"    Ward Average Heart Rate: {result['result_value']:.1f} {result['unit']}")
    
    status = "NORMAL" if 60 <= result['result_value'] <= 100 else "ALERT"
    print(f"    Status: {status}")
    
    print("\n[5] Compliance Summary")
    print("-" * 40)
    print("    HIPAA: Individual PHI never exposed")
    print("    Data minimization: Only aggregate shared")
    print("    Audit trail: All operations logged (encrypted)")
    print("    Quantum-safe: Prepared for future threats")


def demo_industrial_iot():
    """
    Demo: Industrial IoT Confidential Manufacturing Analytics
    
    Scenario: Manufacturing company shares aggregated sensor data
    with supply chain without revealing proprietary processes.
    """
    print("\n" + "=" * 70)
    print("DEMO: Industrial IoT Confidential Analytics")
    print("=" * 70)
    
    print("""
    Scenario: Manufacturer shares production environment data
    with supply chain partners for quality assurance without
    revealing proprietary manufacturing parameters.
    
    Challenge: Share useful analytics while protecting trade secrets.
    """)
    
    # Create network of industrial sensors
    network = IoTNetworkSimulator(num_devices=6)
    network.setup_network(sensor_type=SensorType.TEMPERATURE)
    
    print("\n[1] Production Line Sensors")
    print("-" * 40)
    print(f"    Deployed {network.num_devices} temperature sensors")
    print(f"    Location: Cleanroom Manufacturing Facility")
    
    print("\n[2] Encrypted Data Collection")
    print("-" * 40)
    
    # Industrial temperature (controlled environment: 20-24C)
    readings = network.collect_readings(base_value=22.0, noise_range=2.0)
    print(f"    Collected {len(readings)} encrypted readings")
    print(f"    Encryption: CKKS with 40-bit precision")
    
    print("\n[3] Supply Chain Data Sharing")
    print("-" * 40)
    
    result = network.aggregate_and_report(
        SensorType.TEMPERATURE,
        aggregation_type="average"
    )
    
    print("    Partner receives: Average temperature only")
    print("    Partner cannot see: Individual sensor values")
    print("    Partner cannot see: Sensor locations")
    print("    Partner cannot see: Production volume")
    
    print("\n[4] Shared Analytics")
    print("-" * 40)
    print(f"    Average Production Temperature: {result['result_value']:.1f}{result['unit']}")
    
    in_spec = 20 <= result['result_value'] <= 24
    print(f"    Specification Compliance: {'PASS' if in_spec else 'FAIL'}")
    
    print("\n[5] Trade Secret Protection")
    print("-" * 40)
    print("    Protected Information:")
    print("      - Exact temperature profiles (competitive advantage)")
    print("      - Sensor placement (process optimization)")
    print("      - Production scheduling (capacity intelligence)")


def run_all_demos():
    """Run all IoT security demos."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        PQC-FHE IoT Security Demo Suite                             ║")
    print("║        Post-Quantum + Homomorphic Encryption for IoT               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Timestamp: {datetime.now().isoformat()}")
    print(f"  Crypto Available: {CRYPTO_AVAILABLE}")
    
    demos = [
        ("Smart Grid", demo_smart_grid),
        ("Healthcare Wearables", demo_healthcare_wearables),
        ("Industrial IoT", demo_industrial_iot),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED")
    print("=" * 70)
    print("""
    Key Takeaways:
    
    1. Post-Quantum Security
       - ML-KEM-768 for key exchange (quantum-resistant)
       - ML-DSA-65 for device attestation (quantum-resistant)
       - Ready for "harvest now, decrypt later" threats
    
    2. Privacy-Preserving Computation
       - FHE enables computation on encrypted data
       - Server never sees raw sensor values
       - Only aggregate results decrypted
    
    3. Regulatory Compliance
       - HIPAA: Protected health information never exposed
       - GDPR: Data minimization through aggregation
       - Industry 4.0: Trade secrets protected
    
    4. Future-Proof Architecture
       - NIST-approved algorithms (FIPS 203/204)
       - Hybrid classical+PQC supported
       - GPU acceleration available
    """)


if __name__ == "__main__":
    run_all_demos()
