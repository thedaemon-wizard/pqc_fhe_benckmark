#!/usr/bin/env python3
"""
Enterprise Data Module v2.2.0
==============================

Provides realistic data for enterprise use case demonstrations.
Includes healthcare, finance, IoT, and blockchain examples.

Data Sources:
- Healthcare: Synthetic vital signs based on clinical ranges
- Finance: Historical market data patterns
- IoT: Realistic sensor data with temporal patterns
- Blockchain: Real transaction format examples
"""

import random
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math


# =============================================================================
# HEALTHCARE DATA
# =============================================================================

@dataclass
class VitalSigns:
    """Patient vital signs data structure"""
    timestamp: str
    heart_rate: float       # BPM (60-100 normal)
    systolic_bp: float      # mmHg (90-120 normal)
    diastolic_bp: float     # mmHg (60-80 normal)
    temperature: float      # Celsius (36.1-37.2 normal)
    spo2: float            # % (95-100 normal)
    respiratory_rate: float # breaths/min (12-20 normal)


class HealthcareDataGenerator:
    """Generate realistic healthcare data for privacy-preserving analytics."""
    
    # Clinical reference ranges (CDC/WHO standards)
    NORMAL_RANGES = {
        "heart_rate": (60, 100),
        "systolic_bp": (90, 120),
        "diastolic_bp": (60, 80),
        "temperature": (36.1, 37.2),
        "spo2": (95, 100),
        "respiratory_rate": (12, 20)
    }
    
    @staticmethod
    def generate_vital_signs(
        patient_id: str = "P001",
        num_readings: int = 24,
        condition: str = "normal"
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic vital signs time series.
        
        Args:
            patient_id: Patient identifier
            num_readings: Number of readings (e.g., 24 for hourly over 1 day)
            condition: "normal", "hypertensive", "fever", "hypoxic"
        
        Returns:
            List of vital sign dictionaries
        """
        readings = []
        base_time = datetime.now() - timedelta(hours=num_readings)
        
        # Condition-specific adjustments
        adjustments = {
            "normal": {"hr": 0, "bp": 0, "temp": 0, "spo2": 0},
            "hypertensive": {"hr": 15, "bp": 30, "temp": 0, "spo2": 0},
            "fever": {"hr": 20, "bp": 5, "temp": 1.5, "spo2": -2},
            "hypoxic": {"hr": 25, "bp": -10, "temp": 0, "spo2": -8}
        }
        adj = adjustments.get(condition, adjustments["normal"])
        
        for i in range(num_readings):
            timestamp = base_time + timedelta(hours=i)
            
            # Add circadian rhythm variation (lower at night, higher during day)
            hour = timestamp.hour
            circadian_factor = math.sin((hour - 6) * math.pi / 12) * 0.1
            
            # Generate values with realistic variation
            hr_base = 75 + adj["hr"]
            bp_sys_base = 115 + adj["bp"]
            bp_dia_base = 75 + adj["bp"] * 0.5
            temp_base = 36.6 + adj["temp"]
            spo2_base = 98 + adj["spo2"]
            rr_base = 16
            
            reading = {
                "timestamp": timestamp.isoformat(),
                "patient_id": patient_id,
                "heart_rate": round(hr_base + random.gauss(0, 5) + hr_base * circadian_factor, 1),
                "systolic_bp": round(bp_sys_base + random.gauss(0, 8) + bp_sys_base * circadian_factor, 1),
                "diastolic_bp": round(bp_dia_base + random.gauss(0, 5) + bp_dia_base * circadian_factor, 1),
                "temperature": round(temp_base + random.gauss(0, 0.2), 2),
                "spo2": round(max(85, min(100, spo2_base + random.gauss(0, 1))), 1),
                "respiratory_rate": round(rr_base + random.gauss(0, 2) + adj["hr"] * 0.1, 1)
            }
            readings.append(reading)
        
        return readings
    
    @staticmethod
    def get_aggregation_example() -> Dict[str, Any]:
        """Get example data for FHE aggregation demonstration."""
        # Generate 10 patient readings for averaging
        bp_readings = [
            120, 125, 118, 122, 130,  # Patient group 1
            115, 128, 119, 124, 121   # Patient group 2
        ]
        
        return {
            "description": "Blood pressure readings from 10 patients for private mean calculation",
            "data": bp_readings,
            "expected_mean": sum(bp_readings) / len(bp_readings),
            "fhe_operation": "multiply by 0.1 (for averaging 10 values)",
            "clinical_interpretation": {
                "normal_range": "90-120 mmHg systolic",
                "result_classification": "Stage 1 Hypertension" if sum(bp_readings) / len(bp_readings) > 120 else "Normal"
            }
        }


# =============================================================================
# FINANCE DATA
# =============================================================================

class FinanceDataGenerator:
    """Generate realistic financial data for confidential analytics."""
    
    # Sample portfolio based on typical institutional allocation
    SAMPLE_PORTFOLIO = {
        "AAPL": {"shares": 1500, "sector": "Technology", "price_range": (170, 195)},
        "MSFT": {"shares": 1200, "sector": "Technology", "price_range": (370, 420)},
        "GOOGL": {"shares": 800, "sector": "Technology", "price_range": (140, 175)},
        "JPM": {"shares": 2000, "sector": "Financial", "price_range": (180, 210)},
        "JNJ": {"shares": 1800, "sector": "Healthcare", "price_range": (155, 175)},
        "XOM": {"shares": 2500, "sector": "Energy", "price_range": (105, 125)},
        "PG": {"shares": 1600, "sector": "Consumer", "price_range": (160, 180)},
        "BRK.B": {"shares": 1000, "sector": "Financial", "price_range": (350, 420)}
    }
    
    @staticmethod
    def generate_portfolio_values(use_current_prices: bool = True) -> Dict[str, Any]:
        """
        Generate portfolio valuation data.
        
        Returns:
            Portfolio with current values and sector allocation
        """
        holdings = []
        total_value = 0.0
        sector_totals = {}
        
        for symbol, info in FinanceDataGenerator.SAMPLE_PORTFOLIO.items():
            # Generate realistic price within range
            price = random.uniform(*info["price_range"])
            value = info["shares"] * price
            total_value += value
            
            sector = info["sector"]
            sector_totals[sector] = sector_totals.get(sector, 0) + value
            
            holdings.append({
                "symbol": symbol,
                "shares": info["shares"],
                "price": round(price, 2),
                "value": round(value, 2),
                "sector": sector
            })
        
        # Calculate sector percentages
        sector_allocation = {
            sector: round(value / total_value * 100, 2)
            for sector, value in sector_totals.items()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "holdings": holdings,
            "total_value": round(total_value, 2),
            "sector_allocation": sector_allocation,
            "fhe_compatible_values": [h["value"] for h in holdings]
        }
    
    @staticmethod
    def get_growth_projection_example() -> Dict[str, Any]:
        """Get example for FHE growth projection."""
        portfolio_values = [150000.0, 89000.0, 234000.0, 178000.0, 92000.0]
        growth_rate = 1.08  # 8% annual return
        
        return {
            "description": "Portfolio growth projection without exposing individual positions",
            "current_values": portfolio_values,
            "growth_rate": growth_rate,
            "projected_values": [round(v * growth_rate, 2) for v in portfolio_values],
            "total_current": sum(portfolio_values),
            "total_projected": round(sum(portfolio_values) * growth_rate, 2),
            "fhe_operation": f"multiply each encrypted value by {growth_rate}"
        }
    
    @staticmethod
    def generate_risk_metrics() -> Dict[str, Any]:
        """Generate risk metrics for FHE computation."""
        # Daily returns (percentage) over 20 days
        returns = [
            0.52, -0.31, 0.78, -0.15, 0.43,
            -0.62, 0.89, 0.21, -0.44, 0.67,
            -0.28, 0.55, -0.19, 0.72, 0.33,
            -0.51, 0.41, -0.38, 0.64, -0.22
        ]
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return {
            "description": "Daily portfolio returns for private variance calculation",
            "returns": returns,
            "mean_return": round(mean_return, 4),
            "variance": round(variance, 4),
            "volatility": round(math.sqrt(variance), 4),
            "fhe_operations": [
                "Step 1: Compute mean (sum * (1/n))",
                "Step 2: Subtract mean from each value",
                "Step 3: Square differences",
                "Step 4: Sum and divide by n"
            ]
        }


# =============================================================================
# IOT SENSOR DATA
# =============================================================================

class IoTDataGenerator:
    """Generate realistic IoT sensor data for secure aggregation."""
    
    # Sensor types with typical ranges
    SENSOR_TYPES = {
        "temperature": {"unit": "°C", "range": (18, 28), "noise": 0.5},
        "humidity": {"unit": "%", "range": (30, 70), "noise": 2.0},
        "pressure": {"unit": "hPa", "range": (1000, 1025), "noise": 1.0},
        "co2": {"unit": "ppm", "range": (400, 1000), "noise": 20},
        "light": {"unit": "lux", "range": (100, 1000), "noise": 50},
        "motion": {"unit": "count", "range": (0, 50), "noise": 5}
    }
    
    @staticmethod
    def generate_sensor_network_data(
        num_sensors: int = 10,
        num_readings: int = 12,
        sensor_type: str = "temperature"
    ) -> Dict[str, Any]:
        """
        Generate sensor network data for secure aggregation.
        
        Args:
            num_sensors: Number of sensors in network
            num_readings: Readings per sensor
            sensor_type: Type of sensor
        
        Returns:
            Sensor network data with individual and aggregate values
        """
        spec = IoTDataGenerator.SENSOR_TYPES.get(sensor_type, IoTDataGenerator.SENSOR_TYPES["temperature"])
        base_time = datetime.now() - timedelta(hours=num_readings)
        
        sensors = []
        all_readings = []
        
        for sensor_id in range(num_sensors):
            # Each sensor has slight calibration offset
            calibration_offset = random.gauss(0, spec["noise"] * 0.5)
            sensor_readings = []
            
            for t in range(num_readings):
                timestamp = base_time + timedelta(hours=t)
                
                # Base value with time variation
                hour = timestamp.hour
                time_factor = math.sin((hour - 6) * math.pi / 12)
                
                base_value = (spec["range"][0] + spec["range"][1]) / 2
                range_span = (spec["range"][1] - spec["range"][0]) / 4
                
                value = base_value + time_factor * range_span + calibration_offset + random.gauss(0, spec["noise"])
                value = max(spec["range"][0], min(spec["range"][1], value))
                
                sensor_readings.append(round(value, 2))
                all_readings.append(round(value, 2))
            
            sensors.append({
                "sensor_id": f"SENSOR_{sensor_id:03d}",
                "type": sensor_type,
                "readings": sensor_readings,
                "mean": round(sum(sensor_readings) / len(sensor_readings), 2)
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "sensor_type": sensor_type,
            "unit": spec["unit"],
            "num_sensors": num_sensors,
            "num_readings": num_readings,
            "sensors": sensors,
            "aggregate_mean": round(sum(all_readings) / len(all_readings), 2),
            "aggregate_min": min(all_readings),
            "aggregate_max": max(all_readings),
            "fhe_compatible_values": [s["mean"] for s in sensors]  # Per-sensor averages for aggregation
        }
    
    @staticmethod
    def get_calibration_example() -> Dict[str, Any]:
        """Get example for FHE sensor calibration."""
        raw_readings = [22.5, 23.1, 22.8, 24.2, 22.9, 23.5, 22.7, 23.8]
        calibration_factor = 1.02  # 2% calibration adjustment
        
        return {
            "description": "Temperature sensor calibration without exposing raw readings",
            "raw_readings": raw_readings,
            "calibration_factor": calibration_factor,
            "calibrated_readings": [round(r * calibration_factor, 2) for r in raw_readings],
            "fhe_operation": f"multiply encrypted readings by {calibration_factor}",
            "use_case": "Apply factory calibration to field sensor data securely"
        }


# =============================================================================
# BLOCKCHAIN DATA
# =============================================================================

class BlockchainDataGenerator:
    """Generate realistic blockchain transaction data for PQC signing."""
    
    @staticmethod
    def generate_transaction(
        tx_type: str = "transfer",
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Generate realistic blockchain transaction.
        
        Args:
            tx_type: "transfer", "contract", "nft"
            chain: "ethereum", "bitcoin", "polygon"
        
        Returns:
            Transaction data ready for PQC signing
        """
        # Generate realistic addresses
        from_addr = "0x" + hashlib.sha256(f"sender_{random.randint(1,1000)}".encode()).hexdigest()[:40]
        to_addr = "0x" + hashlib.sha256(f"receiver_{random.randint(1,1000)}".encode()).hexdigest()[:40]
        
        # Chain-specific parameters
        chain_params = {
            "ethereum": {"gas_price": 30, "gas_limit": 21000, "chain_id": 1},
            "polygon": {"gas_price": 100, "gas_limit": 21000, "chain_id": 137},
            "bitcoin": {"fee_rate": 20, "version": 2}
        }
        params = chain_params.get(chain, chain_params["ethereum"])
        
        # Transaction types
        if tx_type == "transfer":
            value = round(random.uniform(0.1, 10.0), 4)
            data = "0x"
        elif tx_type == "contract":
            value = 0
            # Simulated contract call data
            data = "0xa9059cbb" + "0" * 56 + hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
        elif tx_type == "nft":
            value = round(random.uniform(0.01, 2.0), 4)
            data = "0x23b872dd" + "0" * 56 + hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
        else:
            value = round(random.uniform(0.1, 5.0), 4)
            data = "0x"
        
        nonce = random.randint(1, 1000)
        
        tx = {
            "chain": chain,
            "type": tx_type,
            "nonce": nonce,
            "from": from_addr,
            "to": to_addr,
            "value": value,
            "value_wei": int(value * 10**18),
            "gas_price": params.get("gas_price", 30),
            "gas_limit": params.get("gas_limit", 21000),
            "chain_id": params.get("chain_id", 1),
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create signable message
        tx_message = f"{tx['chain']}:{tx['nonce']}:{tx['from']}:{tx['to']}:{tx['value_wei']}:{tx['data']}"
        tx["signable_message"] = tx_message
        tx["message_hash"] = hashlib.sha256(tx_message.encode()).hexdigest()
        
        return tx
    
    @staticmethod
    def generate_batch_transactions(count: int = 5) -> Dict[str, Any]:
        """Generate batch of transactions for throughput testing."""
        tx_types = ["transfer", "contract", "nft"]
        chains = ["ethereum", "polygon"]
        
        transactions = []
        for i in range(count):
            tx = BlockchainDataGenerator.generate_transaction(
                tx_type=random.choice(tx_types),
                chain=random.choice(chains)
            )
            transactions.append(tx)
        
        return {
            "batch_id": hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "count": count,
            "transactions": transactions,
            "signable_messages": [tx["signable_message"] for tx in transactions]
        }
    
    @staticmethod
    def get_signing_example() -> Dict[str, Any]:
        """Get example for PQC transaction signing."""
        tx = BlockchainDataGenerator.generate_transaction("transfer", "ethereum")
        
        return {
            "description": "Quantum-resistant blockchain transaction signing",
            "transaction": tx,
            "pqc_benefits": [
                "Resistant to Shor's algorithm attacks",
                "Future-proof against quantum computers",
                "NIST standardized (FIPS 204)",
                "3,309 byte signatures (ML-DSA-65)"
            ],
            "comparison": {
                "ECDSA": {"signature_size": 64, "quantum_safe": False},
                "ML-DSA-65": {"signature_size": 3309, "quantum_safe": True},
                "Falcon-512": {"signature_size": 666, "quantum_safe": True}
            }
        }


# =============================================================================
# COMBINED DEMO DATA
# =============================================================================

def get_all_demo_data() -> Dict[str, Any]:
    """Get all demo data for enterprise examples."""
    return {
        "healthcare": {
            "vital_signs": HealthcareDataGenerator.generate_vital_signs(num_readings=12),
            "aggregation_example": HealthcareDataGenerator.get_aggregation_example()
        },
        "finance": {
            "portfolio": FinanceDataGenerator.generate_portfolio_values(),
            "growth_projection": FinanceDataGenerator.get_growth_projection_example(),
            "risk_metrics": FinanceDataGenerator.generate_risk_metrics()
        },
        "iot": {
            "sensor_network": IoTDataGenerator.generate_sensor_network_data(),
            "calibration_example": IoTDataGenerator.get_calibration_example()
        },
        "blockchain": {
            "transaction": BlockchainDataGenerator.generate_transaction(),
            "batch_transactions": BlockchainDataGenerator.generate_batch_transactions(3),
            "signing_example": BlockchainDataGenerator.get_signing_example()
        }
    }


def get_fhe_demo_vectors() -> Dict[str, Any]:
    """Get ready-to-use FHE demonstration vectors."""
    return {
        "healthcare_bp": {
            "description": "Blood pressure readings for private mean",
            "data": [120, 125, 118, 122, 130, 115, 128, 119, 124, 121],
            "operation": "multiply",
            "operand": 0.1,
            "expected_result": 122.2
        },
        "finance_portfolio": {
            "description": "Portfolio values for growth projection",
            "data": [150000.0, 89000.0, 234000.0, 178000.0, 92000.0],
            "operation": "multiply",
            "operand": 1.08,
            "expected_result": [162000.0, 96120.0, 252720.0, 192240.0, 99360.0]
        },
        "iot_calibration": {
            "description": "Sensor readings for calibration",
            "data": [22.5, 23.1, 22.8, 24.2, 22.9, 23.5],
            "operation": "multiply",
            "operand": 1.02,
            "expected_result": [22.95, 23.562, 23.256, 24.684, 23.358, 23.97]
        }
    }
