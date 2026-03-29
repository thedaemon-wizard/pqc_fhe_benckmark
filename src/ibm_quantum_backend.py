"""
IBM Quantum QPU Noise Parameter Fetcher with Caching and Fallback.

Provides real quantum hardware noise parameters (T1, T2, gate errors,
readout errors) from IBM Quantum Platform via qiskit-ibm-runtime.

Features:
    - Dynamic QPU backend discovery from IBM Quantum Platform
    - Real-time noise parameter fetching (T1/T2/gate errors/readout errors)
    - 1-hour parameter cache to reduce API calls
    - JSON file cache for offline usage (data/ibm_backends_cache.json)
    - Fallback chain: API → JSON cache → KNOWN_PROCESSORS
    - least_busy backend selection support
    - Processor-specific basis_gates (Heron=CZ, Eagle=ECR, Nighthawk=CZ)
    - EnhancedNoiseSimulator-compatible profile format output
    - Qiskit NoiseModel construction from real QPU parameters
    - Singleton pattern via get_ibm_manager()

References:
    - IBM Quantum Platform: https://quantum.ibm.com/
    - IBM Quantum Docs: processor-types, build-noise-models, qpu-information
    - IBM Heron r1 (133Q, ibm_torino): CZ-based, T1~160µs, T2~100µs (Dec 2023)
    - IBM Heron r2 (156Q, ibm_fez/kingston/marrakesh): CZ, T1~250µs, T2~150µs (Jul 2024)
    - IBM Heron r3 (156Q): Improved coherence and gate fidelity (Jul 2025)
    - IBM Nighthawk r1 (120Q): Grid topology, enhanced connectivity (Dec 2025)
    - IBM Pinnacle Architecture (Feb 2026): qLDPC codes

Usage:
    >>> from src.ibm_quantum_backend import get_ibm_manager
    >>> mgr = get_ibm_manager()  # Singleton
    >>> result = mgr.startup_connect_and_discover()
    >>> backends = mgr.list_backends()
    >>> params = mgr.get_noise_params('ibm_torino')
    >>> least_busy = mgr.get_least_busy(min_num_qubits=100)
"""

import os
import json
import time
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Check qiskit-ibm-runtime availability
IBM_RUNTIME_AVAILABLE = False
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    logger.info("qiskit-ibm-runtime not installed — IBM Quantum features use fallback only")

# Check qiskit-aer noise availability
QISKIT_AER_AVAILABLE = False
try:
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
        ReadoutError,
    )
    import numpy as np
    QISKIT_AER_AVAILABLE = True
except ImportError:
    logger.info("qiskit-aer not available — NoiseModel construction disabled")


class IBMQuantumBackendManager:
    """
    Fetches real QPU noise parameters from IBM Quantum Platform.

    Connects to IBM Quantum using credentials from .env file:
        IBM_QUANTUM_TOKEN    - API token
        IBM_QUANTUM_INSTANCE - CRN instance (crn:v1:bluemix:...)
        IBM_QUANTUM_CHANNEL  - Channel (ibm_quantum_platform)
        IBM_BACKEND_NAME     - Default backend (least_busy)

    Falls back to IBM Heron r2 (ibm_fez) published specifications when API
    connection is unavailable.
    """

    # =========================================================================
    # Fallback Parameters: IBM Heron r2 (ibm_fez) Published Specifications
    # Source: IBM Quantum Jul 2024, Heron r2 processor data sheet
    # Ref: https://quantum.cloud.ibm.com/docs/guides/processor-types
    # =========================================================================
    HERON_R2_FALLBACK: Dict[str, Any] = {
        'name': 'ibm_fez',
        'processor_type': 'Heron r2',
        'num_qubits': 156,
        'basis_gates': ['cz', 'id', 'rz', 'sx', 'x'],
        'T1_us_median': 250.0,
        'T2_us_median': 150.0,
        'single_qubit_error_median': 2.4e-4,
        'two_qubit_error_median': 3.8e-3,  # CZ gate
        'readout_error_median': 1.2e-2,
        'gate_time_single_ns': 60.0,
        'gate_time_cz_ns': 84.0,
        'source': 'fallback_heron_r2_specs',
        'timestamp': None,
    }

    # =========================================================================
    # Processor family → basis gates mapping
    # Heron: CZ-based two-qubit gate
    # Eagle: ECR-based two-qubit gate
    # Nighthawk: CZ-based (Heron technology stack with grid topology)
    # Reference: https://quantum.cloud.ibm.com/docs/guides/processor-types
    # =========================================================================
    PROCESSOR_BASIS_GATES: Dict[str, List[str]] = {
        'Heron': ['cz', 'id', 'rz', 'sx', 'x'],
        'Eagle': ['ecr', 'id', 'rz', 'sx', 'x'],
        'Nighthawk': ['cz', 'id', 'rz', 'sx', 'x'],
    }

    # Minimum fallback known processors (used when API + JSON cache both fail)
    # Ref: https://quantum.cloud.ibm.com/docs/guides/processor-types
    # Heron r1 (Dec 2023): 133Q, ibm_torino
    # Heron r2 (Jul 2024): 156Q, ibm_fez/ibm_kingston/ibm_marrakesh
    # Eagle r3: 127Q, ibm_brisbane/ibm_sherbrooke
    KNOWN_PROCESSORS: Dict[str, Dict[str, Any]] = {
        'ibm_torino': {
            'processor_type': 'Heron r1',
            'processor_family': 'Heron',
            'num_qubits': 133,
            'T1_us_median': 160.0,
            'T2_us_median': 100.0,
            'single_qubit_error_median': 3.0e-4,
            'two_qubit_error_median': 5.0e-3,
            'readout_error_median': 1.5e-2,
            'gate_time_single_ns': 60.0,
            'gate_time_cz_ns': 84.0,
            'basis_gates': ['cz', 'id', 'rz', 'sx', 'x'],
        },
        'ibm_fez': {
            'processor_type': 'Heron r2',
            'processor_family': 'Heron',
            'num_qubits': 156,
            'T1_us_median': 250.0,
            'T2_us_median': 150.0,
            'single_qubit_error_median': 2.4e-4,
            'two_qubit_error_median': 3.8e-3,
            'readout_error_median': 1.2e-2,
            'gate_time_single_ns': 60.0,
            'gate_time_cz_ns': 84.0,
            'basis_gates': ['cz', 'id', 'rz', 'sx', 'x'],
        },
        'ibm_kingston': {
            'processor_type': 'Heron r2',
            'processor_family': 'Heron',
            'num_qubits': 156,
            'T1_us_median': 250.0,
            'T2_us_median': 150.0,
            'single_qubit_error_median': 2.4e-4,
            'two_qubit_error_median': 2.03e-3,  # Per IBM Open Plan announcement
            'readout_error_median': 1.2e-2,
            'gate_time_single_ns': 60.0,
            'gate_time_cz_ns': 84.0,
            'basis_gates': ['cz', 'id', 'rz', 'sx', 'x'],
        },
        'ibm_marrakesh': {
            'processor_type': 'Heron r2',
            'processor_family': 'Heron',
            'num_qubits': 156,
            'T1_us_median': 250.0,
            'T2_us_median': 150.0,
            'single_qubit_error_median': 2.4e-4,
            'two_qubit_error_median': 3.8e-3,
            'readout_error_median': 1.2e-2,
            'gate_time_single_ns': 60.0,
            'gate_time_cz_ns': 84.0,
            'basis_gates': ['cz', 'id', 'rz', 'sx', 'x'],
        },
        'ibm_brisbane': {
            'processor_type': 'Eagle r3',
            'processor_family': 'Eagle',
            'num_qubits': 127,
            'T1_us_median': 200.0,
            'T2_us_median': 120.0,
            'single_qubit_error_median': 3.0e-4,
            'two_qubit_error_median': 7.5e-3,
            'readout_error_median': 1.5e-2,
            'gate_time_single_ns': 60.0,
            'gate_time_cz_ns': 660.0,  # ECR gate time
            'basis_gates': ['ecr', 'id', 'rz', 'sx', 'x'],
        },
        'ibm_sherbrooke': {
            'processor_type': 'Eagle r3',
            'processor_family': 'Eagle',
            'num_qubits': 127,
            'T1_us_median': 220.0,
            'T2_us_median': 130.0,
            'single_qubit_error_median': 2.8e-4,
            'two_qubit_error_median': 6.8e-3,
            'readout_error_median': 1.3e-2,
            'gate_time_single_ns': 60.0,
            'gate_time_cz_ns': 660.0,
            'basis_gates': ['ecr', 'id', 'rz', 'sx', 'x'],
        },
    }

    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize IBM Quantum Backend Manager.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
        """
        load_dotenv()
        self._service: Optional[Any] = None
        self._cache: Dict[str, tuple] = {}  # {backend_name: (timestamp, params)}
        self._cache_ttl = cache_ttl
        self._connected = False
        self._connection_error: Optional[str] = None

    @property
    def connected(self) -> bool:
        """Whether we have an active IBM Quantum connection."""
        return self._connected

    def connect(self) -> bool:
        """
        Connect to IBM Quantum Platform using .env credentials.

        Returns:
            True if connection successful, False otherwise.
        """
        if not IBM_RUNTIME_AVAILABLE:
            self._connection_error = "qiskit-ibm-runtime not installed"
            logger.warning("Cannot connect: qiskit-ibm-runtime not installed")
            return False

        token = os.getenv('IBM_QUANTUM_TOKEN')
        instance = os.getenv('IBM_QUANTUM_INSTANCE')
        channel = os.getenv('IBM_QUANTUM_CHANNEL', 'ibm_quantum_platform')

        if not token:
            self._connection_error = "IBM_QUANTUM_TOKEN not set in .env"
            logger.warning("Cannot connect: IBM_QUANTUM_TOKEN not set")
            return False

        try:
            self._service = QiskitRuntimeService(
                channel=channel,
                token=token,
                instance=instance,
            )
            self._connected = True
            self._connection_error = None
            logger.info("Connected to IBM Quantum Platform successfully")
            return True
        except Exception as e:
            self._connected = False
            self._connection_error = str(e)
            logger.warning(f"IBM Quantum connection failed: {e}")
            return False

    def list_backends(self) -> List[Dict[str, Any]]:
        """
        List available QPU backends with basic info.

        Returns list of dicts with: name, num_qubits, processor_type, basis_gates, source.
        Fallback chain: API → JSON cache → KNOWN_PROCESSORS.
        """
        # 1) Try IBM Quantum API
        if self._connected and self._service:
            try:
                backends = self._service.backends(operational=True)
                result = []
                for b in backends:
                    try:
                        # Extract processor family from backend
                        proc_type = 'unknown'
                        proc_family = 'Heron'  # Default
                        if hasattr(b, 'processor_type') and isinstance(
                            getattr(b, 'processor_type', None), dict
                        ):
                            proc_type_dict = b.processor_type
                            proc_family = proc_type_dict.get('family', 'Heron')
                            revision = proc_type_dict.get('revision', '')
                            proc_type = f"{proc_family} {revision}".strip() if revision else proc_family

                        # Get basis gates from backend or processor family
                        basis_gates = list(b.operation_names) if hasattr(
                            b, 'operation_names'
                        ) else self.PROCESSOR_BASIS_GATES.get(proc_family, ['cz', 'id', 'rz', 'sx', 'x'])

                        result.append({
                            'name': b.name,
                            'num_qubits': b.num_qubits,
                            'processor_type': proc_type,
                            'processor_family': proc_family,
                            'basis_gates': basis_gates,
                            'status': 'operational',
                            'source': 'ibm_quantum_api',
                        })
                    except Exception:
                        result.append({
                            'name': b.name,
                            'num_qubits': getattr(b, 'num_qubits', 0),
                            'processor_type': 'unknown',
                            'processor_family': 'Heron',
                            'basis_gates': ['cz', 'id', 'rz', 'sx', 'x'],
                            'status': 'operational',
                            'source': 'ibm_quantum_api',
                        })
                if result:
                    logger.info(f"Listed {len(result)} IBM Quantum backends from API")
                    return result
            except Exception as e:
                logger.warning(f"Failed to list backends from API: {e}")

        # 2) Fallback: Try JSON cache file
        json_cache = self._load_backends_cache()
        if json_cache and json_cache.get('backends'):
            result = []
            for b in json_cache['backends']:
                result.append({
                    'name': b['name'],
                    'num_qubits': b.get('num_qubits', 0),
                    'processor_type': b.get('processor_type', 'unknown'),
                    'processor_family': b.get('processor_family', 'Heron'),
                    'basis_gates': b.get('basis_gates', ['cz', 'id', 'rz', 'sx', 'x']),
                    'status': 'cached',
                    'source': 'json_cache',
                    'cache_timestamp': json_cache.get('timestamp', ''),
                })
            logger.info(f"Using {len(result)} backends from JSON cache ({json_cache.get('timestamp', 'unknown')})")
            return result

        # 3) Fallback: KNOWN_PROCESSORS (hardcoded minimum)
        result = []
        for name, info in self.KNOWN_PROCESSORS.items():
            result.append({
                'name': name,
                'num_qubits': info['num_qubits'],
                'processor_type': info['processor_type'],
                'processor_family': info.get('processor_family', 'Heron'),
                'basis_gates': info.get('basis_gates', ['cz', 'id', 'rz', 'sx', 'x']),
                'status': 'fallback',
                'source': 'fallback_specs',
            })
        logger.info(f"Using {len(result)} fallback backends (API + JSON cache unavailable)")
        return result

    def get_noise_params(self, backend_name: str = 'ibm_fez') -> Dict[str, Any]:
        """
        Fetch noise parameters (T1/T2/gate errors/readout errors) for a QPU.

        Uses 1-hour cache to reduce API calls. Falls back to known specs.

        Args:
            backend_name: IBM Quantum backend name (e.g., 'ibm_torino')

        Returns:
            Dict with T1_us_median, T2_us_median, single_qubit_error_median,
            two_qubit_error_median, readout_error_median, gate times, etc.
        """
        # Check cache
        if backend_name in self._cache:
            ts, params = self._cache[backend_name]
            if time.time() - ts < self._cache_ttl:
                logger.debug(f"Using cached params for {backend_name}")
                return params

        # Try fetching from API
        if self._connected and self._service:
            try:
                params = self._fetch_backend_params(backend_name)
                if params:
                    self._cache[backend_name] = (time.time(), params)
                    return params
            except Exception as e:
                logger.warning(f"Failed to fetch params for {backend_name}: {e}")

        # Fallback: check JSON cache for this specific backend's noise params
        json_cache = self._load_backends_cache()
        if json_cache and json_cache.get('noise_params', {}).get(backend_name):
            cached_params = json_cache['noise_params'][backend_name]
            cached_params['source'] = 'json_cache'
            cached_params['timestamp'] = time.time()
            self._cache[backend_name] = (time.time(), cached_params)
            logger.info(f"Using JSON-cached noise params for {backend_name}")
            return cached_params

        # Fallback to known specs
        if backend_name in self.KNOWN_PROCESSORS:
            known = self.KNOWN_PROCESSORS[backend_name]
            params = {
                'name': backend_name,
                **known,
                'basis_gates': known.get('basis_gates', ['cz', 'id', 'rz', 'sx', 'x']),
                'source': 'fallback_specs',
                'timestamp': time.time(),
            }
            self._cache[backend_name] = (time.time(), params)
            return params

        # Default: Heron r2 fallback (ibm_fez, 156Q)
        fallback = dict(self.HERON_R2_FALLBACK)
        fallback['timestamp'] = time.time()
        return fallback

    def _fetch_backend_params(self, backend_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real noise parameters from IBM Quantum API.

        Uses backend.properties() or backend.target to extract:
        - T1/T2 times per qubit → compute medians
        - Gate error rates → compute medians
        - Readout error rates → compute medians
        """
        if not self._service:
            return None

        try:
            backend = self._service.backend(backend_name)
        except Exception as e:
            logger.warning(f"Backend '{backend_name}' not found: {e}")
            return None

        params = {
            'name': backend_name,
            'num_qubits': backend.num_qubits,
            'basis_gates': list(backend.operation_names) if hasattr(
                backend, 'operation_names'
            ) else ['cz', 'id', 'rz', 'sx', 'x'],
            'source': 'ibm_quantum_api',
            'timestamp': time.time(),
        }

        # Extract from backend.target (Qiskit 2.x preferred)
        target = backend.target if hasattr(backend, 'target') else None
        if target is not None:
            t1_values = []
            t2_values = []
            sq_errors = []
            tq_errors = []
            readout_errors = []

            for qubit in range(backend.num_qubits):
                try:
                    qubit_props = target.qubit_properties
                    if qubit_props and qubit < len(qubit_props) and qubit_props[qubit]:
                        t1 = qubit_props[qubit].t1
                        t2 = qubit_props[qubit].t2
                        if t1 is not None:
                            t1_values.append(t1 * 1e6)  # seconds → µs
                        if t2 is not None:
                            t2_values.append(t2 * 1e6)
                except Exception:
                    pass

            # Gate errors from target
            for op_name in target.operation_names:
                try:
                    for qargs in target.qargs_for_operation_name(op_name):
                        props = target[op_name].get(qargs, None)
                        if props and props.error is not None:
                            if len(qargs) == 1:
                                sq_errors.append(props.error)
                            elif len(qargs) == 2:
                                tq_errors.append(props.error)
                except Exception:
                    pass

            # Compute medians
            if t1_values:
                params['T1_us_median'] = round(statistics.median(t1_values), 2)
                params['T1_us_min'] = round(min(t1_values), 2)
                params['T1_us_max'] = round(max(t1_values), 2)
            if t2_values:
                params['T2_us_median'] = round(statistics.median(t2_values), 2)
                params['T2_us_min'] = round(min(t2_values), 2)
                params['T2_us_max'] = round(max(t2_values), 2)
            if sq_errors:
                params['single_qubit_error_median'] = float(
                    f"{statistics.median(sq_errors):.6e}"
                )
            if tq_errors:
                params['two_qubit_error_median'] = float(
                    f"{statistics.median(tq_errors):.6e}"
                )

            # Readout errors (from measure operation)
            try:
                for qargs in target.qargs_for_operation_name('measure'):
                    props = target['measure'].get(qargs, None)
                    if props and props.error is not None:
                        readout_errors.append(props.error)
                if readout_errors:
                    params['readout_error_median'] = float(
                        f"{statistics.median(readout_errors):.6e}"
                    )
            except Exception:
                pass

            # Gate times
            try:
                sx_times = []
                cz_times = []
                for op_name in target.operation_names:
                    for qargs in target.qargs_for_operation_name(op_name):
                        props = target[op_name].get(qargs, None)
                        if props and props.duration is not None:
                            if op_name in ('sx', 'x', 'rz', 'id') and len(qargs) == 1:
                                sx_times.append(props.duration * 1e9)  # s → ns
                            elif len(qargs) == 2:
                                cz_times.append(props.duration * 1e9)
                if sx_times:
                    params['gate_time_single_ns'] = round(statistics.median(sx_times), 1)
                if cz_times:
                    params['gate_time_cz_ns'] = round(statistics.median(cz_times), 1)
            except Exception:
                pass

        # Fill missing values from known processors
        if backend_name in self.KNOWN_PROCESSORS:
            known = self.KNOWN_PROCESSORS[backend_name]
            for key in ['T1_us_median', 'T2_us_median', 'single_qubit_error_median',
                        'two_qubit_error_median', 'readout_error_median',
                        'gate_time_single_ns', 'gate_time_cz_ns', 'processor_type']:
                if key not in params:
                    params[key] = known[key]

        logger.info(f"Fetched noise params for {backend_name} from API")
        return params

    def get_noise_profile_for_sector(
        self, backend_name: str = 'ibm_fez'
    ) -> Dict[str, Any]:
        """
        Convert QPU params to EnhancedNoiseSimulator-compatible profile format.

        Returns a dict with the same keys as EnhancedNoiseSimulator.NOISE_PROFILES entries.

        Args:
            backend_name: IBM Quantum backend name

        Returns:
            Dict compatible with EnhancedNoiseSimulator NOISE_PROFILES format
        """
        params = self.get_noise_params(backend_name)

        # Determine basis_gates for this backend
        basis_gates = params.get('basis_gates', ['cz', 'id', 'rz', 'sx', 'x'])
        proc_family = params.get('processor_family', '')
        if not basis_gates and proc_family:
            basis_gates = self.PROCESSOR_BASIS_GATES.get(proc_family, ['cz', 'id', 'rz', 'sx', 'x'])

        profile = {
            'description': f"IBM {params.get('processor_type', 'QPU')} ({backend_name}, "
                          f"{params.get('num_qubits', '?')} qubits) — "
                          f"{'real params' if params.get('source') == 'ibm_quantum_api' else 'fallback specs'}",
            'single_qubit_error': params.get('single_qubit_error_median', 2.4e-4),
            'two_qubit_error': params.get('two_qubit_error_median', 3.8e-3),
            'thermal_relaxation_t1_us': params.get('T1_us_median', 250.0),
            'thermal_relaxation_t2_us': params.get('T2_us_median', 150.0),
            'readout_error_prob': params.get('readout_error_median', 1.2e-2),
            'gate_time_single_ns': params.get('gate_time_single_ns', 60.0),
            'gate_time_cx_ns': params.get('gate_time_cz_ns', 84.0),
            'basis_gates': basis_gates,
            'processor_family': proc_family,
            'source': params.get('source', 'fallback_specs'),
            'backend_name': backend_name,
        }

        return profile

    def build_noise_model(self, backend_name: str = 'ibm_fez') -> Optional[Any]:
        """
        Build a Qiskit NoiseModel from real QPU parameters.

        Constructs a noise model with:
        - Thermal relaxation errors (T1/T2) on all qubits
        - Depolarizing errors on single and two-qubit gates
        - Readout errors

        Args:
            backend_name: IBM Quantum backend name

        Returns:
            qiskit_aer.noise.NoiseModel or None if Aer not available
        """
        if not QISKIT_AER_AVAILABLE:
            logger.warning("qiskit-aer not available — cannot build NoiseModel")
            return None

        params = self.get_noise_params(backend_name)

        t1_us = params.get('T1_us_median', 250.0)
        t2_us = params.get('T2_us_median', 150.0)
        sq_error = params.get('single_qubit_error_median', 2.4e-4)
        tq_error = params.get('two_qubit_error_median', 3.8e-3)
        readout_err = params.get('readout_error_median', 1.2e-2)
        gate_time_single = params.get('gate_time_single_ns', 60.0)
        gate_time_2q = params.get('gate_time_cz_ns', 84.0)

        # Determine 2Q gate list from basis_gates
        basis_gates = params.get('basis_gates', ['cz', 'id', 'rz', 'sx', 'x'])
        proc_family = params.get('processor_family', '')
        if not basis_gates and proc_family:
            basis_gates = self.PROCESSOR_BASIS_GATES.get(proc_family, ['cz', 'id', 'rz', 'sx', 'x'])

        # Identify the native 2Q gate(s) for this processor
        two_qubit_gates = [g for g in basis_gates if g in ('cz', 'ecr', 'cx')]
        if not two_qubit_gates:
            two_qubit_gates = ['cz']  # Default fallback

        noise_model = NoiseModel()

        # 1) Single-qubit depolarizing errors
        sq_depol = depolarizing_error(sq_error, 1)
        for gate in ['sx', 'x', 'rz', 'id']:
            noise_model.add_all_qubit_quantum_error(sq_depol, gate)

        # 2) Two-qubit depolarizing errors (only on native 2Q gates)
        tq_depol = depolarizing_error(tq_error, 2)
        for gate in two_qubit_gates:
            noise_model.add_all_qubit_quantum_error(tq_depol, gate)

        # 3) Thermal relaxation errors
        t1_ns = t1_us * 1000
        t2_ns = t2_us * 1000
        # Ensure T2 <= 2*T1 (physical constraint)
        t2_ns = min(t2_ns, 2 * t1_ns)

        thermal_1q = thermal_relaxation_error(t1_ns, t2_ns, gate_time_single)
        thermal_2q_0 = thermal_relaxation_error(t1_ns, t2_ns, gate_time_2q)
        thermal_2q_1 = thermal_relaxation_error(t1_ns, t2_ns, gate_time_2q)
        thermal_2q = thermal_2q_0.expand(thermal_2q_1)

        for gate in ['sx', 'x']:
            noise_model.add_all_qubit_quantum_error(thermal_1q, gate)
        for gate in two_qubit_gates:
            noise_model.add_all_qubit_quantum_error(thermal_2q, gate)

        # 4) Readout errors
        p1given0 = readout_err
        p0given1 = readout_err
        ro_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        noise_model.add_all_qubit_readout_error(ro_error)

        logger.info(
            f"Built NoiseModel for {backend_name}: "
            f"T1={t1_us}µs, T2={t2_us}µs, "
            f"SQ_err={sq_error:.2e}, 2Q_err={tq_error:.2e}, "
            f"RO_err={readout_err:.2e}, "
            f"2Q_gates={two_qubit_gates}"
        )

        return noise_model

    # =========================================================================
    # JSON Cache File Management
    # =========================================================================

    def _save_backends_cache(self, backends: List[Dict[str, Any]]) -> bool:
        """
        Save dynamically discovered backends + noise params to JSON cache file.

        Saves to data/ibm_backends_cache.json for offline fallback use.

        Args:
            backends: List of backend info dicts from list_backends()

        Returns:
            True if save succeeded, False otherwise.
        """
        try:
            cache_path = Path(__file__).parent.parent / 'data' / 'ibm_backends_cache.json'
            cache_path.parent.mkdir(exist_ok=True)

            # Collect cached noise params
            noise_params = {}
            for name, (_, params) in self._cache.items():
                noise_params[name] = params

            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'connected': self._connected,
                'backends_count': len(backends),
                'backends': backends,
                'noise_params': noise_params,
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

            logger.info(f"Saved {len(backends)} backends to {cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save backends cache: {e}")
            return False

    def _load_backends_cache(self) -> Optional[Dict[str, Any]]:
        """
        Load backends from JSON cache file.

        Reads from data/ibm_backends_cache.json.

        Returns:
            Dict with 'backends', 'noise_params', 'timestamp' or None if unavailable.
        """
        try:
            cache_path = Path(__file__).parent.parent / 'data' / 'ibm_backends_cache.json'
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)
                logger.debug(f"Loaded JSON cache from {cache_path} ({data.get('timestamp', 'unknown')})")
                return data
        except Exception as e:
            logger.warning(f"Failed to load backends cache: {e}")
        return None

    # =========================================================================
    # Startup Discovery and least_busy
    # =========================================================================

    def startup_connect_and_discover(self) -> Dict[str, Any]:
        """
        Connect to IBM Quantum API and discover all available QPU backends.

        This is the recommended initialization method. It:
        1. Connects to IBM Quantum Platform
        2. Lists all operational backends
        3. Pre-caches noise parameters for each backend
        4. Saves everything to data/ibm_backends_cache.json

        Returns:
            Dict with connection status, discovered backends count, etc.
        """
        result = {
            'connected': False,
            'backends_discovered': 0,
            'noise_params_cached': 0,
            'json_cache_saved': False,
            'errors': [],
        }

        # Step 1: Connect
        connected = self.connect()
        result['connected'] = connected

        # Step 2: List backends (API → JSON cache → KNOWN_PROCESSORS)
        backends = self.list_backends()
        result['backends_discovered'] = len(backends)
        result['backends'] = [b['name'] for b in backends]
        result['source'] = backends[0].get('source', 'unknown') if backends else 'none'

        # Step 3: Pre-cache noise params for all discovered backends
        cached_count = 0
        for b in backends:
            try:
                self.get_noise_params(b['name'])
                cached_count += 1
            except Exception as e:
                result['errors'].append(f"Failed to cache params for {b['name']}: {e}")
        result['noise_params_cached'] = cached_count

        # Step 4: Save to JSON cache file
        result['json_cache_saved'] = self._save_backends_cache(backends)

        logger.info(
            f"IBM Quantum startup: connected={connected}, "
            f"backends={len(backends)}, cached={cached_count}, "
            f"source={result['source']}"
        )
        return result

    def get_least_busy(self, min_num_qubits: int = 5) -> Optional[str]:
        """
        Get the least busy operational backend name.

        Uses IBM Quantum Runtime service.least_busy() when connected,
        otherwise falls back to the first known processor.

        Args:
            min_num_qubits: Minimum number of qubits required

        Returns:
            Backend name string, or None if no backend available.
        """
        if self._connected and self._service:
            try:
                backend = self._service.least_busy(
                    operational=True, simulator=False,
                    min_num_qubits=min_num_qubits,
                )
                logger.info(f"Least busy backend: {backend.name}")
                return backend.name
            except Exception as e:
                logger.warning(f"least_busy API call failed: {e}")

        # Fallback: return first backend from list that meets qubit requirement
        backends = self.list_backends()
        for b in backends:
            if b.get('num_qubits', 0) >= min_num_qubits:
                logger.info(f"Least busy fallback: {b['name']}")
                return b['name']

        # Last resort
        return backends[0]['name'] if backends else 'ibm_fez'

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get connection status and configuration summary.

        Returns:
            Dict with connection state, backend info, and credentials status.
        """
        # Check JSON cache existence
        cache_path = Path(__file__).parent.parent / 'data' / 'ibm_backends_cache.json'
        json_cache_exists = cache_path.exists()
        json_cache_timestamp = None
        if json_cache_exists:
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                    json_cache_timestamp = data.get('timestamp')
            except Exception:
                pass

        return {
            'connected': self._connected,
            'connection_error': self._connection_error,
            'runtime_available': IBM_RUNTIME_AVAILABLE,
            'aer_available': QISKIT_AER_AVAILABLE,
            'token_configured': bool(os.getenv('IBM_QUANTUM_TOKEN')),
            'instance_configured': bool(os.getenv('IBM_QUANTUM_INSTANCE')),
            'cached_backends': list(self._cache.keys()),
            'cache_ttl_seconds': self._cache_ttl,
            'json_cache_exists': json_cache_exists,
            'json_cache_timestamp': json_cache_timestamp,
        }


# =============================================================================
# Module-level Singleton
# =============================================================================

_global_ibm_manager: Optional[IBMQuantumBackendManager] = None


def get_ibm_manager() -> IBMQuantumBackendManager:
    """
    Get the global singleton IBMQuantumBackendManager instance.

    Returns the same instance every time, ensuring:
    - Cache is shared across all endpoints
    - Single API connection is reused
    - JSON cache state is consistent

    Returns:
        IBMQuantumBackendManager singleton instance.
    """
    global _global_ibm_manager
    if _global_ibm_manager is None:
        _global_ibm_manager = IBMQuantumBackendManager()
    return _global_ibm_manager


def reset_ibm_manager() -> None:
    """
    Reset the global singleton (for testing purposes only).
    """
    global _global_ibm_manager
    _global_ibm_manager = None
