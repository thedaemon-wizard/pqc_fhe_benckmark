# Hybrid Workflow Tutorial

This tutorial demonstrates how to combine Post-Quantum Cryptography (PQC) with Fully Homomorphic Encryption (FHE) for complete privacy-preserving systems.

## Overview

The hybrid approach provides:

- **Quantum-resistant key exchange** (ML-KEM) for secure channel establishment
- **Quantum-resistant signatures** (ML-DSA) for authentication
- **Homomorphic encryption** (CKKS) for computation on encrypted data

This tutorial covers:

1. Architecture overview
2. Complete hybrid workflow implementation
3. Client-server communication
4. Real-world use cases
5. Production deployment considerations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hybrid PQC + FHE Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐     ML-KEM Key Exchange      ┌──────────┐            │
│  │  Client  │◄────────────────────────────►│  Server  │            │
│  │  (Bob)   │                              │  (Alice) │            │
│  └────┬─────┘                              └────┬─────┘            │
│       │                                         │                   │
│       │  ┌─────────────────────────────────┐   │                   │
│       │  │  1. PQC Key Exchange (ML-KEM)   │   │                   │
│       │  │  2. ML-DSA Signature Auth       │   │                   │
│       │  │  3. FHE Context Setup           │   │                   │
│       │  │  4. Encrypted Data Transfer     │   │                   │
│       │  │  5. Homomorphic Computation     │   │                   │
│       │  │  6. Signed Result Return        │   │                   │
│       │  └─────────────────────────────────┘   │                   │
│       │                                         │                   │
│       ▼                                         ▼                   │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                  Encrypted Channel                        │      │
│  │  • Session key derived from ML-KEM                        │      │
│  │  • FHE ciphertexts transmitted                            │      │
│  │  • Results signed with ML-DSA                             │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Step 1: Basic Hybrid Setup

```python
"""
Basic Hybrid PQC + FHE Setup
Demonstrates combining PQCKeyManager with FHEEngine
"""

from pqc_fhe_integration import PQCKeyManager, FHEEngine, HybridCryptoManager

# Initialize components
pqc_manager = PQCKeyManager()
fhe_engine = FHEEngine()

# Or use the integrated HybridCryptoManager
hybrid = HybridCryptoManager()

print("Hybrid PQC + FHE system initialized")
print(f"  PQC Algorithms: ML-KEM-768, ML-DSA-65")
print(f"  FHE Scheme: CKKS")
```

## Step 2: Complete Hybrid Protocol

### HybridSecureChannel Class

```python
"""
Complete Hybrid Secure Channel Implementation
Combines PQC key exchange, authentication, and FHE
"""

from pqc_fhe_integration import PQCKeyManager, FHEEngine
import hashlib
import json
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


@dataclass
class SecureMessage:
    """Encrypted message with metadata."""
    ciphertext: bytes
    signature: bytes
    timestamp: int
    sender_id: str
    message_type: str


class HybridSecureChannel:
    """
    Complete hybrid secure channel with:
    - ML-KEM for key exchange
    - ML-DSA for authentication
    - CKKS FHE for homomorphic operations
    """
    
    def __init__(self, identity: str):
        self.identity = identity
        self.pqc = PQCKeyManager()
        self.fhe = FHEEngine()
        
        # Generate long-term identity keys (ML-DSA)
        self.signing_public, self.signing_secret = self.pqc.generate_keypair(
            algorithm="ML-DSA-65"
        )
        
        # Session state
        self.session_key: Optional[bytes] = None
        self.peer_signing_key: Optional[bytes] = None
        self.peer_identity: Optional[str] = None
        
    def get_identity_bundle(self) -> Dict[str, Any]:
        """Get identity information for peer verification."""
        return {
            "identity": self.identity,
            "signing_public_key": base64.b64encode(self.signing_public).decode(),
            "supported_algorithms": {
                "kem": ["ML-KEM-768", "ML-KEM-1024"],
                "signature": ["ML-DSA-65", "ML-DSA-87"],
                "fhe": ["CKKS"]
            }
        }
    
    def initiate_session(self, peer_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initiate secure session with peer.
        
        Returns session initiation message to send to peer.
        """
        # Store peer identity
        self.peer_identity = peer_bundle["identity"]
        self.peer_signing_key = base64.b64decode(peer_bundle["signing_public_key"])
        
        # Generate ephemeral KEM keypair
        self.kem_public, self.kem_secret = self.pqc.generate_keypair(
            algorithm="ML-KEM-768"
        )
        
        # Create session request
        timestamp = int(time.time())
        message_data = {
            "type": "session_init",
            "sender": self.identity,
            "kem_public_key": base64.b64encode(self.kem_public).decode(),
            "timestamp": timestamp,
            "nonce": base64.b64encode(hashlib.sha256(
                str(timestamp).encode() + self.identity.encode()
            ).digest()[:16]).decode()
        }
        
        # Sign the request
        message_bytes = json.dumps(message_data, sort_keys=True).encode()
        signature = self.pqc.sign(message_bytes, self.signing_secret)
        
        return {
            "message": message_data,
            "signature": base64.b64encode(signature).decode()
        }
    
    def respond_to_session(
        self, 
        init_message: Dict[str, Any],
        peer_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Respond to session initiation.
        
        Verifies peer signature, performs key encapsulation,
        and returns response with ciphertext.
        """
        # Store peer info
        self.peer_identity = peer_bundle["identity"]
        self.peer_signing_key = base64.b64decode(peer_bundle["signing_public_key"])
        
        # Verify signature
        message_bytes = json.dumps(init_message["message"], sort_keys=True).encode()
        signature = base64.b64decode(init_message["signature"])
        
        if not self.pqc.verify(message_bytes, signature, self.peer_signing_key):
            raise SecurityError("Invalid signature on session init!")
        
        # Verify timestamp (within 5 minutes)
        timestamp = init_message["message"]["timestamp"]
        if abs(time.time() - timestamp) > 300:
            raise SecurityError("Session init timestamp too old!")
        
        # Extract peer's KEM public key
        peer_kem_public = base64.b64decode(init_message["message"]["kem_public_key"])
        
        # Perform key encapsulation
        ciphertext, shared_secret = self.pqc.encapsulate(peer_kem_public)
        
        # Derive session key
        self.session_key = self._derive_session_key(
            shared_secret,
            init_message["message"]["nonce"].encode()
        )
        
        # Create response
        response_timestamp = int(time.time())
        response_data = {
            "type": "session_response",
            "sender": self.identity,
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "timestamp": response_timestamp,
            "init_nonce": init_message["message"]["nonce"]
        }
        
        # Sign response
        response_bytes = json.dumps(response_data, sort_keys=True).encode()
        response_signature = self.pqc.sign(response_bytes, self.signing_secret)
        
        return {
            "message": response_data,
            "signature": base64.b64encode(response_signature).decode()
        }
    
    def complete_session(self, response: Dict[str, Any]) -> bool:
        """
        Complete session setup by processing peer's response.
        
        Returns True if session established successfully.
        """
        # Verify signature
        message_bytes = json.dumps(response["message"], sort_keys=True).encode()
        signature = base64.b64decode(response["signature"])
        
        if not self.pqc.verify(message_bytes, signature, self.peer_signing_key):
            raise SecurityError("Invalid signature on session response!")
        
        # Extract ciphertext
        ciphertext = base64.b64decode(response["message"]["ciphertext"])
        
        # Decapsulate to get shared secret
        shared_secret = self.pqc.decapsulate(ciphertext, self.kem_secret)
        
        # Derive session key (using same nonce as init)
        nonce = response["message"]["init_nonce"].encode()
        self.session_key = self._derive_session_key(shared_secret, nonce)
        
        return True
    
    def _derive_session_key(self, shared_secret: bytes, nonce: bytes) -> bytes:
        """Derive session key using HKDF-like construction."""
        # In production, use proper HKDF
        return hashlib.sha256(
            shared_secret + nonce + b"session_v1"
        ).digest()
    
    def encrypt_data_fhe(self, data: list) -> bytes:
        """Encrypt data using FHE for homomorphic operations."""
        if not self.session_key:
            raise RuntimeError("Session not established!")
        
        # FHE encrypt the data
        ciphertext = self.fhe.encrypt(data)
        
        # Serialize (implementation-specific)
        return self._serialize_fhe_ciphertext(ciphertext)
    
    def decrypt_data_fhe(self, encrypted_data: bytes) -> list:
        """Decrypt FHE ciphertext."""
        if not self.session_key:
            raise RuntimeError("Session not established!")
        
        # Deserialize
        ciphertext = self._deserialize_fhe_ciphertext(encrypted_data)
        
        # Decrypt
        return self.fhe.decrypt(ciphertext)
    
    def compute_on_encrypted(
        self, 
        encrypted_data: bytes,
        operation: str,
        operand: Any = None
    ) -> bytes:
        """
        Perform homomorphic operation on encrypted data.
        
        Args:
            encrypted_data: FHE ciphertext
            operation: Operation name (add, multiply, square, etc.)
            operand: Optional operand (another ciphertext or scalar)
        """
        ct = self._deserialize_fhe_ciphertext(encrypted_data)
        
        if operation == "add":
            if isinstance(operand, bytes):
                ct_operand = self._deserialize_fhe_ciphertext(operand)
                result = self.fhe.add(ct, ct_operand)
            else:
                result = self.fhe.add_scalar(ct, float(operand))
        elif operation == "multiply":
            if isinstance(operand, bytes):
                ct_operand = self._deserialize_fhe_ciphertext(operand)
                result = self.fhe.multiply(ct, ct_operand)
            else:
                result = self.fhe.multiply_scalar(ct, float(operand))
        elif operation == "square":
            result = self.fhe.square(ct)
        elif operation == "negate":
            result = self.fhe.negate(ct)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return self._serialize_fhe_ciphertext(result)
    
    def create_signed_message(self, content: Dict[str, Any]) -> SecureMessage:
        """Create a signed message."""
        timestamp = int(time.time())
        message_bytes = json.dumps(content, sort_keys=True).encode()
        
        signature = self.pqc.sign(
            message_bytes + timestamp.to_bytes(8, 'big'),
            self.signing_secret
        )
        
        return SecureMessage(
            ciphertext=message_bytes,
            signature=signature,
            timestamp=timestamp,
            sender_id=self.identity,
            message_type=content.get("type", "data")
        )
    
    def verify_signed_message(self, message: SecureMessage) -> Dict[str, Any]:
        """Verify and extract signed message content."""
        # Verify signature
        signed_data = message.ciphertext + message.timestamp.to_bytes(8, 'big')
        
        if not self.pqc.verify(signed_data, message.signature, self.peer_signing_key):
            raise SecurityError("Invalid message signature!")
        
        # Verify timestamp
        if abs(time.time() - message.timestamp) > 300:
            raise SecurityError("Message timestamp too old!")
        
        return json.loads(message.ciphertext.decode())
    
    def _serialize_fhe_ciphertext(self, ciphertext) -> bytes:
        """Serialize FHE ciphertext for transmission."""
        # Implementation depends on FHE library
        # This is a placeholder - real implementation would serialize properly
        return str(ciphertext).encode()
    
    def _deserialize_fhe_ciphertext(self, data: bytes):
        """Deserialize FHE ciphertext."""
        # Implementation depends on FHE library
        # This is a placeholder
        return data


class SecurityError(Exception):
    """Security-related error."""
    pass
```

## Step 3: Client-Server Example

### Server Implementation

```python
"""
Hybrid Secure Server
Accepts encrypted data, performs computation, returns signed results
"""

class HybridServer:
    """Server that processes encrypted data."""
    
    def __init__(self, server_id: str = "computation_server"):
        self.channel = HybridSecureChannel(server_id)
        self.clients: Dict[str, Dict] = {}
        
    def get_server_identity(self) -> Dict[str, Any]:
        """Get server identity bundle for clients."""
        return self.channel.get_identity_bundle()
    
    def handle_session_init(
        self, 
        client_id: str,
        client_bundle: Dict[str, Any],
        init_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle session initialization from client."""
        
        # Create new channel for this client
        client_channel = HybridSecureChannel(self.channel.identity)
        client_channel.signing_public = self.channel.signing_public
        client_channel.signing_secret = self.channel.signing_secret
        
        # Respond to session
        response = client_channel.respond_to_session(init_message, client_bundle)
        
        # Store client channel
        self.clients[client_id] = {
            "channel": client_channel,
            "connected_at": time.time()
        }
        
        return response
    
    def process_computation_request(
        self,
        client_id: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process computation request on encrypted data.
        
        Request format:
        {
            "operation": "mean" | "sum" | "polynomial" | ...,
            "encrypted_data": base64_encoded_ciphertext,
            "parameters": {...}
        }
        """
        if client_id not in self.clients:
            raise ValueError("Unknown client")
        
        channel = self.clients[client_id]["channel"]
        operation = request["operation"]
        
        # Decode encrypted data
        encrypted_data = base64.b64decode(request["encrypted_data"])
        
        # Perform computation based on operation
        if operation == "sum":
            result = self._compute_sum(channel, encrypted_data)
        elif operation == "mean":
            n = request["parameters"].get("count", 1)
            result = self._compute_mean(channel, encrypted_data, n)
        elif operation == "polynomial":
            coefficients = request["parameters"]["coefficients"]
            result = self._compute_polynomial(channel, encrypted_data, coefficients)
        elif operation == "custom":
            ops = request["parameters"]["operations"]
            result = self._execute_operations(channel, encrypted_data, ops)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Create signed response
        response_content = {
            "type": "computation_result",
            "operation": operation,
            "result": base64.b64encode(result).decode(),
            "computation_time": time.time()
        }
        
        signed_response = channel.create_signed_message(response_content)
        
        return {
            "result": response_content,
            "signature": base64.b64encode(signed_response.signature).decode(),
            "timestamp": signed_response.timestamp
        }
    
    def _compute_sum(self, channel: HybridSecureChannel, data: bytes) -> bytes:
        """Compute sum of encrypted values."""
        # In real implementation, would handle multiple ciphertexts
        return data
    
    def _compute_mean(
        self, 
        channel: HybridSecureChannel, 
        data: bytes, 
        n: int
    ) -> bytes:
        """Compute mean by dividing by n."""
        return channel.compute_on_encrypted(data, "multiply", 1.0 / n)
    
    def _compute_polynomial(
        self,
        channel: HybridSecureChannel,
        data: bytes,
        coefficients: list
    ) -> bytes:
        """Evaluate polynomial on encrypted data."""
        # Horner's method
        result = channel.compute_on_encrypted(
            data, "multiply", coefficients[-1]
        )
        
        for coef in reversed(coefficients[:-1]):
            # result = result * x
            result = channel.compute_on_encrypted(result, "multiply", data)
            # result = result + coef
            result = channel.compute_on_encrypted(result, "add", coef)
        
        return result
    
    def _execute_operations(
        self,
        channel: HybridSecureChannel,
        data: bytes,
        operations: list
    ) -> bytes:
        """Execute a sequence of operations."""
        result = data
        for op in operations:
            result = channel.compute_on_encrypted(
                result,
                op["operation"],
                op.get("operand")
            )
        return result
```

### Client Implementation

```python
"""
Hybrid Secure Client
Encrypts data, sends to server, receives signed results
"""

class HybridClient:
    """Client that submits encrypted data for processing."""
    
    def __init__(self, client_id: str):
        self.channel = HybridSecureChannel(client_id)
        self.server_bundle: Optional[Dict] = None
        self.connected = False
    
    def get_client_identity(self) -> Dict[str, Any]:
        """Get client identity bundle."""
        return self.channel.get_identity_bundle()
    
    def connect_to_server(self, server_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initiate connection to server.
        
        Returns session init message to send to server.
        """
        self.server_bundle = server_bundle
        return self.channel.initiate_session(server_bundle)
    
    def complete_connection(self, server_response: Dict[str, Any]) -> bool:
        """Complete connection with server response."""
        success = self.channel.complete_session(server_response)
        self.connected = success
        return success
    
    def submit_computation(
        self,
        data: list,
        operation: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Submit encrypted data for computation.
        
        Args:
            data: Plaintext data to encrypt
            operation: Operation to perform
            parameters: Additional parameters
            
        Returns:
            Request dict to send to server
        """
        if not self.connected:
            raise RuntimeError("Not connected to server!")
        
        # Encrypt data with FHE
        encrypted_data = self.channel.encrypt_data_fhe(data)
        
        return {
            "operation": operation,
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "parameters": parameters or {}
        }
    
    def process_result(self, response: Dict[str, Any]) -> Any:
        """
        Process signed result from server.
        
        Verifies signature and decrypts result.
        """
        if not self.connected:
            raise RuntimeError("Not connected!")
        
        # Verify signature
        result_data = response["result"]
        signature = base64.b64decode(response["signature"])
        
        message = SecureMessage(
            ciphertext=json.dumps(result_data, sort_keys=True).encode(),
            signature=signature,
            timestamp=response["timestamp"],
            sender_id=self.server_bundle["identity"],
            message_type="computation_result"
        )
        
        verified_content = self.channel.verify_signed_message(message)
        
        # Decrypt result
        encrypted_result = base64.b64decode(verified_content["result"])
        plaintext_result = self.channel.decrypt_data_fhe(encrypted_result)
        
        return plaintext_result
```

### Complete Workflow Example

```python
def demonstrate_hybrid_workflow():
    """Complete hybrid workflow demonstration."""
    
    print("=" * 60)
    print("HYBRID PQC + FHE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # === Setup ===
    print("\n[1] Initializing server and client...")
    server = HybridServer("analytics_server")
    client = HybridClient("data_client_001")
    
    # === Identity Exchange ===
    print("\n[2] Exchanging identity bundles...")
    server_bundle = server.get_server_identity()
    client_bundle = client.get_client_identity()
    
    print(f"    Server ID: {server_bundle['identity']}")
    print(f"    Client ID: {client_bundle['identity']}")
    
    # === Session Establishment ===
    print("\n[3] Establishing secure session...")
    
    # Client initiates
    init_message = client.connect_to_server(server_bundle)
    print("    Client: Session init sent (signed with ML-DSA)")
    
    # Server responds
    server_response = server.handle_session_init(
        client_bundle['identity'],
        client_bundle,
        init_message
    )
    print("    Server: Session response sent (ML-KEM encapsulation)")
    
    # Client completes
    client.complete_connection(server_response)
    print("    Client: Session established!")
    print("    Shared session key derived via ML-KEM")
    
    # === Data Processing ===
    print("\n[4] Submitting encrypted computation request...")
    
    # Client has private data
    private_data = [100.0, 200.0, 300.0, 400.0, 500.0]
    print(f"    Private data: {private_data}")
    
    # Client submits encrypted computation request
    request = client.submit_computation(
        data=private_data,
        operation="mean",
        parameters={"count": len(private_data)}
    )
    print("    Data encrypted with CKKS FHE")
    print("    Request sent to server")
    
    # === Server Processing ===
    print("\n[5] Server processing (on encrypted data)...")
    
    # Server processes without seeing plaintext
    response = server.process_computation_request(
        client_bundle['identity'],
        request
    )
    print("    Computation performed homomorphically")
    print("    Result signed with ML-DSA")
    
    # === Result Retrieval ===
    print("\n[6] Client retrieving result...")
    
    # Client verifies and decrypts
    result = client.process_result(response)
    print("    Signature verified")
    print("    Result decrypted")
    
    # === Verification ===
    print("\n[7] Results:")
    expected = sum(private_data) / len(private_data)
    print(f"    Expected mean: {expected}")
    print(f"    Computed mean: {result}")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nSecurity guarantees:")
    print("  - Key exchange: ML-KEM-768 (quantum-resistant)")
    print("  - Authentication: ML-DSA-65 (quantum-resistant)")
    print("  - Data privacy: CKKS FHE (computation on encrypted data)")
    print("  - Server never saw plaintext data!")


if __name__ == "__main__":
    demonstrate_hybrid_workflow()
```

## Step 4: REST API Integration

### Using the Hybrid REST API

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def hybrid_api_example():
    """Example using the hybrid REST API."""
    
    # 1. Get server public keys
    response = requests.get(f"{BASE_URL}/api/v1/keys/public")
    server_keys = response.json()
    print(f"Server keys retrieved")
    
    # 2. Establish session
    session_request = {
        "client_id": "api_client_001",
        "kem_algorithm": "ML-KEM-768",
        "signature_algorithm": "ML-DSA-65"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/session/initiate",
        json=session_request
    )
    session = response.json()
    print(f"Session established: {session['session_id']}")
    
    # 3. Encrypt and submit data
    data_request = {
        "session_id": session["session_id"],
        "operation": "statistics",
        "data": [100, 200, 300, 400, 500],  # Will be encrypted client-side
        "parameters": {
            "compute": ["mean", "variance", "sum"]
        }
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/compute/encrypted",
        json=data_request
    )
    result = response.json()
    
    # 4. Verify and display results
    print(f"Results:")
    print(f"  Mean: {result['results']['mean']}")
    print(f"  Variance: {result['results']['variance']}")
    print(f"  Sum: {result['results']['sum']}")
    print(f"  Signature verified: {result['signature_valid']}")


# CLI equivalent
"""
# Start server
pqc-fhe-api --host 0.0.0.0 --port 8000

# Generate keys
pqc-fhe keygen --algorithm ML-KEM-768 --output-public client_kem.pub

# Encrypt data
pqc-fhe fhe-encrypt --values 100 200 300 400 500 --output data.enc

# Submit computation (using curl)
curl -X POST http://localhost:8000/api/v1/compute \
    -H "Content-Type: application/json" \
    -d '{"session_id": "...", "data_file": "data.enc", "operation": "mean"}'
"""
```

## Step 5: Use Cases

### Financial Analytics

```python
"""
Private portfolio analysis with hybrid security.
"""

def private_portfolio_analysis():
    """Analyze portfolio without exposing holdings."""
    
    hybrid = HybridCryptoManager()
    
    # Client's private portfolio (shares owned)
    portfolio = {
        "AAPL": 100,
        "GOOGL": 50,
        "MSFT": 75,
        "AMZN": 30
    }
    
    # Public prices (known to server)
    prices = {
        "AAPL": 150.0,
        "GOOGL": 140.0,
        "MSFT": 380.0,
        "AMZN": 175.0
    }
    
    # Encrypt holdings
    encrypted_holdings = {
        ticker: hybrid.fhe_encrypt([float(shares)])
        for ticker, shares in portfolio.items()
    }
    
    # Server computes portfolio value (without seeing holdings)
    total_value_encrypted = None
    for ticker, ct_shares in encrypted_holdings.items():
        ct_value = hybrid.fhe_multiply_scalar(ct_shares, prices[ticker])
        if total_value_encrypted is None:
            total_value_encrypted = ct_value
        else:
            total_value_encrypted = hybrid.fhe_add(
                total_value_encrypted, ct_value
            )
    
    # Client decrypts total value
    total_value = hybrid.fhe_decrypt(total_value_encrypted)[0]
    
    # Verify
    expected = sum(shares * prices[ticker] 
                   for ticker, shares in portfolio.items())
    
    print(f"Portfolio Value Analysis (Private)")
    print(f"  Computed total: ${total_value:,.2f}")
    print(f"  Expected total: ${expected:,.2f}")
    print(f"  Server never saw individual holdings!")
```

### Healthcare Data

```python
"""
Private health metrics computation.
"""

def private_health_metrics():
    """Compute health metrics without exposing patient data."""
    
    hybrid = HybridCryptoManager()
    
    # Patient's private health data
    patient_data = {
        "heart_rate_readings": [72, 75, 71, 78, 74, 73, 76],
        "blood_pressure_systolic": [120, 118, 122, 119, 121],
        "blood_glucose": [95, 102, 98, 100, 97]
    }
    
    results = {}
    
    for metric, readings in patient_data.items():
        # Encrypt readings
        ct_readings = [hybrid.fhe_encrypt([float(r)]) for r in readings]
        
        # Compute encrypted mean
        ct_sum = ct_readings[0]
        for ct in ct_readings[1:]:
            ct_sum = hybrid.fhe_add(ct_sum, ct)
        ct_mean = hybrid.fhe_multiply_scalar(ct_sum, 1.0 / len(readings))
        
        # Decrypt result
        results[metric] = hybrid.fhe_decrypt(ct_mean)[0]
    
    print("Health Metrics (Computed Privately)")
    print(f"  Average heart rate: {results['heart_rate_readings']:.1f} bpm")
    print(f"  Average systolic BP: {results['blood_pressure_systolic']:.1f} mmHg")
    print(f"  Average blood glucose: {results['blood_glucose']:.1f} mg/dL")
```

## Production Considerations

### Security Checklist

```markdown
- [ ] Use ML-KEM-768 or higher for key exchange
- [ ] Use ML-DSA-65 or higher for signatures  
- [ ] Enable authentication for all sessions
- [ ] Implement rate limiting
- [ ] Use secure random number generation
- [ ] Implement key rotation policies
- [ ] Enable audit logging
- [ ] Use TLS for transport layer
- [ ] Implement proper error handling (no information leakage)
- [ ] Regular security audits
```

### Performance Optimization

```python
# Batch multiple values in single ciphertext (SIMD)
batch_data = [v1, v2, v3, v4, v5, v6, v7, v8]
ct_batch = fhe.encrypt(batch_data)  # Single encryption

# Precompute common operations
precomputed_weights = [fhe.encode(w) for w in model_weights]

# Use scalar operations when possible (faster than ct-ct)
ct_result = fhe.multiply_scalar(ct, 2.0)  # Faster
# vs
# ct_result = fhe.multiply(ct, ct_two)  # Slower
```

## Next Steps

- [Enterprise Integration Tutorial](enterprise_integration.md) - Production deployment
- [Security Best Practices](../security/best_practices.md) - Hardening guidelines
- [API Reference](../api/overview.md) - Complete API documentation
