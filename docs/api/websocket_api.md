# WebSocket API Reference

Real-time quantum-safe communication with FHE computation support.

## Overview

The PQC-FHE WebSocket API enables secure, real-time bidirectional communication
with post-quantum cryptographic protection and homomorphic encryption capabilities.

### Key Features

| Feature | Description |
|---------|-------------|
| **PQC Key Exchange** | ML-KEM-768 session establishment |
| **FHE Operations** | Real-time encrypted computation |
| **Hybrid Security** | Combined classical + PQC protection |
| **Channel Security** | End-to-end encryption on all messages |

### Protocol Flow

```
Client                                           Server
   |                                                |
   |-------- WebSocket Connect ------------------>  |
   |                                                |
   |<------- session.created (session_id) -------  |
   |                                                |
   |-------- pqc.initiate (algorithm, pubkey) -->  |
   |                                                |
   |<------- pqc.response (ciphertext) ----------  |
   |                                                |
   |-------- pqc.confirm (shared_key_hash) ----->  |
   |                                                |
   |<------- pqc.established -------------------   |
   |                                                |
   |======== Secure Channel Established ==========  |
   |                                                |
   |-------- fhe.operation (encrypted_data) ---->  |
   |                                                |
   |<------- fhe.result (encrypted_result) ------  |
   |                                                |
```

## Connection

### Endpoint

```
wss://your-server.com/ws/v1/secure
```

### Connection Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `client_id` | string | Yes | Unique client identifier |
| `auth_token` | string | Yes | JWT authentication token |
| `protocol_version` | string | No | API version (default: "1.0") |

### Example Connection

```python
import asyncio
import websockets
import json

async def connect():
    uri = "wss://localhost:8443/ws/v1/secure"
    
    headers = {
        "Authorization": "Bearer <jwt_token>",
        "X-Client-ID": "client-12345"
    }
    
    async with websockets.connect(uri, extra_headers=headers) as ws:
        # Wait for session creation
        response = await ws.recv()
        session = json.loads(response)
        print(f"Session created: {session['session_id']}")
        
        # Continue with PQC key exchange...
```

## Message Format

### Request Structure

```json
{
    "type": "request",
    "id": "msg-uuid-12345",
    "action": "pqc.initiate",
    "timestamp": "2025-01-15T10:30:00Z",
    "payload": {
        // Action-specific data
    }
}
```

### Response Structure

```json
{
    "type": "response",
    "id": "msg-uuid-12345",
    "action": "pqc.initiate",
    "status": "success",
    "timestamp": "2025-01-15T10:30:01Z",
    "payload": {
        // Response data
    }
}
```

### Event Structure

```json
{
    "type": "event",
    "event": "session.timeout",
    "timestamp": "2025-01-15T11:30:00Z",
    "data": {
        // Event data
    }
}
```

## Session Management

### session.create

Automatically triggered upon connection.

**Response:**

```json
{
    "type": "event",
    "event": "session.created",
    "timestamp": "2025-01-15T10:30:00Z",
    "data": {
        "session_id": "sess-abc123",
        "expires_at": "2025-01-15T11:30:00Z",
        "capabilities": [
            "pqc.ml_kem_768",
            "pqc.ml_dsa_65",
            "fhe.ckks",
            "hybrid.aes_256_gcm"
        ],
        "server_public_key": "base64_encoded_server_public_key"
    }
}
```

### session.refresh

Extend session lifetime.

**Request:**

```json
{
    "type": "request",
    "id": "msg-001",
    "action": "session.refresh",
    "payload": {}
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-001",
    "action": "session.refresh",
    "status": "success",
    "payload": {
        "session_id": "sess-abc123",
        "expires_at": "2025-01-15T12:30:00Z"
    }
}
```

### session.terminate

Gracefully close session.

**Request:**

```json
{
    "type": "request",
    "id": "msg-002",
    "action": "session.terminate",
    "payload": {
        "reason": "client_logout"
    }
}
```

## PQC Key Exchange

### pqc.initiate

Initiate post-quantum key exchange.

**Request:**

```json
{
    "type": "request",
    "id": "msg-003",
    "action": "pqc.initiate",
    "payload": {
        "algorithm": "ML-KEM-768",
        "client_public_key": "base64_encoded_kyber_public_key",
        "hybrid_mode": true,
        "classical_algorithm": "X25519"
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-003",
    "action": "pqc.initiate",
    "status": "success",
    "payload": {
        "algorithm": "ML-KEM-768",
        "ciphertext": "base64_encoded_kyber_ciphertext",
        "server_ephemeral_public": "base64_x25519_public_if_hybrid",
        "key_id": "key-xyz789"
    }
}
```

### pqc.confirm

Confirm key exchange completion.

**Request:**

```json
{
    "type": "request",
    "id": "msg-004",
    "action": "pqc.confirm",
    "payload": {
        "key_id": "key-xyz789",
        "shared_key_hash": "sha256_first_16_bytes_of_derived_key",
        "ready_for_secure_channel": true
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-004",
    "action": "pqc.confirm",
    "status": "success",
    "payload": {
        "secure_channel_established": true,
        "channel_id": "chan-secure-001",
        "cipher_suite": "AES-256-GCM",
        "key_derivation": "HKDF-SHA384"
    }
}
```

### Supported Algorithms

| Algorithm | Security Level | Key Size | Ciphertext Size |
|-----------|---------------|----------|-----------------|
| ML-KEM-512 | NIST Level 1 | 800 bytes | 768 bytes |
| ML-KEM-768 | NIST Level 3 | 1,184 bytes | 1,088 bytes |
| ML-KEM-1024 | NIST Level 5 | 1,568 bytes | 1,568 bytes |

## FHE Operations

### fhe.setup

Initialize FHE context for session.

**Request:**

```json
{
    "type": "request",
    "id": "msg-010",
    "action": "fhe.setup",
    "payload": {
        "scheme": "CKKS",
        "poly_modulus_degree": 8192,
        "coeff_modulus_bits": [60, 40, 40, 60],
        "scale": 1099511627776,
        "security_level": 128
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-010",
    "action": "fhe.setup",
    "status": "success",
    "payload": {
        "context_id": "fhe-ctx-001",
        "public_key_id": "fhe-pk-001",
        "relin_key_id": "fhe-rk-001",
        "galois_key_id": "fhe-gk-001",
        "max_depth": 3,
        "slot_count": 4096
    }
}
```

### fhe.encrypt

Encrypt data using FHE.

**Request:**

```json
{
    "type": "request",
    "id": "msg-011",
    "action": "fhe.encrypt",
    "payload": {
        "context_id": "fhe-ctx-001",
        "data": [1.5, 2.3, 4.7, 8.1],
        "data_type": "float_array",
        "batch_encode": true
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-011",
    "action": "fhe.encrypt",
    "status": "success",
    "payload": {
        "ciphertext_id": "ct-001",
        "ciphertext": "base64_encoded_ciphertext",
        "size_bytes": 262144,
        "noise_budget": 120
    }
}
```

### fhe.compute

Perform computation on encrypted data.

**Request:**

```json
{
    "type": "request",
    "id": "msg-012",
    "action": "fhe.compute",
    "payload": {
        "context_id": "fhe-ctx-001",
        "operation": "polynomial",
        "operands": ["ct-001", "ct-002"],
        "parameters": {
            "expression": "a * b + 0.5 * a",
            "variable_mapping": {
                "a": "ct-001",
                "b": "ct-002"
            }
        }
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-012",
    "action": "fhe.compute",
    "status": "success",
    "payload": {
        "result_ciphertext_id": "ct-003",
        "result_ciphertext": "base64_encoded_result",
        "operations_performed": ["multiply", "multiply_plain", "add"],
        "remaining_depth": 1,
        "noise_budget": 45,
        "computation_time_ms": 125
    }
}
```

### Supported FHE Operations

| Operation | Description | Depth Cost |
|-----------|-------------|------------|
| `add` | Add two ciphertexts | 0 |
| `add_plain` | Add plaintext to ciphertext | 0 |
| `multiply` | Multiply two ciphertexts | 1 |
| `multiply_plain` | Multiply by plaintext | 0* |
| `square` | Square ciphertext | 1 |
| `rotate` | Rotate slots | 0 |
| `polynomial` | Evaluate polynomial | varies |

*Note: multiply_plain with large plaintext may consume depth.

### fhe.decrypt

Decrypt FHE result (requires secret key on client).

**Request:**

```json
{
    "type": "request",
    "id": "msg-013",
    "action": "fhe.decrypt",
    "payload": {
        "context_id": "fhe-ctx-001",
        "ciphertext_id": "ct-003",
        "expected_size": 4
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-013",
    "action": "fhe.decrypt",
    "status": "success",
    "payload": {
        "decrypted_data": [5.25, 9.43, 19.74, 34.02],
        "precision_bits": 40
    }
}
```

## Hybrid Secure Messaging

### secure.send

Send encrypted message through secure channel.

**Request:**

```json
{
    "type": "request",
    "id": "msg-020",
    "action": "secure.send",
    "payload": {
        "channel_id": "chan-secure-001",
        "encrypted_data": "base64_aes_gcm_encrypted_payload",
        "iv": "base64_initialization_vector",
        "tag": "base64_authentication_tag",
        "sequence_number": 42
    }
}
```

**Response:**

```json
{
    "type": "response",
    "id": "msg-020",
    "action": "secure.send",
    "status": "success",
    "payload": {
        "delivered": true,
        "server_sequence": 42
    }
}
```

### secure.receive

Receive encrypted message (event-based).

**Event:**

```json
{
    "type": "event",
    "event": "secure.message",
    "data": {
        "channel_id": "chan-secure-001",
        "encrypted_data": "base64_aes_gcm_encrypted_payload",
        "iv": "base64_initialization_vector",
        "tag": "base64_authentication_tag",
        "sequence_number": 43,
        "timestamp": "2025-01-15T10:35:00Z"
    }
}
```

## Error Handling

### Error Response Format

```json
{
    "type": "error",
    "id": "msg-xxx",
    "code": "PQC_KEY_EXCHANGE_FAILED",
    "message": "Key exchange verification failed",
    "details": {
        "reason": "hash_mismatch",
        "expected": "abc123...",
        "received": "def456..."
    },
    "timestamp": "2025-01-15T10:30:05Z"
}
```

### Error Codes

| Code | HTTP Equiv | Description |
|------|------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Invalid or expired token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `SESSION_NOT_FOUND` | 404 | Session expired or invalid |
| `PQC_KEY_EXCHANGE_FAILED` | 400 | Key exchange error |
| `FHE_CONTEXT_ERROR` | 400 | Invalid FHE context |
| `FHE_COMPUTATION_ERROR` | 400 | FHE operation failed |
| `NOISE_BUDGET_EXHAUSTED` | 400 | FHE noise too high |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `SERVER_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Server overloaded |

## Rate Limiting

### Limits

| Operation Type | Limit | Window |
|---------------|-------|--------|
| Connection | 10 | per minute |
| PQC Key Exchange | 5 | per minute |
| FHE Setup | 2 | per minute |
| FHE Encrypt | 100 | per minute |
| FHE Compute | 50 | per minute |
| Secure Messages | 1000 | per minute |

### Rate Limit Headers

Included in responses when approaching limits:

```json
{
    "rate_limit": {
        "limit": 100,
        "remaining": 23,
        "reset_at": "2025-01-15T10:31:00Z"
    }
}
```

## Complete Client Example

```python
import asyncio
import websockets
import json
import base64
import hashlib
from datetime import datetime
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import oqs  # liboqs Python bindings

class PQCFHEWebSocketClient:
    """
    Post-Quantum Secure WebSocket Client with FHE Support
    
    References:
    - NIST FIPS 203 (ML-KEM)
    - DESILO FHE Library Documentation
    """
    
    def __init__(self, server_url: str, client_id: str, auth_token: str):
        self.server_url = server_url
        self.client_id = client_id
        self.auth_token = auth_token
        self.session_id = None
        self.channel_key = None
        self.ws = None
        self.msg_counter = 0
    
    async def connect(self):
        """Establish WebSocket connection"""
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "X-Client-ID": self.client_id
        }
        
        self.ws = await websockets.connect(
            self.server_url,
            extra_headers=headers,
            ping_interval=30,
            ping_timeout=10
        )
        
        # Wait for session creation
        response = await self._recv()
        if response.get("event") == "session.created":
            self.session_id = response["data"]["session_id"]
            return response["data"]
        
        raise ConnectionError("Failed to create session")
    
    async def establish_pqc_channel(self, algorithm: str = "ML-KEM-768"):
        """
        Establish post-quantum secure channel
        
        NIST FIPS 203 compliant key encapsulation
        """
        # Initialize ML-KEM (Kyber)
        kem = oqs.KeyEncapsulation(algorithm.replace("-", ""))
        client_public_key = kem.generate_keypair()
        
        # Send key exchange initiation
        response = await self._send_request("pqc.initiate", {
            "algorithm": algorithm,
            "client_public_key": base64.b64encode(client_public_key).decode(),
            "hybrid_mode": False
        })
        
        # Decapsulate shared secret
        ciphertext = base64.b64decode(response["payload"]["ciphertext"])
        shared_secret = kem.decap_secret(ciphertext)
        
        # Derive channel key using HKDF
        self.channel_key = self._derive_key(shared_secret, b"channel-key")
        
        # Confirm key exchange
        key_hash = hashlib.sha256(self.channel_key[:16]).hexdigest()[:32]
        confirm_response = await self._send_request("pqc.confirm", {
            "key_id": response["payload"]["key_id"],
            "shared_key_hash": key_hash,
            "ready_for_secure_channel": True
        })
        
        return confirm_response["payload"]
    
    async def setup_fhe(self, poly_degree: int = 8192):
        """Initialize FHE context"""
        return await self._send_request("fhe.setup", {
            "scheme": "CKKS",
            "poly_modulus_degree": poly_degree,
            "coeff_modulus_bits": [60, 40, 40, 60],
            "scale": 2**40,
            "security_level": 128
        })
    
    async def fhe_encrypt(self, context_id: str, data: list):
        """Encrypt data using FHE"""
        return await self._send_request("fhe.encrypt", {
            "context_id": context_id,
            "data": data,
            "data_type": "float_array",
            "batch_encode": True
        })
    
    async def fhe_compute(self, context_id: str, operation: str,
                          operands: list, parameters: dict = None):
        """Perform FHE computation"""
        return await self._send_request("fhe.compute", {
            "context_id": context_id,
            "operation": operation,
            "operands": operands,
            "parameters": parameters or {}
        })
    
    async def send_secure_message(self, data: bytes):
        """Send encrypted message through secure channel"""
        if not self.channel_key:
            raise RuntimeError("Secure channel not established")
        
        # Encrypt with AES-GCM
        aesgcm = AESGCM(self.channel_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        
        return await self._send_request("secure.send", {
            "channel_id": f"chan-{self.session_id}",
            "encrypted_data": base64.b64encode(ciphertext[:-16]).decode(),
            "iv": base64.b64encode(nonce).decode(),
            "tag": base64.b64encode(ciphertext[-16:]).decode(),
            "sequence_number": self.msg_counter
        })
    
    async def _send_request(self, action: str, payload: dict):
        """Send request and wait for response"""
        self.msg_counter += 1
        msg_id = f"msg-{self.msg_counter:06d}"
        
        request = {
            "type": "request",
            "id": msg_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload
        }
        
        await self.ws.send(json.dumps(request))
        
        # Wait for matching response
        while True:
            response = await self._recv()
            if response.get("id") == msg_id:
                if response.get("status") == "success":
                    return response
                else:
                    raise RuntimeError(f"Request failed: {response}")
    
    async def _recv(self):
        """Receive and parse message"""
        data = await self.ws.recv()
        return json.loads(data)
    
    def _derive_key(self, secret: bytes, info: bytes) -> bytes:
        """Derive key using HKDF-SHA384"""
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        
        hkdf = HKDF(
            algorithm=hashes.SHA384(),
            length=32,
            salt=None,
            info=info
        )
        return hkdf.derive(secret)
    
    async def close(self):
        """Close connection gracefully"""
        if self.ws:
            await self._send_request("session.terminate", {
                "reason": "client_close"
            })
            await self.ws.close()


# Usage Example
async def main():
    client = PQCFHEWebSocketClient(
        server_url="wss://localhost:8443/ws/v1/secure",
        client_id="demo-client",
        auth_token="your-jwt-token"
    )
    
    try:
        # Connect and establish session
        session = await client.connect()
        print(f"Session: {session['session_id']}")
        
        # Establish PQC secure channel
        channel = await client.establish_pqc_channel("ML-KEM-768")
        print(f"Secure channel: {channel['channel_id']}")
        
        # Setup FHE context
        fhe_setup = await client.setup_fhe()
        context_id = fhe_setup["payload"]["context_id"]
        print(f"FHE context: {context_id}")
        
        # Encrypt sensitive data
        ct1 = await client.fhe_encrypt(context_id, [1.5, 2.0, 3.5, 4.0])
        ct2 = await client.fhe_encrypt(context_id, [0.5, 1.0, 1.5, 2.0])
        
        # Compute on encrypted data
        result = await client.fhe_compute(
            context_id,
            operation="polynomial",
            operands=[ct1["payload"]["ciphertext_id"],
                     ct2["payload"]["ciphertext_id"]],
            parameters={
                "expression": "a + b",
                "variable_mapping": {
                    "a": ct1["payload"]["ciphertext_id"],
                    "b": ct2["payload"]["ciphertext_id"]
                }
            }
        )
        print(f"Computation complete: {result['payload']['result_ciphertext_id']}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Security Considerations

### Transport Security

- **TLS 1.3 Required**: All WebSocket connections must use TLS 1.3
- **Certificate Pinning**: Recommended for production deployments
- **Perfect Forward Secrecy**: Enabled through ephemeral key exchange

### Key Management

- **Session Keys**: Rotated every 1 hour or 1000 messages
- **Key Derivation**: HKDF-SHA384 with unique context strings
- **Key Zeroization**: All keys securely erased on session close

### Message Integrity

- **Sequence Numbers**: Prevents replay attacks
- **Timestamps**: Reject messages older than 5 minutes
- **Authentication Tags**: AEAD ensures integrity

### Rate Limiting

- **Per-Connection**: Limits prevent resource exhaustion
- **Adaptive**: Increases during suspected attacks
- **Graceful Degradation**: Non-blocking when limits approached

## Monitoring

### Health Check Endpoint

```
GET /ws/health
```

```json
{
    "status": "healthy",
    "active_sessions": 1234,
    "active_fhe_contexts": 567,
    "memory_usage_mb": 2048,
    "uptime_seconds": 86400
}
```

### Metrics (Prometheus)

```prometheus
# WebSocket connections
pqc_fhe_ws_connections_total{status="active"} 1234
pqc_fhe_ws_connections_total{status="closed"} 5678

# PQC operations
pqc_fhe_ws_pqc_exchanges_total{algorithm="ML-KEM-768"} 9012
pqc_fhe_ws_pqc_exchange_duration_seconds{quantile="0.99"} 0.025

# FHE operations
pqc_fhe_ws_fhe_operations_total{operation="encrypt"} 45678
pqc_fhe_ws_fhe_operations_total{operation="compute"} 23456
pqc_fhe_ws_fhe_operation_duration_seconds{operation="compute",quantile="0.99"} 0.150
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-15 | Initial release |
| 1.1 | 2025-02-01 | Added hybrid mode support |
| 1.2 | 2025-03-01 | FHE bootstrap operations |
