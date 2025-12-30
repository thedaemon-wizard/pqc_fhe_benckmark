# REST API Reference

Complete reference for the PQC-FHE REST API endpoints.

## Base URL

```
https://api.example.com/api/v1
```

## Authentication

All endpoints require Bearer token authentication:

```http
Authorization: Bearer <your-api-token>
```

## Response Format

All responses use JSON format:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "INVALID_KEY",
    "message": "The provided key is invalid"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## PQC Key Management

### Generate KEM Keypair

Generate a new key encapsulation mechanism keypair.

```http
POST /pqc/kem/keygen
```

**Request Body:**

```json
{
  "algorithm": "ML-KEM-768"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | `string` | No | Algorithm (default: ML-KEM-768) |

**Supported Algorithms:** `ML-KEM-512`, `ML-KEM-768`, `ML-KEM-1024`

**Response:**

```json
{
  "success": true,
  "data": {
    "public_key": "base64-encoded-public-key",
    "secret_key": "base64-encoded-secret-key",
    "algorithm": "ML-KEM-768",
    "key_id": "kem-abc123"
  }
}
```

**Example:**

```bash
curl -X POST https://api.example.com/api/v1/pqc/kem/keygen \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "ML-KEM-768"}'
```

---

### Encapsulate

Encapsulate a shared secret using a public key.

```http
POST /pqc/kem/encapsulate
```

**Request Body:**

```json
{
  "public_key": "base64-encoded-public-key",
  "algorithm": "ML-KEM-768"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `public_key` | `string` | Yes | Base64-encoded public key |
| `algorithm` | `string` | No | Algorithm (default: ML-KEM-768) |

**Response:**

```json
{
  "success": true,
  "data": {
    "ciphertext": "base64-encoded-ciphertext",
    "shared_secret": "base64-encoded-shared-secret"
  }
}
```

**Example:**

```bash
curl -X POST https://api.example.com/api/v1/pqc/kem/encapsulate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "public_key": "'$PUBLIC_KEY'",
    "algorithm": "ML-KEM-768"
  }'
```

---

### Decapsulate

Decapsulate to recover the shared secret.

```http
POST /pqc/kem/decapsulate
```

**Request Body:**

```json
{
  "ciphertext": "base64-encoded-ciphertext",
  "secret_key": "base64-encoded-secret-key",
  "algorithm": "ML-KEM-768"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ciphertext` | `string` | Yes | Base64-encoded ciphertext |
| `secret_key` | `string` | Yes | Base64-encoded secret key |
| `algorithm` | `string` | No | Algorithm (default: ML-KEM-768) |

**Response:**

```json
{
  "success": true,
  "data": {
    "shared_secret": "base64-encoded-shared-secret"
  }
}
```

---

### Generate Signature Keypair

Generate a new digital signature keypair.

```http
POST /pqc/sig/keygen
```

**Request Body:**

```json
{
  "algorithm": "ML-DSA-65"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | `string` | No | Algorithm (default: ML-DSA-65) |

**Supported Algorithms:** `ML-DSA-44`, `ML-DSA-65`, `ML-DSA-87`

**Response:**

```json
{
  "success": true,
  "data": {
    "public_key": "base64-encoded-public-key",
    "secret_key": "base64-encoded-secret-key",
    "algorithm": "ML-DSA-65",
    "key_id": "sig-xyz789"
  }
}
```

---

### Sign Message

Sign a message using a secret key.

```http
POST /pqc/sig/sign
```

**Request Body:**

```json
{
  "message": "base64-encoded-message",
  "secret_key": "base64-encoded-secret-key",
  "algorithm": "ML-DSA-65"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | `string` | Yes | Base64-encoded message |
| `secret_key` | `string` | Yes | Base64-encoded secret key |
| `algorithm` | `string` | No | Algorithm (default: ML-DSA-65) |

**Response:**

```json
{
  "success": true,
  "data": {
    "signature": "base64-encoded-signature"
  }
}
```

---

### Verify Signature

Verify a signature against a message.

```http
POST /pqc/sig/verify
```

**Request Body:**

```json
{
  "message": "base64-encoded-message",
  "signature": "base64-encoded-signature",
  "public_key": "base64-encoded-public-key",
  "algorithm": "ML-DSA-65"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | `string` | Yes | Base64-encoded message |
| `signature` | `string` | Yes | Base64-encoded signature |
| `public_key` | `string` | Yes | Base64-encoded public key |
| `algorithm` | `string` | No | Algorithm (default: ML-DSA-65) |

**Response:**

```json
{
  "success": true,
  "data": {
    "valid": true
  }
}
```

---

## FHE Operations

### Encrypt Data

Encrypt data using FHE.

```http
POST /fhe/encrypt
```

**Request Body:**

```json
{
  "data": [1.0, 2.0, 3.0, 4.0],
  "context_id": "ctx-123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | `number` or `array` | Yes | Data to encrypt |
| `context_id` | `string` | No | Existing context ID |

**Response:**

```json
{
  "success": true,
  "data": {
    "ciphertext": "base64-encoded-ciphertext",
    "ciphertext_id": "ct-abc123",
    "context_id": "ctx-123",
    "slot_count": 4096,
    "scale": 1099511627776
  }
}
```

**Example:**

```bash
curl -X POST https://api.example.com/api/v1/fhe/encrypt \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": [100.0, 200.0, 300.0]}'
```

---

### Decrypt Data

Decrypt FHE encrypted data.

```http
POST /fhe/decrypt
```

**Request Body:**

```json
{
  "ciphertext": "base64-encoded-ciphertext",
  "length": 4
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ciphertext` | `string` | Yes | Base64-encoded ciphertext |
| `length` | `integer` | No | Number of elements to return |

**Response:**

```json
{
  "success": true,
  "data": {
    "values": [1.0, 2.0, 3.0, 4.0]
  }
}
```

---

### Compute on Encrypted Data

Perform homomorphic computation.

```http
POST /fhe/compute
```

**Request Body:**

```json
{
  "operation": "add",
  "operands": [
    "base64-encoded-ciphertext-a",
    "base64-encoded-ciphertext-b"
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `operation` | `string` | Yes | Operation to perform |
| `operands` | `array` | Yes | Base64-encoded ciphertexts |
| `params` | `object` | No | Operation-specific parameters |

**Supported Operations:**

| Operation | Operands | Params | Description |
|-----------|----------|--------|-------------|
| `add` | 2 | - | Add ciphertexts |
| `subtract` | 2 | - | Subtract ciphertexts |
| `multiply` | 2 | - | Multiply ciphertexts |
| `square` | 1 | - | Square ciphertext |
| `negate` | 1 | - | Negate ciphertext |
| `add_plain` | 1 | `{"plain": 5.0}` | Add plaintext |
| `multiply_plain` | 1 | `{"plain": 2.0}` | Multiply by plaintext |
| `sum` | 1 | `{"length": 100}` | Sum elements |
| `mean` | 1 | `{"length": 100}` | Compute mean |
| `polynomial` | 1 | `{"coeffs": [1,2,3]}` | Evaluate polynomial |

**Response:**

```json
{
  "success": true,
  "data": {
    "result": "base64-encoded-result-ciphertext",
    "operation": "add",
    "noise_budget": 45
  }
}
```

**Example - Compute Mean:**

```bash
curl -X POST https://api.example.com/api/v1/fhe/compute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "mean",
    "operands": ["'$CIPHERTEXT'"],
    "params": {"length": 100}
  }'
```

---

### Get Context Info

Get information about FHE context.

```http
GET /fhe/context/{context_id}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "context_id": "ctx-123",
    "poly_modulus_degree": 8192,
    "coeff_mod_bit_sizes": [60, 40, 40, 60],
    "scale": 1099511627776,
    "max_slots": 4096,
    "max_multiplicative_depth": 3
  }
}
```

---

## Hybrid Operations

### Create Session

Create a new hybrid session.

```http
POST /hybrid/sessions
```

**Request Body:**

```json
{
  "local_kem_public_key": "base64-encoded-key",
  "local_sig_public_key": "base64-encoded-key",
  "peer_kem_public_key": "base64-encoded-key",
  "peer_sig_public_key": "base64-encoded-key",
  "sign_handshake": true
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "session_id": "sess-abc123",
    "ciphertext": "base64-encoded-ciphertext",
    "signature": "base64-encoded-signature",
    "expires_at": "2025-01-15T11:30:00Z"
  }
}
```

---

### Accept Session

Accept an incoming session.

```http
POST /hybrid/sessions/accept
```

**Request Body:**

```json
{
  "local_kem_secret_key": "base64-encoded-key",
  "local_sig_public_key": "base64-encoded-key",
  "peer_sig_public_key": "base64-encoded-key",
  "ciphertext": "base64-encoded-ciphertext",
  "signature": "base64-encoded-signature",
  "verify_signature": true
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "session_id": "sess-abc123",
    "shared_secret": "base64-encoded-shared-secret",
    "established_at": "2025-01-15T10:30:00Z",
    "expires_at": "2025-01-15T11:30:00Z"
  }
}
```

---

### Encrypt Message

Encrypt a message for a session.

```http
POST /hybrid/sessions/{session_id}/encrypt
```

**Request Body:**

```json
{
  "message": "base64-encoded-message",
  "sign": true,
  "signing_key": "base64-encoded-secret-key"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "ciphertext": "base64-encoded-ciphertext",
    "nonce": "base64-encoded-nonce",
    "signature": "base64-encoded-signature"
  }
}
```

---

### Decrypt Message

Decrypt a message from a session.

```http
POST /hybrid/sessions/{session_id}/decrypt
```

**Request Body:**

```json
{
  "ciphertext": "base64-encoded-ciphertext",
  "nonce": "base64-encoded-nonce",
  "signature": "base64-encoded-signature",
  "verify_signature": true,
  "peer_sig_public_key": "base64-encoded-key"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "message": "base64-encoded-message",
    "signature_valid": true
  }
}
```

---

## System Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "components": {
    "pqc": "healthy",
    "fhe": "healthy",
    "database": "healthy"
  }
}
```

---

### Get System Info

```http
GET /info
```

**Response:**

```json
{
  "success": true,
  "data": {
    "version": "1.0.0",
    "pqc_algorithms": {
      "kem": ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"],
      "signature": ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
    },
    "fhe_parameters": {
      "supported_degrees": [4096, 8192, 16384, 32768],
      "default_scale_bits": 40
    }
  }
}
```

---

### Benchmark

Run performance benchmark.

```http
POST /benchmark
```

**Request Body:**

```json
{
  "operations": ["keygen", "encapsulate", "sign"],
  "iterations": 100
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": {
      "kem_keygen": {
        "mean_ms": 0.12,
        "std_ms": 0.02,
        "iterations": 100
      },
      "encapsulate": {
        "mean_ms": 0.08,
        "std_ms": 0.01,
        "iterations": 100
      },
      "sign": {
        "mean_ms": 0.45,
        "std_ms": 0.05,
        "iterations": 100
      }
    }
  }
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request |
| `INVALID_KEY` | 400 | Invalid key format |
| `INVALID_CIPHERTEXT` | 400 | Invalid ciphertext |
| `INVALID_SIGNATURE` | 400 | Invalid signature |
| `UNAUTHORIZED` | 401 | Missing or invalid auth token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `SESSION_EXPIRED` | 410 | Session has expired |
| `COMPUTATION_ERROR` | 422 | FHE computation failed |
| `NOISE_BUDGET_EXHAUSTED` | 422 | FHE noise budget exhausted |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

| Endpoint Category | Limit |
|-------------------|-------|
| Key Generation | 100/minute |
| Encryption/Decryption | 1000/minute |
| FHE Computation | 500/minute |
| Session Management | 200/minute |

---

## SDKs

- Python: `pip install pqc-fhe-sdk`
- JavaScript: `npm install @pqc-fhe/sdk`
- Go: `go get github.com/pqc-fhe/sdk-go`
- Rust: `cargo add pqc-fhe-sdk`

---

## Related Documentation

- [PQC Manager API](pqc_manager.md)
- [FHE Engine API](fhe_engine.md)
- [Hybrid Manager API](hybrid_manager.md)
- [WebSocket API Reference](websocket_api.md)
