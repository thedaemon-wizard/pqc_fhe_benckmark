# PQC-FHE REST API Documentation

## Overview

RESTful API for Post-Quantum Cryptography and Fully Homomorphic Encryption operations.

**Base URL**: `http://localhost:8000`

**Authentication**: API Key via `X-API-Key` header (disabled in DEBUG mode)

**Content-Type**: `application/json`

---

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | System information |
| `/pqc/keygen` | POST | Generate PQC key pair |
| `/pqc/encapsulate` | POST | ML-KEM encapsulation |
| `/pqc/sign` | POST | ML-DSA signature |
| `/pqc/verify` | POST | Verify signature |
| `/fhe/encrypt` | POST | FHE encryption |
| `/fhe/compute` | POST | Homomorphic computation |
| `/fhe/decrypt` | POST | FHE decryption |
| `/hybrid/secure-channel` | POST | Establish PQC channel |

---

## System Endpoints

### GET /health

Health check endpoint.

**Response**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00.000000",
  "uptime_seconds": 3600.5
}
```

### GET /info

Get system capabilities and configuration.

**Response**

```json
{
  "api_version": "1.0.0",
  "pqc_available": true,
  "fhe_available": true,
  "supported_kem_algorithms": ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"],
  "supported_sig_algorithms": ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"],
  "fhe_slot_count": 8192
}
```

---

## PQC Endpoints

### POST /pqc/keygen

Generate a Post-Quantum Cryptography key pair.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | string | No | Algorithm name (default: "ML-KEM-768") |
| `key_id` | string | No | Optional key identifier |

**Supported Algorithms**

| Algorithm | Type | Security Level | Public Key Size | Secret Key Size |
|-----------|------|----------------|-----------------|-----------------|
| ML-KEM-512 | KEM | Level 1 (128-bit) | 800 bytes | 1,632 bytes |
| ML-KEM-768 | KEM | Level 3 (192-bit) | 1,184 bytes | 2,400 bytes |
| ML-KEM-1024 | KEM | Level 5 (256-bit) | 1,568 bytes | 3,168 bytes |
| ML-DSA-44 | Signature | Level 1 (128-bit) | 1,312 bytes | 2,560 bytes |
| ML-DSA-65 | Signature | Level 3 (192-bit) | 1,952 bytes | 4,032 bytes |
| ML-DSA-87 | Signature | Level 5 (256-bit) | 2,592 bytes | 4,896 bytes |

**Example Request**

```bash
curl -X POST http://localhost:8000/pqc/keygen \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "algorithm": "ML-KEM-768",
    "key_id": "user-key-001"
  }'
```

**Response**

```json
{
  "key_id": "user-key-001",
  "algorithm": "ML-KEM-768",
  "public_key": "base64-encoded-public-key...",
  "secret_key": "base64-encoded-secret-key...",
  "public_key_size": 1184,
  "secret_key_size": 2400,
  "timestamp": "2025-01-15T10:30:00.000000"
}
```

---

### POST /pqc/encapsulate

Perform ML-KEM key encapsulation to generate a shared secret.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `public_key` | string | Yes | Base64-encoded public key |
| `algorithm` | string | No | ML-KEM variant (default: "ML-KEM-768") |

**Example Request**

```bash
curl -X POST http://localhost:8000/pqc/encapsulate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "public_key": "base64-encoded-public-key...",
    "algorithm": "ML-KEM-768"
  }'
```

**Response**

```json
{
  "ciphertext": "base64-encoded-ciphertext...",
  "shared_secret": "base64-encoded-32-byte-secret...",
  "ciphertext_size": 1088
}
```

**Notes**

- The `ciphertext` should be sent to the key owner
- The `shared_secret` is identical on both sides after decapsulation
- Use the shared secret as a symmetric encryption key (AES-256)

---

### POST /pqc/sign

Create a digital signature using ML-DSA.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Base64-encoded message |
| `secret_key` | string | Yes | Base64-encoded secret key |
| `algorithm` | string | No | ML-DSA variant (default: "ML-DSA-65") |

**Example Request**

```bash
curl -X POST http://localhost:8000/pqc/sign \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "'$(echo -n "Hello World" | base64)'",
    "secret_key": "base64-encoded-secret-key...",
    "algorithm": "ML-DSA-65"
  }'
```

**Response**

```json
{
  "signature": "base64-encoded-signature...",
  "signature_size": 3309,
  "algorithm": "ML-DSA-65"
}
```

---

### POST /pqc/verify

Verify a digital signature.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Base64-encoded message |
| `signature` | string | Yes | Base64-encoded signature |
| `public_key` | string | Yes | Base64-encoded public key |
| `algorithm` | string | No | ML-DSA variant (default: "ML-DSA-65") |

**Example Request**

```bash
curl -X POST http://localhost:8000/pqc/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "'$(echo -n "Hello World" | base64)'",
    "signature": "base64-encoded-signature...",
    "public_key": "base64-encoded-public-key...",
    "algorithm": "ML-DSA-65"
  }'
```

**Response**

```json
{
  "valid": true,
  "algorithm": "ML-DSA-65"
}
```

---

## FHE Endpoints

### POST /fhe/encrypt

Encrypt an array of floating-point values using CKKS FHE.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | array[float] | Yes | Array of floating-point values |
| `precision` | integer | No | Scale bits for precision (default: 40) |

**Example Request**

```bash
curl -X POST http://localhost:8000/fhe/encrypt \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "data": [1.5, 2.7, 3.14, 4.0, 5.5],
    "precision": 40
  }'
```

**Response**

```json
{
  "ciphertext_id": "a1b2c3d4e5f6g7h8",
  "ciphertext": "base64-encoded-ciphertext-reference...",
  "slot_count": 8192,
  "level": 50,
  "data_length": 5
}
```

**Notes**

- Maximum data length is `slot_count` (typically 8192)
- Higher precision requires more levels but provides better accuracy
- Store the `ciphertext` value for subsequent operations

---

### POST /fhe/compute

Perform homomorphic computation on encrypted data.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `operation` | string | Yes | Operation type |
| `ciphertext1` | string | Yes | First ciphertext (Base64) |
| `ciphertext2` | string | Conditional | Second ciphertext for binary operations |
| `scalar` | float | Conditional | Scalar value for scalar operations |

**Supported Operations**

| Operation | Formula | Level Cost | Requirements |
|-----------|---------|------------|--------------|
| `add` | ct1 + ct2 | 0 | ciphertext2 |
| `multiply` | ct1 * ct2 | 1 | ciphertext2 |
| `square` | ct1^2 | 1 | - |
| `add_scalar` | ct1 + scalar | 0 | scalar |
| `multiply_scalar` | ct1 * scalar | 0 | scalar |
| `negate` | -ct1 | 0 | - |

**Example Request - Addition**

```bash
curl -X POST http://localhost:8000/fhe/compute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "operation": "add",
    "ciphertext1": "base64-ciphertext-1...",
    "ciphertext2": "base64-ciphertext-2..."
  }'
```

**Example Request - Scalar Multiplication**

```bash
curl -X POST http://localhost:8000/fhe/compute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "operation": "multiply_scalar",
    "ciphertext1": "base64-ciphertext-1...",
    "scalar": 2.5
  }'
```

**Response**

```json
{
  "result_ciphertext": "base64-encoded-result-ciphertext...",
  "operation": "add",
  "result_level": 49
}
```

**Notes**

- Monitor `result_level` - when it approaches 0, bootstrap is needed
- Multiplication consumes 1 level; addition is free
- Chain operations carefully to minimize level consumption

---

### POST /fhe/decrypt

Decrypt a ciphertext to recover plaintext values.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ciphertext` | string | Yes | Base64-encoded ciphertext |
| `length` | integer | No | Expected data length |

**Example Request**

```bash
curl -X POST http://localhost:8000/fhe/decrypt \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "ciphertext": "base64-encoded-ciphertext...",
    "length": 5
  }'
```

**Response**

```json
{
  "data": [1.4999999, 2.7000001, 3.1399999, 4.0000001, 5.4999998],
  "length": 5
}
```

**Notes**

- CKKS is an approximate scheme - expect small rounding errors
- Specify `length` to truncate padding values

---

## Hybrid Endpoint

### POST /hybrid/secure-channel

Establish a quantum-resistant secure communication channel.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `client_kem_public_key` | string | Yes | Client's ML-KEM public key |
| `client_sig_public_key` | string | Yes | Client's ML-DSA public key |
| `security_level` | string | No | "level1", "level3" (default), or "level5" |

**Security Levels**

| Level | KEM Algorithm | Signature Algorithm | Security |
|-------|---------------|---------------------|----------|
| level1 | ML-KEM-512 | ML-DSA-44 | 128-bit |
| level3 | ML-KEM-768 | ML-DSA-65 | 192-bit |
| level5 | ML-KEM-1024 | ML-DSA-87 | 256-bit |

**Example Request**

```bash
curl -X POST http://localhost:8000/hybrid/secure-channel \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "client_kem_public_key": "base64-encoded-client-kem-pk...",
    "client_sig_public_key": "base64-encoded-client-sig-pk...",
    "security_level": "level3"
  }'
```

**Response**

```json
{
  "channel_id": "abc123def456ghi789jkl012mno345pq",
  "server_kem_public_key": "base64-encoded-server-kem-pk...",
  "server_sig_public_key": "base64-encoded-server-sig-pk...",
  "kem_ciphertext": "base64-encoded-encapsulated-secret...",
  "signature": "base64-encoded-server-signature...",
  "established_at": "2025-01-15T10:30:00.000000"
}
```

**Channel Establishment Flow**

1. Client generates ML-KEM and ML-DSA key pairs
2. Client sends public keys to server
3. Server generates its own key pairs
4. Server encapsulates shared secret to client's KEM public key
5. Server signs channel data with its signature key
6. Client verifies signature and decapsulates shared secret
7. Both parties now share a quantum-resistant session key

---

## Error Handling

All endpoints return standard HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid API key |
| 500 | Internal Server Error |

**Error Response Format**

```json
{
  "detail": "Error message description"
}
```

---

## Rate Limiting

- Default: 60 requests per minute
- Configure via `MAX_REQUESTS_PER_MINUTE` environment variable

---

## Example Workflows

### Workflow 1: Secure Key Exchange

```python
import requests
import base64

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "your-api-key"}

# 1. Generate client keys
client_kem = requests.post(f"{BASE_URL}/pqc/keygen", 
    json={"algorithm": "ML-KEM-768"}, headers=HEADERS).json()

client_sig = requests.post(f"{BASE_URL}/pqc/keygen",
    json={"algorithm": "ML-DSA-65"}, headers=HEADERS).json()

# 2. Establish secure channel
channel = requests.post(f"{BASE_URL}/hybrid/secure-channel",
    json={
        "client_kem_public_key": client_kem["public_key"],
        "client_sig_public_key": client_sig["public_key"],
        "security_level": "level3"
    }, headers=HEADERS).json()

print(f"Channel ID: {channel['channel_id']}")
```

### Workflow 2: Privacy-Preserving Computation

```python
# 1. Encrypt sensitive data
data = [100.0, 200.0, 300.0, 400.0]
enc_resp = requests.post(f"{BASE_URL}/fhe/encrypt",
    json={"data": data}, headers=HEADERS).json()

ct1 = enc_resp["ciphertext"]

# 2. Perform computation (e.g., scale by 1.1)
compute_resp = requests.post(f"{BASE_URL}/fhe/compute",
    json={
        "operation": "multiply_scalar",
        "ciphertext1": ct1,
        "scalar": 1.1
    }, headers=HEADERS).json()

ct_result = compute_resp["result_ciphertext"]

# 3. Decrypt result
decrypt_resp = requests.post(f"{BASE_URL}/fhe/decrypt",
    json={"ciphertext": ct_result, "length": 4}, headers=HEADERS).json()

print(f"Result: {decrypt_resp['data']}")
# [110.0, 220.0, 330.0, 440.0]
```

---

## References

- [NIST FIPS 203](https://csrc.nist.gov/pubs/fips/203/final) - ML-KEM Standard
- [NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final) - ML-DSA Standard
- [DESILO FHE Documentation](https://fhe.desilo.dev/)
- [Open Quantum Safe](https://openquantumsafe.org/)
