# Quick Start Guide

Get up and running with PQC-FHE Integration in just a few minutes.

## Prerequisites

- Python 3.9+
- pip installed

## Step 1: Install the Library

```bash
pip install pqc-fhe-integration
```

## Step 2: Post-Quantum Key Exchange

### Generate Keys

```python
from pqc_fhe_integration import PQCKeyManager

# Initialize the key manager
pqc = PQCKeyManager()

# Generate ML-KEM-768 keypair (NIST Level 3 security)
public_key, secret_key = pqc.generate_kem_keypair("ML-KEM-768")
print(f"Public key size: {len(public_key)} bytes")
print(f"Secret key size: {len(secret_key)} bytes")
```

### Key Encapsulation

```python
# Alice generates a keypair
alice_pk, alice_sk = pqc.generate_kem_keypair("ML-KEM-768")

# Bob encapsulates a shared secret using Alice's public key
ciphertext, shared_secret_bob = pqc.encapsulate(alice_pk, "ML-KEM-768")

# Alice decapsulates to get the same shared secret
shared_secret_alice = pqc.decapsulate(ciphertext, alice_sk, "ML-KEM-768")

# Both parties now have the same shared secret
assert shared_secret_alice == shared_secret_bob
print("Key exchange successful!")
```

## Step 3: Digital Signatures

```python
# Generate a signing keypair
sign_pk, sign_sk = pqc.generate_signature_keypair("ML-DSA-65")

# Sign a message
message = b"Hello, quantum-resistant world!"
signature = pqc.sign(message, sign_sk, "ML-DSA-65")
print(f"Signature size: {len(signature)} bytes")

# Verify the signature
is_valid = pqc.verify(message, signature, sign_pk, "ML-DSA-65")
print(f"Signature valid: {is_valid}")
```

## Step 4: Homomorphic Encryption

### Basic Encryption/Decryption

```python
from pqc_fhe_integration import FHEEngine

# Initialize the FHE engine
fhe = FHEEngine()

# Encrypt some values
values = [1.5, 2.5, 3.5, 4.5]
encrypted = fhe.encrypt(values)

# Decrypt
decrypted = fhe.decrypt(encrypted)
print(f"Original: {values}")
print(f"Decrypted: {decrypted}")
```

### Homomorphic Computation

```python
# Encrypt two vectors
a = fhe.encrypt([1.0, 2.0, 3.0])
b = fhe.encrypt([4.0, 5.0, 6.0])

# Addition on encrypted data
sum_ct = fhe.add(a, b)
print(f"Sum: {fhe.decrypt(sum_ct)}")  # [5.0, 7.0, 9.0]

# Multiplication on encrypted data
prod_ct = fhe.multiply(a, b)
print(f"Product: {fhe.decrypt(prod_ct)}")  # [4.0, 10.0, 18.0]

# Scalar operations
scaled = fhe.multiply_scalar(a, 2.0)
print(f"Scaled: {fhe.decrypt(scaled)}")  # [2.0, 4.0, 6.0]
```

## Step 5: Hybrid PQC + FHE Workflow

```python
from pqc_fhe_integration import HybridCryptoManager

# Initialize hybrid manager
hybrid = HybridCryptoManager()

# Complete secure computation workflow
result = hybrid.secure_compute(
    data=[100.0, 200.0, 300.0],
    operation="sum",
    pqc_algorithm="ML-KEM-768",
    sign_algorithm="ML-DSA-65"
)

print(f"Encrypted result: {result['ciphertext_hash']}")
print(f"Signature valid: {result['signature_valid']}")
```

## Step 6: Using the CLI

```bash
# Generate keys
pqc-fhe keygen --algorithm ML-KEM-768 --output ./keys

# Sign a file
pqc-fhe sign --secret-key ./keys/*_secret.json \
  --message document.txt \
  --output signature.bin

# Verify signature
pqc-fhe verify --public-key ./keys/*_public.json \
  --message document.txt \
  --signature signature.bin

# FHE operations
pqc-fhe fhe-encrypt --values 1.0 2.0 3.0 --output ct.json
pqc-fhe fhe-compute --operation add_scalar --input1 ct.json --scalar 10 --output result.json
pqc-fhe fhe-decrypt --input result.json
```

## Step 7: Using the REST API

### Start the Server

```bash
# Using Python
python -m pqc_fhe_integration.api.server

# Using Docker
docker run -p 8000:8000 pqc-fhe/integration:latest
```

### Make API Calls

```bash
# Generate keys
curl -X POST http://localhost:8000/api/v1/pqc/keygen \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "ML-KEM-768"}'

# Encrypt data
curl -X POST http://localhost:8000/api/v1/fhe/encrypt \
  -H "Content-Type: application/json" \
  -d '{"values": [1.0, 2.0, 3.0]}'
```

## Next Steps

- [Configuration Guide](configuration.md) - Advanced configuration options
- [API Reference](../api/overview.md) - Complete API documentation
- [Tutorials](../tutorials/pqc_key_exchange.md) - In-depth tutorials
- [Security Best Practices](../security/best_practices.md) - Production deployment
