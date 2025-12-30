# PQC Key Exchange Tutorial

This tutorial demonstrates post-quantum cryptographic key exchange using ML-KEM (FIPS 203), the NIST-standardized key encapsulation mechanism.

## Overview

ML-KEM (Module-Lattice-Based Key Encapsulation Mechanism) provides quantum-resistant key exchange. This tutorial covers:

1. Understanding ML-KEM security levels
2. Generating keypairs
3. Key encapsulation (sender side)
4. Key decapsulation (receiver side)
5. Using shared secrets for encryption
6. Best practices and security considerations

## Prerequisites

```bash
# Install the library
pip install pqc-fhe-lib

# Verify installation
python -c "from pqc_fhe_integration import PQCKeyManager; print('OK')"
```

## ML-KEM Security Levels

| Algorithm    | Security Level | Public Key Size | Ciphertext Size | Shared Secret |
|-------------|----------------|-----------------|-----------------|---------------|
| ML-KEM-512  | NIST Level 1   | 800 bytes       | 768 bytes       | 32 bytes      |
| ML-KEM-768  | NIST Level 3   | 1,184 bytes     | 1,088 bytes     | 32 bytes      |
| ML-KEM-1024 | NIST Level 5   | 1,568 bytes     | 1,568 bytes     | 32 bytes      |

**Recommendation**: Use ML-KEM-768 for most applications. Use ML-KEM-1024 for high-security requirements.

## Step 1: Basic Key Exchange

### Generate Keypair (Receiver)

```python
from pqc_fhe_integration import PQCKeyManager

# Initialize key manager
key_manager = PQCKeyManager()

# Generate ML-KEM-768 keypair (receiver side)
public_key, secret_key = key_manager.generate_keypair(
    algorithm="ML-KEM-768"
)

print(f"Public key size: {len(public_key)} bytes")
print(f"Secret key size: {len(secret_key)} bytes")

# Share public_key with the sender (can be transmitted openly)
```

### Encapsulate (Sender)

```python
# Sender receives the public key and creates a shared secret
ciphertext, shared_secret_sender = key_manager.encapsulate(public_key)

print(f"Ciphertext size: {len(ciphertext)} bytes")
print(f"Shared secret: {shared_secret_sender.hex()[:32]}...")

# Send ciphertext to the receiver
```

### Decapsulate (Receiver)

```python
# Receiver uses their secret key to derive the same shared secret
shared_secret_receiver = key_manager.decapsulate(ciphertext, secret_key)

# Verify both parties have the same secret
assert shared_secret_sender == shared_secret_receiver
print("Key exchange successful!")
print(f"Shared secret: {shared_secret_receiver.hex()[:32]}...")
```

## Step 2: Complete Key Exchange Protocol

Here's a complete implementation of a secure key exchange protocol:

```python
"""
Complete PQC Key Exchange Protocol
Implements a secure handshake between Alice and Bob
"""

from pqc_fhe_integration import PQCKeyManager
import hashlib
import os

class SecureChannel:
    """Secure communication channel using PQC key exchange."""
    
    def __init__(self, algorithm: str = "ML-KEM-768"):
        self.key_manager = PQCKeyManager()
        self.algorithm = algorithm
        self.shared_secret = None
        self.session_key = None
    
    def generate_keypair(self):
        """Generate keypair for this endpoint."""
        self.public_key, self.secret_key = self.key_manager.generate_keypair(
            algorithm=self.algorithm
        )
        return self.public_key
    
    def initiate_exchange(self, peer_public_key: bytes):
        """
        Initiate key exchange (sender side).
        Returns ciphertext to send to peer.
        """
        ciphertext, self.shared_secret = self.key_manager.encapsulate(
            peer_public_key
        )
        self._derive_session_key()
        return ciphertext
    
    def complete_exchange(self, ciphertext: bytes):
        """
        Complete key exchange (receiver side).
        Derives shared secret from ciphertext.
        """
        self.shared_secret = self.key_manager.decapsulate(
            ciphertext, self.secret_key
        )
        self._derive_session_key()
    
    def _derive_session_key(self):
        """Derive session key from shared secret using HKDF."""
        # Simple KDF using SHA-256 (use HKDF in production)
        self.session_key = hashlib.sha256(
            self.shared_secret + b"session_key_v1"
        ).digest()
    
    def get_session_key(self) -> bytes:
        """Get the derived session key for symmetric encryption."""
        if self.session_key is None:
            raise RuntimeError("Key exchange not completed")
        return self.session_key


# Example usage
def demonstrate_key_exchange():
    """Demonstrate complete key exchange between Alice and Bob."""
    
    # === Setup ===
    alice = SecureChannel(algorithm="ML-KEM-768")
    bob = SecureChannel(algorithm="ML-KEM-768")
    
    print("=== PQC Key Exchange Protocol ===\n")
    
    # === Step 1: Bob generates keypair and shares public key ===
    print("Step 1: Bob generates keypair")
    bob_public_key = bob.generate_keypair()
    print(f"  Bob's public key: {len(bob_public_key)} bytes")
    print(f"  (Public key can be shared openly)\n")
    
    # === Step 2: Alice encapsulates using Bob's public key ===
    print("Step 2: Alice encapsulates shared secret")
    ciphertext = alice.initiate_exchange(bob_public_key)
    print(f"  Ciphertext: {len(ciphertext)} bytes")
    print(f"  (Ciphertext sent to Bob)\n")
    
    # === Step 3: Bob decapsulates to get shared secret ===
    print("Step 3: Bob decapsulates ciphertext")
    bob.complete_exchange(ciphertext)
    print(f"  Bob derived shared secret\n")
    
    # === Step 4: Both parties now have the same session key ===
    print("Step 4: Verify session keys match")
    alice_key = alice.get_session_key()
    bob_key = bob.get_session_key()
    
    assert alice_key == bob_key, "Session keys don't match!"
    print(f"  Alice's session key: {alice_key.hex()[:32]}...")
    print(f"  Bob's session key:   {bob_key.hex()[:32]}...")
    print(f"  Keys match: {alice_key == bob_key}")
    
    return alice_key


if __name__ == "__main__":
    session_key = demonstrate_key_exchange()
```

## Step 3: Hybrid Key Exchange (Classical + PQC)

For defense-in-depth, combine classical and post-quantum algorithms:

```python
"""
Hybrid Key Exchange: X25519 + ML-KEM-768
Provides security even if one algorithm is broken
"""

from pqc_fhe_integration import PQCKeyManager
import hashlib

# Note: This example uses simulated X25519 for illustration
# In production, use cryptography library for X25519

class HybridKeyExchange:
    """
    Hybrid key exchange combining:
    - X25519 (classical ECDH)
    - ML-KEM-768 (post-quantum)
    
    Security: Both algorithms must be broken to compromise the key.
    """
    
    def __init__(self):
        self.pqc_manager = PQCKeyManager()
        
    def generate_keypairs(self):
        """Generate both classical and PQC keypairs."""
        # PQC keypair
        self.pqc_public, self.pqc_secret = self.pqc_manager.generate_keypair(
            algorithm="ML-KEM-768"
        )
        
        # Classical keypair (simulated X25519)
        # In production: use cryptography.hazmat.primitives.asymmetric.x25519
        self.classical_secret = hashlib.sha256(
            bytes([i for i in range(32)])
        ).digest()
        self.classical_public = hashlib.sha256(
            self.classical_secret + b"public"
        ).digest()
        
        return {
            'pqc_public': self.pqc_public,
            'classical_public': self.classical_public
        }
    
    def encapsulate(self, peer_keys: dict) -> tuple:
        """
        Create hybrid ciphertext and shared secret.
        
        Args:
            peer_keys: Dict with 'pqc_public' and 'classical_public'
            
        Returns:
            Tuple of (hybrid_ciphertext, combined_shared_secret)
        """
        # PQC encapsulation
        pqc_ciphertext, pqc_secret = self.pqc_manager.encapsulate(
            peer_keys['pqc_public']
        )
        
        # Classical ECDH (simulated)
        classical_secret = hashlib.sha256(
            self.classical_secret + peer_keys['classical_public']
        ).digest()
        
        # Combine secrets using KDF
        combined_secret = self._combine_secrets(pqc_secret, classical_secret)
        
        hybrid_ciphertext = {
            'pqc': pqc_ciphertext,
            'classical': self.classical_public
        }
        
        return hybrid_ciphertext, combined_secret
    
    def decapsulate(self, hybrid_ciphertext: dict) -> bytes:
        """
        Decapsulate hybrid ciphertext.
        
        Args:
            hybrid_ciphertext: Dict with 'pqc' and 'classical' components
            
        Returns:
            Combined shared secret
        """
        # PQC decapsulation
        pqc_secret = self.pqc_manager.decapsulate(
            hybrid_ciphertext['pqc'],
            self.pqc_secret
        )
        
        # Classical ECDH (simulated)
        classical_secret = hashlib.sha256(
            self.classical_secret + hybrid_ciphertext['classical']
        ).digest()
        
        # Combine secrets
        return self._combine_secrets(pqc_secret, classical_secret)
    
    def _combine_secrets(self, pqc_secret: bytes, classical_secret: bytes) -> bytes:
        """
        Combine PQC and classical secrets using HKDF-like construction.
        
        Security: The combined secret is secure if EITHER input is secure.
        """
        # Concatenate and hash (use proper HKDF in production)
        combined = hashlib.sha256(
            pqc_secret + classical_secret + b"hybrid_v1"
        ).digest()
        return combined


def demonstrate_hybrid_exchange():
    """Demonstrate hybrid key exchange."""
    
    print("=== Hybrid Key Exchange (X25519 + ML-KEM-768) ===\n")
    
    alice = HybridKeyExchange()
    bob = HybridKeyExchange()
    
    # Step 1: Both parties generate keypairs
    alice_keys = alice.generate_keypairs()
    bob_keys = bob.generate_keypairs()
    
    print("Step 1: Keypairs generated")
    print(f"  Alice PQC public key: {len(alice_keys['pqc_public'])} bytes")
    print(f"  Bob PQC public key: {len(bob_keys['pqc_public'])} bytes\n")
    
    # Step 2: Alice encapsulates to Bob
    ciphertext, alice_secret = alice.encapsulate(bob_keys)
    print("Step 2: Alice creates hybrid ciphertext")
    print(f"  PQC ciphertext: {len(ciphertext['pqc'])} bytes")
    print(f"  Classical component: {len(ciphertext['classical'])} bytes\n")
    
    # Step 3: Bob decapsulates
    bob_secret = bob.decapsulate(ciphertext)
    print("Step 3: Bob decapsulates")
    
    # Verify
    print("\nVerification:")
    print(f"  Alice secret: {alice_secret.hex()[:32]}...")
    print(f"  Bob secret:   {bob_secret.hex()[:32]}...")
    print(f"  Secrets match: {alice_secret == bob_secret}")


if __name__ == "__main__":
    demonstrate_hybrid_exchange()
```

## Step 4: Key Exchange with Authentication

Add digital signatures to prevent man-in-the-middle attacks:

```python
"""
Authenticated Key Exchange using ML-KEM + ML-DSA
Prevents man-in-the-middle attacks
"""

from pqc_fhe_integration import PQCKeyManager
import hashlib
import time

class AuthenticatedKeyExchange:
    """
    Authenticated key exchange with identity verification.
    
    Protocol:
    1. Both parties have long-term signing keys
    2. Ephemeral KEM keys are generated per session
    3. KEM public keys are signed for authentication
    4. Key exchange proceeds with verified keys
    """
    
    def __init__(self, identity: str):
        self.identity = identity
        self.key_manager = PQCKeyManager()
        
        # Generate long-term signing keys
        self.signing_public, self.signing_secret = self.key_manager.generate_keypair(
            algorithm="ML-DSA-65"
        )
        
    def get_identity_key(self) -> dict:
        """Get identity (signing) public key for peer verification."""
        return {
            'identity': self.identity,
            'signing_key': self.signing_public
        }
    
    def create_signed_kem_key(self) -> dict:
        """
        Create ephemeral KEM keypair and sign the public key.
        
        Returns signed package for key exchange initiation.
        """
        # Generate ephemeral KEM keypair
        self.kem_public, self.kem_secret = self.key_manager.generate_keypair(
            algorithm="ML-KEM-768"
        )
        
        # Create message to sign
        timestamp = int(time.time()).to_bytes(8, 'big')
        message = self.identity.encode() + timestamp + self.kem_public
        
        # Sign with long-term key
        signature = self.key_manager.sign(message, self.signing_secret)
        
        return {
            'identity': self.identity,
            'kem_public': self.kem_public,
            'timestamp': timestamp,
            'signature': signature
        }
    
    def verify_and_encapsulate(
        self, 
        peer_package: dict,
        peer_signing_key: bytes
    ) -> tuple:
        """
        Verify peer's signed KEM key and encapsulate.
        
        Args:
            peer_package: Signed KEM key package from peer
            peer_signing_key: Peer's trusted signing public key
            
        Returns:
            Tuple of (response_package, shared_secret)
        """
        # Reconstruct signed message
        message = (
            peer_package['identity'].encode() +
            peer_package['timestamp'] +
            peer_package['kem_public']
        )
        
        # Verify signature
        is_valid = self.key_manager.verify(
            message,
            peer_package['signature'],
            peer_signing_key
        )
        
        if not is_valid:
            raise SecurityError("Invalid signature - possible MITM attack!")
        
        # Check timestamp freshness (within 5 minutes)
        timestamp = int.from_bytes(peer_package['timestamp'], 'big')
        if abs(time.time() - timestamp) > 300:
            raise SecurityError("Timestamp too old - possible replay attack!")
        
        print(f"  Verified signature from {peer_package['identity']}")
        
        # Encapsulate using verified KEM key
        ciphertext, shared_secret = self.key_manager.encapsulate(
            peer_package['kem_public']
        )
        
        # Sign our response
        response_message = self.identity.encode() + ciphertext
        response_signature = self.key_manager.sign(
            response_message, 
            self.signing_secret
        )
        
        response = {
            'identity': self.identity,
            'ciphertext': ciphertext,
            'signature': response_signature
        }
        
        return response, shared_secret
    
    def verify_and_decapsulate(
        self,
        response: dict,
        peer_signing_key: bytes
    ) -> bytes:
        """
        Verify response signature and decapsulate.
        
        Args:
            response: Signed ciphertext from peer
            peer_signing_key: Peer's trusted signing public key
            
        Returns:
            Shared secret
        """
        # Verify signature
        message = response['identity'].encode() + response['ciphertext']
        is_valid = self.key_manager.verify(
            message,
            response['signature'],
            peer_signing_key
        )
        
        if not is_valid:
            raise SecurityError("Invalid response signature!")
        
        print(f"  Verified response from {response['identity']}")
        
        # Decapsulate
        shared_secret = self.key_manager.decapsulate(
            response['ciphertext'],
            self.kem_secret
        )
        
        return shared_secret


class SecurityError(Exception):
    """Security-related error."""
    pass


def demonstrate_authenticated_exchange():
    """Demonstrate authenticated key exchange."""
    
    print("=== Authenticated Key Exchange (ML-KEM + ML-DSA) ===\n")
    
    # Setup: Both parties create identities
    alice = AuthenticatedKeyExchange("alice@example.com")
    bob = AuthenticatedKeyExchange("bob@example.com")
    
    print("Step 1: Exchange identity keys (out-of-band)")
    alice_identity = alice.get_identity_key()
    bob_identity = bob.get_identity_key()
    print(f"  Alice identity: {alice_identity['identity']}")
    print(f"  Bob identity: {bob_identity['identity']}\n")
    
    # Step 2: Alice initiates with signed KEM key
    print("Step 2: Alice creates signed KEM key")
    alice_kem_package = alice.create_signed_kem_key()
    print(f"  KEM public key: {len(alice_kem_package['kem_public'])} bytes")
    print(f"  Signature: {len(alice_kem_package['signature'])} bytes\n")
    
    # Step 3: Bob verifies and encapsulates
    print("Step 3: Bob verifies Alice and encapsulates")
    bob_response, bob_secret = bob.verify_and_encapsulate(
        alice_kem_package,
        alice_identity['signing_key']
    )
    print(f"  Ciphertext: {len(bob_response['ciphertext'])} bytes\n")
    
    # Step 4: Alice verifies and decapsulates
    print("Step 4: Alice verifies Bob and decapsulates")
    alice_secret = alice.verify_and_decapsulate(
        bob_response,
        bob_identity['signing_key']
    )
    
    # Verify
    print("\nVerification:")
    print(f"  Shared secrets match: {alice_secret == bob_secret}")
    print(f"  Session key: {alice_secret.hex()[:32]}...")


if __name__ == "__main__":
    demonstrate_authenticated_exchange()
```

## CLI Examples

### Generate Keys

```bash
# Generate ML-KEM-768 keypair
pqc-fhe keygen --algorithm ML-KEM-768 \
    --output-public bob_public.pem \
    --output-secret bob_secret.pem

# Generate with JSON output
pqc-fhe keygen --algorithm ML-KEM-768 --json
```

### Perform Encapsulation

```bash
# Encapsulate using public key
pqc-fhe encapsulate --public-key bob_public.pem \
    --output-ciphertext ciphertext.bin \
    --output-secret shared_secret.bin
```

### Perform Decapsulation

```bash
# Decapsulate using secret key
pqc-fhe decapsulate --ciphertext ciphertext.bin \
    --secret-key bob_secret.pem \
    --output shared_secret_bob.bin

# Verify secrets match
diff shared_secret.bin shared_secret_bob.bin && echo "Keys match!"
```

## Security Considerations

### Key Storage

```python
# Bad: Storing keys in plaintext
with open('secret_key.bin', 'wb') as f:
    f.write(secret_key)  # INSECURE!

# Good: Use encrypted storage
from cryptography.fernet import Fernet

def store_key_encrypted(secret_key: bytes, password: bytes):
    """Store secret key with password encryption."""
    import hashlib
    import base64
    
    # Derive encryption key from password
    key = base64.urlsafe_b64encode(
        hashlib.scrypt(password, salt=b'pqc_salt', n=2**14, r=8, p=1, dklen=32)
    )
    
    fernet = Fernet(key)
    encrypted = fernet.encrypt(secret_key)
    
    with open('secret_key.enc', 'wb') as f:
        f.write(encrypted)
```

### Key Rotation

```python
def rotate_keys(old_channel: SecureChannel) -> SecureChannel:
    """Rotate to new keypair while maintaining session."""
    new_channel = SecureChannel()
    new_public = new_channel.generate_keypair()
    
    # Sign new key with old identity (if using signatures)
    # Transfer trust to new key
    # Securely delete old keys
    
    return new_channel
```

### Recommendations

1. **Key Size**: Use ML-KEM-768 minimum (NIST Level 3)
2. **Key Storage**: Never store secret keys in plaintext
3. **Key Rotation**: Rotate ephemeral keys per session
4. **Hybrid Mode**: Consider X25519 + ML-KEM for transition period
5. **Authentication**: Always use signatures to prevent MITM
6. **Timestamps**: Include timestamps to prevent replay attacks
7. **Secure Deletion**: Overwrite secret keys before deletion

## Next Steps

- [FHE Computation Tutorial](fhe_computation.md) - Use shared secrets with FHE
- [Hybrid Workflow Tutorial](hybrid_workflow.md) - Combine PQC key exchange with FHE
- [Security Best Practices](../security/best_practices.md) - Production deployment guidelines
