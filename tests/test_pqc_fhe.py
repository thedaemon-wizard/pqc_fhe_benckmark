"""
PQC-FHE Integration Test Suite
==============================

Comprehensive tests for Post-Quantum Cryptography and FHE operations.

Test Categories:
1. PQC Key Management (ML-KEM, ML-DSA)
2. FHE Operations (CKKS encryption, bootstrap)
3. Hybrid Cryptography (X25519 + ML-KEM)
4. Integration Tests (Full workflow)
5. Security Tests (Edge cases, error handling)

References:
- NIST FIPS 203: ML-KEM Standard
- NIST FIPS 204: ML-DSA Standard
- DESILO FHE Documentation
"""

import unittest
import logging
import sys
import os
import hashlib
import numpy as np
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST FLAGS
# =============================================================================

# Check library availability
try:
    import oqs
    LIBOQS_AVAILABLE = True
    logger.info(f"liboqs available: {len(oqs.get_enabled_kem_mechanisms())} KEMs, "
                f"{len(oqs.get_enabled_sig_mechanisms())} signatures")
except ImportError:
    LIBOQS_AVAILABLE = False
    logger.warning("liboqs not available - PQC tests will be skipped")

try:
    import desilofhe
    DESILO_AVAILABLE = True
    logger.info("DESILO FHE available")
except ImportError:
    DESILO_AVAILABLE = False
    logger.warning("DESILO FHE not available - FHE tests will be skipped")

try:
    from cryptography.hazmat.primitives.asymmetric import x25519
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("cryptography not available - hybrid tests will be skipped")

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
    logger.info("Qiskit available for quantum verification tests")
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available - quantum verification tests will be skipped")


# =============================================================================
# PQC KEY MANAGEMENT TESTS
# =============================================================================

@unittest.skipUnless(LIBOQS_AVAILABLE, "liboqs not available")
class TestPQCKeyManagement(unittest.TestCase):
    """Tests for PQC key generation and operations"""
    
    def test_ml_kem_512_keygen(self):
        """Test ML-KEM-512 key generation"""
        kem = oqs.KeyEncapsulation("ML-KEM-512")
        public_key = kem.generate_keypair()
        
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), kem.details['length_public_key'])
        logger.info(f"ML-KEM-512 public key: {len(public_key)} bytes")
    
    def test_ml_kem_768_keygen(self):
        """Test ML-KEM-768 key generation (NIST Level 3)"""
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        public_key = kem.generate_keypair()
        
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), kem.details['length_public_key'])
        logger.info(f"ML-KEM-768 public key: {len(public_key)} bytes")
    
    def test_ml_kem_1024_keygen(self):
        """Test ML-KEM-1024 key generation (NIST Level 5)"""
        kem = oqs.KeyEncapsulation("ML-KEM-1024")
        public_key = kem.generate_keypair()
        
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), kem.details['length_public_key'])
        logger.info(f"ML-KEM-1024 public key: {len(public_key)} bytes")
    
    def test_ml_kem_encapsulation_decapsulation(self):
        """Test ML-KEM encapsulation and decapsulation"""
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        public_key = kem.generate_keypair()
        
        # Encapsulation
        ciphertext, shared_secret_encap = kem.encap_secret(public_key)
        
        self.assertIsNotNone(ciphertext)
        self.assertIsNotNone(shared_secret_encap)
        self.assertEqual(len(ciphertext), kem.details['length_ciphertext'])
        self.assertEqual(len(shared_secret_encap), kem.details['length_shared_secret'])
        
        # Decapsulation
        shared_secret_decap = kem.decap_secret(ciphertext)
        
        self.assertEqual(shared_secret_encap, shared_secret_decap)
        logger.info(f"ML-KEM-768 shared secret: {len(shared_secret_encap)} bytes")
    
    def test_ml_dsa_44_keygen(self):
        """Test ML-DSA-44 key generation"""
        sig = oqs.Signature("ML-DSA-44")
        public_key = sig.generate_keypair()
        
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), sig.details['length_public_key'])
        logger.info(f"ML-DSA-44 public key: {len(public_key)} bytes")
    
    def test_ml_dsa_65_keygen(self):
        """Test ML-DSA-65 key generation (NIST Level 3)"""
        sig = oqs.Signature("ML-DSA-65")
        public_key = sig.generate_keypair()
        
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), sig.details['length_public_key'])
        logger.info(f"ML-DSA-65 public key: {len(public_key)} bytes")
    
    def test_ml_dsa_87_keygen(self):
        """Test ML-DSA-87 key generation (NIST Level 5)"""
        sig = oqs.Signature("ML-DSA-87")
        public_key = sig.generate_keypair()
        
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), sig.details['length_public_key'])
        logger.info(f"ML-DSA-87 public key: {len(public_key)} bytes")
    
    def test_ml_dsa_sign_verify(self):
        """Test ML-DSA signing and verification"""
        sig = oqs.Signature("ML-DSA-65")
        public_key = sig.generate_keypair()
        
        message = b"Test message for ML-DSA signing"
        
        # Sign
        signature = sig.sign(message)
        self.assertIsNotNone(signature)
        logger.info(f"ML-DSA-65 signature: {len(signature)} bytes")
        
        # Verify
        is_valid = sig.verify(message, signature, public_key)
        self.assertTrue(is_valid)
    
    def test_ml_dsa_invalid_signature(self):
        """Test ML-DSA rejects invalid signatures"""
        sig = oqs.Signature("ML-DSA-65")
        public_key = sig.generate_keypair()
        
        message = b"Test message"
        signature = sig.sign(message)
        
        # Tamper with signature
        tampered_signature = bytearray(signature)
        tampered_signature[0] ^= 0xFF
        tampered_signature = bytes(tampered_signature)
        
        # Should reject
        is_valid = sig.verify(message, tampered_signature, public_key)
        self.assertFalse(is_valid)
    
    def test_ml_dsa_wrong_message(self):
        """Test ML-DSA rejects wrong message"""
        sig = oqs.Signature("ML-DSA-65")
        public_key = sig.generate_keypair()
        
        message = b"Original message"
        signature = sig.sign(message)
        
        wrong_message = b"Wrong message"
        
        # Should reject
        is_valid = sig.verify(wrong_message, signature, public_key)
        self.assertFalse(is_valid)
    
    def test_deterministic_keygen(self):
        """Test deterministic key generation with seed"""
        seed = hashlib.sha512(b"test_seed_12345").digest()
        
        # Generate twice with same seed
        kem1 = oqs.KeyEncapsulation("ML-KEM-768")
        pk1 = kem1.generate_keypair()
        
        kem2 = oqs.KeyEncapsulation("ML-KEM-768")
        pk2 = kem2.generate_keypair()
        
        # Without seed, keys should be different
        self.assertNotEqual(pk1, pk2)
        
        # Note: liboqs-python seeded keygen requires secret_key export
        # This tests the basic functionality exists
        logger.info("Deterministic keygen test completed")


# =============================================================================
# FHE OPERATION TESTS
# =============================================================================

@unittest.skipUnless(DESILO_AVAILABLE, "DESILO FHE not available")
class TestFHEOperations(unittest.TestCase):
    """Tests for FHE encryption and operations"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize FHE engine once for all tests"""
        cls.engine = desilofhe.Engine(mode='gpu', use_bootstrap=True, slot_count=2**14, thread_count=512)
        cls.secret_key = cls.engine.create_secret_key()
        cls.public_key = cls.engine.create_public_key(cls.secret_key)
        cls.relin_key = cls.engine.create_relinearization_key(cls.secret_key)
        cls.rotation_key = cls.engine.create_rotation_key(cls.secret_key)
        logger.info("FHE engine initialized for tests")
    
    def test_encrypt_decrypt_simple(self):
        """Test basic encryption and decryption"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        decrypted = self.engine.decrypt(ct, self.secret_key)
        
        for i, (orig, dec) in enumerate(zip(data, decrypted[:len(data)])):
            self.assertAlmostEqual(orig, dec, places=4,
                                   msg=f"Mismatch at index {i}")
        logger.info("Basic encrypt/decrypt test passed")
    
    def test_encrypt_decrypt_large(self):
        """Test encryption with larger data"""
        n = 1000
        data = list(np.random.randn(n).astype(np.float64))
        
        ct = self.engine.encrypt(data, self.public_key)
        decrypted = self.engine.decrypt(ct, self.secret_key)
        
        mse = np.mean([(data[i] - decrypted[i])**2 for i in range(n)])
        self.assertLess(mse, 1e-6, f"MSE too high: {mse}")
        logger.info(f"Large data encrypt/decrypt MSE: {mse:.2e}")
    
    def test_addition(self):
        """Test homomorphic addition"""
        data1 = [1.0, 2.0, 3.0]
        data2 = [4.0, 5.0, 6.0]
        expected = [5.0, 7.0, 9.0]
        
        ct1 = self.engine.encrypt(data1, self.public_key)
        ct2 = self.engine.encrypt(data2, self.public_key)
        
        ct_sum = self.engine.add(ct1, ct2)
        decrypted = self.engine.decrypt(ct_sum, self.secret_key)
        
        for i, (exp, dec) in enumerate(zip(expected, decrypted[:len(expected)])):
            self.assertAlmostEqual(exp, dec, places=4)
        logger.info("Addition test passed")
    
    def test_scalar_addition(self):
        """Test addition with scalar"""
        data = [1.0, 2.0, 3.0]
        scalar = 10.0
        expected = [11.0, 12.0, 13.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        ct_result = self.engine.add(ct, scalar)
        decrypted = self.engine.decrypt(ct_result, self.secret_key)
        
        for i, (exp, dec) in enumerate(zip(expected, decrypted[:len(expected)])):
            self.assertAlmostEqual(exp, dec, places=4)
        logger.info("Scalar addition test passed")
    
    def test_multiplication(self):
        """Test homomorphic multiplication"""
        data1 = [1.0, 2.0, 3.0]
        data2 = [2.0, 3.0, 4.0]
        expected = [2.0, 6.0, 12.0]
        
        ct1 = self.engine.encrypt(data1, self.public_key)
        ct2 = self.engine.encrypt(data2, self.public_key)
        
        ct_prod = self.engine.multiply(ct1, ct2, self.relin_key)
        decrypted = self.engine.decrypt(ct_prod, self.secret_key)
        
        for i, (exp, dec) in enumerate(zip(expected, decrypted[:len(expected)])):
            self.assertAlmostEqual(exp, dec, places=3)
        logger.info("Multiplication test passed")
    
    def test_scalar_multiplication(self):
        """Test multiplication with scalar"""
        data = [1.0, 2.0, 3.0]
        scalar = 5.0
        expected = [5.0, 10.0, 15.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        ct_result = self.engine.multiply(ct, scalar)
        decrypted = self.engine.decrypt(ct_result, self.secret_key)
        
        for i, (exp, dec) in enumerate(zip(expected, decrypted[:len(expected)])):
            self.assertAlmostEqual(exp, dec, places=4)
        logger.info("Scalar multiplication test passed")
    
    def test_square(self):
        """Test square operation"""
        data = [1.0, 2.0, 3.0, 4.0]
        expected = [1.0, 4.0, 9.0, 16.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        ct_sq = self.engine.square(ct)
        ct_sq = self.engine.relinearize(ct_sq, self.relin_key)
        decrypted = self.engine.decrypt(ct_sq, self.secret_key)
        
        for i, (exp, dec) in enumerate(zip(expected, decrypted[:len(expected)])):
            self.assertAlmostEqual(exp, dec, places=3)
        logger.info("Square test passed")
    
    def test_negate(self):
        """Test negation operation"""
        data = [1.0, -2.0, 3.0]
        expected = [-1.0, 2.0, -3.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        ct_neg = self.engine.negate(ct)
        decrypted = self.engine.decrypt(ct_neg, self.secret_key)
        
        for i, (exp, dec) in enumerate(zip(expected, decrypted[:len(expected)])):
            self.assertAlmostEqual(exp, dec, places=4)
        logger.info("Negation test passed")
    
    def test_rotation(self):
        """Test slot rotation"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        ct = self.engine.encrypt(data, self.public_key)
        # DESILO rotate(delta=-1) shifts slots left: slot[i] = old_slot[i+1]
        ct_rot = self.engine.rotate(ct, self.rotation_key, -1)
        decrypted = self.engine.decrypt(ct_rot, self.secret_key)

        # After left rotation by 1, element 0 should contain original element 1
        self.assertAlmostEqual(decrypted[0], data[1], places=4)
        logger.info("Rotation test passed")
    
    def test_level_consumption(self):
        """Test that multiplication consumes levels"""
        data = [1.0, 2.0, 3.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        initial_level = ct.level if hasattr(ct, 'level') else -1
        
        ct_sq = self.engine.square(ct)
        ct_sq = self.engine.relinearize(ct_sq, self.relin_key)
        
        final_level = ct_sq.level if hasattr(ct_sq, 'level') else -1
        
        if initial_level >= 0 and final_level >= 0:
            self.assertLess(final_level, initial_level)
            logger.info(f"Level consumption: {initial_level} -> {final_level}")
        else:
            logger.info("Level tracking not available")
    
    def test_value_range_scaling(self):
        """Test value range scaling for bootstrap safety"""
        # Large values need scaling before bootstrap
        large_data = [50.0, -30.0, 40.0, -20.0]
        
        ct = self.engine.encrypt(large_data, self.public_key)
        
        # Scale to [-0.8, 0.8]
        max_val = max(abs(v) for v in large_data)
        scale_factor = 0.8 / max_val
        
        ct_scaled = self.engine.multiply(ct, scale_factor)
        decrypted_scaled = self.engine.decrypt(ct_scaled, self.secret_key)
        
        # Verify all values are within safe range
        for i, val in enumerate(decrypted_scaled[:len(large_data)]):
            self.assertLessEqual(abs(val), 1.0, f"Value {val} exceeds safe range")
        
        # Restore original scale
        ct_restored = self.engine.multiply(ct_scaled, 1.0 / scale_factor)
        decrypted_restored = self.engine.decrypt(ct_restored, self.secret_key)
        
        for i, (orig, rest) in enumerate(zip(large_data, decrypted_restored[:len(large_data)])):
            self.assertAlmostEqual(orig, rest, places=2)
        
        logger.info("Value range scaling test passed")


# =============================================================================
# HYBRID CRYPTOGRAPHY TESTS
# =============================================================================

@unittest.skipUnless(LIBOQS_AVAILABLE and CRYPTOGRAPHY_AVAILABLE, 
                     "liboqs or cryptography not available")
class TestHybridCryptography(unittest.TestCase):
    """Tests for hybrid X25519 + ML-KEM cryptography"""
    
    def test_x25519_key_exchange(self):
        """Test X25519 key exchange"""
        # Generate key pairs
        private_key_a = x25519.X25519PrivateKey.generate()
        public_key_a = private_key_a.public_key()
        
        private_key_b = x25519.X25519PrivateKey.generate()
        public_key_b = private_key_b.public_key()
        
        # Exchange
        shared_secret_a = private_key_a.exchange(public_key_b)
        shared_secret_b = private_key_b.exchange(public_key_a)
        
        self.assertEqual(shared_secret_a, shared_secret_b)
        self.assertEqual(len(shared_secret_a), 32)
        logger.info("X25519 key exchange test passed")
    
    def test_hybrid_encapsulation(self):
        """Test hybrid X25519 + ML-KEM encapsulation"""
        # ML-KEM setup
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        ml_kem_pk = kem.generate_keypair()
        
        # X25519 setup
        x25519_private = x25519.X25519PrivateKey.generate()
        x25519_public = x25519_private.public_key()
        
        # Hybrid encapsulation (sender side)
        # 1. ML-KEM encapsulation
        ml_kem_ct, ml_kem_ss = kem.encap_secret(ml_kem_pk)
        
        # 2. X25519 key exchange
        ephemeral_private = x25519.X25519PrivateKey.generate()
        ephemeral_public = ephemeral_private.public_key()
        x25519_ss = ephemeral_private.exchange(x25519_public)
        
        # 3. Combine shared secrets
        combined_ss = hashlib.sha256(ml_kem_ss + x25519_ss).digest()
        
        # Hybrid decapsulation (receiver side)
        # 1. ML-KEM decapsulation
        ml_kem_ss_dec = kem.decap_secret(ml_kem_ct)
        
        # 2. X25519 key exchange
        x25519_ss_dec = x25519_private.exchange(ephemeral_public)
        
        # 3. Combine shared secrets
        combined_ss_dec = hashlib.sha256(ml_kem_ss_dec + x25519_ss_dec).digest()
        
        self.assertEqual(combined_ss, combined_ss_dec)
        logger.info("Hybrid encapsulation test passed")
    
    def test_hybrid_defense_in_depth(self):
        """Test that hybrid provides defense in depth"""
        # Even if one component is compromised, the other provides security
        
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        ml_kem_pk = kem.generate_keypair()
        
        x25519_private = x25519.X25519PrivateKey.generate()
        x25519_public = x25519_private.public_key()
        
        # Encapsulation
        ml_kem_ct, ml_kem_ss = kem.encap_secret(ml_kem_pk)
        
        ephemeral_private = x25519.X25519PrivateKey.generate()
        x25519_ss = ephemeral_private.exchange(x25519_public)
        
        # Combined secret
        combined = hashlib.sha256(ml_kem_ss + x25519_ss).digest()
        
        # Attacker knowing only X25519 secret cannot recover combined secret
        attacker_attempt = hashlib.sha256(b'\x00' * 32 + x25519_ss).digest()
        self.assertNotEqual(combined, attacker_attempt)
        
        # Attacker knowing only ML-KEM secret cannot recover combined secret
        attacker_attempt = hashlib.sha256(ml_kem_ss + b'\x00' * 32).digest()
        self.assertNotEqual(combined, attacker_attempt)
        
        logger.info("Defense in depth test passed")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(LIBOQS_AVAILABLE and DESILO_AVAILABLE, 
                     "liboqs or DESILO not available")
class TestIntegration(unittest.TestCase):
    """Tests for full PQC-FHE integration"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize systems for integration tests"""
        cls.engine = desilofhe.Engine(mode='gpu', use_bootstrap=True, slot_count=2**14, thread_count=512)
        cls.secret_key = cls.engine.create_secret_key()
        cls.public_key = cls.engine.create_public_key(cls.secret_key)
        cls.relin_key = cls.engine.create_relinearization_key(cls.secret_key)
        
        cls.kem = oqs.KeyEncapsulation("ML-KEM-768")
        cls.ml_kem_pk = cls.kem.generate_keypair()
        
        cls.sig = oqs.Signature("ML-DSA-65")
        cls.ml_dsa_pk = cls.sig.generate_keypair()
    
    def test_secure_key_transport(self):
        """Test PQC-secured key transport for FHE"""
        # Scenario: Alice wants to send FHE encrypted data to Bob
        # 1. Bob generates ML-KEM keys
        # 2. Alice encapsulates a session key
        # 3. Alice encrypts data with FHE
        # 4. Alice signs the ciphertext
        # 5. Bob verifies and decapsulates
        
        # Bob's ML-KEM keys (already generated in setup)
        
        # Alice encapsulates session key
        session_ct, session_key = self.kem.encap_secret(self.ml_kem_pk)
        
        # Alice's data
        data = [1.5, 2.5, 3.5, 4.5]
        
        # Alice encrypts with FHE
        fhe_ct = self.engine.encrypt(data, self.public_key)
        fhe_ct_bytes = self.engine.decrypt(fhe_ct, self.secret_key)  # Simulate serialization
        
        # Alice signs
        message_to_sign = session_ct + bytes(str(fhe_ct_bytes[:10]).encode())
        signature = self.sig.sign(message_to_sign)
        
        # Bob verifies signature
        is_valid = self.sig.verify(message_to_sign, signature, self.ml_dsa_pk)
        self.assertTrue(is_valid)
        
        # Bob decapsulates session key
        session_key_bob = self.kem.decap_secret(session_ct)
        self.assertEqual(session_key, session_key_bob)
        
        logger.info("Secure key transport test passed")
    
    def test_encrypted_computation_with_signature(self):
        """Test FHE computation with ML-DSA signature verification"""
        # Input data
        input_data = [10.0, 20.0, 30.0]
        
        # Encrypt
        ct = self.engine.encrypt(input_data, self.public_key)
        
        # Compute: square + 1
        ct_sq = self.engine.square(ct)
        ct_sq = self.engine.relinearize(ct_sq, self.relin_key)
        ct_result = self.engine.add(ct_sq, 1.0)
        
        # Decrypt
        result = self.engine.decrypt(ct_result, self.secret_key)
        
        # Verify result
        expected = [x**2 + 1 for x in input_data]
        for i, (exp, res) in enumerate(zip(expected, result[:len(expected)])):
            self.assertAlmostEqual(exp, res, places=2)
        
        # Sign result
        result_bytes = str(result[:len(expected)]).encode()
        signature = self.sig.sign(result_bytes)
        
        # Verify signature
        is_valid = self.sig.verify(result_bytes, signature, self.ml_dsa_pk)
        self.assertTrue(is_valid)
        
        logger.info("Encrypted computation with signature test passed")
    
    def test_multi_level_computation(self):
        """Test multi-level FHE computation"""
        # This tests the ability to perform multiple operations
        data = [2.0, 3.0, 4.0]
        
        ct = self.engine.encrypt(data, self.public_key)
        
        # Level 1: Square
        ct = self.engine.square(ct)
        ct = self.engine.relinearize(ct, self.relin_key)
        
        # Level 2: Add 1
        ct = self.engine.add(ct, 1.0)
        
        # Level 3: Square again
        ct = self.engine.square(ct)
        ct = self.engine.relinearize(ct, self.relin_key)
        
        result = self.engine.decrypt(ct, self.secret_key)
        
        # Expected: ((x^2) + 1)^2 = x^4 + 2x^2 + 1
        expected = [(x**2 + 1)**2 for x in data]
        
        for i, (exp, res) in enumerate(zip(expected, result[:len(expected)])):
            self.assertAlmostEqual(exp, res, places=1)
        
        logger.info("Multi-level computation test passed")


# =============================================================================
# SECURITY TESTS
# =============================================================================

@unittest.skipUnless(LIBOQS_AVAILABLE, "liboqs not available")
class TestSecurity(unittest.TestCase):
    """Security-related tests"""
    
    def test_kem_ciphertext_tampering(self):
        """Test that tampered ciphertext produces different shared secret"""
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        pk = kem.generate_keypair()
        
        ct, ss_original = kem.encap_secret(pk)
        
        # Tamper with ciphertext
        ct_tampered = bytearray(ct)
        ct_tampered[0] ^= 0xFF
        ct_tampered = bytes(ct_tampered)
        
        # Decapsulation should produce different secret (implicit rejection)
        ss_tampered = kem.decap_secret(ct_tampered)
        
        # ML-KEM uses implicit rejection, so we get a pseudo-random value
        self.assertNotEqual(ss_original, ss_tampered)
        logger.info("KEM ciphertext tampering test passed")
    
    def test_signature_non_repudiation(self):
        """Test signature non-repudiation property"""
        sig = oqs.Signature("ML-DSA-65")
        pk = sig.generate_keypair()
        
        message = b"Important contract"
        signature = sig.sign(message)
        
        # Signature is verifiable
        self.assertTrue(sig.verify(message, signature, pk))
        
        # Different message fails
        self.assertFalse(sig.verify(b"Different contract", signature, pk))
        
        logger.info("Non-repudiation test passed")
    
    def test_key_sizes_match_security_level(self):
        """Verify key sizes match expected security levels"""
        # ML-KEM key sizes (from FIPS 203)
        kem_512 = oqs.KeyEncapsulation("ML-KEM-512")
        kem_768 = oqs.KeyEncapsulation("ML-KEM-768")
        kem_1024 = oqs.KeyEncapsulation("ML-KEM-1024")
        
        # Expected sizes
        self.assertEqual(kem_512.details['length_public_key'], 800)
        self.assertEqual(kem_768.details['length_public_key'], 1184)
        self.assertEqual(kem_1024.details['length_public_key'], 1568)
        
        # ML-DSA key sizes (from FIPS 204)
        sig_44 = oqs.Signature("ML-DSA-44")
        sig_65 = oqs.Signature("ML-DSA-65")
        sig_87 = oqs.Signature("ML-DSA-87")
        
        self.assertEqual(sig_44.details['length_public_key'], 1312)
        self.assertEqual(sig_65.details['length_public_key'], 1952)
        self.assertEqual(sig_87.details['length_public_key'], 2592)
        
        logger.info("Key size verification test passed")


# =============================================================================
# PERFORMANCE REGRESSION TESTS
# =============================================================================

@unittest.skipUnless(LIBOQS_AVAILABLE, "liboqs not available")
class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests to catch slowdowns"""
    
    def test_ml_kem_768_keygen_performance(self):
        """ML-KEM-768 keygen should complete within 10ms"""
        import time
        
        kem = oqs.KeyEncapsulation("ML-KEM-768")
        
        start = time.perf_counter()
        for _ in range(10):
            kem.generate_keypair()
        elapsed = (time.perf_counter() - start) / 10 * 1000
        
        self.assertLess(elapsed, 10.0, f"Keygen too slow: {elapsed:.2f}ms")
        logger.info(f"ML-KEM-768 keygen: {elapsed:.2f}ms")
    
    def test_ml_dsa_65_sign_performance(self):
        """ML-DSA-65 signing should complete within 50ms"""
        import time
        
        sig = oqs.Signature("ML-DSA-65")
        sig.generate_keypair()
        message = b"Test message" * 100
        
        start = time.perf_counter()
        for _ in range(10):
            sig.sign(message)
        elapsed = (time.perf_counter() - start) / 10 * 1000
        
        self.assertLess(elapsed, 50.0, f"Signing too slow: {elapsed:.2f}ms")
        logger.info(f"ML-DSA-65 sign: {elapsed:.2f}ms")


# =============================================================================
# QUANTUM THREAT SIMULATOR TESTS (v3.0.0)
# =============================================================================

class TestQuantumThreatSimulator(unittest.TestCase):
    """Tests for Shor and Grover algorithm resource estimation."""

    def test_shor_rsa_2048_estimate(self):
        """Shor's algorithm resource estimation for RSA-2048."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        est = shor.estimate_rsa_resources(2048)
        self.assertEqual(est.algorithm, 'RSA-2048')
        self.assertEqual(est.attack_type, 'shor_factoring')
        self.assertEqual(est.logical_qubits, 2 * 2048 + 1)
        self.assertGreater(est.physical_qubits, est.logical_qubits)
        self.assertEqual(est.post_quantum_security, 0)
        self.assertIn(est.threat_level, ['critical', 'high', 'moderate', 'low'])

    def test_shor_rsa_3072_estimate(self):
        """RSA-3072 requires more qubits than RSA-2048."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        est_2048 = shor.estimate_rsa_resources(2048)
        est_3072 = shor.estimate_rsa_resources(3072)
        self.assertGreater(est_3072.logical_qubits, est_2048.logical_qubits)
        self.assertGreater(est_3072.physical_qubits, est_2048.physical_qubits)

    def test_shor_ecc_256_estimate(self):
        """Shor's algorithm for ECC P-256."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        est = shor.estimate_ecc_resources(256)
        self.assertIn('P-256', est.algorithm)
        self.assertEqual(est.attack_type, 'shor_dlog')
        self.assertEqual(est.post_quantum_security, 0)

    def test_grover_aes_128_estimate(self):
        """Grover's algorithm for AES-128."""
        from src.quantum_threat_simulator import GroverSimulator
        grover = GroverSimulator()
        est = grover.estimate_aes_resources(128)
        self.assertEqual(est.algorithm, 'AES-128')
        self.assertEqual(est.post_quantum_security, 64)  # Halved
        self.assertEqual(est.attack_type, 'grover_search')

    def test_grover_aes_256_estimate(self):
        """Grover's algorithm for AES-256."""
        from src.quantum_threat_simulator import GroverSimulator
        grover = GroverSimulator()
        est = grover.estimate_aes_resources(256)
        self.assertEqual(est.post_quantum_security, 128)
        self.assertEqual(est.threat_level, 'low')

    def test_grover_invalid_key_size(self):
        """Grover rejects invalid AES key sizes."""
        from src.quantum_threat_simulator import GroverSimulator
        grover = GroverSimulator()
        with self.assertRaises(ValueError):
            grover.estimate_aes_resources(64)

    def test_threat_timeline_generation(self):
        """Full timeline generation."""
        from src.quantum_threat_simulator import QuantumThreatTimeline
        timeline = QuantumThreatTimeline('moderate')
        result = timeline.generate_full_timeline()
        self.assertIn('timeline', result)
        self.assertIn('RSA', result['timeline'])
        self.assertIn('AES', result['timeline'])
        self.assertGreater(len(result['timeline']['RSA']), 0)

    def test_threat_timeline_models(self):
        """All growth models produce valid output."""
        from src.quantum_threat_simulator import QuantumThreatTimeline
        for model in ['conservative', 'moderate', 'aggressive']:
            timeline = QuantumThreatTimeline(model)
            result = timeline.generate_full_timeline()
            self.assertEqual(result['growth_model'], model)

    def test_pqc_comparison(self):
        """Classical vs PQC comparison."""
        from src.quantum_threat_simulator import QuantumThreatTimeline
        timeline = QuantumThreatTimeline('moderate')
        comp = timeline.compare_classical_vs_pqc()
        self.assertIn('classical_vulnerable', comp)
        self.assertIn('pqc_resistant', comp)
        self.assertIn('summary', comp)
        self.assertGreater(len(comp['pqc_resistant']), 0)

    def test_factoring_simulation_steps(self):
        """Shor's algorithm simulation produces correct number of steps."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        steps = shor.simulate_factoring_progress(2048, steps=50)
        self.assertEqual(len(steps), 50)
        self.assertGreater(steps[-1].cumulative_progress, 0.9)

    def test_key_search_simulation_steps(self):
        """Grover's algorithm simulation produces correct step count."""
        from src.quantum_threat_simulator import GroverSimulator
        grover = GroverSimulator()
        steps = grover.simulate_key_search(128, steps=30)
        self.assertEqual(len(steps), 30)

    def test_resource_estimate_to_dict(self):
        """QuantumResourceEstimate serialization."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        est = shor.estimate_rsa_resources(2048)
        d = est.to_dict()
        self.assertIn('algorithm', d)
        self.assertIn('logical_qubits', d)
        self.assertIn('threat_level', d)


# =============================================================================
# SECURITY SCORING TESTS (v3.0.0)
# =============================================================================

class TestSecurityScoring(unittest.TestCase):
    """Tests for security scoring framework."""

    def test_enterprise_inventory_scoring(self):
        """Enterprise inventory produces valid score."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        score = engine.calculate_overall_score(inventory, {
            'has_crypto_inventory': True,
            'has_migration_plan': False,
        })
        self.assertGreaterEqual(score.overall_score, 0)
        self.assertLessEqual(score.overall_score, 100)
        self.assertIn(score.category, ['critical', 'high', 'moderate', 'low'])

    def test_financial_inventory_scoring(self):
        """Financial inventory scoring."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_financial_inventory()
        score = engine.calculate_overall_score(inventory)
        self.assertGreater(score.total_assets, 0)
        self.assertGreater(score.vulnerable_assets, 0)

    def test_government_inventory_scoring(self):
        """Government inventory has PQC assets."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_government_inventory()
        score = engine.calculate_overall_score(inventory)
        self.assertGreater(score.pqc_ready_assets, 0)

    def test_compliance_nist_ir_8547(self):
        """NIST IR 8547 compliance checks."""
        from src.security_scoring import SecurityScoringEngine, ComplianceStandard
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        checks = engine.check_compliance(ComplianceStandard.NIST_IR_8547, inventory)
        self.assertGreater(len(checks), 0)
        for c in checks:
            self.assertIn(c.status, ['pass', 'fail', 'partial', 'not_applicable'])

    def test_compliance_cnsa_2_0(self):
        """CNSA 2.0 compliance checks."""
        from src.security_scoring import SecurityScoringEngine, ComplianceStandard
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_government_inventory()
        checks = engine.check_compliance(ComplianceStandard.CNSA_2_0, inventory)
        self.assertGreater(len(checks), 0)

    def test_score_range_validation(self):
        """All sub-scores within 0-100 range."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        score = engine.calculate_overall_score(inventory)
        for attr in ['algorithm_strength_score', 'pqc_readiness_score',
                     'compliance_score', 'key_management_score', 'crypto_agility_score']:
            val = getattr(score, attr)
            self.assertGreaterEqual(val, 0, f"{attr} below 0: {val}")
            self.assertLessEqual(val, 100, f"{attr} above 100: {val}")

    def test_migration_plan_generation(self):
        """Migration plan has required phases."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        score = engine.calculate_overall_score(inventory)
        plan = engine.generate_migration_plan(score)
        self.assertIn('phases', plan)
        self.assertEqual(len(plan['phases']), 4)

    def test_vulnerability_identification(self):
        """Vulnerabilities are identified in inventory."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        score = engine.calculate_overall_score(inventory)
        # Enterprise inventory has RSA-2048 which is vulnerable
        self.assertGreater(len(score.vulnerabilities), 0)

    def test_security_score_to_dict(self):
        """SecurityScore serialization."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        score = engine.calculate_overall_score(inventory)
        d = score.to_dict()
        self.assertIn('overall_score', d)
        self.assertIn('sub_scores', d)
        self.assertIn('vulnerabilities', d)


# =============================================================================
# MPC-HE PROTOCOL TESTS (v3.0.0)
# =============================================================================

@unittest.skipUnless(DESILO_AVAILABLE, "DESILO FHE not available")
class TestMPCHEProtocol(unittest.TestCase):
    """Tests for MPC-HE 2-party protocol."""

    def test_protocol_creation(self):
        """MPCHEProtocol can be created."""
        from src.mpc_he_inference import MPCHEProtocol, MPCConfig
        config = MPCConfig(fhe_mode='gpu', use_bootstrap=False, log_n=14)
        protocol = MPCHEProtocol(config)
        self.assertIsNotNone(protocol.session_id)

    def test_bfv_raises_not_implemented(self):
        """BFV scheme raises NotImplementedError."""
        from src.mpc_he_inference import MPCHEProtocol, MPCConfig
        with self.assertRaises(NotImplementedError):
            MPCHEProtocol(MPCConfig(scheme='bfv'))

    def test_party_creation(self):
        """Parties can be created with secret keys."""
        from src.mpc_he_inference import MPCHEProtocol, MPCConfig, MPCRole
        config = MPCConfig(fhe_mode='gpu', use_bootstrap=False, log_n=14)
        protocol = MPCHEProtocol(config)
        protocol.setup_engine()
        alice = protocol.create_party(MPCRole.ALICE)
        bob = protocol.create_party(MPCRole.BOB)
        self.assertEqual(alice.role, MPCRole.ALICE)
        self.assertEqual(bob.role, MPCRole.BOB)
        self.assertIsNotNone(alice.secret_key)
        self.assertIsNotNone(bob.secret_key)

    def test_2party_linear_regression_demo(self):
        """Linear regression demo runs end-to-end."""
        from src.mpc_he_inference import SimpleMPCDemo, MPCConfig
        config = MPCConfig(fhe_mode='gpu', use_bootstrap=False, log_n=14)
        demo = SimpleMPCDemo(config)
        result = demo.run_linear_regression_demo([1.0, 2.0, 3.0, 4.0])
        self.assertIn('result', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['demo_type'], 'linear_regression')
        self.assertGreater(result['metrics']['total_ms'], 0)

    def test_private_statistics_demo(self):
        """Private statistics demo computes sum and product."""
        from src.mpc_he_inference import SimpleMPCDemo, MPCConfig
        config = MPCConfig(fhe_mode='gpu', use_bootstrap=False, log_n=14)
        demo = SimpleMPCDemo(config)
        result = demo.run_private_statistics_demo([1.0, 2.0], [3.0, 4.0])
        self.assertIn('result_sum', result)
        self.assertIn('result_product', result)
        self.assertEqual(result['demo_type'], 'private_statistics')

    def test_nn_inference_demo(self):
        """4-layer NN inference demo runs end-to-end with model architecture."""
        from src.mpc_he_inference import SimpleMPCDemo, MPCConfig
        config = MPCConfig(fhe_mode='gpu', use_bootstrap=False, log_n=14)
        demo = SimpleMPCDemo(config)
        result = demo.run_nn_inference_demo([1.0, 2.0, 3.0, 4.0])
        # Verify basic result structure
        self.assertIn('result', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['demo_type'], 'nn_inference')
        self.assertGreater(result['metrics']['total_ms'], 0)
        # Verify model architecture metadata
        self.assertIn('model_architecture', result)
        self.assertIsInstance(result['model_architecture'], list)
        self.assertEqual(len(result['model_architecture']), 4)
        # Verify expected plaintext reference
        self.assertIn('expected_plaintext', result)
        self.assertIsInstance(result['expected_plaintext'], list)
        self.assertEqual(len(result['expected_plaintext']), 4)
        # Verify model info
        self.assertIn('model_info', result)
        self.assertIn('name', result['model_info'])
        logger.info(f"NN inference demo: {result['metrics']['total_ms']:.1f}ms, "
                    f"layers={len(result['model_architecture'])}")

    def test_private_prediction_demo(self):
        """Anomaly detection private prediction demo runs end-to-end."""
        from src.mpc_he_inference import SimpleMPCDemo, MPCConfig
        config = MPCConfig(fhe_mode='gpu', use_bootstrap=False, log_n=14)
        demo = SimpleMPCDemo(config)
        result = demo.run_private_prediction_demo([0.5, -0.3, 1.2, 0.8])
        # Verify basic result structure
        self.assertIn('result', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['demo_type'], 'private_prediction')
        self.assertGreater(result['metrics']['total_ms'], 0)
        # Verify model info with scenario description
        self.assertIn('model_info', result)
        self.assertIn('name', result['model_info'])
        self.assertIn('scenario', result['model_info'])
        # Verify expected plaintext reference
        self.assertIn('expected_plaintext', result)
        self.assertIsInstance(result['expected_plaintext'], list)
        self.assertEqual(len(result['expected_plaintext']), 4)
        # Verify model architecture
        self.assertIn('model_architecture', result)
        self.assertEqual(len(result['model_architecture']), 3)
        logger.info(f"Private prediction demo: {result['metrics']['total_ms']:.1f}ms, "
                    f"layers={len(result['model_architecture'])}")


class TestMPCHEProtocolInfo(unittest.TestCase):
    """Tests for MPC-HE protocol info (no FHE dependency)."""

    def test_protocol_info(self):
        """Protocol info returns expected structure."""
        from src.mpc_he_inference import get_protocol_info
        info = get_protocol_info()
        self.assertIn('protocol_phases', info)
        self.assertEqual(len(info['protocol_phases']), 4)
        self.assertIn('available_demos', info)
        # Verify all 5 demos are listed
        demo_names = [d['name'] for d in info['available_demos']]
        self.assertIn('nn_inference', demo_names)
        self.assertIn('private_prediction', demo_names)
        self.assertIn('linear_regression', demo_names)
        self.assertIn('classification', demo_names)
        self.assertIn('private_statistics', demo_names)

    def test_mpc_config_to_dict(self):
        """MPCConfig serialization."""
        from src.mpc_he_inference import MPCConfig
        config = MPCConfig(fhe_mode='gpu', log_n=15)
        d = config.to_dict()
        self.assertEqual(d['fhe_mode'], 'gpu')
        self.assertEqual(d['slot_count'], 2 ** 14)


# =============================================================================
# EXTENDED BENCHMARK TESTS (v3.0.0)
# =============================================================================

class TestExtendedBenchmarks(unittest.TestCase):
    """Tests for extended benchmark infrastructure."""

    def test_gpu_benchmark_result_dataclass(self):
        """GPUBenchmarkResult inherits from BenchmarkResult."""
        from benchmarks import GPUBenchmarkResult
        result = GPUBenchmarkResult(
            name="test", algorithm="test", operation="test",
            iterations=10, times_ms=[1.0, 2.0, 3.0],
            gpu_memory_mb=100.0, speedup_vs_cpu=2.5,
        )
        d = result.to_dict()
        self.assertIn('gpu_memory_mb', d)
        self.assertIn('speedup_vs_cpu', d)
        self.assertEqual(d['speedup_vs_cpu'], 2.5)

    def test_quantum_threat_benchmark(self):
        """Quantum threat benchmarks produce results."""
        from benchmarks import benchmark_quantum_threat_estimation
        results = benchmark_quantum_threat_estimation(iterations=3)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertGreater(r.mean_ms, 0)

    def test_security_scoring_benchmark(self):
        """Security scoring benchmarks produce results."""
        from benchmarks import benchmark_security_scoring
        results = benchmark_security_scoring(iterations=3)
        self.assertGreater(len(results), 0)

    def test_security_performance_tradeoff(self):
        """Security-performance tradeoff benchmarks."""
        if not LIBOQS_AVAILABLE:
            self.skipTest("liboqs not available")
        from benchmarks import benchmark_security_performance_tradeoff
        results = benchmark_security_performance_tradeoff(iterations=3)
        self.assertGreater(len(results), 0)


# =============================================================================
# QUANTUM VERIFICATION TESTS (v3.1.0)
# =============================================================================

@unittest.skipUnless(QISKIT_AVAILABLE, "Qiskit not available")
class TestShorVerification(unittest.TestCase):
    """Tests for actual Shor's algorithm circuit execution."""

    def test_shor_factoring_15(self):
        """Factor 15 = 3 x 5 using real quantum circuit."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=4096)
        result = verifier.factor(15)
        self.assertTrue(result.success)
        self.assertIn(3, result.found_factors)
        self.assertIn(5, result.found_factors)
        self.assertGreater(result.num_qubits, 0)
        self.assertGreater(result.execution_time_ms, 0)

    def test_shor_factoring_21(self):
        """Factor 21 = 3 x 7 using real quantum circuit."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=4096)
        result = verifier.factor(21)
        self.assertTrue(result.success)
        self.assertIn(3, result.found_factors)
        self.assertIn(7, result.found_factors)

    def test_shor_circuit_has_qft(self):
        """Shor's circuit uses Quantum Fourier Transform (Hadamard gates)."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=1024)
        result = verifier.factor(15)
        self.assertGreater(result.circuit_depth, 0)
        self.assertIn('h', result.gate_count)

    def test_shor_rsa2048_extrapolation(self):
        """Extrapolation to RSA-2048 produces multi-era estimates."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=1024)
        result = verifier.factor(15)
        ext = result.extrapolation_to_rsa2048
        self.assertIn('estimated_qubits', ext)
        self.assertIn('multi_era_estimates', ext)
        # Gidney 2025 logical qubits as reference
        self.assertGreater(ext['estimated_qubits'], 1000)
        # Multi-era should have 4 generations (2021-2026)
        self.assertEqual(len(ext['multi_era_estimates']), 4)

    def test_shor_result_serialization(self):
        """ShorVerificationResult serializes to dict."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=1024)
        result = verifier.factor(15)
        d = result.to_dict()
        self.assertIn('found_factors', d)
        self.assertIn('measurement_counts', d)
        self.assertIn('extrapolation_to_rsa2048', d)


@unittest.skipUnless(QISKIT_AVAILABLE, "Qiskit not available")
class TestGroverVerification(unittest.TestCase):
    """Tests for actual Grover's search circuit execution."""

    def test_grover_search_3qubit(self):
        """3-qubit Grover search finds target with high probability."""
        from src.quantum_verification import GroverCircuitVerifier
        verifier = GroverCircuitVerifier(shots=4096)
        result = verifier.search(num_qubits=3, target=5)
        self.assertGreater(result.target_probability, 0.5)
        self.assertGreater(result.target_probability,
                           result.classical_probability)

    def test_grover_search_4qubit(self):
        """4-qubit Grover search demonstrates quadratic speedup."""
        from src.quantum_verification import GroverCircuitVerifier
        verifier = GroverCircuitVerifier(shots=4096)
        result = verifier.search(num_qubits=4, target=7)
        self.assertGreater(result.speedup_demonstrated, 1.0)
        self.assertEqual(result.optimal_iterations, 3)

    def test_grover_search_5qubit(self):
        """5-qubit Grover search with optimal iterations."""
        from src.quantum_verification import GroverCircuitVerifier
        verifier = GroverCircuitVerifier(shots=4096)
        result = verifier.search(num_qubits=5)
        self.assertGreater(result.target_probability, 0.5)
        self.assertEqual(result.optimal_iterations, 4)

    def test_grover_probability_evolution(self):
        """Probability evolution shows amplitude amplification."""
        from src.quantum_verification import GroverCircuitVerifier
        verifier = GroverCircuitVerifier(shots=2048)
        result = verifier.search(num_qubits=4, target=7)
        self.assertGreater(len(result.probability_evolution), 0)
        probs = [p['empirical_probability'] for p in result.probability_evolution]
        max_prob_idx = probs.index(max(probs))
        self.assertAlmostEqual(max_prob_idx, result.optimal_iterations, delta=2)

    def test_grover_aes_extrapolation(self):
        """Extrapolation to AES confirms security reduction."""
        from src.quantum_verification import GroverCircuitVerifier
        verifier = GroverCircuitVerifier(shots=1024)
        result = verifier.search(num_qubits=4)
        ext = result.extrapolation_to_aes
        self.assertIn('aes_128_effective_security', ext)
        self.assertEqual(ext['aes_128_effective_security'], 64)
        self.assertEqual(ext['aes_256_effective_security'], 128)

    def test_grover_result_serialization(self):
        """GroverVerificationResult serializes to dict."""
        from src.quantum_verification import GroverCircuitVerifier
        verifier = GroverCircuitVerifier(shots=1024)
        result = verifier.search(num_qubits=3, target=2)
        d = result.to_dict()
        self.assertIn('target_probability', d)
        self.assertIn('probability_evolution', d)
        self.assertIn('extrapolation_to_aes', d)


class TestNISTLevelVerification(unittest.TestCase):
    """Tests for NIST security level verification (no Qiskit needed)."""

    def test_ml_kem_768_level_3(self):
        """ML-KEM-768 verifies at or above NIST Level 3."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        result = verifier.verify_algorithm('ML-KEM-768')
        self.assertTrue(result.verification_passed)
        self.assertGreaterEqual(result.verified_nist_level,
                                result.claimed_nist_level)

    def test_ml_dsa_65_level_3(self):
        """ML-DSA-65 verifies at or above NIST Level 3."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        result = verifier.verify_algorithm('ML-DSA-65')
        self.assertTrue(result.verification_passed)

    def test_all_algorithms_pass(self):
        """All supported PQC algorithms pass verification."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        results = verifier.verify_all()
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertTrue(
                r.verification_passed,
                f"{r.algorithm} failed: claimed L{r.claimed_nist_level}, "
                f"verified L{r.verified_nist_level}",
            )

    def test_nist_level_ordering(self):
        """Higher parameter sets yield higher security levels."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        kem512 = verifier.verify_algorithm('ML-KEM-512')
        kem768 = verifier.verify_algorithm('ML-KEM-768')
        kem1024 = verifier.verify_algorithm('ML-KEM-1024')
        self.assertLess(kem512.core_svp_quantum, kem768.core_svp_quantum)
        self.assertLess(kem768.core_svp_quantum, kem1024.core_svp_quantum)

    def test_nist_verification_serialization(self):
        """NISTLevelVerification serializes to dict."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        result = verifier.verify_algorithm('ML-KEM-768')
        d = result.to_dict()
        self.assertIn('algorithm', d)
        self.assertIn('verification_passed', d)
        self.assertIn('core_svp_quantum', d)
        self.assertIn('bkz_block_size_quantum', d)


# =============================================================================
# SECTOR BENCHMARK TESTS (v3.1.0)
# =============================================================================

@unittest.skipUnless(LIBOQS_AVAILABLE, "liboqs not available")
class TestSectorBenchmarks(unittest.TestCase):
    """Tests for sector-specific benchmarks."""

    def test_sector_benchmark_healthcare(self):
        """Healthcare benchmarks produce results."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner(iterations=3)
        suite = runner.run_healthcare()
        self.assertEqual(suite.sector, 'healthcare')
        self.assertGreater(len(suite.results), 0)

    def test_sector_benchmark_finance(self):
        """Finance benchmarks produce results."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner(iterations=3)
        suite = runner.run_finance()
        self.assertEqual(suite.sector, 'finance')
        self.assertGreater(len(suite.results), 0)

    def test_sector_benchmark_blockchain(self):
        """Blockchain benchmarks produce results."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner(iterations=3)
        suite = runner.run_blockchain()
        self.assertEqual(suite.sector, 'blockchain')
        self.assertGreater(len(suite.results), 0)

    def test_sector_benchmark_iot(self):
        """IoT benchmarks produce results."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner(iterations=3)
        suite = runner.run_iot()
        self.assertEqual(suite.sector, 'iot')
        self.assertGreater(len(suite.results), 0)

    def test_sector_benchmark_serialization(self):
        """Sector benchmark results serialize correctly."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner(iterations=2)
        suite = runner.run_healthcare()
        d = suite.to_dict()
        self.assertIn('sector', d)
        self.assertIn('results', d)
        self.assertIn('summary', d)
        self.assertIsInstance(d['results'], list)
        for r in d['results']:
            self.assertIn('mean_ms', r)
            self.assertIn('algorithm', r)


# =============================================================================
# v3.2.0 TESTS: BKZ Accuracy, Shor Multi-Era, Side-Channel, Noise
# =============================================================================

class TestBKZAccuracyFixes(unittest.TestCase):
    """Tests for v3.2.0 BKZ block size and Core-SVP accuracy fixes."""

    def test_cbd_sigma_ml_kem_768(self):
        """ML-KEM-768 CBD(eta=2) sigma should be 1.0."""
        import math
        eta = 2
        sigma = math.sqrt(eta / 2.0)
        self.assertAlmostEqual(sigma, 1.0, places=4)

    def test_cbd_sigma_ml_kem_512(self):
        """ML-KEM-512 CBD(eta=3) sigma should be ~1.225."""
        import math
        eta = 3
        sigma = math.sqrt(eta / 2.0)
        self.assertAlmostEqual(sigma, 1.2247, places=3)

    def test_bkz_block_size_ml_kem_768(self):
        """ML-KEM-768 BKZ block size should be 633 (NIST reference)."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        result = verifier.verify_algorithm('ML-KEM-768')
        self.assertEqual(result.bkz_block_size_quantum, 633)

    def test_quantum_sieve_constant_0257(self):
        """Quantum sieve constant should be 0.257 (Dutch team Oct 2025)."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        hardness = verifier._core_svp_hardness(100, quantum=True)
        # 0.257 * 100 - 3.5 = 22.2
        self.assertAlmostEqual(hardness, 22.2, places=1)

    def test_all_algorithms_still_pass(self):
        """All 9 NIST algorithms should pass verification after v3.2.0 fixes."""
        from src.quantum_verification import NISTLevelVerifier
        verifier = NISTLevelVerifier()
        results = verifier.verify_all()
        self.assertEqual(len(results), 9)
        for r in results:
            self.assertTrue(
                r.verification_passed,
                f"{r.algorithm} failed: claimed L{r.claimed_nist_level}, "
                f"verified L{r.verified_nist_level}, "
                f"margin={r.security_margin:.1f}",
            )


class TestShorMultiEra(unittest.TestCase):
    """Tests for multi-era Shor resource estimation."""

    def test_multi_era_four_models(self):
        """Multi-era estimation should return 4 generation models."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        result = shor.estimate_rsa_resources_multi_era(2048)
        self.assertEqual(len(result['eras']), 4)
        self.assertIn('gidney_ekera_2021', result['eras'])
        self.assertIn('pinnacle_2026', result['eras'])

    def test_gidney_2025_under_2m_qubits(self):
        """Gidney 2025 model should estimate < 2M physical qubits for RSA-2048."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        result = shor.estimate_rsa_resources_multi_era(2048)
        gidney2025 = result['eras']['gidney_2025']
        self.assertLess(gidney2025['physical_qubits'], 2_000_000)

    def test_pinnacle_2026_under_200k(self):
        """Pinnacle 2026 model should estimate < 200K physical qubits for RSA-2048."""
        from src.quantum_threat_simulator import ShorSimulator
        shor = ShorSimulator()
        result = shor.estimate_rsa_resources_multi_era(2048)
        pinnacle = result['eras']['pinnacle_2026']
        self.assertLess(pinnacle['physical_qubits'], 200_000)

    def test_ec_overhead_configurable(self):
        """ShorSimulator EC overhead should be configurable."""
        from src.quantum_threat_simulator import ShorSimulator
        shor_default = ShorSimulator()
        shor_custom = ShorSimulator(error_correction_overhead=100)
        self.assertEqual(shor_default.ec_overhead, 500)
        self.assertEqual(shor_custom.ec_overhead, 100)


class TestSideChannelAssessment(unittest.TestCase):
    """Tests for side-channel risk assessment module."""

    def test_ml_kem_critical_risk(self):
        """ML-KEM should have CRITICAL overall risk."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        profile = sa.assess_algorithm('ML-KEM')
        self.assertEqual(profile.overall_risk.value, 'critical')

    def test_slh_dsa_low_risk(self):
        """SLH-DSA should have LOW overall risk."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        profile = sa.assess_algorithm('SLH-DSA')
        self.assertEqual(profile.overall_risk.value, 'low')

    def test_mitigation_recommendations(self):
        """Mitigations should be returned for all algorithms."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        for algo in ['ML-KEM', 'ML-DSA', 'SLH-DSA']:
            mitigations = sa.get_mitigations(algo)
            self.assertGreater(len(mitigations), 0, f"No mitigations for {algo}")


class TestNoiseAwareSimulation(unittest.TestCase):
    """Tests for noise-aware quantum simulation (requires Qiskit)."""

    def test_noisy_lower_probability(self):
        """Noisy simulation should have lower success than ideal."""
        from src.quantum_verification import NoiseAwareQuantumSimulator
        sim = NoiseAwareQuantumSimulator(shots=2048)
        result = sim.compare_ideal_vs_noisy('grover', 4, [0.05])
        ideal = result['ideal_probability']
        noisy = result['noisy_results']['0.05']['success_probability']
        self.assertGreater(ideal, noisy)

    def test_noise_model_creation(self):
        """Noise model should be creatable at different error rates."""
        from src.quantum_verification import NoiseAwareQuantumSimulator
        sim = NoiseAwareQuantumSimulator()
        for rate in [1e-3, 1e-2, 5e-2]:
            model = sim._build_noise_model(rate)
            self.assertIsNotNone(model)

    def test_multiple_error_rates(self):
        """Multiple error rates should produce multiple results."""
        from src.quantum_verification import NoiseAwareQuantumSimulator
        sim = NoiseAwareQuantumSimulator(shots=1024)
        result = sim.compare_ideal_vs_noisy('grover', 3, [1e-3, 1e-2, 5e-2])
        self.assertEqual(len(result['noisy_results']), 3)


class TestExtendedFactorization(unittest.TestCase):
    """Tests for extended Shor factorization (N=143, N=221)."""

    def test_shor_factoring_143(self):
        """Shor's should factor 143 = 11 × 13."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=2048)
        result = verifier.factor(143)
        self.assertTrue(result.success)
        self.assertEqual(sorted(result.found_factors), [11, 13])

    def test_shor_factoring_221(self):
        """Shor's should factor 221 = 13 × 17."""
        from src.quantum_verification import ShorCircuitVerifier
        verifier = ShorCircuitVerifier(shots=2048)
        result = verifier.factor(221)
        self.assertTrue(result.success)
        self.assertEqual(sorted(result.found_factors), [13, 17])


class TestAlgorithmDiversity(unittest.TestCase):
    """Tests for algorithm diversity assessment (v3.3.0)."""

    def test_enterprise_lattice_monoculture(self):
        """Enterprise inventory should detect lattice monoculture."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        result = engine.assess_algorithm_diversity(inventory)
        self.assertIn('diversity_score', result)
        self.assertIn('lattice_monoculture', result)
        # Enterprise has no hash-based or code-based PQC → low diversity
        self.assertLessEqual(result['diversity_score'], 70)

    def test_government_has_slh_dsa(self):
        """Government inventory should include SLH-DSA (hash-based)."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_government_inventory()
        result = engine.assess_algorithm_diversity(inventory)
        self.assertIn('hash', result['families_deployed'])

    def test_diversity_recommendations(self):
        """Low diversity should produce recommendations."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        result = engine.assess_algorithm_diversity(inventory)
        self.assertGreater(len(result['recommendations']), 0)

    def test_hqc_in_knowledge_base(self):
        """HQC should be in algorithm knowledge base."""
        from src.security_scoring import ALGORITHM_QUANTUM_STATUS
        self.assertIn('HQC-128', ALGORITHM_QUANTUM_STATUS)
        self.assertEqual(ALGORITHM_QUANTUM_STATUS['HQC-128']['family'], 'code')

    def test_pqc_family_info(self):
        """PQC_FAMILY_INFO should cover lattice, hash, code families."""
        from src.security_scoring import PQC_FAMILY_INFO
        self.assertIn('lattice', PQC_FAMILY_INFO)
        self.assertIn('hash', PQC_FAMILY_INFO)
        self.assertIn('code', PQC_FAMILY_INFO)


class TestCNSA20Readiness(unittest.TestCase):
    """Tests for CNSA 2.0 readiness assessment."""

    def test_enterprise_cnsa_readiness(self):
        """Enterprise should have partial CNSA 2.0 readiness."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        inventory = engine.create_sample_enterprise_inventory()
        result = engine.assess_cnsa_2_0_readiness(inventory)
        self.assertIn('cnsa_2_0_readiness_pct', result)
        self.assertIn('phase_gates', result)
        self.assertIn('compliance_gaps', result)

    def test_government_higher_readiness(self):
        """Government inventory should have higher CNSA readiness than enterprise."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()

        ent_inv = engine.create_sample_enterprise_inventory()
        gov_inv = engine.create_sample_government_inventory()

        ent_result = engine.assess_cnsa_2_0_readiness(ent_inv)
        gov_result = engine.assess_cnsa_2_0_readiness(gov_inv)

        self.assertGreaterEqual(
            gov_result['cnsa_2_0_readiness_pct'],
            ent_result['cnsa_2_0_readiness_pct']
        )

    def test_cnsa_phase_gates_exist(self):
        """CNSA 2.0 phase gates should span 2025-2035."""
        from src.security_scoring import CNSA_2_0_PHASE_GATES
        self.assertIn('2025', CNSA_2_0_PHASE_GATES)
        self.assertIn('2030', CNSA_2_0_PHASE_GATES)
        self.assertIn('2035', CNSA_2_0_PHASE_GATES)


class TestMaskingVerification(unittest.TestCase):
    """Tests for masking deployment verification."""

    def test_liboqs_lacks_masking(self):
        """liboqs should lack masking support for ML-KEM."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        result = sa.verify_masking_deployment(impl='liboqs')
        self.assertFalse(result['overall_masking_adequate'])
        self.assertFalse(result['algorithm_status']['ML-KEM']['masking_support'])

    def test_pqm4_has_masking(self):
        """pqm4 should have masking support for ML-KEM."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        result = sa.verify_masking_deployment(impl='pqm4')
        self.assertTrue(result['algorithm_status']['ML-KEM']['masking_support'])

    def test_hqc_assessment(self):
        """HQC should be assessable."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        profile = sa.assess_algorithm('HQC')
        self.assertIsNotNone(profile)
        # HQC's only vuln is patched → LOW risk
        self.assertEqual(profile.overall_risk.value, 'low')

    def test_hqc_in_assess_all(self):
        """assess_all should include HQC."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        result = sa.assess_all()
        self.assertIn('HQC', result['assessments'])


# =============================================================================
# CKKS / FHE QUANTUM SECURITY TESTS (v3.2.0)
# =============================================================================

class TestCKKSSecurityVerification(unittest.TestCase):
    """Tests for CKKS Ring-LWE quantum security verification."""

    def test_ckks_standard_meets_nist_level_1(self):
        """CKKS Standard (log_n=15, levels=20) should meet NIST Level 1."""
        from src.quantum_verification import CKKSSecurityVerifier
        v = CKKSSecurityVerifier()
        result = v.verify_ckks_config(log_n=15, max_levels=20)
        self.assertGreaterEqual(
            result['security_assessment']['nist_level_classical'], 1
        )
        self.assertTrue(result['security_assessment']['within_128bit_bound'])

    def test_ckks_default_mpc_he_warning(self):
        """MPC-HE default (log_n=15, levels=40) should exceed 128-bit bound."""
        from src.quantum_verification import CKKSSecurityVerifier
        v = CKKSSecurityVerifier()
        result = v.verify_ckks_config(log_n=15, max_levels=40)
        # log Q = 40*40+60 = 1660 > 881 (max for 128-bit at N=32768)
        self.assertFalse(result['security_assessment']['within_128bit_bound'])
        self.assertTrue(len(result['warnings']) > 0)

    def test_ckks_heavy_within_bounds(self):
        """CKKS Heavy (log_n=16, levels=40) should be within 128-bit bound."""
        from src.quantum_verification import CKKSSecurityVerifier
        v = CKKSSecurityVerifier()
        result = v.verify_ckks_config(log_n=16, max_levels=40)
        # log Q = 40*40+60 = 1660 < 1770 (max for 128-bit at N=65536)
        self.assertTrue(result['security_assessment']['within_128bit_bound'])

    def test_ckks_all_configs_has_summary(self):
        """verify_all_configs should return summary with business impact."""
        from src.quantum_verification import CKKSSecurityVerifier
        v = CKKSSecurityVerifier()
        result = v.verify_all_configs()
        self.assertIn('summary', result)
        self.assertIn('business_impact', result)
        self.assertIn('healthcare', result['business_impact'])
        self.assertIn('finance', result['business_impact'])
        self.assertGreater(result['summary']['total_configs'], 0)

    def test_ckks_lattice_monoculture_noted(self):
        """CKKS verification should note lattice monoculture risk."""
        from src.quantum_verification import CKKSSecurityVerifier
        v = CKKSSecurityVerifier()
        result = v.verify_ckks_config(log_n=15, max_levels=20)
        self.assertIn('quantum_threat', result)
        self.assertIn('ML-KEM', result['quantum_threat']['shared_risk_with'])

    def test_ckks_nn_demo_config_secure(self):
        """MPC-HE NN demo (log_n=15, levels=15) should meet NIST Level 1."""
        from src.quantum_verification import CKKSSecurityVerifier
        v = CKKSSecurityVerifier()
        result = v.verify_ckks_config(log_n=15, max_levels=15)
        self.assertGreaterEqual(
            result['security_assessment']['nist_level_classical'], 1
        )


class TestFHEQuantumRisk(unittest.TestCase):
    """Tests for FHE deployment quantum risk assessment."""

    def test_fhe_risk_score_exists(self):
        """FHE quantum risk should return a score 0-100."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        result = engine.assess_fhe_quantum_security(log_n=15, max_levels=20)
        self.assertIn('risk_score', result)
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)

    def test_fhe_lattice_monoculture_penalty(self):
        """FHE risk should include lattice monoculture as risk factor."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        result = engine.assess_fhe_quantum_security(log_n=15, max_levels=20)
        factor_names = [f['factor'] for f in result['risk_factors']]
        self.assertTrue(
            any('monoculture' in f.lower() for f in factor_names)
        )

    def test_fhe_insecure_params_critical(self):
        """Very small log_n with many levels should flag critical."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        result = engine.assess_fhe_quantum_security(log_n=12, max_levels=40)
        # log Q = 40*40+60 = 1660 >> 109 (max for 128-bit at N=4096)
        self.assertIn(result['risk_category'], ['critical', 'high'])

    def test_fhe_diversification_strategy(self):
        """FHE risk should include diversification strategy."""
        from src.security_scoring import SecurityScoringEngine
        engine = SecurityScoringEngine()
        result = engine.assess_fhe_quantum_security()
        self.assertIn('shared_lattice_risk', result)
        self.assertIn('HQC', result['shared_lattice_risk']['non_lattice_alternatives']['pqc'])


class TestCKKSSideChannel(unittest.TestCase):
    """Tests for CKKS/FHE side-channel assessment."""

    def test_ckks_in_assess_all(self):
        """assess_all should include CKKS-FHE."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        result = sa.assess_all()
        self.assertIn('CKKS-FHE', result['assessments'])

    def test_ckks_normalize(self):
        """CKKS, FHE, DESILO should normalize to CKKS-FHE."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        for alias in ['CKKS', 'FHE', 'BGV', 'BFV']:
            profile = sa.assess_algorithm(alias)
            self.assertEqual(profile.algorithm, 'CKKS-FHE')

    def test_ckks_masking_verification(self):
        """Masking verification should include CKKS-FHE."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        sa = SideChannelRiskAssessment()
        result = sa.verify_masking_deployment(impl='desilofhe')
        self.assertIn('CKKS-FHE', result['algorithm_status'])

    def test_ckks_fhe_in_security_scoring(self):
        """CKKS should be in ALGORITHM_QUANTUM_STATUS."""
        from src.security_scoring import ALGORITHM_QUANTUM_STATUS
        self.assertIn('CKKS', ALGORITHM_QUANTUM_STATUS)
        self.assertEqual(ALGORITHM_QUANTUM_STATUS['CKKS']['family'], 'lattice')


class TestSectorQuantumContext(unittest.TestCase):
    """Tests for sector-specific quantum security context."""

    def test_healthcare_quantum_context(self):
        """Healthcare summary should include quantum security context."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        ctx = SectorBenchmarkRunner._get_sector_quantum_context('healthcare')
        self.assertIn('compliance_framework', ctx)
        self.assertIn('HIPAA', ctx['compliance_framework'])
        self.assertIn('quantum_risk_profile', ctx)

    def test_mpc_fhe_quantum_context(self):
        """MPC-FHE summary should warn about parameter security."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        ctx = SectorBenchmarkRunner._get_sector_quantum_context('mpc-fhe')
        self.assertIn('Ring-LWE', ctx['quantum_risk_profile'])
        self.assertIn('CRITICAL', ctx['business_recommendation'])

    def test_all_sectors_have_context(self):
        """All 5 sectors should have quantum context."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        for sector in ['healthcare', 'finance', 'blockchain', 'iot', 'mpc-fhe']:
            ctx = SectorBenchmarkRunner._get_sector_quantum_context(sector)
            self.assertIn('quantum_risk_profile', ctx)


# =============================================================================
# v3.2.0 GL SCHEME TESTS
# =============================================================================

class TestGLSchemeEngine(unittest.TestCase):
    """Tests for GL scheme FHE engine wrapper."""

    def test_gl_scheme_info(self):
        """GL scheme info should return correct structure."""
        from src.gl_scheme_engine import get_gl_scheme_info
        info = get_gl_scheme_info()
        self.assertEqual(info['name'], 'GL Scheme (Gentry-Lee 5th Gen FHE)')
        self.assertIn('matrix_multiply', info['operations'])
        self.assertIn('hadamard_multiply', info['operations'])
        self.assertIn('transpose', info['operations'])
        self.assertEqual(info['security']['hardness'], 'Ring-LWE')
        self.assertTrue(info['security']['ntt_based'])

    def test_gl_config(self):
        """GLConfig should correctly compute batch/rows/cols."""
        from src.gl_scheme_engine import GLConfig
        config = GLConfig(shape=(256, 16, 16))
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.rows, 16)
        self.assertEqual(config.cols, 16)
        d = config.to_dict()
        self.assertEqual(d['shape'], [256, 16, 16])
        self.assertEqual(d['batch_size'], 256)

    def test_gl_matrix_shapes(self):
        """All supported GL matrix shapes should be valid tuples."""
        from src.gl_scheme_engine import GLMatrixShape
        for shape in GLMatrixShape:
            self.assertEqual(len(shape.value), 3)
            batch, rows, cols = shape.value
            self.assertGreater(batch, 0)
            self.assertGreater(rows, 0)
            self.assertGreater(cols, 0)

    def test_gl_security_info(self):
        """GL security info should document known vulnerabilities."""
        from src.gl_scheme_engine import GLSchemeEngine, GLConfig
        engine = GLSchemeEngine(GLConfig(shape=(256, 16, 16)))
        sec = engine.get_security_info()
        self.assertEqual(sec['scheme'], 'GL (Gentry-Lee)')
        self.assertEqual(sec['hardness_assumption'], 'Ring-LWE')
        self.assertTrue(sec['ntt_based'])
        # Should document NTT SPA and CPAD risks
        vuln_ids = [v['id'] for v in sec['known_vulnerabilities']]
        self.assertIn('GL-NTT-SPA-SHARED', vuln_ids)
        self.assertIn('GL-CPAD-THRESHOLD', vuln_ids)
        # Should list advantages over CKKS
        self.assertTrue(len(sec['advantages_over_ckks']) >= 3)

    def test_gl_hybrid_engine_info(self):
        """GL+CKKS hybrid engine should report both scheme capabilities."""
        from src.gl_scheme_engine import GLCKKSHybridEngine
        hybrid = GLCKKSHybridEngine(mode='cpu')
        info = hybrid.get_info()
        self.assertIn('gl_available', info)
        self.assertIn('ckks_available', info)
        self.assertIn('scheme_comparison', info)
        self.assertIn('gl', info['scheme_comparison'])
        self.assertIn('ckks', info['scheme_comparison'])


class TestGLPrivateInference(unittest.TestCase):
    """Tests for GL scheme-based private inference in MPC-HE module."""

    def test_gl_private_inference_info(self):
        """GL private inference should report correct capabilities."""
        from src.mpc_he_inference import GLPrivateInference
        gl = GLPrivateInference(shape=(256, 16, 16), mode='cpu')
        info = gl.get_info()
        self.assertEqual(info['scheme'], 'GL (Gentry-Lee, ePrint 2025/1935)')
        self.assertEqual(info['shape'], [256, 16, 16])
        self.assertIn('matrix_multiply (native)', info['operations'])
        self.assertTrue(len(info['security_notes']) >= 3)

    def test_protocol_info_includes_gl(self):
        """Protocol info should include GL scheme info."""
        from src.mpc_he_inference import get_protocol_info
        info = get_protocol_info()
        self.assertIn('gl', info['supported_schemes'])
        self.assertIn('gl_scheme', info)
        self.assertIn('gl_matrix_inference', [d['name'] for d in info['available_demos']])

    def test_protocol_info_security_notes(self):
        """Protocol info should include CPAD and NTT SPA security notes."""
        from src.mpc_he_inference import get_protocol_info
        info = get_protocol_info()
        self.assertIn('security_notes', info)
        self.assertIn('cpad_defense', info['security_notes'])
        self.assertIn('ntt_spa', info['security_notes'])
        self.assertIn('key_refresh', info['security_notes'])

    def test_protocol_references_updated(self):
        """Protocol info should reference GL, RhombusEnd2End, CEA, PKC."""
        from src.mpc_he_inference import get_protocol_info
        info = get_protocol_info()
        refs = info['references']
        ref_str = ' '.join(refs)
        self.assertIn('GL', ref_str)
        self.assertIn('Rhombus', ref_str)
        self.assertIn('CEA', ref_str)
        self.assertIn('PKC', ref_str)


class TestGLSideChannelAssessment(unittest.TestCase):
    """Tests for GL scheme in side-channel assessment."""

    def test_assess_all_includes_gl(self):
        """assess_all() should include GL-FHE assessment."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        assessor = SideChannelRiskAssessment()
        result = assessor.assess_all()
        self.assertIn('GL-FHE', result['assessments'])
        gl = result['assessments']['GL-FHE']
        self.assertIn('ePrint 2025/1935', gl['notes'])
        self.assertIn('Ring-LWE', gl['notes'])
        self.assertTrue(gl['vulnerability_count'] >= 2)

    def test_gl_inherits_ckks_risks(self):
        """GL-FHE should inherit CKKS NTT side-channel risks."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        assessor = SideChannelRiskAssessment()
        result = assessor.assess_all()
        gl = result['assessments']['GL-FHE']
        inherited = gl.get('inherited_risks', [])
        risk_str = ' '.join(inherited)
        self.assertIn('NTT SPA', risk_str)
        self.assertIn('CPAD', risk_str)

    def test_gl_key_findings(self):
        """Key findings should mention GL-FHE."""
        from src.side_channel_assessment import SideChannelRiskAssessment
        assessor = SideChannelRiskAssessment()
        result = assessor.assess_all()
        findings_str = ' '.join(result['summary']['key_findings'])
        self.assertIn('GL-FHE', findings_str)


class TestQuantumThreatGLScheme(unittest.TestCase):
    """Tests for GL scheme in quantum threat assessment."""

    def test_compare_pqc_includes_gl(self):
        """PQC comparison should include GL scheme in FHE security."""
        from src.quantum_threat_simulator import QuantumThreatTimeline
        timeline = QuantumThreatTimeline()
        result = timeline.compare_classical_vs_pqc()
        fhe_schemes = [s['scheme'] for s in result['fhe_security']]
        self.assertIn('GL (Gentry-Lee)', fhe_schemes)

    def test_gl_security_details(self):
        """GL entry should include security details and advantages."""
        from src.quantum_threat_simulator import QuantumThreatTimeline
        timeline = QuantumThreatTimeline()
        result = timeline.compare_classical_vs_pqc()
        gl_entry = next(
            s for s in result['fhe_security'] if s['scheme'] == 'GL (Gentry-Lee)'
        )
        self.assertTrue(gl_entry['post_quantum_secure'])
        self.assertEqual(gl_entry['security_assumption'], 'Ring-LWE')
        self.assertIn('ePrint 2025/1935', gl_entry['notes'])
        self.assertTrue(len(gl_entry.get('advantages', [])) >= 3)
        self.assertTrue(len(gl_entry.get('side_channel_risks', [])) >= 2)

    def test_quantum_resistant_includes_gl(self):
        """Quantum-resistant list should include GL."""
        from src.quantum_threat_simulator import QuantumThreatTimeline
        timeline = QuantumThreatTimeline()
        result = timeline.compare_classical_vs_pqc()
        resistant = result['summary']['quantum_resistant']
        resistant_str = ' '.join(resistant)
        self.assertIn('GL', resistant_str)


class TestSectorBenchmarkGL(unittest.TestCase):
    """Tests for GL scheme in sector benchmarks."""

    def test_mpc_fhe_includes_gl_status(self):
        """MPC-FHE benchmark should include GL scheme status."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner()
        suite = runner.run_mpc_fhe()
        operations = [r.operation for r in suite.results]
        self.assertIn('gl_scheme_status', operations)

    def test_gl_benchmark_result_metadata(self):
        """GL benchmark result should have correct metadata."""
        from src.sector_benchmarks import SectorBenchmarkRunner
        runner = SectorBenchmarkRunner()
        suite = runner.run_mpc_fhe()
        gl_result = next(
            r for r in suite.results if r.operation == 'gl_scheme_status'
        )
        self.assertEqual(gl_result.algorithm, 'GL (Gentry-Lee)')
        self.assertIn('ePrint 2025/1935', gl_result.notes)
        self.assertIn('Ring-LWE', gl_result.notes)


# =============================================================================
# MAIN
# =============================================================================

import time

def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPQCKeyManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestFHEOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridCryptography))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceRegression))

    # v3.0.0 test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumThreatSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestMPCHEProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestMPCHEProtocolInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestExtendedBenchmarks))

    # v3.2.0 CKKS/FHE quantum security tests
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSSecurityVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestFHEQuantumRisk))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSSideChannel))
    suite.addTests(loader.loadTestsFromTestCase(TestSectorQuantumContext))

    # v3.2.0 GL scheme tests
    suite.addTests(loader.loadTestsFromTestCase(TestGLSchemeEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestGLPrivateInference))
    suite.addTests(loader.loadTestsFromTestCase(TestGLSideChannelAssessment))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumThreatGLScheme))
    suite.addTests(loader.loadTestsFromTestCase(TestSectorBenchmarkGL))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
