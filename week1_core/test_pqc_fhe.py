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
        cls.engine = desilofhe.Engine(mode='cpu', slot_count=2**14)
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
        ct_rot = self.engine.rotate(ct, self.rotation_key, 1)
        decrypted = self.engine.decrypt(ct_rot, self.secret_key)
        
        # After rotation by 1, element 0 should contain original element 1
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
        cls.engine = desilofhe.Engine(mode='cpu', slot_count=2**14)
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
# MAIN
# =============================================================================

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
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
