#!/usr/bin/env python3
"""
Test script for Smart Default feature in PQC-FHE API.

This script demonstrates that the Smart Default feature works correctly
by testing PQC signature operations without manually copying keys.

Usage:
    1. Start the API server: uvicorn api.server:app --reload
    2. Run this script: python test_smart_default.py
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"


def test_pqc_signature_smart_default():
    """Test PQC signature workflow with Smart Default."""
    print("=" * 60)
    print("  PQC Signature Smart Default Test")
    print("=" * 60)
    print()
    
    # Step 1: Generate keypair
    print("[Step 1] Generating ML-DSA-65 keypair...")
    try:
        resp = requests.post(f"{BASE_URL}/pqc/sig/keypair")
        resp.raise_for_status()
        keypair = resp.json()
        print(f"  ✓ Keypair generated")
        print(f"    Public key size: {keypair['public_key_size']} bytes")
        print(f"    Secret key size: {keypair['secret_key_size']} bytes")
        print(f"    Mock mode: {keypair['is_mock']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    print()
    
    # Step 2: Sign message using Smart Default (placeholder secret_key)
    print("[Step 2] Signing message with Smart Default...")
    print('  Request: {"message": "Hello World", "secret_key": "string"}')
    try:
        resp = requests.post(
            f"{BASE_URL}/pqc/sig/sign",
            json={"message": "Hello World", "secret_key": "string"}
        )
        resp.raise_for_status()
        sign_result = resp.json()
        print(f"  ✓ Message signed successfully!")
        print(f"    Signature size: {sign_result['signature_size']} bytes")
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ Failed: {e.response.json()['detail']}")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    print()
    
    # Step 3: Verify signature using Smart Default (placeholder signature and public_key)
    print("[Step 3] Verifying signature with Smart Default...")
    print('  Request: {"message": "Hello World", "signature": "string", "public_key": "string"}')
    try:
        resp = requests.post(
            f"{BASE_URL}/pqc/sig/verify",
            json={
                "message": "Hello World",
                "signature": "string",
                "public_key": "string"
            }
        )
        resp.raise_for_status()
        verify_result = resp.json()
        if verify_result['valid']:
            print(f"  ✓ Signature verified: VALID")
        else:
            print(f"  ✗ Signature verified: INVALID")
            return False
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ Failed: {e.response.json()['detail']}")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    print()
    
    return True


def test_fhe_smart_default():
    """Test FHE workflow with Smart Default."""
    print("=" * 60)
    print("  FHE Smart Default Test")
    print("=" * 60)
    print()
    
    # Step 1: Encrypt data
    print("[Step 1] Encrypting data [1.0, 2.0, 3.0, 4.0, 5.0]...")
    try:
        resp = requests.post(
            f"{BASE_URL}/fhe/encrypt",
            json={"data": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        resp.raise_for_status()
        encrypt_result = resp.json()
        print(f"  ✓ Data encrypted")
        print(f"    Ciphertext ID: {encrypt_result['ciphertext_id'][:16]}...")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    print()
    
    # Step 2: Multiply with Smart Default (placeholder ciphertext_id)
    print("[Step 2] Multiplying by 2.0 with Smart Default...")
    print('  Request: {"ciphertext_id": "c40e1e56fda432fa80efdf0486959354", "scalar": 2.0}')
    try:
        resp = requests.post(
            f"{BASE_URL}/fhe/multiply",
            json={
                "ciphertext_id": "c40e1e56fda432fa80efdf0486959354",
                "scalar": 2.0
            }
        )
        resp.raise_for_status()
        multiply_result = resp.json()
        print(f"  ✓ Multiplication completed")
        print(f"    Result ID: {multiply_result['result_ciphertext_id'][:16]}...")
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ Failed: {e.response.json()['detail']}")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    print()
    
    # Step 3: Decrypt with Smart Default
    print("[Step 3] Decrypting with Smart Default...")
    print('  Request: {"ciphertext_id": "c40e1e56fda432fa80efdf0486959354"}')
    try:
        resp = requests.post(
            f"{BASE_URL}/fhe/decrypt",
            json={"ciphertext_id": "c40e1e56fda432fa80efdf0486959354"}
        )
        resp.raise_for_status()
        decrypt_result = resp.json()
        print(f"  ✓ Decryption completed")
        print(f"    Result: {[round(x, 2) for x in decrypt_result['data'][:5]]}")
        print(f"    Expected: [2.0, 4.0, 6.0, 8.0, 10.0]")
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ Failed: {e.response.json()['detail']}")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    print()
    
    return True


def main():
    """Run all Smart Default tests."""
    print()
    print("Testing Smart Default Feature")
    print("Make sure the API server is running at:", BASE_URL)
    print()
    
    # Check server is running
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        print("✓ Server is running")
        print()
    except Exception as e:
        print(f"✗ Cannot connect to server at {BASE_URL}")
        print(f"  Error: {e}")
        print()
        print("Please start the server first:")
        print("  cd pqc_fhe_portfolio_v2.1.2_complete")
        print("  uvicorn api.server:app --reload")
        sys.exit(1)
    
    # Run tests
    pqc_pass = test_pqc_signature_smart_default()
    fhe_pass = test_fhe_smart_default()
    
    # Summary
    print("=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print(f"  PQC Signature Smart Default: {'PASS ✓' if pqc_pass else 'FAIL ✗'}")
    print(f"  FHE Smart Default:           {'PASS ✓' if fhe_pass else 'FAIL ✗'}")
    print("=" * 60)
    
    if pqc_pass and fhe_pass:
        print()
        print("All Smart Default tests passed!")
        print("You can now use Swagger UI without manually copying IDs.")
        sys.exit(0)
    else:
        print()
        print("Some tests failed. Please check server logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
