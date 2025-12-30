# Security Best Practices

Comprehensive security guidelines for deploying and operating PQC-FHE systems.

## Overview

This document covers security best practices across:

1. Key Management
2. FHE Operations
3. Network Security
4. Access Control
5. Audit and Compliance
6. Incident Response

---

## Key Management

### Key Generation

**DO:**

```python
# Use cryptographically secure random number generation
from pqc_fhe import PQCKeyManager

manager = PQCKeyManager(
    kem_algorithm="ML-KEM-768",  # NIST Level 3
    sig_algorithm="ML-DSA-65"   # NIST Level 3
)

# Generate fresh keys for each identity
public_key, secret_key = manager.generate_kem_keypair()
```

**DON'T:**

```python
# Never use deterministic or weak randomness
import random
random.seed(12345)  # NEVER DO THIS

# Never reuse keys across different purposes
# KEM keys for encapsulation only
# Signature keys for signing only
```

### Key Storage

**Recommended Storage Methods:**

| Storage Type | Use Case | Security Level |
|--------------|----------|----------------|
| HSM | Production secrets | Highest |
| AWS KMS / GCP KMS | Cloud deployments | High |
| HashiCorp Vault | Multi-cloud | High |
| Encrypted file | Development | Medium |

**HSM Integration Example:**

```python
from pqc_fhe.storage import HSMKeyStore

# Initialize HSM connection
hsm = HSMKeyStore(
    slot_id=0,
    pin="secure-pin",
    library_path="/usr/lib/softhsm/libsofthsm2.so"
)

# Store key in HSM
key_handle = hsm.store_key(
    key_id="ml-kem-prod-001",
    key_data=secret_key,
    key_type="ML-KEM-768"
)

# Retrieve key from HSM
secret_key = hsm.retrieve_key(key_handle)
```

**Environment Variable Protection:**

```bash
# Never store keys directly in environment variables
# Use secret management services instead

# Bad
export SECRET_KEY="actual-key-bytes"

# Good - use reference to secret manager
export SECRET_KEY_REF="vault:secret/pqc/prod#secret_key"
```

### Key Rotation

**Rotation Schedule:**

| Key Type | Rotation Period | Reason |
|----------|-----------------|--------|
| KEM keys | 90 days | Limit exposure window |
| Signature keys | 365 days | Certificate lifecycle |
| Session keys | Per-session | Forward secrecy |
| FHE context | Per-computation | Noise budget |

**Rotation Implementation:**

```python
from datetime import datetime, timedelta
from pqc_fhe import PQCKeyManager

class KeyRotationManager:
    def __init__(self, max_age_days: int = 90):
        self.max_age = timedelta(days=max_age_days)
        self.manager = PQCKeyManager()
    
    def should_rotate(self, key_created_at: datetime) -> bool:
        age = datetime.utcnow() - key_created_at
        return age > self.max_age
    
    def rotate_key(self, old_key_id: str):
        # Generate new key
        new_pub, new_sec = self.manager.generate_kem_keypair()
        
        # Store new key
        new_key_id = self.store_key(new_pub, new_sec)
        
        # Mark old key for deprecation (don't delete immediately)
        self.mark_deprecated(old_key_id)
        
        # Schedule old key deletion (grace period)
        self.schedule_deletion(old_key_id, days=30)
        
        return new_key_id
```

### Key Destruction

**Secure Key Destruction:**

```python
import ctypes

def secure_zero_memory(buffer: bytes) -> None:
    """
    Securely zero memory to prevent key recovery.
    
    References:
    - NIST SP 800-88: Media Sanitization Guidelines
    - CWE-244: Improper Clearing of Heap Memory
    """
    length = len(buffer)
    
    # Use ctypes to overwrite memory
    ctypes.memset(ctypes.c_char_p(buffer), 0, length)
    
    # Verify zeroing
    for byte in buffer:
        assert byte == 0, "Memory not properly zeroed"


def destroy_secret_key(secret_key: bytes):
    """Destroy a secret key securely"""
    # Overwrite multiple times (DoD 5220.22-M standard)
    patterns = [b'\x00', b'\xff', b'\x00']
    
    for pattern in patterns:
        ctypes.memset(
            ctypes.c_char_p(secret_key), 
            pattern[0], 
            len(secret_key)
        )
    
    # Final verification
    secure_zero_memory(secret_key)
    
    del secret_key
```

---

## FHE Operations

### Parameter Selection

**Security Parameter Guidelines:**

| Security Level | Polynomial Degree | Use Case |
|----------------|-------------------|----------|
| 128-bit | 4096-8192 | Standard operations |
| 192-bit | 16384 | Sensitive data |
| 256-bit | 32768 | Maximum security |

**Recommended Configurations:**

```python
from pqc_fhe import FHEEngine

# Standard security (128-bit)
engine_standard = FHEEngine(
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
    scale=2**40,
    security_level=128
)

# High security (192-bit equivalent)
engine_high = FHEEngine(
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 50, 50, 50, 50, 50, 60],
    scale=2**50,
    security_level=128  # Implicit 192-bit from parameters
)
```

### Noise Budget Management

**Monitor Noise Budget:**

```python
def safe_multiply(engine, ct_a, ct_b, min_budget: int = 10):
    """
    Perform multiplication with noise budget check.
    
    References:
    - Brakerski et al., "Leveled Fully Homomorphic Encryption"
    - SEAL Manual: Managing Noise Budget
    """
    budget_a = engine.get_noise_budget(ct_a)
    budget_b = engine.get_noise_budget(ct_b)
    
    # Estimate budget consumption
    estimated_consumption = 40  # Typical for CKKS multiply
    
    if min(budget_a, budget_b) < estimated_consumption + min_budget:
        raise ValueError(
            f"Insufficient noise budget: {min(budget_a, budget_b)} bits. "
            f"Need at least {estimated_consumption + min_budget} bits."
        )
    
    result = engine.multiply(ct_a, ct_b)
    
    result_budget = engine.get_noise_budget(result)
    if result_budget < min_budget:
        raise ValueError(
            f"Noise budget exhausted after multiply: {result_budget} bits"
        )
    
    return result
```

### Data Leakage Prevention

**Prevent Timing Attacks:**

```python
import time
import secrets

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Constant-time comparison to prevent timing attacks.
    
    References:
    - CERT: MSC03-J (Timing attacks)
    - OpenSSL: CRYPTO_memcmp implementation
    """
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    
    return result == 0


def add_timing_jitter(min_ms: float = 1.0, max_ms: float = 5.0):
    """Add random timing jitter to prevent timing analysis"""
    jitter = secrets.randbelow(int((max_ms - min_ms) * 1000)) / 1000 + min_ms
    time.sleep(jitter / 1000)
```

---

## Network Security

### TLS Configuration

**Recommended TLS Settings:**

```python
import ssl

def create_secure_ssl_context():
    """
    Create a secure SSL context with quantum-safe considerations.
    
    References:
    - NIST SP 800-52 Rev. 2: TLS Guidelines
    - Mozilla SSL Configuration Generator
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # TLS 1.3 only
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    
    # Strong cipher suites
    context.set_ciphers([
        'TLS_AES_256_GCM_SHA384',
        'TLS_CHACHA20_POLY1305_SHA256',
        'TLS_AES_128_GCM_SHA256'
    ])
    
    # Certificate verification
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    
    return context
```

**Nginx Configuration:**

```nginx
# /etc/nginx/conf.d/ssl.conf

ssl_protocols TLSv1.3;
ssl_prefer_server_ciphers off;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

# HSTS
add_header Strict-Transport-Security "max-age=63072000" always;

# Certificate
ssl_certificate /etc/ssl/certs/server.crt;
ssl_certificate_key /etc/ssl/private/server.key;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
```

### Request Validation

**Input Validation:**

```python
from pydantic import BaseModel, validator, constr
from typing import List
import base64
import re

class EncryptRequest(BaseModel):
    """Validated encryption request"""
    
    data: List[float]
    context_id: constr(regex=r'^ctx-[a-f0-9]{8}$') = None
    
    @validator('data')
    def validate_data(cls, v):
        if len(v) > 4096:
            raise ValueError("Data exceeds maximum slot count")
        
        for value in v:
            if not (-1e10 <= value <= 1e10):
                raise ValueError("Value out of safe range")
        
        return v


class KeyRequest(BaseModel):
    """Validated key request"""
    
    public_key: str
    algorithm: str = "ML-KEM-768"
    
    @validator('public_key')
    def validate_public_key(cls, v):
        try:
            decoded = base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        
        # Check expected sizes
        expected_sizes = {
            "ML-KEM-512": 800,
            "ML-KEM-768": 1184,
            "ML-KEM-1024": 1568
        }
        
        # Will be validated against algorithm in full validation
        if len(decoded) not in expected_sizes.values():
            raise ValueError("Invalid key size")
        
        return v
```

---

## Access Control

### Authentication

**JWT Configuration:**

```python
import jwt
from datetime import datetime, timedelta
from typing import Optional

class JWTManager:
    """
    JWT manager with security best practices.
    
    References:
    - RFC 7519: JSON Web Token
    - OWASP: JWT Security Cheat Sheet
    """
    
    def __init__(
        self,
        secret_key: bytes,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 15,
        refresh_token_expire_days: int = 7
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_expire = timedelta(days=refresh_token_expire_days)
    
    def create_access_token(
        self,
        subject: str,
        scopes: List[str] = None
    ) -> str:
        now = datetime.utcnow()
        
        payload = {
            "sub": subject,
            "iat": now,
            "exp": now + self.access_expire,
            "type": "access",
            "scopes": scopes or []
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "require": ["sub", "iat", "exp", "type"],
                    "verify_exp": True,
                    "verify_iat": True
                }
            )
            return payload
        except jwt.InvalidTokenError:
            return None
```

### Authorization

**Role-Based Access Control:**

```python
from enum import Enum
from functools import wraps

class Permission(Enum):
    KEY_READ = "key:read"
    KEY_WRITE = "key:write"
    KEY_DELETE = "key:delete"
    FHE_ENCRYPT = "fhe:encrypt"
    FHE_DECRYPT = "fhe:decrypt"
    FHE_COMPUTE = "fhe:compute"
    ADMIN = "admin"


ROLE_PERMISSIONS = {
    "reader": [Permission.KEY_READ, Permission.FHE_ENCRYPT],
    "operator": [
        Permission.KEY_READ, Permission.KEY_WRITE,
        Permission.FHE_ENCRYPT, Permission.FHE_DECRYPT, Permission.FHE_COMPUTE
    ],
    "admin": [Permission.ADMIN]  # All permissions
}


def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = get_current_user()
            
            if not has_permission(user, permission):
                raise PermissionDeniedError(
                    f"Permission {permission.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def has_permission(user, permission: Permission) -> bool:
    """Check if user has permission"""
    if Permission.ADMIN in get_user_permissions(user):
        return True
    
    return permission in get_user_permissions(user)
```

---

## Audit and Compliance

### Logging Requirements

**Structured Logging:**

```python
import structlog
from typing import Any, Dict

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


def log_cryptographic_operation(
    operation: str,
    actor_id: str,
    resource_id: str,
    outcome: str,
    details: Dict[str, Any] = None
):
    """
    Log cryptographic operations for audit trail.
    
    Compliance:
    - SOC 2 CC6.1: Logical access controls
    - HIPAA 164.312(b): Audit controls
    - PCI-DSS 10.2: Audit trail
    """
    logger.info(
        "cryptographic_operation",
        operation=operation,
        actor_id=actor_id,
        resource_id=resource_id,
        outcome=outcome,
        details=details or {},
        timestamp=datetime.utcnow().isoformat()
    )
```

**Required Audit Events:**

| Event | Data to Log | Retention |
|-------|-------------|-----------|
| Key generation | Key ID, algorithm, owner | 7 years |
| Key access | Key ID, accessor, operation | 7 years |
| Key deletion | Key ID, deleter, reason | 7 years |
| Encryption | Data ID, context, size | 1 year |
| Decryption | Data ID, requestor | 7 years |
| Auth failure | IP, username, method | 1 year |

### Compliance Checklists

**SOC 2 Type II:**

- [ ] Access control policies documented
- [ ] Key management procedures in place
- [ ] Audit logging enabled and tamper-evident
- [ ] Encryption at rest and in transit
- [ ] Incident response plan tested
- [ ] Change management process followed

**HIPAA:**

- [ ] Unique user identification
- [ ] Emergency access procedure
- [ ] Automatic logoff enabled
- [ ] Encryption/decryption mechanisms
- [ ] Audit controls implemented
- [ ] Transmission security ensured

**GDPR:**

- [ ] Data processing purposes documented
- [ ] Lawful basis for processing established
- [ ] Data retention limits defined
- [ ] Data subject rights procedures
- [ ] Breach notification process
- [ ] Privacy by design implemented

---

## Incident Response

### Response Plan

**Incident Severity Levels:**

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P1 | Critical | 15 minutes | Key compromise |
| P2 | High | 1 hour | Service outage |
| P3 | Medium | 4 hours | Performance degradation |
| P4 | Low | 24 hours | Minor bug |

### Key Compromise Response

```python
class IncidentResponse:
    """Key compromise incident response"""
    
    async def handle_key_compromise(
        self,
        compromised_key_id: str,
        discovery_time: datetime,
        reporter: str
    ):
        """
        Handle key compromise incident.
        
        References:
        - NIST SP 800-61: Incident Response
        - CERT Incident Response Guide
        """
        # 1. Containment (Immediate)
        await self.revoke_key(compromised_key_id)
        await self.invalidate_sessions_using_key(compromised_key_id)
        
        # 2. Alert stakeholders
        await self.send_alert(
            severity="P1",
            title="Key Compromise Detected",
            key_id=compromised_key_id,
            discovery_time=discovery_time,
            reporter=reporter
        )
        
        # 3. Generate replacement keys
        new_key_id = await self.generate_replacement_key(compromised_key_id)
        
        # 4. Identify affected data/sessions
        affected = await self.identify_affected_resources(compromised_key_id)
        
        # 5. Re-encrypt affected data if necessary
        for resource in affected:
            await self.re_encrypt_resource(resource, new_key_id)
        
        # 6. Create incident report
        report = await self.create_incident_report(
            key_id=compromised_key_id,
            new_key_id=new_key_id,
            affected_resources=affected,
            timeline={
                "discovery": discovery_time,
                "containment": datetime.utcnow(),
                "recovery": datetime.utcnow()
            }
        )
        
        # 7. Post-incident review scheduled
        await self.schedule_post_incident_review(report)
        
        return report
```

---

## Security Checklist

### Pre-Deployment

- [ ] Security parameters reviewed by cryptographer
- [ ] Key management procedures documented
- [ ] Access control policies implemented
- [ ] Audit logging configured
- [ ] TLS 1.3 enforced
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] Penetration testing completed
- [ ] Vulnerability scanning automated

### Operational

- [ ] Keys rotated on schedule
- [ ] Logs monitored for anomalies
- [ ] Backups tested monthly
- [ ] Incident response plan tested quarterly
- [ ] Security patches applied within SLA
- [ ] Access reviews conducted monthly
- [ ] Compliance audits scheduled

### Post-Incident

- [ ] Root cause analysis completed
- [ ] Affected parties notified
- [ ] Remediation actions documented
- [ ] Lessons learned incorporated
- [ ] Controls updated as needed

---

## References

1. NIST SP 800-57: Key Management Guidelines
2. NIST SP 800-88: Media Sanitization Guidelines
3. NIST SP 800-52: TLS Guidelines
4. NIST SP 800-61: Incident Response
5. OWASP Security Guidelines
6. CIS Benchmarks

---

## Related Documentation

- [Security Overview](overview.md)
- [API Reference](../api/overview.md)
- [Enterprise Integration](../tutorials/enterprise_integration.md)
