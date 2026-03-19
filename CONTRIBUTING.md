# Contributing to PQC-FHE Integration

Thank you for your interest in contributing to the PQC-FHE Integration project!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Security Considerations](#security-considerations)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other contributors

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Docker (optional, for containerized development)
- liboqs C library (for PQC operations)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/pqc-fhe-integration.git
cd pqc-fhe-integration
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
# Install all development dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[dev]"      # Development tools
pip install -e ".[pqc]"      # PQC support
pip install -e ".[api]"      # API dependencies
```

### 3. Install liboqs (for PQC)

```bash
# Ubuntu/Debian
sudo apt-get install cmake ninja-build libssl-dev
git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local ..
ninja && sudo ninja install
sudo ldconfig
```

### 4. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Start API server
python -m api.server
```

---

## Making Contributions

### Types of Contributions

We welcome:

1. **Bug Fixes** - Fix issues and improve stability
2. **Features** - Add new functionality
3. **Documentation** - Improve docs, examples, comments
4. **Tests** - Increase test coverage
5. **Performance** - Optimize algorithms and reduce overhead
6. **Security** - Identify and fix vulnerabilities

### Branch Naming

Use descriptive branch names:

```
feature/add-slh-dsa-support
bugfix/fix-bootstrap-scaling
docs/update-api-reference
test/add-integration-tests
security/fix-key-exposure
```

### Commit Messages

Follow conventional commits:

```
feat: add SLH-DSA signature support
fix: correct bootstrap value scaling for large inputs
docs: update API documentation with new endpoints
test: add integration tests for hybrid channel
perf: optimize FHE multiplication by 20%
security: fix potential key exposure in logging
```

---

## Coding Standards

### Python Style

We use **Black** for formatting and **isort** for imports:

```bash
# Format code
black .
isort .

# Check formatting
black --check .
isort --check-only .
```

### Type Hints

Use type hints for all public functions:

```python
def encrypt_data(
    data: List[float],
    public_key: bytes,
    precision: int = 40
) -> Tuple[bytes, int]:
    """
    Encrypt data using CKKS scheme.
    
    Args:
        data: Array of floating-point values
        public_key: Encryption public key
        precision: Scale bits for precision
        
    Returns:
        Tuple of (ciphertext, level)
    """
    ...
```

### Documentation

- Use Google-style docstrings
- Document all public APIs
- Include usage examples
- Reference relevant papers/standards

### Logging

Use the logging module, not print statements:

```python
import logging

logger = logging.getLogger(__name__)

def some_function():
    logger.info("Processing started")
    logger.debug("Debug details: %s", details)
    logger.warning("Potential issue detected")
    logger.error("Operation failed: %s", error)
```

---

## Testing Guidelines

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test file
pytest tests/test_pqc_fhe.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

```python
import pytest
from pqc_fhe_integration import PQCKeyManager

class TestPQCKeyManager:
    """Test suite for PQC key operations."""
    
    def test_keygen_ml_kem_768(self):
        """Test ML-KEM-768 key generation."""
        km = PQCKeyManager()
        public_key, secret_key = km.generate_keypair("ML-KEM-768")
        
        assert len(public_key) == 1184
        assert len(secret_key) == 2400
    
    @pytest.mark.slow
    def test_keygen_all_algorithms(self):
        """Test all supported algorithms (slow)."""
        ...
    
    @pytest.mark.integration
    def test_end_to_end_encryption(self):
        """Integration test for full encryption flow."""
        ...
```

### Test Coverage

- Aim for >80% coverage on core modules
- All public APIs must have tests
- Include edge cases and error conditions

---

## Security Considerations

### Cryptographic Code

When contributing cryptographic code:

1. **Never implement your own crypto** - Use established libraries
2. **Follow standards** - Reference NIST FIPS documents
3. **Constant-time operations** - Avoid timing side channels
4. **Secure memory** - Clear sensitive data after use
5. **Input validation** - Validate all cryptographic inputs

### Key Handling

```python
# GOOD: Clear sensitive data
def process_secret_key(secret_key: bytes) -> bytes:
    try:
        result = do_operation(secret_key)
        return result
    finally:
        # Clear sensitive data (Python limitation: best effort)
        secret_key = b'\x00' * len(secret_key)

# BAD: Logging sensitive data
logger.debug(f"Using key: {secret_key}")  # NEVER DO THIS
```

### Reporting Vulnerabilities

If you discover a security vulnerability:

1. **Do NOT open a public issue**
2. Email security concerns privately
3. Include detailed reproduction steps
4. Allow time for a fix before disclosure

---

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: `pytest tests/ -v`
2. **Check formatting**: `black --check . && isort --check-only .`
3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Update CHANGELOG.md** with your changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement
- [ ] Security fix

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Coverage maintained/improved

## Security
- [ ] No sensitive data exposed
- [ ] Cryptographic best practices followed
- [ ] Input validation added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Submit PR against `develop` branch
2. Automated CI checks must pass
3. At least one maintainer approval required
4. Security-related PRs require additional review
5. Squash and merge preferred

---

## Questions?

- Open a GitHub Discussion for questions
- Tag maintainers for urgent issues
- Check existing issues before creating new ones

Thank you for contributing to quantum-safe cryptography!
