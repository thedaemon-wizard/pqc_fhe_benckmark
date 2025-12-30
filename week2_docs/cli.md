# CLI Reference

Complete reference for the `pqc-fhe` command-line interface.

## Installation

```bash
pip install pqc-fhe-integration
```

## Global Options

```
pqc-fhe [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --version  Show version and exit
  --json         Output results in JSON format
  -q, --quiet    Suppress non-essential output
  --help         Show help message and exit
```

## Commands

### keygen

Generate post-quantum cryptographic key pairs.

```bash
pqc-fhe keygen [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-a, --algorithm` | `ML-KEM-768` | Algorithm to use |
| `-o, --output` | `./keys` | Output directory |
| `-f, --format` | `json` | Output format (json, pem, binary) |

**Examples:**

```bash
# Generate ML-KEM-768 keypair
pqc-fhe keygen --algorithm ML-KEM-768 --output ./my-keys

# Generate ML-DSA-65 signing keypair in PEM format
pqc-fhe keygen --algorithm ML-DSA-65 --format pem --output ./signing-keys

# JSON output
pqc-fhe --json keygen --algorithm ML-KEM-1024
```

**Supported Algorithms:**

KEM:
- `ML-KEM-512` - NIST Security Level 1
- `ML-KEM-768` - NIST Security Level 3 (Recommended)
- `ML-KEM-1024` - NIST Security Level 5

Signatures:
- `ML-DSA-44` - NIST Security Level 2
- `ML-DSA-65` - NIST Security Level 3 (Recommended)
- `ML-DSA-87` - NIST Security Level 5
- `SLH-DSA-SHA2-128s`, `SLH-DSA-SHA2-128f`
- `SLH-DSA-SHA2-192s`, `SLH-DSA-SHA2-192f`
- `SLH-DSA-SHA2-256s`, `SLH-DSA-SHA2-256f`

---

### encapsulate

Encapsulate a shared secret using KEM.

```bash
pqc-fhe encapsulate [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --public-key` | (required) | Path to public key file |
| `-a, --algorithm` | auto-detect | KEM algorithm |
| `-o, --output` | `./encapsulation` | Output directory |

**Examples:**

```bash
# Encapsulate with ML-KEM-768
pqc-fhe encapsulate --public-key ./keys/abc123_public.json \
  --output ./encap-result

# With explicit algorithm
pqc-fhe encapsulate -p ./keys/pub.bin -a ML-KEM-1024 -o ./output
```

**Output Files:**

- `ciphertext.bin` - The KEM ciphertext
- `shared_secret.bin` - The derived shared secret (32 bytes)

---

### decapsulate

Decapsulate a shared secret using KEM.

```bash
pqc-fhe decapsulate [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --secret-key` | (required) | Path to secret key file |
| `-c, --ciphertext` | (required) | Path to ciphertext file |
| `-a, --algorithm` | `ML-KEM-768` | KEM algorithm |
| `-o, --output` | `./shared_secret.bin` | Output file |

**Examples:**

```bash
pqc-fhe decapsulate \
  --secret-key ./keys/abc123_secret.json \
  --ciphertext ./encap-result/ciphertext.bin \
  --output ./shared_secret.bin
```

---

### sign

Generate a digital signature.

```bash
pqc-fhe sign [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --secret-key` | (required) | Path to signing secret key |
| `-m, --message` | (required) | Message to sign (file or string) |
| `-a, --algorithm` | `ML-DSA-65` | Signature algorithm |
| `-o, --output` | `./signature.bin` | Output file for signature |

**Examples:**

```bash
# Sign a file
pqc-fhe sign \
  --secret-key ./signing-keys/abc123_secret.json \
  --message ./document.pdf \
  --output ./document.sig

# Sign a string
pqc-fhe sign \
  --secret-key ./keys/secret.json \
  --message "Hello, World!" \
  --output ./message.sig
```

---

### verify

Verify a digital signature.

```bash
pqc-fhe verify [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --public-key` | (required) | Path to public key file |
| `-m, --message` | (required) | Original message (file or string) |
| `-S, --signature` | (required) | Path to signature file |
| `-a, --algorithm` | `ML-DSA-65` | Signature algorithm |

**Examples:**

```bash
# Verify a file signature
pqc-fhe verify \
  --public-key ./signing-keys/abc123_public.json \
  --message ./document.pdf \
  --signature ./document.sig

# Exit code: 0 = valid, 1 = invalid
```

**Output:**

Returns exit code `0` if valid, `1` if invalid.

---

### fhe-encrypt

Encrypt data using FHE.

```bash
pqc-fhe fhe-encrypt [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | - | Input file (JSON array) |
| `--values` | - | Values to encrypt (space-separated) |
| `-o, --output` | `./ciphertext.json` | Output file |

**Examples:**

```bash
# Encrypt values directly
pqc-fhe fhe-encrypt --values 1.5 2.5 3.5 4.5 --output ./ct.json

# Encrypt from file
echo '[100.0, 200.0, 300.0]' > data.json
pqc-fhe fhe-encrypt --input data.json --output ./ct.json
```

---

### fhe-decrypt

Decrypt FHE ciphertext.

```bash
pqc-fhe fhe-decrypt [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | Input ciphertext file |
| `-o, --output` | - | Output file (optional) |

**Examples:**

```bash
# Decrypt and display
pqc-fhe fhe-decrypt --input ./ct.json

# Decrypt to file
pqc-fhe fhe-decrypt --input ./ct.json --output ./decrypted.json
```

---

### fhe-compute

Perform homomorphic computation.

```bash
pqc-fhe fhe-compute [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-op, --operation` | (required) | Operation to perform |
| `-i1, --input1` | (required) | First input ciphertext |
| `-i2, --input2` | - | Second input (for binary ops) |
| `--scalar` | - | Scalar value (for scalar ops) |
| `-o, --output` | `./result.json` | Output file |

**Operations:**

| Operation | Description | Requires |
|-----------|-------------|----------|
| `add` | Add two ciphertexts | `--input2` |
| `subtract` | Subtract ciphertexts | `--input2` |
| `multiply` | Multiply ciphertexts | `--input2` |
| `negate` | Negate ciphertext | - |
| `square` | Square ciphertext | - |
| `add_scalar` | Add scalar to ciphertext | `--scalar` |
| `multiply_scalar` | Multiply by scalar | `--scalar` |

**Examples:**

```bash
# Add two ciphertexts
pqc-fhe fhe-compute --operation add \
  --input1 ./ct1.json \
  --input2 ./ct2.json \
  --output ./sum.json

# Multiply by scalar
pqc-fhe fhe-compute --operation multiply_scalar \
  --input1 ./ct.json \
  --scalar 2.5 \
  --output ./scaled.json

# Square a ciphertext
pqc-fhe fhe-compute --operation square \
  --input1 ./ct.json \
  --output ./squared.json
```

---

### benchmark

Run performance benchmarks.

```bash
pqc-fhe benchmark [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --suite` | `quick` | Benchmark suite (quick, full) |
| `-n, --iterations` | `10` | Number of iterations |
| `-o, --output` | `./benchmark_results` | Output directory |

**Examples:**

```bash
# Quick benchmark
pqc-fhe benchmark --suite quick --iterations 10

# Full benchmark with JSON output
pqc-fhe --json benchmark --suite full --iterations 100 --output ./results
```

---

### info

Display system information.

```bash
pqc-fhe info
```

**Output:**

```
============================================================
PQC-FHE CLI v1.0.0
============================================================

Cryptographic Engines:
  PQC Available: Yes
  FHE Available: Yes

Supported KEM Algorithms:
  - ML-KEM-512
  - ML-KEM-768
  - ML-KEM-1024

Supported Signature Algorithms:
  - ML-DSA-44
  - ML-DSA-65
  - ML-DSA-87
  - SLH-DSA-SHA2-128s
  ...
============================================================
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error or verification failed |
| 2 | Invalid arguments |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PQC_FHE_LOG_LEVEL` | Set logging level (DEBUG, INFO, WARNING, ERROR) |
| `PQC_FHE_KEY_DIR` | Default key directory |
| `PQC_FHE_CONFIG` | Path to config file |

## Shell Completion

### Bash

```bash
# Add to ~/.bashrc
eval "$(_PQC_FHE_COMPLETE=bash_source pqc-fhe)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_PQC_FHE_COMPLETE=zsh_source pqc-fhe)"
```

## Scripting Examples

### Key Rotation Script

```bash
#!/bin/bash
# rotate-keys.sh

ALGO=${1:-ML-KEM-768}
OUTPUT_DIR=${2:-./keys}
BACKUP_DIR=${OUTPUT_DIR}/backup/$(date +%Y%m%d)

# Backup old keys
mkdir -p "$BACKUP_DIR"
cp "$OUTPUT_DIR"/*.json "$BACKUP_DIR/" 2>/dev/null

# Generate new keys
pqc-fhe keygen --algorithm "$ALGO" --output "$OUTPUT_DIR"

echo "Keys rotated. Backup at: $BACKUP_DIR"
```

### Batch Encryption Script

```bash
#!/bin/bash
# batch-encrypt.sh

INPUT_DIR=${1:-.}
OUTPUT_DIR=${2:-./encrypted}

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.json; do
  [ -f "$file" ] || continue
  name=$(basename "$file" .json)
  pqc-fhe fhe-encrypt --input "$file" --output "$OUTPUT_DIR/${name}_encrypted.json"
  echo "Encrypted: $file"
done
```
