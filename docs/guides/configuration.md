# Configuration Guide

This guide covers all configuration options for the PQC-FHE Integration library.

## Environment Variables

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PQC_FHE_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PQC_FHE_KEY_DIR` | `./keys` | Default directory for key storage |
| `PQC_FHE_CACHE_DIR` | `./cache` | Cache directory for FHE contexts |

### PQC Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PQC_DEFAULT_KEM` | `ML-KEM-768` | Default KEM algorithm |
| `PQC_DEFAULT_SIG` | `ML-DSA-65` | Default signature algorithm |
| `PQC_KEY_ROTATION_DAYS` | `30` | Key rotation interval in days |

### FHE Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FHE_LOG_N` | `15` | Polynomial ring dimension (2^N) |
| `FHE_SCALE_BITS` | `40` | Scale factor bits |
| `FHE_FIRST_MOD_SIZE` | `60` | First modulus size |
| `FHE_SECURITY_LEVEL` | `128` | Security level in bits |

### API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `API_WORKERS` | `4` | Number of worker processes |
| `API_RATE_LIMIT` | `100` | Requests per minute |
| `WS_PORT` | `8765` | WebSocket server port |

## Configuration File

Create a `config.yaml` file for advanced configuration:

```yaml
# config.yaml
pqc:
  default_kem_algorithm: ML-KEM-768
  default_signature_algorithm: ML-DSA-65
  key_rotation_interval_days: 30
  supported_kem_algorithms:
    - ML-KEM-512
    - ML-KEM-768
    - ML-KEM-1024
  supported_signature_algorithms:
    - ML-DSA-44
    - ML-DSA-65
    - ML-DSA-87
    - SLH-DSA-SHA2-128s
    - SLH-DSA-SHA2-128f

fhe:
  scheme: CKKS
  parameters:
    log_n: 15
    scale_bits: 40
    first_mod_size: 60
    security_level: 128
  bootstrap:
    enabled: false
    precision: 20

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors:
    enabled: true
    origins:
      - "*"
  rate_limiting:
    enabled: true
    requests_per_minute: 100
  tls:
    enabled: true
    cert_file: /path/to/cert.pem
    key_file: /path/to/key.pem

websocket:
  host: 0.0.0.0
  port: 8765
  max_connections: 1000
  ping_interval: 30

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: /var/log/pqc-fhe/app.log
  rotation:
    max_size_mb: 100
    backup_count: 10

storage:
  keys_dir: /var/lib/pqc-fhe/keys
  cache_dir: /var/cache/pqc-fhe
  encryption: true

monitoring:
  prometheus:
    enabled: true
    port: 9090
  metrics:
    - operation_latency
    - operation_count
    - error_count
    - key_generation_count
```

## Loading Configuration

### Python API

```python
from pqc_fhe_integration import PQCKeyManager, FHEEngine
from pqc_fhe_integration.config import load_config

# Load from file
config = load_config("config.yaml")

# Initialize with config
pqc = PQCKeyManager(config=config.pqc)
fhe = FHEEngine(config=config.fhe)
```

### CLI

```bash
# Use custom config file
pqc-fhe --config /path/to/config.yaml keygen --algorithm ML-KEM-768

# Override specific settings
PQC_DEFAULT_KEM=ML-KEM-1024 pqc-fhe keygen
```

### Docker

```bash
# Mount config file
docker run -v /path/to/config.yaml:/app/config.yaml \
  -e PQC_FHE_CONFIG=/app/config.yaml \
  pqc-fhe/integration:latest
```

### Kubernetes (Helm)

```yaml
# values.yaml
config:
  pqc:
    defaultKemAlgorithm: ML-KEM-768
    defaultSignatureAlgorithm: ML-DSA-65
  fhe:
    logN: 15
    scaleBits: 40
  api:
    replicas: 3
    resources:
      requests:
        memory: "512Mi"
        cpu: "500m"
      limits:
        memory: "2Gi"
        cpu: "2000m"
```

## Security Configuration

### TLS Configuration

```yaml
tls:
  min_version: "1.3"
  cipher_suites:
    - TLS_AES_256_GCM_SHA384
    - TLS_CHACHA20_POLY1305_SHA256
  cert_file: /etc/ssl/certs/server.crt
  key_file: /etc/ssl/private/server.key
  client_auth: require  # none, request, require
  ca_file: /etc/ssl/certs/ca.crt
```

### Key Management

```yaml
key_management:
  storage:
    type: file  # file, vault, aws_kms, azure_keyvault
    path: /var/lib/pqc-fhe/keys
    encryption: true
  rotation:
    enabled: true
    interval_days: 30
    retention_count: 5
  backup:
    enabled: true
    schedule: "0 2 * * *"  # 2 AM daily
    destination: s3://backup-bucket/keys
```

## Performance Tuning

### FHE Optimization

```yaml
fhe:
  # Use larger polynomial for more operations
  parameters:
    log_n: 16  # 2^16 = 65536 slots
    
  # Enable GPU acceleration
  gpu:
    enabled: true
    device_id: 0
    
  # Threading
  threading:
    num_threads: 8
    
  # Memory management
  memory:
    max_heap_gb: 16
    cache_size_mb: 512
```

### API Performance

```yaml
api:
  # Connection pooling
  connection_pool:
    min_size: 10
    max_size: 100
    
  # Request handling
  request:
    timeout_seconds: 30
    max_body_size_mb: 10
    
  # Caching
  cache:
    enabled: true
    ttl_seconds: 300
    max_entries: 10000
```

## Monitoring Configuration

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: /metrics
    
  health_check:
    enabled: true
    path: /health
    interval_seconds: 10
    
  tracing:
    enabled: true
    exporter: jaeger
    endpoint: http://jaeger:14268/api/traces
    
  logging:
    structured: true
    include_request_id: true
```

## Example Configurations

### Development

```yaml
# config.dev.yaml
logging:
  level: DEBUG
  
api:
  workers: 1
  cors:
    origins: ["*"]
  rate_limiting:
    enabled: false
```

### Production

```yaml
# config.prod.yaml
logging:
  level: WARNING
  file: /var/log/pqc-fhe/app.log
  
api:
  workers: 8
  tls:
    enabled: true
  rate_limiting:
    enabled: true
    requests_per_minute: 1000

fhe:
  gpu:
    enabled: true
```

## Next Steps

- [API Reference](../api/overview.md) - Explore the API documentation
- [Security Best Practices](../security/best_practices.md) - Security hardening
- [Deployment Guide](../development/architecture.md) - Production deployment
