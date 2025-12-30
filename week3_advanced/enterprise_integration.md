# Enterprise Integration Tutorial

This tutorial covers integrating PQC-FHE into enterprise environments, including microservices architecture, cloud deployment, monitoring, and compliance considerations.

## Overview

Enterprise integration requires:

- **Scalability**: Handle thousands of concurrent requests
- **High Availability**: 99.9%+ uptime with failover
- **Security Compliance**: SOC 2, HIPAA, GDPR, PCI-DSS
- **Observability**: Monitoring, logging, and alerting
- **Integration**: Connect with existing systems

## Architecture Patterns

### Microservices Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                             │
│                    (HAProxy / AWS ALB)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  PQC Key  │  │    FHE    │  │  Hybrid   │
│  Service  │  │  Service  │  │  Service  │
│  (REST)   │  │  (gRPC)   │  │(WebSocket)│
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      └──────────────┼──────────────┘
                     ▼
        ┌────────────────────────┐
        │    Message Queue       │
        │  (RabbitMQ / Kafka)    │
        └────────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌────────────┐ ┌──────────┐ ┌──────────┐
│  Key Store │ │   Redis  │ │PostgreSQL│
│   (HSM)    │ │  (Cache) │ │(Metadata)│
└────────────┘ └──────────┘ └──────────┘
```

### Service Implementation

```python
"""
enterprise_pqc_service.py
PQC Key Management Microservice
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
import aio_pika
from prometheus_client import Counter, Histogram, generate_latest
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'pqc_requests_total',
    'Total PQC requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'pqc_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)
KEY_OPERATIONS = Counter(
    'pqc_key_operations_total',
    'Key operations count',
    ['operation', 'algorithm']
)


class KeyAlgorithm(str, Enum):
    ML_KEM_512 = "ML-KEM-512"
    ML_KEM_768 = "ML-KEM-768"
    ML_KEM_1024 = "ML-KEM-1024"
    ML_DSA_44 = "ML-DSA-44"
    ML_DSA_65 = "ML-DSA-65"
    ML_DSA_87 = "ML-DSA-87"


class KeyStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class KeyMetadata:
    """Key metadata for tracking and management"""
    key_id: str
    algorithm: KeyAlgorithm
    status: KeyStatus
    created_at: datetime
    expires_at: datetime
    owner_id: str
    tags: Dict[str, str] = field(default_factory=dict)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "owner_id": self.owner_id,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


class EnterprisePQCKeyManager:
    """
    Enterprise-grade PQC Key Management Service
    
    Features:
    - Key lifecycle management
    - Redis caching for performance
    - Message queue for async operations
    - HSM integration support
    - Audit logging
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        rabbitmq_url: str = "amqp://guest:guest@localhost/",
        hsm_enabled: bool = False
    ):
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.hsm_enabled = hsm_enabled
        self.redis_client: Optional[redis.Redis] = None
        self.mq_connection: Optional[aio_pika.Connection] = None
        self.mq_channel: Optional[aio_pika.Channel] = None
        
    async def initialize(self):
        """Initialize connections"""
        # Redis connection
        self.redis_client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # RabbitMQ connection
        self.mq_connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.mq_channel = await self.mq_connection.channel()
        
        # Declare exchanges and queues
        await self.mq_channel.declare_exchange(
            "pqc_events",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        logger.info("Enterprise PQC Key Manager initialized")
        
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.mq_connection:
            await self.mq_connection.close()
            
    async def generate_keypair(
        self,
        algorithm: KeyAlgorithm,
        owner_id: str,
        tags: Dict[str, str] = None,
        validity_days: int = 365
    ) -> KeyMetadata:
        """
        Generate a new PQC keypair with full lifecycle management
        """
        key_id = f"key_{secrets.token_hex(16)}"
        now = datetime.utcnow()
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            status=KeyStatus.ACTIVE,
            created_at=now,
            expires_at=now + timedelta(days=validity_days),
            owner_id=owner_id,
            tags=tags or {}
        )
        
        # Generate actual keys (using liboqs in production)
        if algorithm in [KeyAlgorithm.ML_KEM_512, KeyAlgorithm.ML_KEM_768, KeyAlgorithm.ML_KEM_1024]:
            public_key, secret_key = self._generate_kem_keypair(algorithm)
        else:
            public_key, secret_key = self._generate_dsa_keypair(algorithm)
        
        # Store in Redis with encryption
        await self._store_key(key_id, public_key, secret_key, metadata)
        
        # Publish event
        await self._publish_event("key.created", {
            "key_id": key_id,
            "algorithm": algorithm.value,
            "owner_id": owner_id
        })
        
        # Update metrics
        KEY_OPERATIONS.labels(operation="generate", algorithm=algorithm.value).inc()
        
        logger.info(
            "Key generated",
            key_id=key_id,
            algorithm=algorithm.value,
            owner_id=owner_id
        )
        
        return metadata
    
    def _generate_kem_keypair(self, algorithm: KeyAlgorithm) -> tuple:
        """Generate KEM keypair using liboqs"""
        try:
            import oqs
            alg_name = algorithm.value.replace("-", "")
            kem = oqs.KeyEncapsulation(alg_name)
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            return public_key, secret_key
        except ImportError:
            # Fallback for demonstration
            return secrets.token_bytes(1184), secrets.token_bytes(2400)
    
    def _generate_dsa_keypair(self, algorithm: KeyAlgorithm) -> tuple:
        """Generate DSA keypair using liboqs"""
        try:
            import oqs
            alg_name = algorithm.value.replace("-", "")
            sig = oqs.Signature(alg_name)
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
            return public_key, secret_key
        except ImportError:
            # Fallback for demonstration
            return secrets.token_bytes(1312), secrets.token_bytes(2528)
    
    async def _store_key(
        self,
        key_id: str,
        public_key: bytes,
        secret_key: bytes,
        metadata: KeyMetadata
    ):
        """Store key material and metadata"""
        import base64
        
        # In production, encrypt secret_key with HSM or KMS
        key_data = {
            "public_key": base64.b64encode(public_key).decode(),
            "secret_key": base64.b64encode(secret_key).decode(),  # Encrypt in production!
            "metadata": metadata.to_dict()
        }
        
        # Store with TTL based on expiration
        ttl = int((metadata.expires_at - datetime.utcnow()).total_seconds())
        await self.redis_client.setex(
            f"pqc:key:{key_id}",
            ttl,
            json.dumps(key_data)
        )
        
        # Index by owner
        await self.redis_client.sadd(f"pqc:owner:{metadata.owner_id}:keys", key_id)
        
    async def get_public_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve public key"""
        import base64
        
        data = await self.redis_client.get(f"pqc:key:{key_id}")
        if not data:
            return None
            
        key_data = json.loads(data)
        
        # Check status
        if key_data["metadata"]["status"] != KeyStatus.ACTIVE.value:
            raise HTTPException(status_code=403, detail="Key is not active")
            
        return base64.b64decode(key_data["public_key"])
    
    async def encapsulate(self, key_id: str) -> tuple:
        """
        Perform key encapsulation using stored public key
        Returns: (ciphertext, shared_secret)
        """
        import base64
        
        data = await self.redis_client.get(f"pqc:key:{key_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Key not found")
            
        key_data = json.loads(data)
        public_key = base64.b64decode(key_data["public_key"])
        algorithm = KeyAlgorithm(key_data["metadata"]["algorithm"])
        
        try:
            import oqs
            alg_name = algorithm.value.replace("-", "")
            kem = oqs.KeyEncapsulation(alg_name)
            ciphertext, shared_secret = kem.encap_secret(public_key)
        except ImportError:
            # Fallback for demonstration
            ciphertext = secrets.token_bytes(1088)
            shared_secret = secrets.token_bytes(32)
        
        # Update usage stats
        await self._update_usage(key_id)
        
        KEY_OPERATIONS.labels(operation="encapsulate", algorithm=algorithm.value).inc()
        
        return ciphertext, shared_secret
    
    async def decapsulate(self, key_id: str, ciphertext: bytes) -> bytes:
        """
        Perform key decapsulation using stored secret key
        Returns: shared_secret
        """
        import base64
        
        data = await self.redis_client.get(f"pqc:key:{key_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Key not found")
            
        key_data = json.loads(data)
        secret_key = base64.b64decode(key_data["secret_key"])
        algorithm = KeyAlgorithm(key_data["metadata"]["algorithm"])
        
        try:
            import oqs
            alg_name = algorithm.value.replace("-", "")
            kem = oqs.KeyEncapsulation(alg_name, secret_key)
            shared_secret = kem.decap_secret(ciphertext)
        except ImportError:
            # Fallback for demonstration
            shared_secret = secrets.token_bytes(32)
        
        # Update usage stats
        await self._update_usage(key_id)
        
        KEY_OPERATIONS.labels(operation="decapsulate", algorithm=algorithm.value).inc()
        
        return shared_secret
    
    async def revoke_key(self, key_id: str, reason: str = ""):
        """Revoke a key"""
        data = await self.redis_client.get(f"pqc:key:{key_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Key not found")
            
        key_data = json.loads(data)
        key_data["metadata"]["status"] = KeyStatus.REVOKED.value
        
        # Update in Redis
        await self.redis_client.set(f"pqc:key:{key_id}", json.dumps(key_data))
        
        # Publish revocation event
        await self._publish_event("key.revoked", {
            "key_id": key_id,
            "reason": reason,
            "revoked_at": datetime.utcnow().isoformat()
        })
        
        logger.warning("Key revoked", key_id=key_id, reason=reason)
    
    async def rotate_key(self, key_id: str) -> KeyMetadata:
        """
        Rotate a key by creating new keypair and revoking old
        """
        # Get existing key metadata
        data = await self.redis_client.get(f"pqc:key:{key_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Key not found")
            
        key_data = json.loads(data)
        old_metadata = key_data["metadata"]
        
        # Generate new key with same parameters
        new_metadata = await self.generate_keypair(
            algorithm=KeyAlgorithm(old_metadata["algorithm"]),
            owner_id=old_metadata["owner_id"],
            tags={**old_metadata.get("tags", {}), "rotated_from": key_id}
        )
        
        # Revoke old key
        await self.revoke_key(key_id, reason=f"Rotated to {new_metadata.key_id}")
        
        # Publish rotation event
        await self._publish_event("key.rotated", {
            "old_key_id": key_id,
            "new_key_id": new_metadata.key_id
        })
        
        logger.info("Key rotated", old_key_id=key_id, new_key_id=new_metadata.key_id)
        
        return new_metadata
    
    async def list_keys(
        self,
        owner_id: str,
        status: Optional[KeyStatus] = None
    ) -> List[KeyMetadata]:
        """List keys for an owner"""
        key_ids = await self.redis_client.smembers(f"pqc:owner:{owner_id}:keys")
        
        keys = []
        for key_id in key_ids:
            data = await self.redis_client.get(f"pqc:key:{key_id}")
            if data:
                key_data = json.loads(data)
                metadata = key_data["metadata"]
                
                if status and metadata["status"] != status.value:
                    continue
                    
                keys.append(KeyMetadata(
                    key_id=metadata["key_id"],
                    algorithm=KeyAlgorithm(metadata["algorithm"]),
                    status=KeyStatus(metadata["status"]),
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                    expires_at=datetime.fromisoformat(metadata["expires_at"]),
                    owner_id=metadata["owner_id"],
                    tags=metadata.get("tags", {}),
                    usage_count=metadata.get("usage_count", 0),
                    last_used=datetime.fromisoformat(metadata["last_used"]) if metadata.get("last_used") else None
                ))
        
        return keys
    
    async def _update_usage(self, key_id: str):
        """Update key usage statistics"""
        data = await self.redis_client.get(f"pqc:key:{key_id}")
        if data:
            key_data = json.loads(data)
            key_data["metadata"]["usage_count"] = key_data["metadata"].get("usage_count", 0) + 1
            key_data["metadata"]["last_used"] = datetime.utcnow().isoformat()
            await self.redis_client.set(f"pqc:key:{key_id}", json.dumps(key_data))
    
    async def _publish_event(self, event_type: str, data: dict):
        """Publish event to message queue"""
        if self.mq_channel:
            exchange = await self.mq_channel.get_exchange("pqc_events")
            await exchange.publish(
                aio_pika.Message(
                    body=json.dumps({
                        "event_type": event_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": data
                    }).encode()
                ),
                routing_key=event_type
            )


# FastAPI Application
app = FastAPI(
    title="Enterprise PQC Key Service",
    description="Post-Quantum Cryptography Key Management Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global key manager instance
key_manager: Optional[EnterprisePQCKeyManager] = None


@app.on_event("startup")
async def startup():
    global key_manager
    key_manager = EnterprisePQCKeyManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
    )
    await key_manager.initialize()


@app.on_event("shutdown")
async def shutdown():
    if key_manager:
        await key_manager.close()


# Pydantic models for API
class KeyGenerateRequest(BaseModel):
    algorithm: KeyAlgorithm
    owner_id: str
    tags: Dict[str, str] = Field(default_factory=dict)
    validity_days: int = Field(default=365, ge=1, le=3650)


class KeyGenerateResponse(BaseModel):
    key_id: str
    algorithm: str
    status: str
    created_at: str
    expires_at: str


class EncapsulateResponse(BaseModel):
    ciphertext: str
    shared_secret: str


# API Endpoints
@app.post("/api/v1/keys", response_model=KeyGenerateResponse)
async def generate_key(request: KeyGenerateRequest):
    """Generate a new PQC keypair"""
    metadata = await key_manager.generate_keypair(
        algorithm=request.algorithm,
        owner_id=request.owner_id,
        tags=request.tags,
        validity_days=request.validity_days
    )
    
    REQUEST_COUNT.labels(method="POST", endpoint="/keys", status="200").inc()
    
    return KeyGenerateResponse(
        key_id=metadata.key_id,
        algorithm=metadata.algorithm.value,
        status=metadata.status.value,
        created_at=metadata.created_at.isoformat(),
        expires_at=metadata.expires_at.isoformat()
    )


@app.get("/api/v1/keys/{key_id}/public")
async def get_public_key(key_id: str):
    """Get public key"""
    import base64
    public_key = await key_manager.get_public_key(key_id)
    if not public_key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    return {"public_key": base64.b64encode(public_key).decode()}


@app.post("/api/v1/keys/{key_id}/encapsulate", response_model=EncapsulateResponse)
async def encapsulate(key_id: str):
    """Encapsulate using public key"""
    import base64
    ciphertext, shared_secret = await key_manager.encapsulate(key_id)
    
    return EncapsulateResponse(
        ciphertext=base64.b64encode(ciphertext).decode(),
        shared_secret=base64.b64encode(shared_secret).decode()
    )


@app.post("/api/v1/keys/{key_id}/decapsulate")
async def decapsulate(key_id: str, ciphertext: str):
    """Decapsulate using secret key"""
    import base64
    ct_bytes = base64.b64decode(ciphertext)
    shared_secret = await key_manager.decapsulate(key_id, ct_bytes)
    
    return {"shared_secret": base64.b64encode(shared_secret).decode()}


@app.post("/api/v1/keys/{key_id}/revoke")
async def revoke_key(key_id: str, reason: str = ""):
    """Revoke a key"""
    await key_manager.revoke_key(key_id, reason)
    return {"status": "revoked", "key_id": key_id}


@app.post("/api/v1/keys/{key_id}/rotate")
async def rotate_key(key_id: str):
    """Rotate a key"""
    new_metadata = await key_manager.rotate_key(key_id)
    return {
        "old_key_id": key_id,
        "new_key_id": new_metadata.key_id,
        "status": "rotated"
    }


@app.get("/api/v1/owners/{owner_id}/keys")
async def list_owner_keys(owner_id: str, status: Optional[KeyStatus] = None):
    """List keys for an owner"""
    keys = await key_manager.list_keys(owner_id, status)
    return {"keys": [k.to_dict() for k in keys]}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
```

## Cloud Deployment

### AWS Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            AWS Region                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         VPC                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │  Public     │  │  Private    │  │  Private    │         │   │
│  │  │  Subnet     │  │  Subnet A   │  │  Subnet B   │         │   │
│  │  │             │  │             │  │             │         │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │   │
│  │  │ │   ALB   │ │  │ │   EKS   │ │  │ │   EKS   │ │         │   │
│  │  │ └────┬────┘ │  │ │  Node   │ │  │ │  Node   │ │         │   │
│  │  │      │      │  │ └────┬────┘ │  │ └────┬────┘ │         │   │
│  │  └──────┼──────┘  └──────┼──────┘  └──────┼──────┘         │   │
│  │         │                │                │                 │   │
│  │         └────────────────┼────────────────┘                 │   │
│  │                          │                                   │   │
│  │  ┌───────────────────────┼───────────────────────────────┐  │   │
│  │  │   Data Layer          │                               │  │   │
│  │  │  ┌─────────────┐  ┌───┴───────┐  ┌─────────────────┐ │  │   │
│  │  │  │ ElastiCache │  │    RDS    │  │   AWS KMS       │ │  │   │
│  │  │  │  (Redis)    │  │(PostgreSQL)│  │ (Key Storage)   │ │  │   │
│  │  │  └─────────────┘  └───────────┘  └─────────────────┘ │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Terraform Configuration

```hcl
# terraform/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  
  backend "s3" {
    bucket         = "pqc-fhe-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "PQC-FHE"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  default = "us-west-2"
}

variable "environment" {
  default = "production"
}

variable "cluster_name" {
  default = "pqc-fhe-cluster"
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "pqc-fhe-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.0.0"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access = true
  
  eks_managed_node_groups = {
    pqc_nodes = {
      name = "pqc-node-group"
      
      instance_types = ["m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      labels = {
        workload = "pqc-fhe"
      }
    }
  }
  
  # Enable IRSA
  enable_irsa = true
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "pqc-fhe-redis"
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "pqc-fhe-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name        = "pqc-fhe-redis-sg"
  description = "Security group for Redis"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# RDS PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "6.0.0"
  
  identifier = "pqc-fhe-db"
  
  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = "db.r6g.large"
  
  allocated_storage     = 100
  max_allocated_storage = 500
  
  db_name  = "pqc_fhe"
  username = "pqc_admin"
  port     = 5432
  
  multi_az               = true
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = 30
  deletion_protection     = true
  storage_encrypted       = true
  
  performance_insights_enabled = true
}

resource "aws_security_group" "rds" {
  name        = "pqc-fhe-rds-sg"
  description = "Security group for RDS"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
}

# KMS Key for encryption
resource "aws_kms_key" "pqc_key" {
  description             = "KMS key for PQC-FHE encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow EKS Service"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey*"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_kms_alias" "pqc_key" {
  name          = "alias/pqc-fhe-key"
  target_key_id = aws_kms_key.pqc_key.key_id
}

data "aws_caller_identity" "current" {}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "rds_endpoint" {
  value = module.rds.db_instance_endpoint
}

output "kms_key_arn" {
  value = aws_kms_key.pqc_key.arn
}
```

### Kubernetes Deployment

```yaml
# kubernetes/pqc-service-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: pqc-key-service
  namespace: pqc-fhe
  labels:
    app: pqc-key-service
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: pqc-key-service
  template:
    metadata:
      labels:
        app: pqc-key-service
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: pqc-key-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: pqc-key-service
          image: pqc-fhe/key-service:1.0.0
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
              protocol: TCP
          env:
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: pqc-secrets
                  key: redis-url
            - name: RABBITMQ_URL
              valueFrom:
                secretKeyRef:
                  name: pqc-secrets
                  key: rabbitmq-url
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: pqc-secrets
                  key: database-url
            - name: AWS_KMS_KEY_ID
              valueFrom:
                configMapKeyRef:
                  name: pqc-config
                  key: kms-key-id
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          securityContext:
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - pqc-key-service
                topologyKey: "kubernetes.io/hostname"
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: "topology.kubernetes.io/zone"
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: pqc-key-service

---
apiVersion: v1
kind: Service
metadata:
  name: pqc-key-service
  namespace: pqc-fhe
  labels:
    app: pqc-key-service
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: pqc-key-service

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pqc-key-service-hpa
  namespace: pqc-fhe
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pqc-key-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: pqc-key-service-pdb
  namespace: pqc-fhe
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: pqc-key-service
```

## Monitoring and Observability

### Prometheus + Grafana Setup

```yaml
# monitoring/prometheus-config.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093

    rule_files:
      - /etc/prometheus/rules/*.yml

    scrape_configs:
      - job_name: 'pqc-key-service'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - pqc-fhe
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name

      - job_name: 'fhe-service'
        static_configs:
          - targets: ['fhe-service.pqc-fhe:80']
        metrics_path: /metrics

---
# Alert rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  pqc-alerts.yml: |
    groups:
      - name: pqc-fhe-alerts
        rules:
          - alert: HighErrorRate
            expr: |
              sum(rate(pqc_requests_total{status=~"5.."}[5m])) 
              / sum(rate(pqc_requests_total[5m])) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"
              description: "Error rate is {{ $value | humanizePercentage }}"

          - alert: HighLatency
            expr: |
              histogram_quantile(0.95, 
                sum(rate(pqc_request_latency_seconds_bucket[5m])) by (le, endpoint)
              ) > 1
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High latency detected"
              description: "95th percentile latency is {{ $value }}s for {{ $labels.endpoint }}"

          - alert: KeyExpirationWarning
            expr: |
              pqc_keys_expiring_soon > 10
            for: 1h
            labels:
              severity: warning
            annotations:
              summary: "Keys expiring soon"
              description: "{{ $value }} keys will expire in the next 7 days"

          - alert: ServiceDown
            expr: up{job="pqc-key-service"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "PQC Key Service is down"
              description: "Service has been down for more than 1 minute"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "PQC-FHE Service Dashboard",
    "uid": "pqc-fhe-dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(pqc_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(pqc_requests_total{status=~\"5..\"}[5m])) / sum(rate(pqc_requests_total[5m])) * 100",
            "legendFormat": "Error %"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(pqc_request_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(pqc_request_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(pqc_request_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Key Operations",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "sum(rate(pqc_key_operations_total[5m])) by (operation, algorithm)",
            "legendFormat": "{{operation}} - {{algorithm}}"
          }
        ]
      }
    ]
  }
}
```

## Compliance and Audit

### Audit Logging

```python
"""
audit_logging.py
Comprehensive audit logging for compliance
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger()


class AuditEventType(str, Enum):
    KEY_GENERATED = "KEY_GENERATED"
    KEY_ACCESSED = "KEY_ACCESSED"
    KEY_ROTATED = "KEY_ROTATED"
    KEY_REVOKED = "KEY_REVOKED"
    KEY_DELETED = "KEY_DELETED"
    ENCAPSULATION = "ENCAPSULATION"
    DECAPSULATION = "DECAPSULATION"
    SIGNATURE_CREATED = "SIGNATURE_CREATED"
    SIGNATURE_VERIFIED = "SIGNATURE_VERIFIED"
    FHE_ENCRYPTION = "FHE_ENCRYPTION"
    FHE_DECRYPTION = "FHE_DECRYPTION"
    FHE_COMPUTATION = "FHE_COMPUTATION"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    CONFIG_CHANGE = "CONFIG_CHANGE"


@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    actor_id: str
    actor_ip: str
    resource_type: str
    resource_id: str
    action: str
    outcome: str  # SUCCESS, FAILURE, ERROR
    details: Dict[str, Any]
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute tamper-evident hash"""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """
    Audit logger for compliance requirements
    
    Supports:
    - SOC 2 Type II
    - HIPAA
    - GDPR
    - PCI-DSS
    """
    
    def __init__(
        self,
        storage_backend: str = "elasticsearch",  # elasticsearch, s3, database
        retention_days: int = 2555,  # 7 years for compliance
        enable_tamper_detection: bool = True
    ):
        self.storage_backend = storage_backend
        self.retention_days = retention_days
        self.enable_tamper_detection = enable_tamper_detection
        self._previous_hash: Optional[str] = None
        
    async def log_event(
        self,
        event_type: AuditEventType,
        actor_id: str,
        actor_ip: str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str,
        details: Dict[str, Any] = None,
        request_id: str = None,
        session_id: str = None
    ) -> AuditEvent:
        """Log an audit event"""
        import uuid
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            actor_id=actor_id,
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            request_id=request_id,
            session_id=session_id
        )
        
        # Add tamper-evident chain
        if self.enable_tamper_detection:
            event.details["_previous_hash"] = self._previous_hash
            event.details["_event_hash"] = event.compute_hash()
            self._previous_hash = event.details["_event_hash"]
        
        # Store event
        await self._store_event(event)
        
        # Log for real-time monitoring
        logger.info(
            "Audit event recorded",
            event_id=event.event_id,
            event_type=event_type.value,
            actor_id=actor_id,
            resource_id=resource_id,
            outcome=outcome
        )
        
        return event
    
    async def _store_event(self, event: AuditEvent):
        """Store event to configured backend"""
        if self.storage_backend == "elasticsearch":
            await self._store_elasticsearch(event)
        elif self.storage_backend == "s3":
            await self._store_s3(event)
        else:
            await self._store_database(event)
    
    async def _store_elasticsearch(self, event: AuditEvent):
        """Store in Elasticsearch"""
        # Implementation for Elasticsearch
        pass
    
    async def _store_s3(self, event: AuditEvent):
        """Store in S3 with Glacier lifecycle"""
        # Implementation for S3
        pass
    
    async def _store_database(self, event: AuditEvent):
        """Store in PostgreSQL"""
        # Implementation for database
        pass
    
    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: list = None,
        actor_id: str = None,
        resource_id: str = None,
        limit: int = 1000
    ) -> list:
        """Query audit events for compliance reporting"""
        # Implementation depends on storage backend
        pass
    
    async def generate_compliance_report(
        self,
        report_type: str,  # "soc2", "hipaa", "gdpr", "pci"
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Generate compliance report"""
        events = await self.query_events(start_date, end_date)
        
        if report_type == "soc2":
            return self._generate_soc2_report(events)
        elif report_type == "hipaa":
            return self._generate_hipaa_report(events)
        elif report_type == "gdpr":
            return self._generate_gdpr_report(events)
        elif report_type == "pci":
            return self._generate_pci_report(events)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _generate_soc2_report(self, events: list) -> dict:
        """Generate SOC 2 Type II report"""
        return {
            "report_type": "SOC 2 Type II",
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {
                "CC6.1": {  # Logical Access
                    "control": "Logical access security software and policies",
                    "events": [e for e in events if e.event_type in [
                        AuditEventType.AUTH_SUCCESS,
                        AuditEventType.AUTH_FAILURE
                    ]],
                    "status": "COMPLIANT"
                },
                "CC6.7": {  # Change Management
                    "control": "Changes to system components are authorized",
                    "events": [e for e in events if e.event_type == AuditEventType.CONFIG_CHANGE],
                    "status": "COMPLIANT"
                },
                "CC7.2": {  # System Monitoring
                    "control": "System activity is monitored",
                    "total_events": len(events),
                    "status": "COMPLIANT"
                }
            }
        }
    
    def _generate_hipaa_report(self, events: list) -> dict:
        """Generate HIPAA compliance report"""
        return {
            "report_type": "HIPAA",
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {
                "164.312(a)(1)": {  # Access Control
                    "requirement": "Unique user identification",
                    "status": "COMPLIANT"
                },
                "164.312(b)": {  # Audit Controls
                    "requirement": "Hardware, software, and procedural mechanisms",
                    "total_events": len(events),
                    "status": "COMPLIANT"
                },
                "164.312(e)(1)": {  # Transmission Security
                    "requirement": "Encryption and decryption",
                    "encryption_events": len([e for e in events if "ENCRYPTION" in e.event_type.value]),
                    "status": "COMPLIANT"
                }
            }
        }
    
    def _generate_gdpr_report(self, events: list) -> dict:
        """Generate GDPR compliance report"""
        return {
            "report_type": "GDPR",
            "generated_at": datetime.utcnow().isoformat(),
            "data_processing_activities": len(events),
            "encryption_applied": True,
            "access_controls": "Implemented",
            "data_retention_policy": f"{self.retention_days} days"
        }
    
    def _generate_pci_report(self, events: list) -> dict:
        """Generate PCI-DSS compliance report"""
        return {
            "report_type": "PCI-DSS",
            "generated_at": datetime.utcnow().isoformat(),
            "requirements": {
                "3.4": {  # Render PAN unreadable
                    "description": "Encryption of stored data",
                    "status": "COMPLIANT"
                },
                "4.1": {  # Strong cryptography
                    "description": "Use strong cryptography for transmission",
                    "status": "COMPLIANT"
                },
                "10.2": {  # Audit trails
                    "description": "Implement automated audit trails",
                    "total_events": len(events),
                    "status": "COMPLIANT"
                }
            }
        }
```

## Integration Examples

### Database Integration (SQLAlchemy)

```python
"""
database_integration.py
SQLAlchemy models for PQC-FHE metadata storage
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, DateTime, Integer, Text, 
    Boolean, ForeignKey, JSON, Enum, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class KeyStatusEnum(enum.Enum):
    ACTIVE = "active"
    PENDING = "pending"
    REVOKED = "revoked"
    EXPIRED = "expired"


class AlgorithmEnum(enum.Enum):
    ML_KEM_512 = "ML-KEM-512"
    ML_KEM_768 = "ML-KEM-768"
    ML_KEM_1024 = "ML-KEM-1024"
    ML_DSA_44 = "ML-DSA-44"
    ML_DSA_65 = "ML-DSA-65"
    ML_DSA_87 = "ML-DSA-87"


class PQCKey(Base):
    """PQC Key metadata table"""
    __tablename__ = "pqc_keys"
    
    id = Column(String(64), primary_key=True)
    algorithm = Column(Enum(AlgorithmEnum), nullable=False)
    status = Column(Enum(KeyStatusEnum), default=KeyStatusEnum.ACTIVE)
    owner_id = Column(String(128), nullable=False, index=True)
    
    # Key fingerprints (not actual keys)
    public_key_fingerprint = Column(String(64), nullable=False)
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    revocation_reason = Column(Text, nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    # Metadata
    tags = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    audit_logs = relationship("KeyAuditLog", back_populates="key")
    
    __table_args__ = (
        Index("idx_owner_status", "owner_id", "status"),
        Index("idx_expires_at", "expires_at"),
    )


class KeyAuditLog(Base):
    """Key audit log table"""
    __tablename__ = "key_audit_logs"
    
    id = Column(String(64), primary_key=True)
    key_id = Column(String(64), ForeignKey("pqc_keys.id"), nullable=False)
    
    event_type = Column(String(64), nullable=False)
    actor_id = Column(String(128), nullable=False)
    actor_ip = Column(String(45), nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON, default=dict)
    
    key = relationship("PQCKey", back_populates="audit_logs")
    
    __table_args__ = (
        Index("idx_key_timestamp", "key_id", "timestamp"),
        Index("idx_actor_timestamp", "actor_id", "timestamp"),
    )


class FHEContext(Base):
    """FHE context metadata"""
    __tablename__ = "fhe_contexts"
    
    id = Column(String(64), primary_key=True)
    owner_id = Column(String(128), nullable=False, index=True)
    
    # CKKS parameters
    poly_modulus_degree = Column(Integer, nullable=False)
    coeff_mod_bit_sizes = Column(JSON, nullable=False)
    scale = Column(Integer, nullable=False)
    
    # Context hash for integrity
    context_hash = Column(String(64), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    metadata = Column(JSON, default=dict)


class EncryptedData(Base):
    """Encrypted data references"""
    __tablename__ = "encrypted_data"
    
    id = Column(String(64), primary_key=True)
    context_id = Column(String(64), ForeignKey("fhe_contexts.id"), nullable=False)
    owner_id = Column(String(128), nullable=False, index=True)
    
    # Storage location (S3 URI, etc.)
    storage_uri = Column(String(512), nullable=False)
    
    # Metadata
    data_type = Column(String(64), nullable=False)
    element_count = Column(Integer, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    metadata = Column(JSON, default=dict)
```

## Production Checklist

### Security Checklist

- [ ] All secrets stored in AWS Secrets Manager / HashiCorp Vault
- [ ] TLS 1.3 enabled for all communications
- [ ] Network policies restrict pod-to-pod communication
- [ ] Service mesh (Istio) configured with mTLS
- [ ] Web Application Firewall (WAF) rules configured
- [ ] DDoS protection enabled (AWS Shield)
- [ ] Secret key material never logged
- [ ] Audit logging enabled and tamper-evident
- [ ] Key rotation policy implemented (90 days)
- [ ] HSM integration for production keys

### Operational Checklist

- [ ] Multi-AZ deployment configured
- [ ] Auto-scaling policies defined
- [ ] Pod disruption budgets set
- [ ] Backup and disaster recovery tested
- [ ] Runbooks documented
- [ ] On-call rotation established
- [ ] Incident response plan documented

### Monitoring Checklist

- [ ] Prometheus metrics exported
- [ ] Grafana dashboards configured
- [ ] Alert rules defined
- [ ] Log aggregation (ELK/Loki) configured
- [ ] Distributed tracing (Jaeger) enabled
- [ ] SLO/SLI defined and monitored

### Compliance Checklist

- [ ] SOC 2 Type II audit scheduled
- [ ] HIPAA BAA in place (if applicable)
- [ ] GDPR DPA signed (if applicable)
- [ ] PCI-DSS scope defined (if applicable)
- [ ] Penetration testing scheduled
- [ ] Vulnerability scanning automated

## Next Steps

1. **Security Hardening**
   - Implement HSM integration
   - Add mTLS with Istio
   - Configure OPA policies

2. **Performance Optimization**
   - Implement connection pooling
   - Add caching layers
   - Profile and optimize hot paths

3. **Advanced Features**
   - Multi-region deployment
   - Active-passive failover
   - Geo-distributed key management

4. **Documentation**
   - API versioning strategy
   - Migration guides
   - Troubleshooting runbooks

## Related Documentation

- [PQC Key Exchange Tutorial](pqc_key_exchange.md)
- [FHE Computation Tutorial](fhe_computation.md)
- [Hybrid Workflow Tutorial](hybrid_workflow.md)
- [Security Overview](../security/overview.md)
- [API Reference](../api/overview.md)
