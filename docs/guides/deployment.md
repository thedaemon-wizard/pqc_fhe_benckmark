# Production Deployment Guide

This guide covers deploying PQC-FHE systems in production environments with emphasis on security, scalability, and operational excellence.

## Overview

Production deployment of cryptographic systems requires careful attention to:

- **Security hardening**: Protecting key material and cryptographic operations
- **High availability**: Ensuring system resilience and fault tolerance
- **Scalability**: Handling varying workloads efficiently
- **Monitoring**: Observability for security and performance
- **Compliance**: Meeting regulatory requirements (NIST, FIPS)

## Deployment Architecture

### Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Load Balancer (TLS 1.3)                       │
│                     ┌─────────────────────────────────┐                 │
│                     │  HAProxy / AWS ALB / GCP LB     │                 │
│                     └───────────────┬─────────────────┘                 │
└─────────────────────────────────────┼───────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
            ┌───────▼───────┐                   ┌───────▼───────┐
            │  API Gateway  │                   │  API Gateway  │
            │  (Instance 1) │                   │  (Instance 2) │
            └───────┬───────┘                   └───────┬───────┘
                    │                                   │
        ┌───────────┴───────────┐           ┌───────────┴───────────┐
        │                       │           │                       │
┌───────▼───────┐       ┌───────▼───────┐   │               ┌───────▼───────┐
│  PQC Service  │       │  FHE Service  │   │               │  FHE Service  │
│  (Stateless)  │       │  (Stateless)  │   │               │  (Stateless)  │
└───────┬───────┘       └───────┬───────┘   │               └───────┬───────┘
        │                       │           │                       │
        └───────────────────────┴───────────┴───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼───────┐       ┌───────▼───────┐
            │   Key Store   │       │  Metrics DB   │
            │  (Encrypted)  │       │ (Prometheus)  │
            └───────────────┘       └───────────────┘
```

### Component Responsibilities

| Component | Responsibility | Scaling Strategy |
|-----------|---------------|------------------|
| Load Balancer | TLS termination, traffic distribution | Active-passive HA |
| API Gateway | Authentication, rate limiting, routing | Horizontal (stateless) |
| PQC Service | Key generation, encapsulation, signatures | Horizontal (CPU-bound) |
| FHE Service | Encryption, homomorphic operations | Horizontal (compute-intensive) |
| Key Store | Secure key storage | Primary-replica |
| Metrics DB | Monitoring data | Time-series optimized |

## Docker Deployment

### Production Dockerfile

```dockerfile
# Dockerfile.production
# Multi-stage build for minimal attack surface

# Stage 1: Build dependencies
FROM python:3.11-slim-bookworm AS builder

# Security: Run as non-root user
RUN groupadd -r pqcfhe && useradd -r -g pqcfhe pqcfhe

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim-bookworm AS production

# Security labels
LABEL maintainer="security@example.com" \
      version="1.0.0" \
      description="PQC-FHE Production Service" \
      org.opencontainers.image.source="https://github.com/example/pqc-fhe"

# Create non-root user
RUN groupadd -r pqcfhe && useradd -r -g pqcfhe pqcfhe

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=pqcfhe:pqcfhe src/ ./src/
COPY --chown=pqcfhe:pqcfhe config/ ./config/

# Security: Remove unnecessary files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete

# Security: Set restrictive permissions
RUN chmod -R 750 /app && \
    chmod 640 /app/config/*

# Security: Read-only filesystem where possible
RUN mkdir -p /app/tmp /app/logs && \
    chown pqcfhe:pqcfhe /app/tmp /app/logs

# Switch to non-root user
USER pqcfhe

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PQC_FHE_ENV=production \
    PQC_FHE_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose port
EXPOSE 8080

# Entry point with security options
ENTRYPOINT ["python", "-m", "src.api.server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose for Development/Staging

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  pqc-fhe-api:
    build:
      context: .
      dockerfile: Dockerfile.production
    image: pqc-fhe:${VERSION:-latest}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - PQC_FHE_ENV=production
      - PQC_FHE_LOG_LEVEL=INFO
      - PQC_FHE_METRICS_ENABLED=true
      - PQC_FHE_KEY_STORE_URL=redis://redis:6379/0
    secrets:
      - pqc_master_key
      - fhe_secret_key
    networks:
      - pqc-fhe-internal
      - pqc-fhe-external
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:size=100M,mode=1777
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - pqc-fhe-internal
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G

  prometheus:
    image: prom/prometheus:v2.47.0
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - pqc-fhe-internal
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:10.1.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - grafana-data:/var/lib/grafana
    networks:
      - pqc-fhe-internal
      - pqc-fhe-external
    depends_on:
      - prometheus

networks:
  pqc-fhe-internal:
    driver: bridge
    internal: true
  pqc-fhe-external:
    driver: bridge

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

secrets:
  pqc_master_key:
    external: true
  fhe_secret_key:
    external: true
```

## Kubernetes Deployment

### Namespace and RBAC

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pqc-fhe
  labels:
    name: pqc-fhe
    istio-injection: enabled
---
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pqc-fhe-service
  namespace: pqc-fhe
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pqc-fhe-role
  namespace: pqc-fhe
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pqc-fhe-rolebinding
  namespace: pqc-fhe
subjects:
  - kind: ServiceAccount
    name: pqc-fhe-service
    namespace: pqc-fhe
roleRef:
  kind: Role
  name: pqc-fhe-role
  apiGroup: rbac.authorization.k8s.io
```

### ConfigMap and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pqc-fhe-config
  namespace: pqc-fhe
data:
  config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
      workers: 4
      
    pqc:
      default_kem: "ML-KEM-768"
      default_dsa: "ML-DSA-65"
      key_cache_ttl: 3600
      key_cache_max_size: 1000
      
    fhe:
      default_scheme: "CKKS"
      poly_modulus_degree: 8192
      coeff_modulus_bits: [60, 40, 40, 60]
      scale_bits: 40
      security_level: 128
      
    security:
      tls_enabled: true
      tls_cert_path: "/etc/pqc-fhe/tls/tls.crt"
      tls_key_path: "/etc/pqc-fhe/tls/tls.key"
      rate_limit_rps: 100
      rate_limit_burst: 200
      
    observability:
      metrics_enabled: true
      metrics_port: 9090
      tracing_enabled: true
      log_level: "INFO"
      log_format: "json"
---
# k8s/secret.yaml (use external secrets in production)
apiVersion: v1
kind: Secret
metadata:
  name: pqc-fhe-secrets
  namespace: pqc-fhe
type: Opaque
stringData:
  # In production, use external secrets manager (Vault, AWS Secrets Manager)
  MASTER_KEY_ID: "alias/pqc-fhe-master"
  DATABASE_URL: "postgresql://user:pass@host:5432/pqcfhe"
```

### Deployment with Security Context

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pqc-fhe-api
  namespace: pqc-fhe
  labels:
    app: pqc-fhe-api
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
      app: pqc-fhe-api
  template:
    metadata:
      labels:
        app: pqc-fhe-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: pqc-fhe-service
      
      # Pod security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      
      # Anti-affinity for high availability
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: pqc-fhe-api
                topologyKey: kubernetes.io/hostname
      
      # Topology spread for zone distribution
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: pqc-fhe-api
      
      containers:
        - name: pqc-fhe-api
          image: pqc-fhe:v1.0.0
          imagePullPolicy: Always
          
          # Container security context
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP
          
          # Resource management
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          
          # Probes
          livenessProbe:
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 10
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 3
          
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          startupProbe:
            httpGet:
              path: /health/startup
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 30
          
          # Environment variables
          env:
            - name: PQC_FHE_ENV
              value: "production"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
          
          envFrom:
            - secretRef:
                name: pqc-fhe-secrets
          
          # Volume mounts
          volumeMounts:
            - name: config
              mountPath: /etc/pqc-fhe/config
              readOnly: true
            - name: tls
              mountPath: /etc/pqc-fhe/tls
              readOnly: true
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache
      
      volumes:
        - name: config
          configMap:
            name: pqc-fhe-config
        - name: tls
          secret:
            secretName: pqc-fhe-tls
        - name: tmp
          emptyDir:
            medium: Memory
            sizeLimit: 100Mi
        - name: cache
          emptyDir:
            sizeLimit: 1Gi
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pqc-fhe-api
  namespace: pqc-fhe
  labels:
    app: pqc-fhe-api
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 80
      targetPort: 8080
      protocol: TCP
    - name: metrics
      port: 9090
      targetPort: 9090
      protocol: TCP
  selector:
    app: pqc-fhe-api
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pqc-fhe-ingress
  namespace: pqc-fhe
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-burst-multiplier: "5"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - api.pqc-fhe.example.com
      secretName: pqc-fhe-tls-cert
  rules:
    - host: api.pqc-fhe.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: pqc-fhe-api
                port:
                  number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pqc-fhe-api-hpa
  namespace: pqc-fhe
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pqc-fhe-api
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
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

### Pod Disruption Budget

```yaml
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: pqc-fhe-api-pdb
  namespace: pqc-fhe
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: pqc-fhe-api
```

## Cloud Provider Deployments

### AWS Deployment

```python
"""
AWS CDK deployment for PQC-FHE system.

References:
- AWS Well-Architected Framework: Security Pillar
- NIST SP 800-53: Security Controls
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_secretsmanager as secretsmanager,
    aws_kms as kms,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_sns as sns,
    aws_wafv2 as wafv2,
)
from constructs import Construct


class PQCFHEStack(Stack):
    """AWS CDK Stack for PQC-FHE deployment."""
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # VPC with private subnets
        vpc = ec2.Vpc(
            self, "PQCFHEVPC",
            max_azs=3,
            nat_gateways=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                )
            ]
        )
        
        # KMS key for encryption at rest
        encryption_key = kms.Key(
            self, "PQCFHEKey",
            alias="alias/pqc-fhe-master",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
            description="Master encryption key for PQC-FHE system"
        )
        
        # Secrets Manager for sensitive configuration
        master_secret = secretsmanager.Secret(
            self, "MasterSecret",
            secret_name="pqc-fhe/master-config",
            encryption_key=encryption_key,
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"username": "admin"}',
                generate_string_key="password",
                exclude_punctuation=True
            )
        )
        
        # ECS Cluster
        cluster = ecs.Cluster(
            self, "PQCFHECluster",
            vpc=vpc,
            container_insights=True,
            cluster_name="pqc-fhe-cluster"
        )
        
        # CloudWatch Log Group
        log_group = logs.LogGroup(
            self, "PQCFHELogs",
            log_group_name="/ecs/pqc-fhe",
            retention=logs.RetentionDays.THIRTY_DAYS,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self, "PQCFHETask",
            memory_limit_mib=4096,
            cpu=2048,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX
            )
        )
        
        # Container
        container = task_definition.add_container(
            "pqc-fhe-api",
            image=ecs.ContainerImage.from_registry("pqc-fhe:latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="pqc-fhe",
                log_group=log_group
            ),
            environment={
                "PQC_FHE_ENV": "production",
                "AWS_REGION": self.region
            },
            secrets={
                "MASTER_SECRET": ecs.Secret.from_secrets_manager(master_secret)
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60)
            )
        )
        
        container.add_port_mappings(
            ecs.PortMapping(container_port=8080)
        )
        
        # ALB Fargate Service
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "PQCFHEService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=3,
            public_load_balancer=True,
            listener_port=443,
            protocol=elbv2.ApplicationProtocol.HTTPS,
            redirect_http=True,
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                rollback=True
            ),
            enable_execute_command=True
        )
        
        # Configure health check
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3
        )
        
        # Auto Scaling
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=3,
            max_capacity=20
        )
        
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60)
        )
        
        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60)
        )
        
        # WAF Web ACL
        web_acl = wafv2.CfnWebACL(
            self, "PQCFHEWAF",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            scope="REGIONAL",
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name="pqc-fhe-waf",
                sampled_requests_enabled=True
            ),
            rules=[
                # AWS Managed Rules
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSManagedRulesCommonRuleSet",
                    priority=1,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet"
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="common-rules",
                        sampled_requests_enabled=True
                    )
                ),
                # Rate limiting
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimitRule",
                    priority=2,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            limit=2000,
                            aggregate_key_type="IP"
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="rate-limit",
                        sampled_requests_enabled=True
                    )
                )
            ]
        )
        
        # CloudWatch Alarms
        sns_topic = sns.Topic(self, "AlertTopic", topic_name="pqc-fhe-alerts")
        
        # High CPU alarm
        cloudwatch.Alarm(
            self, "HighCPUAlarm",
            metric=fargate_service.service.metric_cpu_utilization(),
            threshold=85,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="CPU utilization is high"
        ).add_alarm_action(cloudwatch.SnsAction(sns_topic))
        
        # Error rate alarm
        cloudwatch.Alarm(
            self, "ErrorRateAlarm",
            metric=fargate_service.target_group.metrics.http_code_target(
                elbv2.HttpCodeTarget.TARGET_5XX_COUNT
            ),
            threshold=10,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="High 5XX error rate"
        ).add_alarm_action(cloudwatch.SnsAction(sns_topic))
```

### GCP Deployment (Terraform)

```hcl
# terraform/gcp/main.tf

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "pqc-fhe-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "pqc_fhe_vpc" {
  name                    = "pqc-fhe-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "pqc_fhe_subnet" {
  name          = "pqc-fhe-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.pqc_fhe_vpc.id
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }
  
  private_ip_google_access = true
}

# GKE Cluster
resource "google_container_cluster" "pqc_fhe_cluster" {
  name     = "pqc-fhe-cluster"
  location = var.region
  
  # Use release channel for automatic upgrades
  release_channel {
    channel = "REGULAR"
  }
  
  # Network configuration
  network    = google_compute_network.pqc_fhe_vpc.name
  subnetwork = google_compute_subnetwork.pqc_fhe_subnet.name
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  # Private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Security
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = var.authorized_network
      display_name = "Authorized Network"
    }
  }
  
  # Binary Authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }
  
  # Logging and monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }
  
  # Initial node pool (to be replaced)
  remove_default_node_pool = true
  initial_node_count       = 1
}

# Node Pool
resource "google_container_node_pool" "pqc_fhe_nodes" {
  name       = "pqc-fhe-node-pool"
  location   = var.region
  cluster    = google_container_cluster.pqc_fhe_cluster.name
  
  node_count = var.min_node_count
  
  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    machine_type = "n2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    # Security
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    
    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Service account
    service_account = google_service_account.gke_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      env         = "production"
      application = "pqc-fhe"
    }
    
    tags = ["pqc-fhe-node"]
  }
}

# Service Account
resource "google_service_account" "gke_sa" {
  account_id   = "pqc-fhe-gke-sa"
  display_name = "PQC-FHE GKE Service Account"
}

# IAM bindings
resource "google_project_iam_member" "gke_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

resource "google_project_iam_member" "gke_sa_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# Cloud Armor Security Policy
resource "google_compute_security_policy" "pqc_fhe_policy" {
  name = "pqc-fhe-security-policy"
  
  # Default rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }
  
  # Rate limiting
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      ban_duration_sec = 600
    }
    description = "Rate limiting rule"
  }
  
  # Block known bad IPs
  rule {
    action   = "deny(403)"
    priority = "100"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('xss-v33-stable')"
      }
    }
    description = "XSS protection"
  }
}

# Secret Manager
resource "google_secret_manager_secret" "pqc_fhe_config" {
  secret_id = "pqc-fhe-config"
  
  replication {
    auto {}
  }
}

# Variables
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "min_node_count" {
  description = "Minimum number of nodes"
  type        = number
  default     = 3
}

variable "max_node_count" {
  description = "Maximum number of nodes"
  type        = number
  default     = 20
}

variable "authorized_network" {
  description = "CIDR for authorized network access"
  type        = string
}

# Outputs
output "cluster_endpoint" {
  value     = google_container_cluster.pqc_fhe_cluster.endpoint
  sensitive = true
}

output "cluster_ca_certificate" {
  value     = google_container_cluster.pqc_fhe_cluster.master_auth[0].cluster_ca_certificate
  sensitive = true
}
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'pqc-fhe-production'
    env: 'production'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  # PQC-FHE API metrics
  - job_name: 'pqc-fhe-api'
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
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'pqc_fhe_.*'
        action: keep
```

### Alert Rules

```yaml
# config/prometheus/rules/pqc_fhe_alerts.yml
groups:
  - name: pqc_fhe_alerts
    rules:
      # High error rate
      - alert: PQCFHEHighErrorRate
        expr: |
          sum(rate(pqc_fhe_requests_total{status="error"}[5m])) /
          sum(rate(pqc_fhe_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
          service: pqc-fhe
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
      
      # High latency
      - alert: PQCFHEHighLatency
        expr: |
          histogram_quantile(0.95, 
            sum(rate(pqc_fhe_request_duration_seconds_bucket[5m])) by (le, operation)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
          service: pqc-fhe
        annotations:
          summary: "High latency for {{ $labels.operation }}"
          description: "P95 latency is {{ $value | humanizeDuration }}"
      
      # Key generation failures
      - alert: PQCKeyGenerationFailure
        expr: |
          increase(pqc_fhe_keygen_failures_total[5m]) > 5
        for: 2m
        labels:
          severity: critical
          service: pqc-fhe
        annotations:
          summary: "PQC key generation failures detected"
          description: "{{ $value }} key generation failures in the last 5 minutes"
      
      # Bootstrap operation failures
      - alert: FHEBootstrapFailure
        expr: |
          increase(pqc_fhe_bootstrap_failures_total[5m]) > 3
        for: 2m
        labels:
          severity: critical
          service: pqc-fhe
        annotations:
          summary: "FHE bootstrap operation failures"
          description: "{{ $value }} bootstrap failures in the last 5 minutes"
      
      # High noise budget consumption
      - alert: FHEHighNoiseConsumption
        expr: |
          pqc_fhe_noise_budget_remaining < 20
        for: 1m
        labels:
          severity: warning
          service: pqc-fhe
        annotations:
          summary: "FHE noise budget running low"
          description: "Remaining noise budget: {{ $value }} bits"
      
      # Memory pressure
      - alert: PQCFHEHighMemoryUsage
        expr: |
          container_memory_usage_bytes{container="pqc-fhe-api"} /
          container_spec_memory_limit_bytes{container="pqc-fhe-api"} > 0.85
        for: 5m
        labels:
          severity: warning
          service: pqc-fhe
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
      
      # Pod restarts
      - alert: PQCFHEPodRestarting
        expr: |
          increase(kube_pod_container_status_restarts_total{container="pqc-fhe-api"}[1h]) > 3
        labels:
          severity: warning
          service: pqc-fhe
        annotations:
          summary: "Pod restarting frequently"
          description: "{{ $value }} restarts in the last hour"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "PQC-FHE Production Dashboard",
    "uid": "pqc-fhe-prod",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(pqc_fhe_requests_total[5m])) by (operation)",
            "legendFormat": "{{ operation }}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(pqc_fhe_requests_total{status=\"error\"}[5m])) / sum(rate(pqc_fhe_requests_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ],
        "thresholds": [
          {"value": 0.01, "colorMode": "warning"},
          {"value": 0.05, "colorMode": "critical"}
        ]
      },
      {
        "title": "Latency (P95)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(pqc_fhe_request_duration_seconds_bucket[5m])) by (le, operation))",
            "legendFormat": "{{ operation }} P95"
          }
        ]
      },
      {
        "title": "PQC Operations",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "sum(increase(pqc_fhe_pqc_operations_total[1h]))",
            "legendFormat": "Total"
          }
        ]
      },
      {
        "title": "FHE Operations",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8},
        "targets": [
          {
            "expr": "sum(increase(pqc_fhe_fhe_operations_total[1h]))",
            "legendFormat": "Total"
          }
        ]
      },
      {
        "title": "Noise Budget Distribution",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
        "targets": [
          {
            "expr": "pqc_fhe_noise_budget_remaining",
            "legendFormat": "{{ instance }}"
          }
        ]
      }
    ]
  }
}
```

## Security Hardening Checklist

### Pre-Deployment Security Checklist

```python
"""
Security hardening checklist for PQC-FHE deployment.

Based on:
- NIST SP 800-53 Rev. 5: Security and Privacy Controls
- CIS Kubernetes Benchmark v1.8
- OWASP Container Security Verification Standard
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import json


class Severity(Enum):
    """Security check severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Status(Enum):
    """Check status."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class SecurityCheck:
    """Security check definition."""
    id: str
    name: str
    description: str
    severity: Severity
    category: str
    remediation: str
    references: List[str]
    status: Status = Status.SKIP
    details: Optional[str] = None


class SecurityChecklist:
    """PQC-FHE security hardening checklist."""
    
    def __init__(self):
        self.checks: List[SecurityCheck] = []
        self._initialize_checks()
    
    def _initialize_checks(self):
        """Initialize security checks."""
        
        # Container Security
        self.checks.extend([
            SecurityCheck(
                id="CS-001",
                name="Non-root container",
                description="Container runs as non-root user",
                severity=Severity.CRITICAL,
                category="Container Security",
                remediation="Set runAsNonRoot: true and runAsUser: 1000 in securityContext",
                references=["CIS Docker Benchmark 4.1", "NIST SP 800-190"]
            ),
            SecurityCheck(
                id="CS-002",
                name="Read-only root filesystem",
                description="Container filesystem is read-only",
                severity=Severity.HIGH,
                category="Container Security",
                remediation="Set readOnlyRootFilesystem: true in securityContext",
                references=["CIS Docker Benchmark 5.12"]
            ),
            SecurityCheck(
                id="CS-003",
                name="No privilege escalation",
                description="Container cannot escalate privileges",
                severity=Severity.CRITICAL,
                category="Container Security",
                remediation="Set allowPrivilegeEscalation: false in securityContext",
                references=["CIS Docker Benchmark 5.25"]
            ),
            SecurityCheck(
                id="CS-004",
                name="Capabilities dropped",
                description="All Linux capabilities are dropped",
                severity=Severity.HIGH,
                category="Container Security",
                remediation="Set capabilities.drop: [ALL] in securityContext",
                references=["CIS Docker Benchmark 5.3"]
            ),
            SecurityCheck(
                id="CS-005",
                name="Seccomp profile",
                description="Seccomp profile is applied",
                severity=Severity.MEDIUM,
                category="Container Security",
                remediation="Set seccompProfile.type: RuntimeDefault",
                references=["CIS Kubernetes Benchmark 5.7.2"]
            ),
        ])
        
        # Network Security
        self.checks.extend([
            SecurityCheck(
                id="NS-001",
                name="TLS 1.3 only",
                description="Only TLS 1.3 connections are accepted",
                severity=Severity.CRITICAL,
                category="Network Security",
                remediation="Configure min TLS version to 1.3 in ingress/load balancer",
                references=["NIST SP 800-52 Rev. 2"]
            ),
            SecurityCheck(
                id="NS-002",
                name="Network policies",
                description="Kubernetes NetworkPolicies restrict traffic",
                severity=Severity.HIGH,
                category="Network Security",
                remediation="Create NetworkPolicy to allow only required traffic",
                references=["CIS Kubernetes Benchmark 5.3.2"]
            ),
            SecurityCheck(
                id="NS-003",
                name="Internal traffic encrypted",
                description="Service mesh encrypts pod-to-pod traffic",
                severity=Severity.HIGH,
                category="Network Security",
                remediation="Enable mTLS in service mesh (Istio/Linkerd)",
                references=["NIST SP 800-204A"]
            ),
        ])
        
        # Secrets Management
        self.checks.extend([
            SecurityCheck(
                id="SM-001",
                name="External secrets manager",
                description="Secrets are stored in external manager (Vault/AWS SM)",
                severity=Severity.CRITICAL,
                category="Secrets Management",
                remediation="Use external secrets operator with Vault or cloud provider",
                references=["NIST SP 800-57"]
            ),
            SecurityCheck(
                id="SM-002",
                name="Secrets rotation",
                description="Automatic secrets rotation is configured",
                severity=Severity.HIGH,
                category="Secrets Management",
                remediation="Configure rotation policy in secrets manager",
                references=["NIST SP 800-57"]
            ),
            SecurityCheck(
                id="SM-003",
                name="No secrets in environment",
                description="Sensitive data not exposed in environment variables",
                severity=Severity.HIGH,
                category="Secrets Management",
                remediation="Mount secrets as files, not environment variables",
                references=["CIS Kubernetes Benchmark 5.4.1"]
            ),
        ])
        
        # Cryptographic Security
        self.checks.extend([
            SecurityCheck(
                id="CR-001",
                name="NIST Level 3 minimum",
                description="PQC algorithms meet NIST security level 3+",
                severity=Severity.CRITICAL,
                category="Cryptographic Security",
                remediation="Use ML-KEM-768 or ML-KEM-1024, ML-DSA-65 or ML-DSA-87",
                references=["NIST FIPS 203", "NIST FIPS 204"]
            ),
            SecurityCheck(
                id="CR-002",
                name="128-bit FHE security",
                description="FHE parameters provide 128-bit security",
                severity=Severity.CRITICAL,
                category="Cryptographic Security",
                remediation="Use poly_modulus_degree >= 8192 with appropriate coeff_modulus",
                references=["Homomorphic Encryption Standard"]
            ),
            SecurityCheck(
                id="CR-003",
                name="Secure key storage",
                description="Private keys stored with HSM or KMS",
                severity=Severity.CRITICAL,
                category="Cryptographic Security",
                remediation="Use hardware security module or cloud KMS for key storage",
                references=["NIST SP 800-57", "FIPS 140-3"]
            ),
            SecurityCheck(
                id="CR-004",
                name="Key zeroization",
                description="Keys are securely erased from memory when not needed",
                severity=Severity.HIGH,
                category="Cryptographic Security",
                remediation="Implement secure memory wiping for key material",
                references=["NIST SP 800-88"]
            ),
        ])
        
        # Access Control
        self.checks.extend([
            SecurityCheck(
                id="AC-001",
                name="RBAC enabled",
                description="Role-based access control is enforced",
                severity=Severity.CRITICAL,
                category="Access Control",
                remediation="Configure Kubernetes RBAC with least privilege",
                references=["CIS Kubernetes Benchmark 5.1"]
            ),
            SecurityCheck(
                id="AC-002",
                name="Service account tokens",
                description="Service accounts have minimal permissions",
                severity=Severity.HIGH,
                category="Access Control",
                remediation="Create dedicated service accounts with minimal RBAC",
                references=["CIS Kubernetes Benchmark 5.1.5"]
            ),
            SecurityCheck(
                id="AC-003",
                name="API authentication",
                description="All API endpoints require authentication",
                severity=Severity.CRITICAL,
                category="Access Control",
                remediation="Implement JWT or mTLS authentication for all endpoints",
                references=["OWASP API Security Top 10"]
            ),
        ])
        
        # Logging and Monitoring
        self.checks.extend([
            SecurityCheck(
                id="LM-001",
                name="Audit logging",
                description="All cryptographic operations are logged",
                severity=Severity.HIGH,
                category="Logging and Monitoring",
                remediation="Enable comprehensive audit logging for all operations",
                references=["NIST SP 800-92"]
            ),
            SecurityCheck(
                id="LM-002",
                name="Anomaly detection",
                description="Automated anomaly detection is configured",
                severity=Severity.MEDIUM,
                category="Logging and Monitoring",
                remediation="Configure alerting rules for unusual patterns",
                references=["NIST SP 800-137"]
            ),
            SecurityCheck(
                id="LM-003",
                name="Log integrity",
                description="Logs are protected from tampering",
                severity=Severity.HIGH,
                category="Logging and Monitoring",
                remediation="Use immutable log storage with integrity verification",
                references=["NIST SP 800-92"]
            ),
        ])
    
    def run_checks(self) -> dict:
        """Run all security checks and return results."""
        results = {
            "total": len(self.checks),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "skipped": 0,
            "by_category": {},
            "by_severity": {},
            "checks": []
        }
        
        for check in self.checks:
            check_result = {
                "id": check.id,
                "name": check.name,
                "status": check.status.value,
                "severity": check.severity.value,
                "category": check.category,
                "details": check.details
            }
            results["checks"].append(check_result)
            
            # Update counters
            if check.status == Status.PASS:
                results["passed"] += 1
            elif check.status == Status.FAIL:
                results["failed"] += 1
            elif check.status == Status.WARN:
                results["warnings"] += 1
            else:
                results["skipped"] += 1
            
            # Update category stats
            if check.category not in results["by_category"]:
                results["by_category"][check.category] = {"pass": 0, "fail": 0}
            if check.status == Status.PASS:
                results["by_category"][check.category]["pass"] += 1
            elif check.status == Status.FAIL:
                results["by_category"][check.category]["fail"] += 1
            
            # Update severity stats
            if check.severity.value not in results["by_severity"]:
                results["by_severity"][check.severity.value] = {"pass": 0, "fail": 0}
            if check.status == Status.PASS:
                results["by_severity"][check.severity.value]["pass"] += 1
            elif check.status == Status.FAIL:
                results["by_severity"][check.severity.value]["fail"] += 1
        
        return results
    
    def generate_report(self) -> str:
        """Generate security assessment report."""
        results = self.run_checks()
        
        report = []
        report.append("=" * 60)
        report.append("PQC-FHE SECURITY ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Checks: {results['total']}")
        report.append(f"Passed: {results['passed']}")
        report.append(f"Failed: {results['failed']}")
        report.append(f"Warnings: {results['warnings']}")
        report.append(f"Skipped: {results['skipped']}")
        report.append("")
        report.append("-" * 60)
        report.append("RESULTS BY SEVERITY")
        report.append("-" * 60)
        
        for severity in ["critical", "high", "medium", "low"]:
            if severity in results["by_severity"]:
                stats = results["by_severity"][severity]
                report.append(f"  {severity.upper()}: {stats['pass']} passed, {stats['fail']} failed")
        
        report.append("")
        report.append("-" * 60)
        report.append("FAILED CHECKS (Require Remediation)")
        report.append("-" * 60)
        
        for check in self.checks:
            if check.status == Status.FAIL:
                report.append(f"\n[{check.id}] {check.name}")
                report.append(f"  Severity: {check.severity.value.upper()}")
                report.append(f"  Category: {check.category}")
                report.append(f"  Remediation: {check.remediation}")
                if check.details:
                    report.append(f"  Details: {check.details}")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    checklist = SecurityChecklist()
    
    # Simulate check results (in practice, these would be automated)
    checklist.checks[0].status = Status.PASS  # Non-root container
    checklist.checks[1].status = Status.PASS  # Read-only filesystem
    checklist.checks[2].status = Status.FAIL  # No privilege escalation
    checklist.checks[2].details = "allowPrivilegeEscalation not set to false"
    
    print(checklist.generate_report())
```

## Deployment Automation

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy PQC-FHE

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'
      
      - name: Run Bandit security linter
        run: |
          pip install bandit
          bandit -r src/ -ll -ii

  build:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    permissions:
      contents: read
      packages: write
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=
      
      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.production
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/pqc-fhe-api \
            pqc-fhe-api=${{ needs.build.outputs.image-tag }} \
            -n pqc-fhe-staging
          kubectl rollout status deployment/pqc-fhe-api \
            -n pqc-fhe-staging --timeout=300s
      
      - name: Run smoke tests
        run: |
          ./scripts/smoke-test.sh staging

  deploy-production:
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
      
      - name: Deploy canary
        run: |
          kubectl set image deployment/pqc-fhe-api-canary \
            pqc-fhe-api=${{ needs.build.outputs.image-tag }} \
            -n pqc-fhe
          kubectl rollout status deployment/pqc-fhe-api-canary \
            -n pqc-fhe --timeout=300s
      
      - name: Validate canary
        run: |
          sleep 300  # Wait 5 minutes
          ./scripts/validate-canary.sh
      
      - name: Full rollout
        run: |
          kubectl set image deployment/pqc-fhe-api \
            pqc-fhe-api=${{ needs.build.outputs.image-tag }} \
            -n pqc-fhe
          kubectl rollout status deployment/pqc-fhe-api \
            -n pqc-fhe --timeout=600s
      
      - name: Notify success
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "PQC-FHE ${{ github.ref_name }} deployed to production"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

## Operational Runbook

### Health Check Endpoints

```python
"""
Health check endpoint implementations.

Provides liveness, readiness, and startup probes for Kubernetes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import asyncio
import time


router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    checks: Dict[str, dict]


class HealthChecker:
    """Health check implementation."""
    
    def __init__(self):
        self.startup_complete = False
        self.startup_time: Optional[float] = None
    
    async def check_pqc_service(self) -> dict:
        """Check PQC service health."""
        try:
            # Verify PQC operations work
            start = time.perf_counter()
            # Simulated check - in production, actually test key generation
            await asyncio.sleep(0.001)
            latency = time.perf_counter() - start
            
            return {
                "status": "healthy",
                "latency_ms": round(latency * 1000, 2)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_fhe_service(self) -> dict:
        """Check FHE service health."""
        try:
            start = time.perf_counter()
            # Simulated check - in production, test encryption/decryption
            await asyncio.sleep(0.001)
            latency = time.perf_counter() - start
            
            return {
                "status": "healthy",
                "latency_ms": round(latency * 1000, 2)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_key_store(self) -> dict:
        """Check key store connectivity."""
        try:
            start = time.perf_counter()
            # Check Redis/key store connection
            await asyncio.sleep(0.001)
            latency = time.perf_counter() - start
            
            return {
                "status": "healthy",
                "latency_ms": round(latency * 1000, 2)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


health_checker = HealthChecker()


@router.get("/live")
async def liveness():
    """
    Liveness probe.
    
    Returns 200 if the application is running.
    Used by Kubernetes to determine if pod should be restarted.
    """
    return {"status": "alive"}


@router.get("/ready", response_model=HealthStatus)
async def readiness():
    """
    Readiness probe.
    
    Returns 200 if the application is ready to receive traffic.
    Checks all dependencies.
    """
    checks = {
        "pqc_service": await health_checker.check_pqc_service(),
        "fhe_service": await health_checker.check_fhe_service(),
        "key_store": await health_checker.check_key_store(),
    }
    
    all_healthy = all(c["status"] == "healthy" for c in checks.values())
    
    if not all_healthy:
        raise HTTPException(status_code=503, detail={
            "status": "not_ready",
            "checks": checks
        })
    
    return HealthStatus(
        status="ready",
        timestamp=time.time(),
        version="1.0.0",
        checks=checks
    )


@router.get("/startup")
async def startup():
    """
    Startup probe.
    
    Returns 200 when startup is complete.
    Used by Kubernetes during initial pod startup.
    """
    if not health_checker.startup_complete:
        raise HTTPException(status_code=503, detail="Starting up")
    
    return {
        "status": "started",
        "startup_time": health_checker.startup_time
    }
```

## Summary

This deployment guide covers:

1. **Reference Architecture**: Multi-component design with load balancing and horizontal scaling
2. **Docker Deployment**: Production Dockerfile with security hardening, Docker Compose for orchestration
3. **Kubernetes Deployment**: Complete manifests including RBAC, ConfigMaps, Deployments, Services, HPA, and PDB
4. **Cloud Provider Deployments**: AWS CDK and GCP Terraform configurations
5. **Monitoring**: Prometheus metrics, alert rules, and Grafana dashboards
6. **Security Hardening**: Comprehensive checklist based on NIST and CIS benchmarks
7. **CI/CD Pipeline**: GitHub Actions workflow with security scanning and staged deployment
8. **Operational Runbook**: Health check endpoints and probe implementations

## References

- NIST SP 800-53 Rev. 5: Security and Privacy Controls for Information Systems
- CIS Kubernetes Benchmark v1.8
- CIS Docker Benchmark v1.5
- AWS Well-Architected Framework
- Google Cloud Architecture Framework
- OWASP Container Security Verification Standard
- Homomorphic Encryption Standard (https://homomorphicencryption.org/standard/)
