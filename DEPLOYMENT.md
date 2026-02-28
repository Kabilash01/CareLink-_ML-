# CARELINK AI Microservice - Deployment Guide

## 🚀 Production Deployment Guide

This guide covers production deployment strategies for the CARELINK AI Microservice.

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [AWS Deployment](#aws-deployment)
5. [Azure Deployment](#azure-deployment)
6. [Configuration](#configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Security Hardening](#security-hardening)
9. [Scaling Strategies](#scaling-strategies)
10. [Disaster Recovery](#disaster-recovery)

---

## Pre-Deployment Checklist

Before deploying to production:

- [ ] Train and validate ML model with real data
- [ ] Configure environment variables in `.env`
- [ ] Set up SSL/TLS certificates
- [ ] Configure API authentication
- [ ] Set up monitoring and alerting
- [ ] Configure audit logging
- [ ] Review and customize clinical safety rules
- [ ] Perform security audit
- [ ] Load testing and performance validation
- [ ] Backup and disaster recovery plan
- [ ] HIPAA compliance review (if applicable)
- [ ] Prepare incident response plan

---

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t carelink-ai:v1.0.0 .

# Run container
docker run -d \
  --name carelink-ai \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  -e API_KEY_ENABLED=true \
  -e API_KEY=your-secure-api-key \
  -v /path/to/models:/app/app/models:ro \
  -v /path/to/logs:/app/logs \
  --restart unless-stopped \
  carelink-ai:v1.0.0
```

### Docker Compose (Production)

```yaml
version: '3.8'

services:
  carelink-ai:
    image: carelink-ai:v1.0.0
    ports:
      - "8000:8000"
    environment:
      ENVIRONMENT: production
      LOG_LEVEL: INFO
      API_KEY_ENABLED: "true"
      API_KEY: ${API_KEY}
    volumes:
      - ./models:/app/app/models:ro
      - ./logs:/app/logs
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: carelink-ai
  namespace: healthcare
  labels:
    app: carelink-ai
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: carelink-ai
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: carelink-ai
        version: v1.0.0
    spec:
      containers:
      - name: carelink-ai
        image: carelink-ai:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: carelink-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/app/models
          readOnly: true
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: carelink-models
      - name: logs
        persistentVolumeClaim:
          claimName: carelink-logs
---
apiVersion: v1
kind: Service
metadata:
  name: carelink-ai-service
  namespace: healthcare
spec:
  selector:
    app: carelink-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: Secret
metadata:
  name: carelink-secrets
  namespace: healthcare
type: Opaque
stringData:
  api-key: "your-secure-api-key-here"
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: carelink-ai-hpa
  namespace: healthcare
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: carelink-ai
  minReplicas: 3
  maxReplicas: 10
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
```

---

## AWS Deployment

### Using ECS (Elastic Container Service)

```bash
# Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag carelink-ai:v1.0.0 <account-id>.dkr.ecr.us-east-1.amazonaws.com/carelink-ai:v1.0.0
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/carelink-ai:v1.0.0

# Create ECS task definition and service
aws ecs create-service \
  --cluster carelink-cluster \
  --service-name carelink-ai \
  --task-definition carelink-ai:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Using Lambda (Serverless)

For low-traffic scenarios, deploy as AWS Lambda with API Gateway:

```python
# lambda_handler.py
from mangum import Mangum
from app.main import app

handler = Mangum(app)
```

---

## Azure Deployment

### Azure Container Instances

```bash
az container create \
  --resource-group carelink-rg \
  --name carelink-ai \
  --image carelink-ai:v1.0.0 \
  --cpu 2 \
  --memory 4 \
  --port 8000 \
  --environment-variables \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
  --secure-environment-variables \
    API_KEY=your-secure-api-key
```

### Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group carelink-rg \
  --name carelink-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Deploy application
kubectl apply -f kubernetes/
```

---

## Configuration

### Environment Variables

Production environment variables:

```bash
# Required
ENVIRONMENT=production
APP_VERSION=1.0.0
API_KEY=<generate-strong-key>

# Server
HOST=0.0.0.0
PORT=8000

# Security
API_KEY_ENABLED=true
CORS_ORIGINS=https://app.carelink.com

# Monitoring
ENABLE_METRICS=true
ENABLE_AUDIT_LOGGING=true
LOG_LEVEL=INFO

# Model
MODEL_PATH=app/models/risk_model.pkl
MODEL_VERSION=v1.0.0

# Thresholds
CONFIDENCE_THRESHOLD=0.6
ESCALATION_THRESHOLD=0.75
```

### Secrets Management

Use proper secrets management:

- **Kubernetes**: Use Secrets or external secret managers (Vault, AWS Secrets Manager)
- **Docker**: Use Docker Secrets
- **Cloud**: Use cloud provider's secret management (AWS Secrets Manager, Azure Key Vault)

---

## Monitoring & Logging

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'carelink-ai'
    static_configs:
      - targets: ['carelink-ai:8000']
    metrics_path: '/api/v1/metrics/prometheus'
```

### Grafana Dashboard

Create dashboards to monitor:
- Request rate
- Response time (p50, p95, p99)
- Error rate
- Prediction distribution
- Model performance

### Centralized Logging

Configure log aggregation:
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Cloud**: CloudWatch, Azure Monitor, Google Cloud Logging
- **Third-party**: Datadog, Splunk, New Relic

---

## Security Hardening

### 1. Enable HTTPS

```bash
# Use reverse proxy (Nginx, Traefik)
# Or terminate SSL at load balancer
```

### 2. API Authentication

```python
# Enable in .env
API_KEY_ENABLED=true
API_KEY=<strong-random-key>
```

### 3. Rate Limiting

Configure rate limits per client:
```python
RATE_LIMIT_PER_MINUTE=60
```

### 4. Security Headers

Already implemented in application. Verify:
- X-Content-Type-Options
- X-Frame-Options
- Strict-Transport-Security

### 5. Network Security

- Use VPC/VNET isolation
- Configure security groups
- Enable WAF (Web Application Firewall)

---

## Scaling Strategies

### Horizontal Scaling

- Use Kubernetes HPA
- Configure based on CPU/Memory metrics
- Consider custom metrics (request rate)

### Vertical Scaling

- Increase container resources
- Optimize model inference
- Use GPU for large models (if needed)

### Database Scaling (Audit Logs)

- Use managed database services
- Configure read replicas
- Implement log rotation

---

## Disaster Recovery

### Backup Strategy

1. **Model Backups**: Version and backup trained models
2. **Configuration Backups**: Backup environment configs
3. **Audit Logs**: Implement log retention policy

### Recovery Procedures

```bash
# Rollback deployment
kubectl rollout undo deployment/carelink-ai

# Restore from backup
# Restore model files and configuration
```

### Health Checks

Configure monitoring alerts for:
- Service downtime
- High error rate
- Model performance degradation
- Resource exhaustion

---

## Performance Optimization

### 1. Model Optimization

- Use quantized models
- Implement model caching
- Consider batch inference

### 2. Caching

- Cache feature engineering results
- Cache LLM responses (with appropriate TTL)

### 3. Database Optimization

- Index audit log queries
- Implement connection pooling
- Use read replicas

---

## Compliance & Regulations

### HIPAA Compliance

If handling PHI:
- Encrypt data at rest and in transit
- Implement audit logging
- Configure access controls
- Sign BAA with cloud providers
- Regular security audits

### GDPR Compliance

- Implement data retention policies
- Support data deletion requests
- Maintain consent records

---

## Support & Maintenance

### Monitoring Checklist

- [ ] Set up uptime monitoring
- [ ] Configure error alerting
- [ ] Monitor model performance
- [ ] Track API usage
- [ ] Set up log aggregation

### Regular Maintenance

- Weekly: Review error logs
- Monthly: Model performance review
- Quarterly: Security audit
- Annually: Update dependencies

---

**For questions or issues, contact: engineering@carelink.example.com**
