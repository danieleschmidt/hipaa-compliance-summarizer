# Health Check Configuration

This document describes the health check endpoints and monitoring configuration for the HIPAA Compliance Summarizer.

## Health Check Endpoints

### Primary Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Response Format**: JSON
- **Timeout**: 5 seconds

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00Z",
  "version": "1.2.0",
  "checks": {
    "database": "healthy",
    "cache": "healthy",
    "phi_engine": "healthy",
    "compliance_checker": "healthy"
  },
  "metrics": {
    "uptime_seconds": 86400,
    "memory_usage_percent": 45.2,
    "cpu_usage_percent": 12.8,
    "active_connections": 15
  }
}
```

### Detailed Health Check
- **Endpoint**: `/health/detailed`
- **Method**: GET
- **Authentication**: Required
- **Response**: Comprehensive system status

### Readiness Check
- **Endpoint**: `/health/ready`
- **Method**: GET
- **Purpose**: Kubernetes readiness probe
- **Response**: 200 OK when ready, 503 when not ready

### Liveness Check
- **Endpoint**: `/health/live`
- **Method**: GET
- **Purpose**: Kubernetes liveness probe
- **Response**: Always 200 OK unless critical failure

## Health Check Components

### Database Connectivity
```python
def check_database():
    try:
        # Test database connection
        result = db.execute("SELECT 1")
        return {"status": "healthy", "latency_ms": 12}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### PHI Engine Status
```python
def check_phi_engine():
    try:
        # Test PHI detection with synthetic data
        test_result = phi_detector.test_detection("Test Patient")
        return {"status": "healthy", "confidence": test_result.confidence}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Cache Connectivity
```python
def check_cache():
    try:
        # Test cache read/write
        cache.set("health_check", "ok", ttl=10)
        result = cache.get("health_check")
        return {"status": "healthy" if result == "ok" else "degraded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Monitoring Integration

### Prometheus Metrics
Health check results are exposed as Prometheus metrics:

```prometheus
# Service availability
up{job="hipaa-summarizer"} 1

# Component health status
component_health{component="database"} 1
component_health{component="cache"} 1
component_health{component="phi_engine"} 1

# Response time metrics
health_check_duration_seconds{endpoint="/health"} 0.025
```

### Grafana Dashboards
Health check metrics are visualized in Grafana dashboards:
- Overall system health status
- Component availability over time
- Health check response times
- Alert status and history

### Alert Rules
Critical health check failures trigger immediate alerts:

```yaml
- alert: HealthCheckFailed
  expr: up{job="hipaa-summarizer"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Health check failed"
```

## HIPAA Compliance Considerations

### PHI Safety
- Health checks never expose actual PHI data
- Test data uses synthetic, clearly fake information
- Responses are sanitized to prevent data leakage

### Audit Logging
All health check access is logged for compliance:
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "event": "health_check_accessed",
  "endpoint": "/health",
  "source_ip": "10.0.1.15",
  "user_agent": "Prometheus/2.40.0",
  "response_status": 200,
  "response_time_ms": 25
}
```

### Security Headers
Health check endpoints include security headers:
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Cache-Control: no-cache, no-store, must-revalidate
```

## Deployment Configuration

### Docker Health Check
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Kubernetes Probes
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
```

### Load Balancer Configuration
```nginx
location /health {
    access_log off;
    proxy_pass http://backend;
    proxy_connect_timeout 1s;
    proxy_read_timeout 3s;
}
```

## Troubleshooting

### Common Issues

#### Database Connection Failures
- Verify database credentials
- Check network connectivity
- Review database logs

#### PHI Engine Failures
- Check model file availability
- Verify memory allocation
- Review processing logs

#### Cache Failures
- Check Redis connectivity
- Verify cache configuration
- Review memory usage

### Emergency Procedures

#### Critical Health Check Failure
1. Immediate alert notification
2. Automatic failover (if configured)
3. Incident response activation
4. Service degradation assessment

#### Gradual Performance Degradation
1. Scale monitoring frequency
2. Investigate resource usage
3. Consider load reduction
4. Plan maintenance window

## Best Practices

### Development
- Test health checks in development environment
- Mock external dependencies appropriately
- Validate timeout configurations
- Ensure PHI safety in test data

### Production
- Monitor health check response times
- Set appropriate alert thresholds
- Regular review of health check logic
- Document incident response procedures

### Security
- Limit health check endpoint access
- Use authentication for detailed checks
- Regular security review of endpoints
- Audit access patterns