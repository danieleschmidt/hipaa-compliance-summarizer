# Environment-Based Configuration

This document describes how to configure the HIPAA Compliance Summarizer using environment variables for secure secret management.

## Overview

The application supports loading sensitive configuration from environment variables to avoid storing secrets in configuration files. This follows security best practices and enables secure deployment in containerized and cloud environments.

## Supported Environment Variables

### Required for Production

| Variable | Description | Example |
|----------|-------------|---------|
| `ENCRYPTION_KEY` | Base64-encoded encryption key for data at rest | `base64-encoded-key-here` |

### API Integration

| Variable | Description | Example |
|----------|-------------|---------|
| `EPIC_API_KEY` | API key for Epic EHR integration | `epic_prod_key_123456` |
| `OPENAI_API_KEY` | OpenAI API key for LLM processing | `sk-1234567890abcdef` |
| `AWS_ACCESS_KEY_ID` | AWS access key for cloud services | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for cloud services | `secretkey123` |

### Database and Cache

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` | <!-- pragma: allowlist secret -->
| `REDIS_URL` | Redis connection string for caching | `redis://localhost:6379/0` |

### Cloud and Security

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_KEY_VAULT_URL` | Azure Key Vault URL for secret management | `https://vault.vault.azure.net/` |
| `JWT_SECRET_KEY` | Secret key for JWT token signing | `your-jwt-secret-key` |

### Application Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENVIRONMENT` | Deployment environment | `development` | `production` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |
| `LOG_FORMAT` | Log message format | Standard format | `%(levelname)s: %(message)s` |
| `DEBUG` | Enable debug mode | `false` | `true` |

## Usage Examples

### Development Environment

```bash
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development
export EPIC_API_KEY=test_key_123  # pragma: allowlist secret

# Run batch processing
hipaa-batch-process --input-dir ./docs --output-dir ./output
```

### Production Environment

```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export ENCRYPTION_KEY=YourBase64EncodedEncryptionKey
export DATABASE_URL=postgresql://user:password@db.example.com:5432/hipaa_prod  # pragma: allowlist secret
export EPIC_API_KEY=prod_epic_key_secure  # pragma: allowlist secret
export AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net/

# Run with production configuration
hipaa-batch-process --input-dir /data/input --output-dir /data/output --compliance-level strict
```

### Docker Environment

Create a `.env` file:

```bash
# .env file
ENVIRONMENT=production
ENCRYPTION_KEY=your-encryption-key  # pragma: allowlist secret
DATABASE_URL=postgresql://user:password@postgres:5432/hipaa  # pragma: allowlist secret  
EPIC_API_KEY=your-epic-key  # pragma: allowlist secret
LOG_LEVEL=INFO
```

Run with Docker:

```bash
docker run --env-file .env hipaa-summarizer:latest
```

### Kubernetes Deployment

Create a Kubernetes Secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hipaa-secrets
type: Opaque
stringData:
  ENCRYPTION_KEY: "your-encryption-key"  # pragma: allowlist secret
  DATABASE_URL: "postgresql://user:password@postgres:5432/hipaa"  # pragma: allowlist secret
  EPIC_API_KEY: "your-epic-key"  # pragma: allowlist secret
```

Reference in deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hipaa-summarizer
spec:
  template:
    spec:
      containers:
      - name: app
        image: hipaa-summarizer:latest
        envFrom:
        - secretRef:
            name: hipaa-secrets
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
```

## Configuration Validation

The application automatically validates configuration at startup:

```python
from hipaa_compliance_summarizer.startup import validate_environment

# Validate development environment
validation = validate_environment(require_production_secrets=False)

# Validate production environment  
validation = validate_environment(require_production_secrets=True)

if not validation["valid"]:
    print("Configuration errors:")
    for error in validation["errors"]:
        print(f"  - {error}")
```

## Security Features

### Automatic Masking

Sensitive values are automatically masked in logs:

```python
from hipaa_compliance_summarizer.config import mask_sensitive_config

config = {
    "epic_api_key": "secret_key_12345",  # pragma: allowlist secret
    "database_url": "postgresql://user:password@host/db",  # pragma: allowlist secret
    "non_secret": "public_value"  # pragma: allowlist secret
}

masked = mask_sensitive_config(config)
# Output: {"epic_api_key": "secret***", "database_url": "postgresql://user:***@host/db", "non_secret": "public_value"}  # pragma: allowlist secret
```

### Pre-commit Secret Scanning

The repository includes pre-commit hooks that prevent secrets from being committed:

```bash
# Install pre-commit hooks
pre-commit install

# Test with a file containing secrets
echo 'API_KEY = "secret123"' > test.py  # pragma: allowlist secret
git add test.py
git commit -m "test"  # This will be blocked!
```

### Environment Variable Precedence

Configuration loading follows this precedence order:

1. `HIPAA_CONFIG_YAML` environment variable (YAML string)
2. `HIPAA_CONFIG_PATH` environment variable (file path)
3. Individual environment variables (for secrets)
4. Default configuration file (`config/hipaa_config.yml`)

## Best Practices

### 1. Use Different Secrets for Each Environment

```bash
# Development
export EPIC_API_KEY=dev_key_123  # pragma: allowlist secret

# Production  
export EPIC_API_KEY=prod_key_456  # pragma: allowlist secret
```

### 2. Rotate Secrets Regularly

Set up automatic secret rotation using cloud provider tools:

```bash
# Example with AWS Secrets Manager rotation
aws secretsmanager rotate-secret --secret-id hipaa/encryption-key  # pragma: allowlist secret
```

### 3. Use Secret Management Services

Instead of environment variables, use dedicated secret management:

```python
# Example integration with Azure Key Vault
from azure.keyvault.secrets import SecretClient

def get_secret_from_vault(secret_name):  # pragma: allowlist secret
    vault_url = os.environ["AZURE_KEY_VAULT_URL"]
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value
```

### 4. Validate Configuration in CI/CD

Add configuration validation to your deployment pipeline:

```bash
# In your CI/CD pipeline
python -c "
from hipaa_compliance_summarizer.startup import check_required_environment_for_production
try:
    check_required_environment_for_production()
    print('✓ Configuration validation passed')
except RuntimeError as e:
    print('✗ Configuration validation failed:')
    print(e)
    exit(1)
"
```

## Troubleshooting

### Common Configuration Issues

1. **Missing required secrets in production:**
   ```
   RuntimeError: Production environment validation failed:
   Required secret 'encryption_key' is not configured
   ```
   Solution: Set the `ENCRYPTION_KEY` environment variable.

2. **Invalid URL format:**
   ```
   Invalid URL format for 'database_url': invalid-url
   ```
   Solution: Ensure URLs include scheme (https://, postgresql://, etc.)

3. **Pre-commit hook blocking commits:**
   ```
   ERROR: Potential secrets about to be committed to git repo!
   ```
   Solution: Use environment variables instead of hardcoded secrets, or add `# pragma: allowlist secret` for false positives.

### Debug Configuration Loading

Enable debug logging to see configuration loading:

```bash
export LOG_LEVEL=DEBUG
python -c "
from hipaa_compliance_summarizer.config import get_secret_config
config = get_secret_config()
print('Loaded config keys:', list(config.keys()))
"
```