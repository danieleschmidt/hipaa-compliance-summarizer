# Manual Setup Requirements

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## Repository Configuration Required

### 1. GitHub Actions Workflows

**Location**: Copy files from `docs/workflows/examples/` to `.github/workflows/`

```bash
# Copy workflow templates to active directory
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/deploy.yml .github/workflows/
```

**Required Repository Secrets**:
```bash
# Container Registry
DOCKER_HUB_USERNAME
DOCKER_HUB_ACCESS_TOKEN

# Security Scanning
SONAR_TOKEN
SNYK_TOKEN
CODECOV_TOKEN

# Deployment
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
STAGING_DEPLOY_KEY
PRODUCTION_DEPLOY_KEY

# Notifications
SLACK_WEBHOOK_URL
TEAMS_WEBHOOK_URL
```

**Required Repository Variables**:
```bash
AWS_REGION=us-west-2
STAGING_URL=https://staging.hipaa-summarizer.com
PRODUCTION_URL=https://hipaa-summarizer.com
MONITORING_DASHBOARD=https://grafana.hipaa-summarizer.com
```

### 2. Branch Protection Rules

Configure the following branch protection rules for `main`:

- ✅ Require pull request reviews before merging (minimum 2 reviewers)
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require linear history
- ✅ Restrict pushes that create files
- ✅ Require deployments to succeed before merging

**Required Status Checks**:
- CI/CD Pipeline / security-preflight
- CI/CD Pipeline / code-quality
- CI/CD Pipeline / test
- CI/CD Pipeline / security-scan
- CI/CD Pipeline / container-build

### 3. Repository Settings

**General Settings**:
- Description: "Healthcare-focused LLM agent for HIPAA-compliant PHI detection and document processing"
- Homepage: https://hipaa-summarizer.com
- Topics: `healthcare`, `hipaa`, `phi-detection`, `compliance`, `python`, `machine-learning`

**Security Settings**:
- ✅ Private vulnerability reporting
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Code scanning alerts

**Actions Settings**:
- Workflow permissions: "Read and write permissions"
- ✅ Allow GitHub Actions to create and approve pull requests

## HIPAA Compliance Configuration

### 1. Security Requirements
- All secrets must be encrypted at rest
- Access logs must be maintained for 7 years
- PHI processing must be audited and logged
- TLS 1.2+ required for all communications

### 2. Monitoring Requirements
- Real-time PHI detection monitoring
- Security incident alerting
- Compliance metric tracking
- Automated vulnerability scanning

### 3. Documentation Requirements
- Risk assessment documentation
- Security procedures documentation
- Incident response plans
- Staff training records

## Automation Setup

### 1. Metrics Collection
```bash
# Run automated metrics collection
python scripts/automation/metrics-collector.py
```

### 2. Dependency Management
```bash
# Setup automated dependency updates
python scripts/automation/dependency-updater.py
```

### 3. Release Automation
```bash
# Configure semantic release
npm install -g semantic-release
```

## Contact Information

**Technical Support**: tech-support@hipaa-summarizer.com
**Security Team**: security@hipaa-summarizer.com
**Compliance Officer**: compliance@hipaa-summarizer.com

---

**Note**: This setup is required due to GitHub App permission limitations. All templates and automation scripts are provided in the repository for manual implementation.