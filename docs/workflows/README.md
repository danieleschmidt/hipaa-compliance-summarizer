# GitHub Actions Workflow Templates

This directory contains advanced CI/CD workflow templates that need to be manually copied to `.github/workflows/` directory due to GitHub security restrictions.

## Workflow Files to Copy

### 1. Advanced CI/CD Pipeline
**Source**: See `AUTONOMOUS_SDLC_ENHANCEMENTS.md` for complete implementation  
**Target**: `.github/workflows/advanced-ci.yml`

Features:
- Multi-stage security scanning (Bandit, Safety, pip-audit, Semgrep)
- SBOM generation with CycloneDX  
- Cross-platform testing matrix
- Performance benchmarking integration
- Container security scanning with Trivy
- Automated deployment pipelines

### 2. Supply Chain Security
**Source**: See `AUTONOMOUS_SDLC_ENHANCEMENTS.md` for complete implementation  
**Target**: `.github/workflows/supply-chain-security.yml`  

Features:
- SLSA Level 3 provenance generation
- Container image signing with Cosign
- Dependency vulnerability scanning
- License compliance checking
- Code signing for releases

## Setup Instructions

1. **Copy Workflow Files**:
   The complete workflow YAML files are documented in the enhancement report.
   Copy them manually to `.github/workflows/` directory.

2. **Configure Repository Secrets**:
   - `CODECOV_TOKEN`: For coverage reporting
   - `DOCKER_HUB_USERNAME`: For container registry
   - `DOCKER_HUB_ACCESS_TOKEN`: For container registry  
   - `STAGING_DEPLOY_KEY`: For staging deployment
   - `PRODUCTION_DEPLOY_KEY`: For production deployment

3. **Configure Repository Variables**:
   - `STAGING_URL`: Staging environment URL
   - `PRODUCTION_URL`: Production environment URL
   - `MONITORING_DASHBOARD`: Link to monitoring dashboard

4. **Enable Required Permissions**:
   - Go to Settings > Actions > General
   - Set "Workflow permissions" to "Read and write permissions"
   - Enable "Allow GitHub Actions to create and approve pull requests"

## Healthcare Compliance Notes

These workflows are specifically designed for healthcare applications with:
- PHI-safe processing validation
- HIPAA compliance verification  
- Healthcare regulation adherence checks
- Clinical workflow smoke tests
- Supply chain security for healthcare vendors

**IMPORTANT**: Review all workflow configurations for your specific healthcare compliance requirements before enabling in production.
EOF < /dev/null
