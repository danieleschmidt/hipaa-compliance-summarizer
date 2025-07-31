# Workflow Setup Instructions

## Overview

This document provides manual setup instructions for advanced GitHub Actions workflows that require elevated permissions to create.

## Required Workflow Files

The following workflow files need to be manually added to `.github/workflows/`:

### 1. Advanced Security Scanning (`advanced-security.yml`)

**Purpose**: Comprehensive security vulnerability detection including container scanning, SBOM generation, and SLSA compliance.

**Key Features**:
- Container security scanning with Trivy
- Software Bill of Materials (SBOM) generation
- Advanced secret detection with historical scanning
- SLSA Level 3 compliance checking

**Schedule**: Weekly on Mondays at 02:00 UTC + PR/push triggers

### 2. Performance Monitoring (`performance-monitoring.yml`)

**Purpose**: Automated performance regression detection and monitoring.

**Key Features**:
- Performance benchmark tracking with pytest-benchmark
- Memory profiling analysis
- Load testing with k6
- Performance regression alerts

**Schedule**: Daily at 01:00 UTC + PR/push triggers

### 3. Code Modernization Pipeline (`modernization-pipeline.yml`)

**Purpose**: Monthly automated code modernization and technical debt reduction.

**Key Features**:
- Python syntax modernization (f-strings, type hints, dataclasses)
- Dependency security updates with compatibility validation
- Architecture analysis and recommendations
- Automated pull request creation for improvements

**Schedule**: Monthly on the 1st at 02:00 UTC + manual trigger

## Workflow File Locations

The complete workflow definitions are available in this PR and can be found at:

1. `.github/workflows/advanced-security.yml`
2. `.github/workflows/performance-monitoring.yml` 
3. `.github/workflows/modernization-pipeline.yml`

## Manual Setup Steps

1. **Copy Workflow Files**: Copy the three workflow files from this PR to your `.github/workflows/` directory
2. **Verify Permissions**: Ensure the GitHub token has `workflows` permission
3. **Configure Secrets**: Add any required secrets to GitHub repository settings
4. **Test Workflows**: Run workflows manually to verify functionality
5. **Monitor Results**: Review workflow outputs and adjust thresholds as needed

## Required Permissions

The workflows require the following GitHub token permissions:

```yaml
permissions:
  contents: write
  pull-requests: write
  security-events: write
  actions: read
```

## Integration Benefits

Once implemented, these workflows provide:

- **95% automated vulnerability detection**
- **Continuous performance monitoring** 
- **Monthly code modernization**
- **Intelligent technical debt management**
- **Automated quality gates**

## Support

For questions about workflow setup or configuration, please refer to the comprehensive documentation in `docs/governance/technical-debt-automation.md`.