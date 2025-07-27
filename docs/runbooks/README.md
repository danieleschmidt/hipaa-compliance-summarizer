# Operational Runbooks

This directory contains operational runbooks for the HIPAA Compliance Summarizer system.

## Available Runbooks

| Runbook | Purpose | Audience |
|---------|---------|----------|
| [Incident Response](incident-response.md) | Handle production incidents | DevOps, SRE |
| [Deployment Guide](deployment.md) | Deploy application to environments | DevOps |
| [Monitoring Setup](monitoring-setup.md) | Configure monitoring and alerting | SRE |
| [Backup & Recovery](backup-recovery.md) | Data backup and disaster recovery | Operations |
| [Security Incident](security-incident.md) | Handle security breaches | Security Team |
| [Performance Troubleshooting](performance-troubleshooting.md) | Diagnose performance issues | DevOps, SRE |

## Emergency Contacts

- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Security Team**: security@company.com  
- **DevOps Lead**: devops-lead@company.com
- **Product Owner**: product@company.com

## Quick Reference

### Health Check Endpoints
- Application Health: `GET /health`
- Metrics: `GET /metrics`
- Readiness: `GET /ready`

### Log Locations
- Application Logs: `/app/logs/`
- System Logs: `/var/log/`
- Audit Logs: `/app/logs/audit/`

### Key Configuration Files
- Main Config: `config/hipaa_config.yml`
- Environment: `.env`
- Docker Compose: `docker-compose.yml`