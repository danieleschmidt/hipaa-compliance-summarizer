# Security Implementation Guide

Follow these practices to keep data secure:

1. Keep dependencies updated.
2. Limit access to PHI using file permissions and access logs.
3. Rotate encryption keys as defined in `config/hipaa_config.yml`.
   Set `HIPAA_CONFIG_PATH` to load a custom config.
4. Run `ruff` and `bandit` on each commit to detect lint and security issues.
5. Scan dependencies with `pip-audit` during CI.
