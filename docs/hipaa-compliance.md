# HIPAA Compliance Guide

This guide provides an overview of how the project helps maintain HIPAA compliance.

- **PHI Redaction**: Automatically detects and masks Protected Health Information.
- **Audit Logging**: Optional dashboard summarises detected PHI and compliance scores.
- **Streaming Redaction**: `PHIRedactor.redact_file` handles large files efficiently.
- **Progress Indicators**: `BatchProcessor.process_directory` can display per-file progress.
- **Configurable Levels**: Choose strict, standard or minimal compliance modes.

See [security.md](security.md) for operational best practices and [ehr-integration.md](ehr-integration.md) for EHR system examples.

Configuration is loaded from ``config/hipaa_config.yml`` by default. Use the
``HIPAA_CONFIG_PATH`` environment variable to override the location.

Review the `README.md` for basic usage examples.
