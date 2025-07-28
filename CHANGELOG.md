# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security & Error Handling Improvements
- Fix empty except blocks that could silently swallow errors
- Add proper logging to UnicodeDecodeError in file validation
- Improve error handling in config URL parsing with warning logs
- Add error information to cache performance metrics when calculation fails
- Enhance error visibility for debugging and monitoring

### Features
- Add `BatchProcessor` for directory processing
- Document batch processing feature
- Add CLI utilities for summarizing documents and compliance reporting
- Package CLI scripts under `hipaa_compliance_summarizer.cli` with console
  entry points
- Add `--show-dashboard` option to `hipaa-batch-process`
- Batch dashboard provides friendly string output for CLI
- Rename batch process flag to `--compliance-level` for consistency
- Add `--dashboard-json` option to save dashboard metrics as JSON
- Provide `BatchProcessor.save_dashboard` helper for exporting metrics
- Introduce GitHub Actions CI with lint, security scan and tests
- Add CONTRIBUTING guide and initial compliance docs
- Fix truncated `.gitignore`
- Load scoring and PHI patterns from configurable `config/hipaa_config.yml`
- Add logging to processor and batch modules
- Install `pip-audit` in CI for vulnerability scanning
- Stream large files with `PHIRedactor.redact_file`
- Show progress in `BatchProcessor.process_directory`

## v0.0.2 - 2025-07-23

### Security Improvements
- Fix 5 MD5 usage warnings by adding `usedforsecurity=False` parameter for non-security hashing
- Enhance exception handling with proper logging in empty except blocks
- Complete security vulnerability scan with bandit (0 issues found)
- Conduct dependency audit with pip-audit (identified system-level vulnerabilities)
- Generate comprehensive security audit report with remediation recommendations

### Code Quality Enhancements
- Add comprehensive docstrings to all CLI main() functions
- Implement proper exception classes with attributes and context methods:
  - SecurityError with file_path, violation_type, and context tracking
  - ParsingError, FileReadError, EncodingError with enhanced metadata
  - DocumentError, DocumentTypeError with validation details
- Create centralized constants module to eliminate hardcoded values
- Extract test data constants and magic numbers to improve maintainability

### Documentation & Reporting
- Add SECURITY_AUDIT_REPORT.md with detailed vulnerability analysis
- Update TECHNICAL_DEBT_REPORT.md with comprehensive debt tracking
- Create COMPREHENSIVE_BACKLOG.md with WSJF-prioritized task management
- Enhance CLI documentation with detailed argument descriptions

### Infrastructure
- Integrate autonomous continuous backlog execution system
- Implement WSJF (Weighted Shortest Job First) prioritization methodology
- Establish security scanning automation with bandit and pip-audit
