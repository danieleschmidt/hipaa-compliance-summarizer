# Changelog

## v0.0.1 - 2025-06-25

- Applying previous commit.
- Merge pull request #1 from danieleschmidt/codex/generate/update-strategic-development-plan
- docs(review): add comprehensive code review
- Update README.md
- Initial commit

## Unreleased

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
