# Workflow Requirements

## Required GitHub Actions Workflows

The following workflows should be manually created in `.github/workflows/`:

### Core Workflows
- **ci.yml** - Continuous Integration pipeline
- **security.yml** - Security scanning and vulnerability checks
- **release.yml** - Automated release management
- **dependency-update.yml** - Automated dependency updates

### Workflow Requirements
- Python 3.8+ testing matrix
- Code coverage reporting (>80%)
- Security scanning with Bandit
- Type checking with mypy
- Linting with ruff
- Pre-commit hook validation

### Manual Setup Required
1. Configure branch protection rules
2. Set up repository secrets for CI/CD
3. Enable security advisories
4. Configure automated dependency updates

## Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI/CD Best Practices](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Security Workflow Examples](https://github.com/github/super-linter)

## Integration with Existing Tools
- Uses Makefile commands for consistency
- Leverages pyproject.toml configuration
- Integrates with pre-commit hooks

For detailed setup instructions, see [SETUP_REQUIRED.md](../SETUP_REQUIRED.md)