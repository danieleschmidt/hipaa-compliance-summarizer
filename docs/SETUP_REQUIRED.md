# Manual Setup Requirements

## Repository Configuration

### Branch Protection Rules
Configure the following branch protection rules for `main`:
- Require pull request reviews (1+ reviewers)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to specific users/teams

### GitHub Actions Secrets
Add the following repository secrets:
- `PYPI_API_TOKEN` - For package publishing
- `CODECOV_TOKEN` - For coverage reporting
- `SECURITY_EMAIL` - For security notifications

## Workflow Setup

### Required GitHub Actions
Create workflows in `.github/workflows/`:
1. **CI Pipeline** (`ci.yml`)
2. **Security Scanning** (`security.yml`) 
3. **Release Management** (`release.yml`)
4. **Dependency Updates** (`dependency-update.yml`)

## External Integrations

### Monitoring & Security
- **Dependabot** - Enable automated dependency updates
- **CodeQL** - Enable code scanning for security vulnerabilities
- **Secret Scanning** - Enable automatic secret detection

### Development Tools
- **Pre-commit.ci** - Automated pre-commit hook execution
- **Codecov** - Code coverage reporting and tracking

## Project Settings

### Repository Settings
- Enable Issues, Wiki, and Projects as needed
- Configure merge settings (squash, rebase options)
- Set up repository topics and description
- Configure GitHub Pages if documentation hosting needed

### Team Access
- Configure team permissions and access levels
- Set up CODEOWNERS file for review assignments
- Configure notification settings

## Resources
- [GitHub Repository Settings Guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)
- [Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [GitHub Actions Security Guide](https://docs.github.com/en/actions/security-guides)