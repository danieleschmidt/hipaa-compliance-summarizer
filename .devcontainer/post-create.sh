#!/bin/bash

# HIPAA Compliance Summarizer - Development Environment Setup
# This script configures the development environment after container creation

set -e

echo "🚀 Starting HIPAA Compliance Summarizer development environment setup..."

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create necessary directories
print_status "Creating project directories..."
mkdir -p /workspace/{logs,temp,processed,uploads,reports}
mkdir -p /workspace/data/{input,output,cache}
mkdir -p /workspace/.pytest_cache
mkdir -p /workspace/.mypy_cache
mkdir -p /workspace/.ruff_cache

# Set proper permissions
chmod 755 /workspace/{logs,temp,processed,uploads,reports}
chmod 755 /workspace/data/{input,output,cache}

print_success "Project directories created"

# Update package lists
print_status "Updating package lists..."
apt-get update -qq

# Install system dependencies for healthcare document processing
print_status "Installing system dependencies..."
apt-get install -y -qq \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    antiword \
    pandoc \
    imagemagick \
    libmagic1 \
    wkhtmltopdf \
    curl \
    wget \
    jq \
    tree \
    htop \
    vim \
    git-lfs

print_success "System dependencies installed"

# Install Python development tools
print_status "Installing Python development tools..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
print_status "Installing Python dependencies..."
if [ -f "/workspace/requirements.txt" ]; then
    pip install -r /workspace/requirements.txt
    print_success "Production dependencies installed"
else
    print_warning "requirements.txt not found, skipping dependency installation"
fi

# Install development dependencies
print_status "Installing development dependencies..."
pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-xdist>=3.0.0 \
    pytest-mock>=3.10.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.5.0 \
    pre-commit>=3.0.0 \
    bandit>=1.7.0 \
    safety>=2.0.0 \
    pip-audit>=2.6.0 \
    coverage>=7.0.0 \
    flake8>=6.0.0 \
    isort>=5.12.0 \
    pydocstyle>=6.3.0 \
    pylint>=2.17.0 \
    jupyter>=1.0.0 \
    ipykernel>=6.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0

print_success "Development dependencies installed"

# Install the package in development mode
print_status "Installing package in development mode..."
if [ -f "/workspace/pyproject.toml" ] || [ -f "/workspace/setup.py" ]; then
    pip install -e /workspace
    print_success "Package installed in development mode"
else
    print_warning "No setup.py or pyproject.toml found, skipping package installation"
fi

# Set up pre-commit hooks
print_status "Setting up pre-commit hooks..."
if [ -f "/workspace/.pre-commit-config.yaml" ]; then
    cd /workspace && pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
fi

# Set up Git configuration
print_status "Configuring Git..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Configure Git for healthcare compliance (if not already set)
if ! git config --global user.name >/dev/null 2>&1; then
    git config --global user.name "Development Environment"
fi

if ! git config --global user.email >/dev/null 2>&1; then
    git config --global user.email "dev@hipaa-summarizer.local"
fi

print_success "Git configuration completed"

# Create .env file from template if it doesn't exist
print_status "Setting up environment configuration..."
if [ ! -f "/workspace/.env" ] && [ -f "/workspace/.env.example" ]; then
    cp /workspace/.env.example /workspace/.env
    print_success "Environment file created from template"
    print_warning "Please update .env with your specific configuration values"
elif [ -f "/workspace/.env" ]; then
    print_success "Environment file already exists"
else
    print_warning "No .env.example found, please create .env manually"
fi

# Set up logging directories
print_status "Setting up logging configuration..."
mkdir -p /workspace/logs/{application,audit,security,performance}
touch /workspace/logs/application/app.log
touch /workspace/logs/audit/audit.log
touch /workspace/logs/security/security.log
touch /workspace/logs/performance/performance.log

# Set appropriate permissions for log files
chmod 644 /workspace/logs/**/*.log

print_success "Logging directories configured"

# Initialize testing environment
print_status "Initializing testing environment..."
if [ -d "/workspace/tests" ]; then
    cd /workspace && python -m pytest --collect-only >/dev/null 2>&1 && \
        print_success "Test discovery completed successfully" || \
        print_warning "Some issues found during test discovery"
else
    print_warning "Tests directory not found"
fi

# Set up Jupyter kernel
print_status "Setting up Jupyter kernel..."
python -m ipykernel install --user --name hipaa-summarizer --display-name "HIPAA Summarizer"
print_success "Jupyter kernel installed"

# Create sample configuration files for development
print_status "Creating development configuration files..."

# Create a sample test configuration
cat > /workspace/pytest.ini << 'EOF'
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --strict-config
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    security: marks tests as security tests
    performance: marks tests as performance tests
EOF

print_success "Development configuration files created"

# Set up shell aliases for common tasks
print_status "Setting up development aliases..."
cat >> /home/vscode/.bashrc << 'EOF'

# HIPAA Compliance Summarizer Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Project-specific aliases
alias hipaa-test='python -m pytest -v'
alias hipaa-cov='python -m pytest --cov=hipaa_compliance_summarizer --cov-report=html'
alias hipaa-lint='ruff check src/ tests/'
alias hipaa-format='ruff format src/ tests/'
alias hipaa-type='mypy src/'
alias hipaa-security='bandit -r src/'
alias hipaa-audit='pip-audit'
alias hipaa-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'

# Logging helpers
alias logs='tail -f logs/application/app.log'
alias audit-logs='tail -f logs/audit/audit.log'
alias security-logs='tail -f logs/security/security.log'

# Quick development commands
alias run-tests='python -m pytest tests/ -v --tb=short'
alias run-coverage='python -m pytest --cov=hipaa_compliance_summarizer --cov-report=term-missing'
alias run-security='bandit -r src/ && safety check'
alias run-quality='ruff check src/ tests/ && mypy src/'

EOF

print_success "Development aliases configured"

# Set up VS Code workspace settings
print_status "Configuring VS Code workspace..."
mkdir -p /workspace/.vscode

cat > /workspace/.vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.activateEnvironment": false,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit",
            "source.fixAll.ruff": "explicit"
        }
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/.coverage": true,
        "**/htmlcov": true,
        "**/*.egg-info": true
    },
    "editor.rulers": [88],
    "editor.insertSpaces": true,
    "editor.tabSize": 4,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "terminal.integrated.cwd": "/workspace",
    "git.openRepositoryInParentFolders": "never"
}
EOF

print_success "VS Code workspace configured"

# Run initial security and quality checks
print_status "Running initial code quality checks..."
cd /workspace

# Check if we can run basic linting
if command -v ruff >/dev/null 2>&1 && [ -d "src" ]; then
    print_status "Running ruff checks..."
    ruff check src/ --fix || print_warning "Ruff found issues to fix"
    print_success "Ruff checks completed"
fi

# Run security scan if source exists
if command -v bandit >/dev/null 2>&1 && [ -d "src" ]; then
    print_status "Running security scan..."
    bandit -r src/ -f json -o /workspace/security_scan_results.json || print_warning "Security scan found issues"
    print_success "Security scan completed"
fi

# Create a development health check script
print_status "Creating development health check..."
cat > /workspace/dev-health-check.sh << 'EOF'
#!/bin/bash
echo "🏥 HIPAA Compliance Summarizer - Development Environment Health Check"
echo "=================================================================="

echo "📦 Python Environment:"
python --version
echo "📍 Python Path: $(which python)"
echo "🔧 Pip Version: $(pip --version)"

echo ""
echo "🧪 Testing Framework:"
pytest --version 2>/dev/null && echo "✅ pytest available" || echo "❌ pytest not available"

echo ""
echo "🔍 Code Quality Tools:"
ruff --version 2>/dev/null && echo "✅ ruff available" || echo "❌ ruff not available"
mypy --version 2>/dev/null && echo "✅ mypy available" || echo "❌ mypy not available"
bandit --version 2>/dev/null && echo "✅ bandit available" || echo "❌ bandit not available"

echo ""
echo "📁 Project Structure:"
[ -d "src" ] && echo "✅ src/ directory exists" || echo "❌ src/ directory missing"
[ -d "tests" ] && echo "✅ tests/ directory exists" || echo "❌ tests/ directory missing"
[ -f "pyproject.toml" ] && echo "✅ pyproject.toml exists" || echo "❌ pyproject.toml missing"
[ -f ".env" ] && echo "✅ .env file exists" || echo "⚠️  .env file missing (use .env.example)"

echo ""
echo "🔐 Security Configuration:"
[ -f "config/hipaa_config.yml" ] && echo "✅ HIPAA config exists" || echo "❌ HIPAA config missing"

echo ""
echo "🗂️  Log Directories:"
[ -d "logs" ] && echo "✅ logs/ directory exists" || echo "❌ logs/ directory missing"

echo ""
echo "=================================================================="
echo "🎉 Health check completed!"
EOF

chmod +x /workspace/dev-health-check.sh
print_success "Development health check script created"

# Final setup steps
print_status "Finalizing development environment setup..."

# Set proper ownership
chown -R vscode:vscode /workspace || true

# Create a welcome message
cat > /workspace/DEVELOPMENT_SETUP.md << 'EOF'
# Development Environment Setup Complete! 🎉

Welcome to the HIPAA Compliance Summarizer development environment.

## Quick Start Commands

```bash
# Run all tests
hipaa-test

# Run tests with coverage
hipaa-cov

# Format code
hipaa-format

# Check code quality
hipaa-lint

# Run security scan
hipaa-security

# Health check
./dev-health-check.sh
```

## Development Workflow

1. Make your changes in `src/`
2. Add tests in `tests/`
3. Run `hipaa-test` to verify tests pass
4. Run `hipaa-lint` to check code quality
5. Run `hipaa-security` to check for security issues
6. Commit your changes

## Important Files

- `.env` - Environment configuration (copy from `.env.example`)
- `config/hipaa_config.yml` - HIPAA compliance configuration
- `logs/` - Application logs
- `tests/` - Test files

## VS Code Features

The environment includes pre-configured VS Code settings for:
- Python development with Ruff and MyPy
- Automatic formatting on save
- Test discovery and running
- Git integration
- Healthcare-specific extensions

## Security Reminders

- Never commit real PHI data
- Use synthetic test data only
- Follow HIPAA compliance guidelines
- Run security scans regularly

Happy coding! 🚀
EOF

print_success "Development setup documentation created"

# Print final summary
echo ""
echo "🎊 HIPAA Compliance Summarizer development environment setup completed!"
echo ""
echo "📋 Setup Summary:"
echo "   ✅ System dependencies installed"
echo "   ✅ Python development tools configured"
echo "   ✅ Pre-commit hooks set up"
echo "   ✅ VS Code workspace configured"
echo "   ✅ Development directories created"
echo "   ✅ Logging infrastructure ready"
echo "   ✅ Testing environment initialized"
echo "   ✅ Security tools configured"
echo ""
echo "🚀 Next Steps:"
echo "   1. Copy .env.example to .env and configure"
echo "   2. Run './dev-health-check.sh' to verify setup"
echo "   3. Run 'hipaa-test' to execute the test suite"
echo "   4. Review DEVELOPMENT_SETUP.md for workflow guidance"
echo ""
echo "💡 For help, run any of the configured aliases (hipaa-test, hipaa-lint, etc.)"
echo ""

# Source the new bashrc to make aliases available
source /home/vscode/.bashrc || true

print_success "🎉 Ready for HIPAA-compliant healthcare AI development!"