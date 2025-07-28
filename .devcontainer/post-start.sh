#!/bin/bash

# HIPAA Compliance Summarizer - Post-Start Script
# This script runs every time the dev container starts

set -e

echo "🔄 Starting HIPAA Compliance Summarizer development environment..."

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Navigate to workspace
cd /workspace

# Check environment configuration
print_status "Checking environment configuration..."
if [ -f ".env" ]; then
    print_success "Environment configuration found"
else
    print_warning "No .env file found. Please copy .env.example to .env and configure."
fi

# Verify HIPAA configuration
if [ -f "config/hipaa_config.yml" ]; then
    print_success "HIPAA configuration found"
else
    print_warning "HIPAA configuration not found at config/hipaa_config.yml"
fi

# Update pre-commit hooks
print_status "Updating pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit autoupdate --quiet || print_warning "Failed to update pre-commit hooks"
    print_success "Pre-commit hooks updated"
fi

# Quick health check
print_status "Running quick health check..."

# Check Python environment
if python -c "import sys; print(f'Python {sys.version}')" >/dev/null 2>&1; then
    print_success "Python environment is healthy"
else
    print_warning "Python environment issues detected"
fi

# Check if package is installed
if python -c "import hipaa_compliance_summarizer" >/dev/null 2>&1; then
    print_success "HIPAA Compliance Summarizer package is available"
else
    print_warning "Package not installed. Run 'pip install -e .' to install in development mode"
fi

# Check critical directories
for dir in logs temp processed uploads; do
    if [ -d "$dir" ]; then
        print_success "$dir/ directory is ready"
    else
        mkdir -p "$dir"
        print_status "Created $dir/ directory"
    fi
done

# Display useful information
echo ""
echo "🎯 HIPAA Compliance Summarizer Development Environment"
echo "======================================================"
echo "📍 Workspace: /workspace"
echo "🐍 Python: $(python --version 2>&1)"
echo "📦 Package: $(python -c "import hipaa_compliance_summarizer; print('✅ Installed')" 2>/dev/null || echo '❌ Not installed')"
echo "🔧 Config: $([ -f '.env' ] && echo '✅ .env found' || echo '❌ .env missing')"
echo "🏥 HIPAA: $([ -f 'config/hipaa_config.yml' ] && echo '✅ Config found' || echo '❌ Config missing')"
echo ""
echo "🚀 Quick Commands:"
echo "   hipaa-test     - Run test suite"
echo "   hipaa-cov      - Run tests with coverage"
echo "   hipaa-lint     - Check code quality"
echo "   hipaa-format   - Format code"
echo "   hipaa-security - Run security scan"
echo ""
echo "📚 Documentation:"
echo "   cat DEVELOPMENT_SETUP.md - Development workflow guide"
echo "   ./dev-health-check.sh    - Comprehensive health check"
echo ""

print_success "🎉 Development environment is ready!"