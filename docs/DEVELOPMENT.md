# Development Guide

## Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- Git

### Installation
```bash
git clone <repository-url>
cd hipaa-compliance-summarizer
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Development Setup
```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
make lint

# Build documentation
make docs
```

### Docker Development
```bash
docker-compose -f docker-compose.dev.yml up
```

## Resources
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Architecture Documentation](../ARCHITECTURE.md)
- [Security Guidelines](../SECURITY.md)
- [Testing Documentation](../tests/README.md)

For detailed setup instructions, see [SETUP_REQUIRED.md](./SETUP_REQUIRED.md)