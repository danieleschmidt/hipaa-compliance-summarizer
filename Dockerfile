# Multi-stage build for HIPAA Compliance Summarizer
FROM python:3.13-slim as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Install the package
RUN pip install -e .

# Security and vulnerability scanning stage
FROM builder as security-scan

# Install security scanning tools
RUN pip install bandit safety pip-audit

# Run security scans
COPY . ./
RUN bandit -r src/ -f json -o /tmp/bandit-report.json || true
RUN safety check --json --output /tmp/safety-report.json || true
RUN pip-audit --format=json --output=/tmp/pip-audit-report.json || true

# Production stage
FROM python:3.13-slim as production

# Security: Create non-root user
RUN groupadd -r hipaa && useradd -r -g hipaa hipaa

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/home/hipaa/.local/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directory and set ownership
WORKDIR /app
RUN chown -R hipaa:hipaa /app

# Switch to non-root user
USER hipaa

# Copy application from builder stage
COPY --from=builder --chown=hipaa:hipaa /app ./
COPY --from=builder --chown=hipaa:hipaa /usr/local/lib/python3.13/site-packages /home/hipaa/.local/lib/python3.13/site-packages
COPY --from=builder --chown=hipaa:hipaa /usr/local/bin /home/hipaa/.local/bin

# Copy security scan reports
COPY --from=security-scan /tmp/*-report.json /app/security-reports/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/temp && \
    chown -R hipaa:hipaa /app/data /app/logs /app/temp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hipaa_compliance_summarizer; print('OK')" || exit 1

# Expose port for web interface (if implemented)
EXPOSE 8000

# Default command
CMD ["hipaa-summarize", "--help"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install pytest pytest-cov pytest-xdist ruff bandit pre-commit

# Copy all files including tests
COPY . ./

# Install pre-commit hooks
RUN pre-commit install

# Create development user
RUN groupadd -r dev && useradd -r -g dev -s /bin/bash dev
RUN chown -R dev:dev /app
USER dev

# Default to bash for development
CMD ["/bin/bash"]

# Testing stage
FROM development as testing

# Set environment for testing
ENV ENVIRONMENT=test \
    DEBUG=true \
    LOG_LEVEL=DEBUG

# Run the test suite
RUN python -m pytest tests/ -v --cov=hipaa_compliance_summarizer --cov-report=xml --cov-report=term

# Benchmark stage
FROM testing as benchmark

# Install performance testing tools
RUN pip install memory-profiler line-profiler

# Run performance benchmarks
RUN python -m pytest tests/ -k "performance" --benchmark-only || true

# Documentation stage
FROM python:3.13-slim as docs

WORKDIR /app

# Install documentation dependencies
RUN pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin

# Copy documentation source
COPY docs/ ./docs/
COPY README.md ARCHITECTURE.md CONTRIBUTING.md ./

# Build documentation
RUN mkdocs build

# Serve documentation
EXPOSE 8080
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8080"]