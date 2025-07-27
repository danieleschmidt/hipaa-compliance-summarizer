# HIPAA Compliance Summarizer Makefile
.PHONY: help install install-dev test test-fast test-integration test-performance \
        lint format security-scan build clean docs serve-docs \
        docker-build docker-test docker-dev docker-prod docker-clean \
        setup-pre-commit release coverage ci

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := hipaa-compliance-summarizer
COVERAGE_THRESHOLD := 80

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)HIPAA Compliance Summarizer - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /Setup/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /Development/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /Testing/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /Docker/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Production Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; /Production/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Setup Commands
install: ## Setup: Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Setup: Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install pytest pytest-cov pytest-xdist ruff bandit pre-commit \
		mypy types-PyYAML commitizen mkdocs mkdocs-material
	@$(MAKE) setup-pre-commit

setup-pre-commit: ## Setup: Install and configure pre-commit hooks
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg

# Development Commands
lint: ## Development: Run code linting
	@echo "$(GREEN)Running code linting...$(NC)"
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Development: Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	ruff format src/ tests/
	ruff check --fix src/ tests/

security-scan: ## Development: Run security scanning
	@echo "$(GREEN)Running security scans...$(NC)"
	bandit -r src/ -f json -o bandit_report.json
	pip-audit -r requirements.txt --format=json --output=pip_audit_report.json

type-check: ## Development: Run type checking
	@echo "$(GREEN)Running type checking...$(NC)"
	mypy src/ --ignore-missing-imports --check-untyped-defs

clean: ## Development: Clean up build artifacts and cache
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	rm -f bandit_report.json pip_audit_report.json

# Testing Commands
test: ## Testing: Run all tests with coverage
	@echo "$(GREEN)Running full test suite...$(NC)"
	$(PYTEST) tests/ -v \
		--cov=hipaa_compliance_summarizer \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html \
		--cov-fail-under=$(COVERAGE_THRESHOLD)

test-fast: ## Testing: Run tests without coverage (fast)
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(PYTEST) tests/ -v -x --disable-warnings

test-unit: ## Testing: Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/ -v -m "not integration and not performance" \
		--cov=hipaa_compliance_summarizer \
		--cov-report=term-missing

test-integration: ## Testing: Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v --integration

test-performance: ## Testing: Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	$(PYTEST) tests/ -v -m "performance" --benchmark-only

coverage: ## Testing: Generate coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	$(PYTEST) tests/ --cov=hipaa_compliance_summarizer \
		--cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

# Docker Commands
docker-build: ## Docker: Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

docker-dev: ## Docker: Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)Development environment available at http://localhost:8001$(NC)"

docker-test: ## Docker: Run tests in Docker
	@echo "$(GREEN)Running tests in Docker...$(NC)"
	$(DOCKER_COMPOSE) --profile test run --rm hipaa-test

docker-prod: ## Docker: Start production environment
	@echo "$(GREEN)Starting production environment...$(NC)"
	$(DOCKER_COMPOSE) --profile production up -d
	@echo "$(GREEN)Production environment available at http://localhost:8000$(NC)"

docker-monitoring: ## Docker: Start with monitoring stack
	@echo "$(GREEN)Starting with monitoring...$(NC)"
	$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "$(GREEN)Grafana available at http://localhost:3000$(NC)"
	@echo "$(GREEN)Prometheus available at http://localhost:9090$(NC)"

docker-logs: ## Docker: View logs from all services
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Docker: Clean up Docker resources
	@echo "$(GREEN)Cleaning up Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -f

# Documentation Commands
docs: ## Development: Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	mkdocs build

serve-docs: ## Development: Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	mkdocs serve --dev-addr=0.0.0.0:8080

docker-docs: ## Docker: Serve documentation with Docker
	@echo "$(GREEN)Starting documentation server...$(NC)"
	$(DOCKER_COMPOSE) --profile docs up -d
	@echo "$(GREEN)Documentation available at http://localhost:8080$(NC)"

# Production Commands
build: ## Production: Build production package
	@echo "$(GREEN)Building production package...$(NC)"
	$(PYTHON) -m build

release: ## Production: Create a release (requires semantic-release)
	@echo "$(GREEN)Creating release...$(NC)"
	@if command -v semantic-release >/dev/null 2>&1; then \
		semantic-release; \
	else \
		echo "$(RED)semantic-release not found. Install with: npm install -g semantic-release$(NC)"; \
		exit 1; \
	fi

publish: ## Production: Publish to PyPI (requires credentials)
	@echo "$(GREEN)Publishing to PyPI...$(NC)"
	$(PYTHON) -m twine upload dist/*

# CI/CD Commands
ci: ## CI/CD: Run full CI pipeline locally
	@echo "$(GREEN)Running CI pipeline...$(NC)"
	@$(MAKE) clean
	@$(MAKE) install-dev
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security-scan
	@$(MAKE) test
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

pre-commit-all: ## Development: Run pre-commit on all files
	@echo "$(GREEN)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files

# Health checks
health: ## Production: Check application health
	@echo "$(GREEN)Checking application health...$(NC)"
	@if curl -f http://localhost:8000/health >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Application is healthy$(NC)"; \
	else \
		echo "$(RED)✗ Application is not responding$(NC)"; \
		exit 1; \
	fi

# Database operations
db-init: ## Development: Initialize database
	@echo "$(GREEN)Initializing database...$(NC)"
	$(DOCKER_COMPOSE) exec postgres psql -U hipaa -d hipaa_compliance -f /docker-entrypoint-initdb.d/init-db.sql

db-migrate: ## Development: Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	# Add migration commands here when implemented

db-backup: ## Production: Backup database
	@echo "$(GREEN)Creating database backup...$(NC)"
	$(DOCKER_COMPOSE) exec postgres pg_dump -U hipaa hipaa_compliance > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Monitoring and metrics
metrics: ## Production: View application metrics
	@echo "$(GREEN)Opening metrics dashboard...$(NC)"
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:3000; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:3000; \
	else \
		echo "$(GREEN)Metrics available at http://localhost:3000$(NC)"; \
	fi

# Security operations
security-full: ## Development: Run comprehensive security scan
	@echo "$(GREEN)Running comprehensive security scan...$(NC)"
	@$(MAKE) security-scan
	bandit -r src/ -ll
	safety check
	pip-audit -r requirements.txt

# Utility commands
env-check: ## Development: Check environment configuration
	@echo "$(GREEN)Checking environment...$(NC)"
	@$(PYTHON) -c "import sys; print(f'Python: {sys.version}')"
	@$(PIP) --version
	@git --version
	@$(DOCKER) --version
	@$(DOCKER_COMPOSE) --version

version: ## Development: Show project version
	@$(PYTHON) -c "import hipaa_compliance_summarizer; print(hipaa_compliance_summarizer.__version__)" 2>/dev/null || echo "0.0.1"

# Quick commands for common workflows
dev-setup: install-dev setup-pre-commit ## Development: Complete development setup
	@echo "$(GREEN)Development environment ready!$(NC)"

quick-test: lint test-fast ## Development: Quick test (lint + fast tests)
	@echo "$(GREEN)Quick tests passed!$(NC)"

full-check: ci ## Development: Full check (complete CI pipeline)
	@echo "$(GREEN)Full check completed!$(NC)"