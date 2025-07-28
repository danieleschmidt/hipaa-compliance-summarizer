# HIPAA Compliance Summarizer - Test Makefile
# Common testing tasks and automation

.PHONY: test test-unit test-integration test-e2e test-performance test-security test-compliance
.PHONY: test-fast test-slow test-coverage test-coverage-html test-coverage-xml
.PHONY: test-cli test-api test-batch test-phi test-encryption
.PHONY: clean-test lint-tests security-scan-tests install-test-deps
.PHONY: test-report test-metrics test-benchmark help

# Default target
test: test-unit test-integration

# Test categories
test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v -m "unit and not slow"

test-integration:
	@echo "🔗 Running integration tests..."
	pytest tests/integration/ -v -m "integration"

test-e2e:
	@echo "🌐 Running end-to-end tests..."
	pytest tests/e2e/ -v -m "e2e"

test-performance:
	@echo "⚡ Running performance tests..."
	pytest tests/performance/ -v -m "performance" --benchmark-only

test-security:
	@echo "🔒 Running security tests..."
	pytest tests/security/ -v -m "security"

test-compliance:
	@echo "📋 Running compliance tests..."
	pytest tests/compliance/ -v -m "compliance"

# Test speed categories
test-fast:
	@echo "🏃 Running fast tests..."
	pytest -v -m "fast and not slow"

test-slow:
	@echo "🐌 Running slow tests..."
	pytest -v -m "slow" --timeout=600

# Coverage testing
test-coverage:
	@echo "📊 Running tests with coverage..."
	pytest --cov=hipaa_compliance_summarizer --cov-report=term-missing

test-coverage-html:
	@echo "📊 Generating HTML coverage report..."
	pytest --cov=hipaa_compliance_summarizer --cov-report=html:htmlcov
	@echo "Coverage report generated: htmlcov/index.html"

test-coverage-xml:
	@echo "📊 Generating XML coverage report..."
	pytest --cov=hipaa_compliance_summarizer --cov-report=xml

# Component-specific tests
test-cli:
	@echo "⌨️  Running CLI tests..."
	pytest -v -m "cli"

test-api:
	@echo "🌐 Running API tests..."
	pytest -v -m "api"

test-batch:
	@echo "📦 Running batch processing tests..."
	pytest -v -m "batch"

test-phi:
	@echo "🏥 Running PHI detection tests..."
	pytest -v -m "phi_test"

test-encryption:
	@echo "🔐 Running encryption tests..."
	pytest -v -m "encryption_test"

# Healthcare-specific tests
test-clinical:
	@echo "🏥 Running clinical note tests..."
	pytest -v -m "clinical_note"

test-lab:
	@echo "🧪 Running lab report tests..."
	pytest -v -m "lab_report"

test-ehr:
	@echo "🏥 Running EHR integration tests..."
	pytest -v -m "ehr_integration"

# Quality and maintenance
lint-tests:
	@echo "🔍 Linting test code..."
	ruff check tests/
	mypy tests/ --ignore-missing-imports

security-scan-tests:
	@echo "🛡️  Scanning test code for security issues..."
	bandit -r tests/ -f json -o test_security_scan.json

clean-test:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test_security_scan.json
	rm -rf reports/
	find tests/ -name "*.pyc" -delete
	find tests/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

install-test-deps:
	@echo "📦 Installing test dependencies..."
	pip install -r requirements-test.txt

# Reporting and metrics
test-report:
	@echo "📈 Generating comprehensive test report..."
	mkdir -p reports/
	pytest --html=reports/test_report.html --self-contained-html
	pytest --cov=hipaa_compliance_summarizer --cov-report=html:reports/coverage
	@echo "Test report: reports/test_report.html"
	@echo "Coverage report: reports/coverage/index.html"

test-metrics:
	@echo "📊 Collecting test metrics..."
	pytest --collect-only -q | grep "test session starts" -A 1000 | grep "collected" | tail -1

test-benchmark:
	@echo "⏱️  Running benchmark tests..."
	pytest tests/performance/ --benchmark-only --benchmark-html=reports/benchmark.html
	@echo "Benchmark report: reports/benchmark.html"

# Parallel testing
test-parallel:
	@echo "🚀 Running tests in parallel..."
	pytest -n auto

test-parallel-coverage:
	@echo "🚀 Running parallel tests with coverage..."
	pytest -n auto --cov=hipaa_compliance_summarizer --cov-report=term-missing

# Environment-specific tests
test-docker:
	@echo "🐳 Running tests in Docker environment..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down

test-local:
	@echo "💻 Running tests in local environment..."
	pytest -v --tb=short

# Continuous integration targets
ci-test:
	@echo "🔄 Running CI test suite..."
	pytest -v --tb=short --cov=hipaa_compliance_summarizer --cov-report=xml --cov-fail-under=80

ci-security:
	@echo "🔄 Running CI security tests..."
	bandit -r src/ -f json -o security_scan.json
	safety check
	pytest tests/security/ -v

ci-compliance:
	@echo "🔄 Running CI compliance tests..."
	pytest tests/compliance/ -v --compliance-report=reports/compliance.json

# Help target
help:
	@echo "HIPAA Compliance Summarizer - Test Commands"
	@echo "==========================================="
	@echo ""
	@echo "Test Categories:"
	@echo "  test-unit          Run unit tests"
	@echo "  test-integration   Run integration tests"
	@echo "  test-e2e          Run end-to-end tests"
	@echo "  test-performance  Run performance tests"
	@echo "  test-security     Run security tests"
	@echo "  test-compliance   Run compliance tests"
	@echo ""
	@echo "Test Speed:"
	@echo "  test-fast         Run fast tests only"
	@echo "  test-slow         Run slow tests only"
	@echo ""
	@echo "Coverage:"
	@echo "  test-coverage     Run tests with coverage"
	@echo "  test-coverage-html Generate HTML coverage report"
	@echo "  test-coverage-xml Generate XML coverage report"
	@echo ""
	@echo "Components:"
	@echo "  test-cli          Test CLI components"
	@echo "  test-api          Test API components"
	@echo "  test-batch        Test batch processing"
	@echo "  test-phi          Test PHI detection"
	@echo "  test-encryption   Test encryption"
	@echo ""
	@echo "Healthcare Specific:"
	@echo "  test-clinical     Test clinical note processing"
	@echo "  test-lab          Test lab report processing"
	@echo "  test-ehr          Test EHR integration"
	@echo ""
	@echo "Quality:"
	@echo "  lint-tests        Lint test code"
	@echo "  security-scan-tests Security scan test code"
	@echo "  clean-test        Clean test artifacts"
	@echo ""
	@echo "Reporting:"
	@echo "  test-report       Generate comprehensive test report"
	@echo "  test-metrics      Show test metrics"
	@echo "  test-benchmark    Run benchmark tests"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci-test           Run CI test suite"
	@echo "  ci-security       Run CI security tests"
	@echo "  ci-compliance     Run CI compliance tests"