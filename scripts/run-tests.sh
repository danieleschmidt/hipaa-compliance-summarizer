#!/bin/bash

# HIPAA Compliance Summarizer - Test Automation Script
# Comprehensive test execution with reporting and validation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/test-reports"
COVERAGE_DIR="$PROJECT_ROOT/htmlcov"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
HIPAA Compliance Summarizer Test Runner

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -a, --all               Run all tests (default)
    -u, --unit              Run unit tests only
    -i, --integration       Run integration tests only
    -p, --performance       Run performance tests only
    -s, --security          Run security tests only
    -c, --compliance        Run compliance tests only
    -f, --fast              Run fast tests only (skip slow tests)
    --coverage              Generate coverage report
    --benchmark             Run benchmark tests
    --ci                    CI mode (XML output, no interactive)
    --verbose               Verbose output
    --parallel              Run tests in parallel
    --profile               Profile test execution
    --clean                 Clean test artifacts before running

Examples:
    $0                      # Run all tests with coverage
    $0 --unit --coverage    # Run unit tests with coverage
    $0 --fast --parallel    # Run fast tests in parallel
    $0 --ci                 # CI mode with XML reports
    $0 --security --verbose # Run security tests with verbose output

EOF
}

# Parse command line arguments
ARGS=""
RUN_ALL=true
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_PERFORMANCE=false
RUN_SECURITY=false
RUN_COMPLIANCE=false
FAST_ONLY=false
GENERATE_COVERAGE=true
RUN_BENCHMARK=false
CI_MODE=false
VERBOSE=false
PARALLEL=false
PROFILE=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -u|--unit)
            RUN_ALL=false
            RUN_UNIT=true
            shift
            ;;
        -i|--integration)
            RUN_ALL=false
            RUN_INTEGRATION=true
            shift
            ;;
        -p|--performance)
            RUN_ALL=false
            RUN_PERFORMANCE=true
            shift
            ;;
        -s|--security)
            RUN_ALL=false
            RUN_SECURITY=true
            shift
            ;;
        -c|--compliance)
            RUN_ALL=false
            RUN_COMPLIANCE=true
            shift
            ;;
        -f|--fast)
            FAST_ONLY=true
            shift
            ;;
        --coverage)
            GENERATE_COVERAGE=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Setup environment
cd "$PROJECT_ROOT"

log_info "HIPAA Compliance Summarizer Test Runner"
log_info "========================================"
log_info "Project root: $PROJECT_ROOT"
log_info "Timestamp: $TIMESTAMP"

# Clean artifacts if requested
if [[ "$CLEAN" == true ]]; then
    log_info "Cleaning test artifacts..."
    rm -rf "$REPORTS_DIR" "$COVERAGE_DIR" .coverage .pytest_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    log_success "Test artifacts cleaned"
fi

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Verify environment
log_info "Verifying test environment..."

# Check Python environment
if ! python -c "import sys; print(f'Python {sys.version}')" >/dev/null 2>&1; then
    log_error "Python not available or not working"
    exit 1
fi

# Check pytest availability
if ! python -m pytest --version >/dev/null 2>&1; then
    log_error "pytest not available. Install with: pip install pytest"
    exit 1
fi

# Check if package is installed
if ! python -c "import hipaa_compliance_summarizer" >/dev/null 2>&1; then
    log_warning "Package not installed. Installing in development mode..."
    pip install -e .
fi

log_success "Environment verification complete"

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add verbosity
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
else
    PYTEST_CMD="$PYTEST_CMD -q"
fi

# Add parallel execution
if [[ "$PARALLEL" == true ]]; then
    if python -c "import pytest_xdist" >/dev/null 2>&1; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    else
        log_warning "pytest-xdist not available, running in serial mode"
    fi
fi

# Add profiling
if [[ "$PROFILE" == true ]]; then
    if python -c "import pytest_profiling" >/dev/null 2>&1; then
        PYTEST_CMD="$PYTEST_CMD --profile"
    else
        log_warning "pytest-profiling not available, skipping profiling"
    fi
fi

# Add coverage
if [[ "$GENERATE_COVERAGE" == true ]]; then
    if python -c "import pytest_cov" >/dev/null 2>&1; then
        PYTEST_CMD="$PYTEST_CMD --cov=hipaa_compliance_summarizer"
        PYTEST_CMD="$PYTEST_CMD --cov-report=term-missing"
        PYTEST_CMD="$PYTEST_CMD --cov-report=html:$COVERAGE_DIR"
        
        if [[ "$CI_MODE" == true ]]; then
            PYTEST_CMD="$PYTEST_CMD --cov-report=xml:$REPORTS_DIR/coverage.xml"
        fi
    else
        log_warning "pytest-cov not available, skipping coverage"
        GENERATE_COVERAGE=false
    fi
fi

# Add CI mode outputs
if [[ "$CI_MODE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --junit-xml=$REPORTS_DIR/test-results.xml"
fi

# Add benchmark support
if [[ "$RUN_BENCHMARK" == true ]]; then
    if python -c "import pytest_benchmark" >/dev/null 2>&1; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-json=$REPORTS_DIR/benchmark-results.json"
    else
        log_warning "pytest-benchmark not available, skipping benchmarks"
        RUN_BENCHMARK=false
    fi
fi

# Add fast mode
if [[ "$FAST_ONLY" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

# Determine test paths
TEST_PATHS=""

if [[ "$RUN_ALL" == true ]]; then
    TEST_PATHS="tests/"
else
    if [[ "$RUN_UNIT" == true ]]; then
        TEST_PATHS="$TEST_PATHS tests/unit/ tests/test_*.py"
    fi
    if [[ "$RUN_INTEGRATION" == true ]]; then
        TEST_PATHS="$TEST_PATHS tests/integration/"
    fi
    if [[ "$RUN_PERFORMANCE" == true ]]; then
        TEST_PATHS="$TEST_PATHS tests/performance/"
    fi
    if [[ "$RUN_SECURITY" == true ]]; then
        TEST_PATHS="$TEST_PATHS tests/security/"
    fi
    if [[ "$RUN_COMPLIANCE" == true ]]; then
        TEST_PATHS="$TEST_PATHS tests/compliance/"
    fi
fi

# Run pre-test security scan
log_info "Running pre-test security validation..."
if command -v bandit >/dev/null 2>&1; then
    bandit -r src/ -f json -o "$REPORTS_DIR/security-scan.json" -q || log_warning "Security scan found issues"
    log_success "Security scan completed"
else
    log_warning "Bandit not available, skipping security scan"
fi

# Execute tests
log_info "Starting test execution..."
log_info "Command: $PYTEST_CMD $TEST_PATHS"

# Set environment variables for testing
export ENVIRONMENT=test
export DEBUG=true
export LOG_LEVEL=DEBUG
export PYTHONPATH="$PROJECT_ROOT/src"

# Run the tests
START_TIME=$(date +%s)
if eval "$PYTEST_CMD $TEST_PATHS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log_success "Tests completed successfully in ${DURATION}s"
    TEST_RESULT=0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log_error "Tests failed after ${DURATION}s"
    TEST_RESULT=1
fi

# Generate reports summary
log_info "Generating test reports summary..."

if [[ "$GENERATE_COVERAGE" == true && -f "$COVERAGE_DIR/index.html" ]]; then
    log_success "Coverage report generated: $COVERAGE_DIR/index.html"
fi

if [[ "$CI_MODE" == true && -f "$REPORTS_DIR/test-results.xml" ]]; then
    log_success "JUnit XML report generated: $REPORTS_DIR/test-results.xml"
fi

if [[ "$RUN_BENCHMARK" == true && -f "$REPORTS_DIR/benchmark-results.json" ]]; then
    log_success "Benchmark results generated: $REPORTS_DIR/benchmark-results.json"
fi

# Create summary report
SUMMARY_FILE="$REPORTS_DIR/test-summary-$TIMESTAMP.txt"
cat > "$SUMMARY_FILE" << EOF
HIPAA Compliance Summarizer Test Summary
========================================

Execution Time: $TIMESTAMP
Duration: ${DURATION}s
Result: $([ $TEST_RESULT -eq 0 ] && echo "PASSED" || echo "FAILED")

Test Configuration:
- Run All: $RUN_ALL
- Unit Tests: $RUN_UNIT
- Integration Tests: $RUN_INTEGRATION
- Performance Tests: $RUN_PERFORMANCE
- Security Tests: $RUN_SECURITY
- Compliance Tests: $RUN_COMPLIANCE
- Fast Mode: $FAST_ONLY
- Coverage: $GENERATE_COVERAGE
- Parallel: $PARALLEL
- CI Mode: $CI_MODE

Environment:
- Python: $(python --version 2>&1)
- Pytest: $(python -m pytest --version 2>&1)
- Project Root: $PROJECT_ROOT

Reports Generated:
$([ -f "$COVERAGE_DIR/index.html" ] && echo "- Coverage Report: $COVERAGE_DIR/index.html")
$([ -f "$REPORTS_DIR/test-results.xml" ] && echo "- JUnit XML: $REPORTS_DIR/test-results.xml")
$([ -f "$REPORTS_DIR/benchmark-results.json" ] && echo "- Benchmarks: $REPORTS_DIR/benchmark-results.json")
$([ -f "$REPORTS_DIR/security-scan.json" ] && echo "- Security Scan: $REPORTS_DIR/security-scan.json")

EOF

log_success "Test summary saved: $SUMMARY_FILE"

# Final status
echo ""
if [[ $TEST_RESULT -eq 0 ]]; then
    log_success "🎉 All tests passed! Test suite execution completed successfully."
else
    log_error "❌ Test suite execution failed. Check test output for details."
fi

echo ""
log_info "Test artifacts available in: $REPORTS_DIR"
if [[ "$GENERATE_COVERAGE" == true ]]; then
    log_info "Coverage report available at: $COVERAGE_DIR/index.html"
fi

exit $TEST_RESULT