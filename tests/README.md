# HIPAA Compliance Summarizer - Test Suite

This directory contains comprehensive tests for the HIPAA Compliance Summarizer, organized by test type and functionality.

## Test Structure

```
tests/
├── unit/                     # Unit tests for individual components
├── integration/             # Integration tests for component interactions
├── performance/             # Performance and load testing
├── security/               # Security-focused tests
├── compliance/             # HIPAA compliance validation tests
├── fixtures/               # Test data and fixtures
├── conftest.py            # Pytest configuration and shared fixtures
├── utils.py               # Test utilities and helpers
└── README.md              # This documentation
```

## Test Categories

### Unit Tests
- **Location**: `tests/test_*.py` and `tests/unit/`
- **Purpose**: Test individual functions and classes in isolation
- **Coverage**: All core modules should have comprehensive unit tests
- **Mocking**: External dependencies are mocked to ensure isolation

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test component interactions and workflows
- **Scope**: End-to-end PHI detection, redaction, and compliance checking
- **Data**: Uses synthetic healthcare data only

### Performance Tests
- **Location**: `tests/performance/`
- **Purpose**: Validate performance requirements and identify bottlenecks
- **Metrics**: Processing time, memory usage, throughput
- **Load Testing**: Batch processing with large document sets

### Security Tests
- **Location**: `tests/security/`
- **Purpose**: Validate security controls and HIPAA compliance
- **Coverage**: Encryption, access controls, audit logging
- **Compliance**: Ensures no PHI leakage in logs or outputs

### Compliance Tests
- **Location**: `tests/compliance/`
- **Purpose**: Verify HIPAA compliance requirements
- **Validation**: PHI detection accuracy, audit trails, data retention
- **Reporting**: Generate compliance test reports

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hipaa_compliance_summarizer --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/security/

# Run tests with specific markers
pytest -m "not slow"
pytest -m "security"
pytest -m "performance"
```

### Test Configuration
```bash
# Run with verbose output
pytest -v

# Run with detailed coverage
pytest --cov=hipaa_compliance_summarizer --cov-report=term-missing

# Run parallel tests
pytest -n auto

# Run with benchmark reporting
pytest --benchmark-only
```

### Environment-Specific Testing
```bash
# Development environment
ENVIRONMENT=development pytest

# CI/CD environment
ENVIRONMENT=ci pytest --cov --junit-xml=test-results.xml

# Production-like testing
ENVIRONMENT=staging pytest tests/integration/
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.security` - Security tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.compliance` - HIPAA compliance tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.requires_config` - Tests requiring configuration files

### Example Usage
```python
import pytest

@pytest.mark.security
@pytest.mark.slow
def test_encryption_performance():
    """Test encryption performance under load."""
    pass

@pytest.mark.compliance
def test_phi_detection_accuracy():
    """Validate PHI detection meets HIPAA requirements."""
    pass
```

## Test Data and Fixtures

### Synthetic Test Data
All test data uses synthetic information only:
- **No Real PHI**: Never use actual patient data
- **Realistic Patterns**: Data mimics real healthcare documents
- **Compliance Safe**: All test data is HIPAA-compliant for testing

### Available Fixtures
- `test_config` - HIPAA configuration for testing
- `temp_config_file` - Temporary configuration file
- `sample_documents` - Synthetic healthcare documents
- `mock_phi_detector` - Mocked PHI detection service
- `mock_redactor` - Mocked redaction service
- `performance_test_documents` - Large document set for performance testing

### Creating Test Data
```python
def test_with_sample_data(sample_documents):
    """Example test using sample documents fixture."""
    clinical_note = sample_documents["clinical_note"]
    content = clinical_note.read_text()
    assert "John Doe" in content  # Synthetic name
```

## Security Considerations

### PHI Protection in Tests
- **Never commit real PHI data** to version control
- **Use synthetic data only** for all healthcare scenarios
- **Encrypt sensitive test configurations** if needed
- **Review test outputs** to ensure no PHI leakage

### Test Environment Security
- **Isolated test environment** with no production data access
- **Secure test credentials** separate from production
- **Audit test execution** for compliance verification
- **Clean up temporary files** containing synthetic PHI

## Coverage Requirements

### Minimum Coverage Targets
- **Overall Code Coverage**: 80% minimum
- **Core PHI Detection**: 95% minimum
- **Security Modules**: 90% minimum
- **CLI Interfaces**: 70% minimum (due to I/O complexity)

### Coverage Reporting
```bash
# Generate HTML coverage report
pytest --cov=hipaa_compliance_summarizer --cov-report=html
open htmlcov/index.html

# Generate XML coverage for CI/CD
pytest --cov=hipaa_compliance_summarizer --cov-report=xml

# Generate JSON coverage for analysis
pytest --cov=hipaa_compliance_summarizer --cov-report=json
```

## Performance Testing

### Benchmark Tests
- **Document Processing Speed**: <15 seconds per clinical document
- **Batch Processing Throughput**: >100 documents per hour
- **Memory Usage**: <2GB for typical workloads
- **PHI Detection Accuracy**: >98% precision and recall

### Load Testing
```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Generate performance report
pytest tests/performance/ --benchmark-json=benchmark_results.json
```

## Continuous Integration

### CI/CD Test Pipeline
1. **Security Scan**: Static security analysis
2. **Unit Tests**: Fast feedback on code changes
3. **Integration Tests**: Validate component interactions
4. **Performance Tests**: Ensure performance requirements
5. **Compliance Tests**: HIPAA compliance validation
6. **Coverage Report**: Generate and publish coverage metrics

### GitHub Actions Integration
```yaml
- name: Run Tests
  run: |
    pytest --cov=hipaa_compliance_summarizer \
           --cov-report=xml \
           --junit-xml=test-results.xml \
           tests/

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Test Development Guidelines

### Writing New Tests
1. **Follow naming conventions**: `test_*.py` files, `test_*` functions
2. **Use descriptive names**: Test names should explain what is being tested
3. **Include docstrings**: Document test purpose and expected behavior
4. **Use appropriate markers**: Mark tests with relevant pytest markers
5. **Mock external dependencies**: Keep tests isolated and fast

### Test Quality Standards
- **Arrange-Act-Assert** pattern for test structure
- **One assertion per test** when possible
- **Clear test data setup** using fixtures
- **Comprehensive edge case coverage**
- **Performance considerations** for long-running tests

### Example Test Structure
```python
import pytest
from hipaa_compliance_summarizer import PHIDetector

@pytest.mark.unit
def test_phi_detector_identifies_names(mock_phi_detector):
    """Test that PHI detector correctly identifies patient names."""
    # Arrange
    detector = PHIDetector()
    text = "Patient: John Doe"
    
    # Act
    result = detector.detect_phi(text)
    
    # Assert
    assert len(result["entities"]) > 0
    assert any(entity["type"] == "PERSON" for entity in result["entities"])
```

## Troubleshooting

### Common Test Issues
- **Import Errors**: Ensure `PYTHONPATH` includes `src/` directory
- **Configuration Missing**: Copy `.env.example` to `.env` for local testing
- **Fixture Not Found**: Check `conftest.py` for required fixtures
- **Slow Tests**: Use `-m "not slow"` to skip long-running tests

### Debug Test Failures
```bash
# Run with maximum verbosity
pytest -vvv --tb=long

# Drop into debugger on failure
pytest --pdb

# Run only failed tests from last run
pytest --lf

# Show local variables in tracebacks
pytest --tb=long --showlocals
```

## Compliance Reporting

### Test Reports for Audits
- **Coverage Reports**: Demonstrate code quality
- **Security Test Results**: Validate security controls
- **Performance Metrics**: Show system performance
- **Compliance Test Results**: HIPAA requirement validation

### Generating Audit Reports
```bash
# Complete test suite with reporting
pytest --cov=hipaa_compliance_summarizer \
       --cov-report=html \
       --cov-report=xml \
       --junit-xml=test-results.xml \
       --benchmark-json=benchmarks.json \
       tests/
```

## Contributing to Tests

### Adding New Tests
1. **Create test file** in appropriate directory
2. **Use existing fixtures** when possible
3. **Add new fixtures** to `conftest.py` if needed
4. **Update documentation** for significant test additions
5. **Run full test suite** before submitting

### Test Review Checklist
- [ ] Tests use synthetic data only
- [ ] No real PHI in test files
- [ ] Appropriate test markers applied
- [ ] Good test coverage of new code
- [ ] Performance impact considered
- [ ] Documentation updated if needed

For questions about testing, see the [Contributing Guide](../CONTRIBUTING.md) or contact the development team.