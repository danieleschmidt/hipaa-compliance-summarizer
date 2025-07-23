# Integration Testing Guide

This document provides guidance for running and extending the comprehensive integration test suite for the HIPAA Compliance Summarizer.

## Overview

The integration test suite (`tests/test_full_pipeline_integration.py`) validates the complete end-to-end workflow of the PHI processing pipeline, ensuring all components work together correctly for healthcare document processing.

## Test Coverage

### Core Integration Tests
- **Single Document Processing**: Tests complete pipeline for individual documents with PHI
- **Clean Document Handling**: Validates processing of documents without PHI
- **Batch Processing Workflow**: Tests directory-based batch processing with multiple document types
- **Multi-Compliance Level Processing**: Validates different compliance strictness levels
- **Document Type Detection**: Tests automatic document type detection integration
- **Security Validation**: Ensures security checks are properly integrated
- **Error Handling**: Tests graceful error handling throughout the pipeline
- **Compliance Reporting**: Validates integration between processing and reporting components

### Performance Benchmarks
- **Processing Time Baseline**: Ensures documents process within acceptable time limits
- **Batch Processing Throughput**: Measures and validates batch processing performance
- **Memory Usage Stability**: Tests memory stability during extended processing

### Edge Cases
- **Large Document Processing**: Tests handling of documents with significant content
- **Mixed Content Batches**: Validates processing of varied document types and quality
- **Concurrent Processing**: Tests simultaneous processing capabilities

## Running Integration Tests

### Full Suite
```bash
# Run all integration tests
python3 -m pytest tests/test_full_pipeline_integration.py -v

# Run with coverage reporting
python3 -m pytest tests/test_full_pipeline_integration.py --cov=hipaa_compliance_summarizer --cov-report=term-missing
```

### Specific Test Categories
```bash
# Run only core integration tests
python3 -m pytest tests/test_full_pipeline_integration.py::TestFullPipelineIntegration -v

# Run only performance benchmarks
python3 -m pytest tests/test_full_pipeline_integration.py::TestIntegrationPerformanceBenchmarks -v

# Run only edge case tests
python3 -m pytest tests/test_full_pipeline_integration.py::TestIntegrationEdgeCases -v
```

### Individual Tests
```bash
# Test single document processing
python3 -m pytest tests/test_full_pipeline_integration.py::TestFullPipelineIntegration::test_single_document_full_pipeline -v

# Test batch processing
python3 -m pytest tests/test_full_pipeline_integration.py::TestFullPipelineIntegration::test_batch_processing_full_workflow -v
```

## Test Fixtures and Data

The integration tests use realistic healthcare document samples that include:

### Clinical Note Sample
```text
CLINICAL NOTE
Patient: John Doe
DOB: 01/15/1980
SSN: 123-45-6789
MRN: MR123456
Phone: (555) 123-4567
Address: 123 Main St, Anytown, NY 12345

CHIEF COMPLAINT: Chest pain and shortness of breath
HISTORY: 43-year-old male presents with acute onset chest pain...
```

### Insurance Form Sample
```text
INSURANCE CLAIM FORM
Patient Information:
Name: Jane Smith
DOB: 05/22/1975
SSN: 987-65-4321
Policy Number: INS789012
```

### Lab Report Sample
```text
LABORATORY REPORT
Patient: Robert Johnson
DOB: 11/08/1965
MRN: LAB987654
Account: ACC123789
```

## Expected PHI Detection

Based on current patterns, the tests expect detection of:
- **SSN patterns**: `123-45-6789` format
- **Date patterns**: `MM/DD/YYYY` format  
- **Medical Record Numbers**: `MRxxxxxx` format
- **Phone numbers**: `xxx-xxx-xxxx` format
- **Email addresses**: Standard email format

**Note**: Name and address detection patterns are not currently implemented, so tests are designed accordingly.

## Performance Expectations

### Processing Time Limits
- Single document: < 5 seconds
- Large documents (1000+ lines): < 30 seconds
- Batch processing (10 documents): < 60 seconds

### Throughput Requirements  
- Batch processing: > 1 document per second
- Memory stability: No unbounded growth during extended processing

## Adding New Integration Tests

### Test Structure
```python
def test_new_integration_scenario(self, sample_healthcare_documents):
    """Test description of the integration scenario."""
    # Setup
    processor = HIPAAProcessor()
    document = Document(str(sample_healthcare_documents["clinical_note"]), DocumentType.CLINICAL_NOTE)
    
    # Execute
    result = processor.process_document(document)
    
    # Verify
    assert isinstance(result, ProcessingResult)
    assert result.phi_detected_count > 0
    assert "[REDACTED]" in result.redacted.text
```

### Best Practices
1. **Use Realistic Data**: Test fixtures should represent actual healthcare documents
2. **Test End-to-End**: Validate complete workflows, not just individual components
3. **Include Error Cases**: Test both success and failure scenarios
4. **Performance Awareness**: Include timing assertions for critical paths
5. **Security Focus**: Ensure PHI protection is validated at every step

## Troubleshooting

### Common Issues
1. **ImportError**: Ensure package is installed with `pip install -e .`
2. **PHI Pattern Mismatches**: Update test assertions to match current pattern definitions
3. **Performance Failures**: Check system resources and adjust timeout values if needed
4. **File Permission Errors**: Ensure test has write access to temporary directories

### Debug Mode
```bash
# Run with detailed output
python3 -m pytest tests/test_full_pipeline_integration.py -v -s --tb=long

# Run single test with debugging
python3 -m pytest tests/test_full_pipeline_integration.py::TestFullPipelineIntegration::test_single_document_full_pipeline -v -s --pdb
```

## Integration with CI/CD

The integration tests are designed to run in the GitHub Actions CI pipeline:

```yaml
- name: Run Integration Tests
  run: pytest tests/test_full_pipeline_integration.py -n auto --cov=hipaa_compliance_summarizer
```

Tests should complete within the CI time limits and provide comprehensive coverage reporting.

## Future Enhancements

Planned improvements to the integration test suite:
- [ ] Add name and address PHI pattern tests when patterns are implemented
- [ ] Expand performance benchmarking with more document types
- [ ] Add integration tests for external system connections (when available)
- [ ] Include stress testing for high-volume batch processing
- [ ] Add integration tests for compliance certification workflows