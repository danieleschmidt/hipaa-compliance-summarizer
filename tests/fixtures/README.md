# Test Fixtures

This directory contains test data and fixtures for the HIPAA Compliance Summarizer test suite.

## ⚠️ CRITICAL SECURITY NOTICE

**NEVER store real PHI (Protected Health Information) in this directory!**

All test data must be:
- Synthetic/artificially generated
- Completely de-identified 
- Compliant with HIPAA Safe Harbor provisions
- Not derived from real patient records

## Directory Structure

```
fixtures/
├── synthetic_documents/     # Synthetic healthcare documents for testing
├── phi_patterns/           # Test PHI patterns (synthetic only)
├── config_samples/         # Sample configuration files
├── mock_responses/         # Mock API responses
├── test_datasets/          # Synthetic datasets for various test scenarios
├── performance_data/       # Performance benchmarking data
└── compliance_samples/     # Sample compliance reports and outputs
```

## Test Data Guidelines

### Synthetic Document Creation
- Use realistic but fake names, addresses, phone numbers
- Ensure SSNs, MRNs, and account numbers are clearly synthetic
- Include various document types: clinical notes, lab reports, etc.
- Test edge cases: partially redacted documents, malformed data

### PHI Pattern Testing
- Create comprehensive test patterns for all 18 HIPAA identifiers
- Include edge cases and boundary conditions
- Test both positive and negative cases
- Validate pattern performance with large datasets

### Configuration Testing
- Sample HIPAA configurations for different compliance levels
- Test configurations for various deployment scenarios
- Include both valid and invalid configurations for error handling

## Data Privacy & Security

1. **Regular Audits**: All test data is audited quarterly for PHI exposure
2. **Automated Scanning**: Pre-commit hooks scan for potential real PHI
3. **Access Controls**: Test data access is logged and monitored
4. **Data Lifecycle**: Test data is regularly refreshed and sanitized

## Contributing Test Data

When adding new test fixtures:
1. Verify data is completely synthetic
2. Document the purpose and scope of the test data
3. Include appropriate copyright/license information
4. Test data should support multiple test scenarios

## Compliance Verification

All test fixtures undergo:
- ✅ Automated PHI detection scanning
- ✅ Manual review by compliance team
- ✅ Documentation of data source and generation method
- ✅ Regular re-validation of synthetic nature

For questions about test data compliance, contact: compliance@hipaa-summarizer.com