# PHI Pattern Configuration Guide

## Overview

The HIPAA Compliance Summarizer now features a modular PHI pattern configuration system that provides flexible, extensible, and maintainable pattern management for detecting Protected Health Information (PHI).

## Key Features

- **Modular Design**: Organize patterns into logical categories (core, medical, custom)
- **Dynamic Management**: Add, disable, enable, and validate patterns at runtime
- **Backward Compatibility**: Existing code continues to work with legacy pattern definitions
- **Performance Optimized**: Built-in caching and compiled pattern reuse
- **Validation**: Comprehensive pattern validation and error handling
- **Configuration Driven**: Load patterns from YAML files or configuration objects

## Architecture

```
PHIPatternManager
├── PHIPatternCategory (core)
│   ├── PHIPatternConfig (ssn)
│   ├── PHIPatternConfig (email)
│   └── PHIPatternConfig (phone)
├── PHIPatternCategory (medical)
│   ├── PHIPatternConfig (mrn)
│   ├── PHIPatternConfig (dea)
│   └── PHIPatternConfig (insurance_id)
└── PHIPatternCategory (custom)
    └── [User-defined patterns]
```

## Basic Usage

### Using Default Patterns

```python
from hipaa_compliance_summarizer.phi import PHIRedactor

# Automatically loads default patterns
redactor = PHIRedactor()

# Detect PHI in text
text = "Patient SSN: 123-45-6789, email: john@hospital.com"
entities = redactor.detect(text)
print(f"Found {len(entities)} PHI entities")

# Redact PHI
result = redactor.redact(text)
print(result.text)  # "Patient SSN: [REDACTED], email: [REDACTED]"
```

### Adding Custom Patterns

```python
# Add a custom pattern through the redactor
redactor.add_custom_pattern(
    name="employee_id",
    pattern=r"\bEMP-\d{4}\b",
    description="Employee ID pattern",
    confidence_threshold=0.95,
    category="custom"
)

# Test the custom pattern
text = "Employee ID: EMP-1234"
entities = redactor.detect(text)
```

### Managing Patterns

```python
# List all available patterns
patterns = redactor.list_patterns()
for name, details in patterns.items():
    print(f"{name}: {details['description']} (Category: {details['category']})")

# Disable a pattern temporarily
redactor.disable_pattern("phone")

# Re-enable it later
redactor.enable_pattern("phone")

# Get pattern statistics
stats = redactor.get_pattern_statistics()
print(f"Total patterns: {stats['total_patterns']}")
print(f"Enabled patterns: {stats['enabled_patterns']}")
```

## Advanced Configuration

### Direct Pattern Manager Usage

```python
from hipaa_compliance_summarizer.phi_patterns import (
    pattern_manager, 
    PHIPatternConfig, 
    PHIPatternCategory
)

# Create a custom pattern configuration
custom_pattern = PHIPatternConfig(
    name="hospital_id",
    pattern=r"\bHOSP-\d{6}\b",
    description="Hospital ID pattern",
    confidence_threshold=0.98,
    category="institutional"
)

# Add to pattern manager
pattern_manager.add_custom_pattern(custom_pattern, "institutional")

# Load patterns from configuration file
pattern_manager.load_patterns_from_file("my_patterns.yml")
```

### Loading Patterns from YAML

Create a YAML file with custom patterns:

```yaml
# my_patterns.yml
patterns:
  facility_id: "\\bFAC-\\d{4}\\b"
  lab_id: "\\bLAB-[A-Z]{2}\\d{4}\\b"
  prescription_number: "\\bRX-\\d{8}\\b"
  
scoring:
  penalty_per_entity: 0.01
  penalty_cap: 0.2
```

Load the patterns:

```python
from hipaa_compliance_summarizer.phi_patterns import pattern_manager

pattern_manager.load_patterns_from_file("my_patterns.yml")
```

### Creating Custom Pattern Categories

```python
from hipaa_compliance_summarizer.phi_patterns import PHIPatternCategory

# Create a specialized category
radiology_category = PHIPatternCategory(
    name="radiology",
    description="Radiology-specific identifiers"
)

# Add patterns to the category
radiology_patterns = [
    PHIPatternConfig("dicom_id", r"\\bDCM-\\d{10}\\b", "DICOM Study ID"),
    PHIPatternConfig("imaging_id", r"\\bIMG-[A-Z]\\d{6}\\b", "Imaging ID"),
]

for pattern in radiology_patterns:
    radiology_category.add_pattern(pattern)

# Add category to manager
pattern_manager.categories["radiology"] = radiology_category
```

## Pattern Configuration Reference

### PHIPatternConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Unique identifier for the pattern |
| `pattern` | str | Required | Regular expression pattern |
| `description` | str | "" | Human-readable description |
| `confidence_threshold` | float | 0.95 | Confidence threshold (0.0-1.0) |
| `category` | str | "general" | Pattern category for organization |
| `enabled` | bool | True | Whether pattern is active |

### Default Pattern Categories

#### Core Patterns
- **ssn**: Social Security Numbers (`\b\d{3}-\d{2}-\d{4}\b`)
- **phone**: Phone numbers (`\b\d{3}[.-]\d{3}[.-]\d{4}\b`)
- **email**: Email addresses
- **date**: Dates in MM/DD/YYYY format

#### Medical Patterns
- **mrn**: Medical Record Numbers
- **dea**: DEA Numbers
- **insurance_id**: Insurance/Member IDs

## Performance Considerations

### Caching

The system includes multiple levels of caching:

```python
# Check cache statistics
cache_info = PHIRedactor.get_cache_info()
print(f"Pattern compilation cache: {cache_info['pattern_compilation']}")
print(f"PHI detection cache: {cache_info['phi_detection']}")

# Clear caches if needed (for testing or memory management)
PHIRedactor.clear_cache()
```

### Pattern Optimization

- **Compile Once**: Patterns are compiled once and reused
- **Detection Caching**: Results are cached for identical text+pattern combinations
- **Lazy Loading**: Patterns are only compiled when first used

## Error Handling and Validation

### Pattern Validation

```python
# Validate all patterns
errors = pattern_manager.validate_all_patterns()
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

### Error Recovery

```python
try:
    redactor.add_custom_pattern(
        name="invalid",
        pattern="[invalid regex",  # Missing closing bracket
        description="This will fail"
    )
except ValueError as e:
    print(f"Pattern validation failed: {e}")
```

## Migration Guide

### From Legacy Pattern Definitions

**Before (Legacy)**:
```python
patterns = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"
}
redactor = PHIRedactor(patterns=patterns)
```

**After (Modular)**:
```python
# Option 1: Use defaults and add custom patterns
redactor = PHIRedactor()
redactor.add_custom_pattern("custom_id", r"\bCUST-\d+\b", "Custom ID")

# Option 2: Continue using legacy approach (still supported)
redactor = PHIRedactor(patterns=patterns)  # Works exactly as before
```

### Configuration File Migration

**Before**:
```yaml
patterns:
  ssn: "\\b\\d{3}-\\d{2}-\\d{4}\\b"
  phone: "\\b\\d{3}[.-]\\d{3}[.-]\\d{4}\\b"
```

**After** (Enhanced):
```yaml
patterns:
  ssn: "\\b\\d{3}-\\d{2}-\\d{4}\\b"
  phone: "\\b\\d{3}[.-]\\d{3}[.-]\\d{4}\\b"
  # Add new patterns without code changes
  employee_id: "\\bEMP-\\d{4}\\b"
  facility_code: "\\bFAC-[A-Z]{3}\\d{3}\\b"
```

## Best Practices

### Pattern Design

1. **Specificity**: Make patterns as specific as possible to reduce false positives
2. **Capturing Groups**: Use capturing groups for patterns with context (e.g., "MRN: (\\d+)")
3. **Case Insensitivity**: Patterns are compiled with `re.IGNORECASE` by default
4. **Boundary Matching**: Use `\b` for word boundaries to avoid partial matches

### Performance

1. **Pattern Complexity**: Avoid overly complex regex patterns that can cause performance issues
2. **Caching**: Leverage the built-in caching for repeated text processing
3. **Selective Disabling**: Disable unused patterns to improve performance

### Organization

1. **Categorization**: Group related patterns into logical categories
2. **Naming**: Use descriptive names for patterns and categories
3. **Documentation**: Always provide meaningful descriptions for custom patterns

## Troubleshooting

### Common Issues

**Pattern Not Detecting Expected Text**:
```python
# Test pattern directly
import re
pattern = re.compile(r"\bTEST-\d+\b", re.IGNORECASE)
test_text = "ID: TEST-123"
matches = pattern.findall(test_text)
print(f"Matches: {matches}")
```

**Cache Issues**:
```python
# Clear cache after pattern changes
PHIRedactor.clear_cache()
```

**Validation Errors**:
```python
# Check for validation errors
errors = pattern_manager.validate_all_patterns()
for error in errors:
    print(f"Error: {error}")
```

## API Reference

### Key Classes

- `PHIPatternConfig`: Individual pattern configuration
- `PHIPatternCategory`: Container for related patterns  
- `PHIPatternManager`: Central pattern management
- `PHIRedactor`: Enhanced redactor with modular pattern support

### Key Methods

- `add_custom_pattern()`: Add new patterns
- `disable_pattern()`, `enable_pattern()`: Toggle patterns
- `list_patterns()`: Get all pattern details
- `get_pattern_statistics()`: Get usage statistics
- `validate_all_patterns()`: Validate configuration

For complete API documentation, see the docstrings in the source code.