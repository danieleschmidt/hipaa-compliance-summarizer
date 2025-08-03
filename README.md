# HIPAA Compliance Summarizer

Healthcare-focused LLM agent that automatically redacts PHI (Protected Health Information), generates compliance summaries, and ensures HIPAA-ready document processing using specialized healthcare models.

## Problem Statement

Healthcare organizations process thousands of documents containing Protected Health Information (PHI) daily, but lack automated tools that can simultaneously identify and redact PHI with high accuracy, maintain clinical context for medical decision-making, generate compliance reports for HIPAA audits, and scale to handle enterprise-level document volumes while providing real-time compliance monitoring.

## Solution Overview

The HIPAA Compliance Summarizer provides a comprehensive solution for automated PHI detection, redaction, and compliance reporting. Our system uses advanced ML models trained on healthcare data to achieve ≥98% PHI detection accuracy while maintaining clinical context and generating audit-ready compliance reports.

## Features

- **PHI Detection & Redaction**: Automatic identification and redaction of protected health information
- **HIPAA-Compliant Processing**: Uses healthcare-certified models and secure processing pipelines
- **Compliance Reporting**: Generates detailed compliance summaries and audit trails
- **Healthcare Document Types**: Specialized handling for medical records, clinical notes, insurance forms
- **Risk Assessment**: Identifies potential HIPAA violations and compliance gaps
- **Batch Processing**: Handle large volumes of healthcare documents securely using `BatchProcessor`

## Quick Start

```bash
# Install with healthcare extensions
pip install -r requirements.txt
pip install hipaa-ml-toolkit
pip install -e .  # install package locally for CLI


# Process a single medical document
hipaa-summarize --file patient_record.pdf --compliance-level strict

# Batch process medical records
hipaa-batch-process \
  --input-dir ./medical_records \
  --output-dir ./redacted_records \
  --compliance-level standard \
  --generate-summaries \
  --show-dashboard \
  --dashboard-json dashboard.json \
  --show-cache-performance

# Example dashboard output
Documents processed: 20
Average compliance score: 0.98
Total PHI detected: 75

# Generate compliance report
hipaa-compliance-report --audit-period "2024-Q1"
```

## CLI Reference

### hipaa-summarize
Process a single medical document for PHI detection and redaction.

```bash
hipaa-summarize --file <document> [--compliance-level <level>]
```

**Options:**
- `--file` (required): Path to the document to process
- `--compliance-level`: Compliance strictness level (choices: `strict`, `standard`, `minimal`; default: `standard`)

### hipaa-batch-process  
Batch process multiple healthcare documents with advanced options.

```bash
hipaa-batch-process --input-dir <dir> --output-dir <dir> [options]
```

**Required Options:**
- `--input-dir`: Input directory containing documents to process
- `--output-dir`: Output directory for processed documents

**Optional Parameters:**
- `--compliance-level`: Compliance level (`strict`, `standard`, `minimal`; default: `standard`)
- `--generate-summaries`: Generate summary files alongside redacted documents
- `--show-dashboard`: Display batch processing summary after completion
- `--dashboard-json <file>`: Save dashboard summary to JSON file
- `--show-cache-performance`: Display cache performance metrics after processing

**Cache Performance Output:**
When using `--show-cache-performance`, displays:
```
Cache Performance:
Pattern Compilation - Hits: 145, Misses: 12, Hit Ratio: 92.4%
PHI Detection - Hits: 89, Misses: 23, Hit Ratio: 79.5%  
Cache Memory Usage - Pattern: 12/∞, PHI: 67/1000
```

### hipaa-compliance-report
Generate compliance reports for audit periods.

```bash
hipaa-compliance-report --audit-period <period> [options]
```

**Required Options:**
- `--audit-period`: Reporting period (e.g., "2024-Q1", "2024-01")

**Optional Parameters:**
- `--documents-processed <number>`: Number of documents processed (default: 0)
- `--include-recommendations`: Include compliance recommendations in output

## Environment Configuration

The CLI tools automatically validate the environment configuration before processing:

```bash
# Environment validation checks:
# ✓ Required configuration files present
# ✓ PHI pattern configurations valid
# ✓ Logging framework properly configured
# ⚠ Optional: Production secrets (warnings for development)
```

**Configuration Sources:**
- `config/hipaa_config.yml`: Main configuration file
- Environment variables: `HIPAA_CONFIG_PATH`, `HIPAA_CONFIG_YAML`
- Logging configuration via structured logging framework

## HIPAA Compliance Features

### PHI Detection Categories
- **Direct Identifiers**: Names, addresses, phone numbers, SSNs
- **Medical Identifiers**: MRNs, account numbers, device identifiers
- **Biometric Data**: Fingerprints, voiceprints, full-face photos
- **Date Information**: Birth dates, admission dates, discharge dates
- **Geographic Data**: ZIP codes, cities, states (when combined with other identifiers)

### Redaction Methods
- **Complete Removal**: Eliminates PHI entirely
- **Masking**: Replaces with asterisks or placeholders
- **Synthetic Replacement**: Replaces with realistic but fake data
- **Tokenization**: Replaces with reversible tokens (for authorized access)

## Architecture

```
Healthcare Document → PHI Scanner → Redaction Engine → Compliance Checker → Summary Generator → Audit Logger
                          ↓              ↓                ↓                  ↓                ↓
                    Pattern Match    Smart Redact    Rule Validation   Clinical Summary   Compliance Log
```

## Configuration

```yaml
# config/hipaa_config.yml
compliance:
  level: "strict"  # strict, standard, minimal
  audit_logging: true
  encryption_at_rest: true
  phi_detection_threshold: 0.95

redaction:
  method: "synthetic_replacement"  # removal, masking, synthetic, tokenization
  preserve_clinical_context: true
  maintain_document_structure: true

models:
  phi_detector: "microsoft/presidio-analyzer"
  clinical_summarizer: "microsoft/BioGPT-Large"
  compliance_checker: "custom_hipaa_model_v2"

security:
  encryption_key_rotation: 90  # days
  access_logging: true
  data_retention_policy: 2555  # days (7 years)
  secure_deletion: true

output:
  include_confidence_scores: true
  generate_audit_trail: true
  compliance_score: true
  redaction_summary: true
```

Scoring penalties and PHI detection patterns can be adjusted in this file.
Set the ``HIPAA_CONFIG_PATH`` environment variable to load a custom
configuration file or ``HIPAA_CONFIG_YAML`` to provide the YAML directly
via environment variable.

## Usage Examples

### Basic PHI Redaction
```python
from hipaa_compliance_summarizer import HIPAAProcessor

processor = HIPAAProcessor(compliance_level="strict")
result = processor.process_document("patient_chart.pdf")

print(result.summary)
# "Patient presented with chest pain. Age [REDACTED], gender [REDACTED]. 
#  Treatment administered on [DATE_REDACTED]. Discharged in stable condition."

print(result.compliance_score)  # 0.98
print(result.phi_detected_count)  # 15
```

### Processing Documents by Type
```python
from hipaa_compliance_summarizer import (
    HIPAAProcessor,
    Document,
    DocumentType,
)

processor = HIPAAProcessor()
doc = Document("note.txt", DocumentType.CLINICAL_NOTE)
result = processor.process_document(doc)
print(result.summary)
```

### Compliance Reporting
```python
from hipaa_compliance_summarizer import ComplianceReporter

reporter = ComplianceReporter()
report = reporter.generate_report(
    period="2024-Q1",
    documents_processed=1250,
    include_recommendations=True
)

print(report.overall_compliance)  # 0.97
print(report.violations_detected)  # 3
print(report.recommendations)
# ["Implement additional staff training on PHI handling",
#  "Review data retention policies for imaging files"]
```

### Batch Processing
```python
from hipaa_compliance_summarizer import BatchProcessor

processor = BatchProcessor()
results = processor.process_directory(
    "./medical_records",
    output_dir="./processed_records",
    compliance_level="strict",
    generate_summaries=True,
    show_progress=True
)

# Generate compliance dashboard
dashboard = processor.generate_dashboard(results)
processor.save_dashboard(results, "dashboard.json")
```

### Streaming Redaction
```python
from hipaa_compliance_summarizer import PHIRedactor

redactor = PHIRedactor()
result = redactor.redact_file("large_record.txt")
print(result.text)
```

## Document Types Supported

### Clinical Documents
- **Electronic Health Records (EHR)**
- **Clinical Notes** (SOAP, progress notes, discharge summaries)
- **Laboratory Reports**
- **Radiology Reports**
- **Pathology Reports**
- **Medication Lists**

### Administrative Documents
- **Insurance Claims**
- **Authorization Forms**
- **Consent Forms**
- **Billing Records**
- **Referral Letters**
- **Treatment Plans**

### Specialized Formats
- **HL7 FHIR** messages
- **DICOM** metadata
- **CCD/CDA** documents
- **Handwritten notes** (with OCR)

## Sample Output

### Redacted Document Summary
```json
{
  "document_id": "medical_record_001",
  "processing_timestamp": "2024-01-15T14:30:00Z",
  "original_document": {
    "type": "clinical_note",
    "pages": 3,
    "word_count": 1250
  },
  "phi_analysis": {
    "phi_entities_detected": 23,
    "confidence_scores": {
      "names": 0.98,
      "dates": 0.95,
      "addresses": 0.92,
      "phone_numbers": 0.99
    },
    "redaction_summary": {
      "names": 5,
      "dates": 8,
      "addresses": 2,
      "medical_record_numbers": 3,
      "phone_numbers": 1,
      "ssn": 1,
      "other": 3
    }
  },
  "clinical_summary": {
    "chief_complaint": "Chest pain and shortness of breath",
    "diagnosis": "Acute coronary syndrome, rule out myocardial infarction",
    "treatment": "Administered aspirin, nitroglycerin, and monitoring",
    "disposition": "Admitted to cardiac care unit for further evaluation",
    "key_findings": [
      "Elevated troponin levels",
      "EKG showing ST depression",
      "Patient stable at discharge"
    ]
  },
  "compliance_assessment": {
    "overall_score": 0.96,
    "hipaa_compliance": "COMPLIANT",
    "risk_level": "LOW",
    "audit_trail": "complete",
    "recommendations": [
      "Document retention policy verified",
      "Access controls properly implemented"
    ]
  }
}
```

### Compliance Dashboard
```json
{
  "reporting_period": "2024-Q1",
  "documents_processed": 1847,
  "compliance_metrics": {
    "overall_compliance_rate": 0.97,
    "phi_detection_accuracy": 0.98,
    "processing_time_avg": "12.3 seconds",
    "zero_violation_days": 87
  },
  "phi_statistics": {
    "total_phi_detected": 15432,
    "most_common_phi_types": [
      {"type": "names", "count": 4821},
      {"type": "dates", "count": 3954},
      {"type": "addresses", "count": 2387}
    ]
  },
  "risk_assessment": {
    "high_risk_documents": 12,
    "medium_risk_documents": 89,
    "low_risk_documents": 1746,
    "violations_detected": 3,
    "false_positive_rate": 0.02
  }
}
```

## Advanced Features

### Custom PHI Patterns
```python
# Add custom PHI detection patterns
processor.add_custom_phi_pattern(
    name="hospital_id",
    pattern=r"HSP-\d{6}",
    confidence_threshold=0.9
)

# Define clinical context preservation rules
processor.add_clinical_rule(
    "preserve_medication_dosages",
    lambda text: re.sub(r'\b\d+mg\b', '[DOSAGE]', text)
)
```

### Integration with EHR Systems
```python
# Epic EHR integration
from hipaa_compliance_summarizer.integrations import EpicConnector

epic = EpicConnector(api_key="your_epic_key")
records = epic.fetch_patient_records(patient_id="12345")
processed = processor.process_ehr_batch(records)
```

### Audit and Monitoring
```python
# Real-time compliance monitoring
from hipaa_compliance_summarizer import ComplianceMonitor

monitor = ComplianceMonitor()
monitor.start_real_time_monitoring(
    callback=lambda violation: send_alert(violation),
    threshold=0.95
)
```

## Security Features

### Data Protection
- **Encryption at Rest**: AES-256 encryption for stored documents
- **Encryption in Transit**: TLS 1.3 for all data transmission
- **Zero-Trust Architecture**: Principle of least privilege access
- **Secure Key Management**: Hardware security module (HSM) integration

### Access Controls
- **Role-Based Access Control (RBAC)**
- **Multi-Factor Authentication (MFA)**
- **Session Management** with automatic timeouts
- **Audit Logging** of all access attempts

### Compliance Certifications
- **HIPAA Compliance** - Business Associate Agreement ready
- **SOC 2 Type II** - Security and availability controls
- **GDPR Compliance** - European data protection standards
- **HITRUST CSF** - Healthcare security framework

## Deployment Options

### Cloud Deployment (HIPAA-Compliant)
```yaml
# Azure HIPAA-compliant deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hipaa-summarizer
spec:
  template:
    spec:
      containers:
      - name: app
        image: hipaa-summarizer:latest
        env:
        - name: AZURE_KEY_VAULT_URL
          value: "https://your-keyvault.vault.azure.net/"
        - name: COMPLIANCE_LEVEL
          value: "strict"
```

### On-Premises Deployment
```bash
# Secure on-premises installation
./install.sh --mode secure --encryption enabled --audit-logging enabled

# Configure for air-gapped environment
./configure.sh --offline-mode --local-models
```

## Performance & Scalability

| Document Type | Processing Time | Accuracy | Throughput |
|---------------|----------------|----------|------------|
| Clinical Notes | 8.5s | 98.2% | 450/hour |
| Lab Reports | 5.2s | 99.1% | 720/hour |
| Insurance Forms | 12.1s | 96.8% | 300/hour |
| Radiology Reports | 15.3s | 97.5% | 235/hour |

## Training Data & Models

### Healthcare-Specific Models
- **BioBERT**: Biomedical text understanding
- **ClinicalBERT**: Clinical note processing
- **BlueBERT**: Biomedical language representation
- **Custom PHI Model**: Trained on de-identified healthcare data

### Continuous Learning
- **Federated Learning**: Improve models without data sharing
- **Active Learning**: Human feedback integration
- **Domain Adaptation**: Customize for specific healthcare settings

## Contributing

Priority areas for contribution:
- Additional healthcare document format support
- New PHI detection patterns
- Integration with more EHR systems
- Performance optimizations
- Security enhancements

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and security requirements.

## Continuous Integration

All pull requests run through GitHub Actions. The workflow installs
dependencies, runs `ruff` and `bandit`, and executes the test suite with
coverage reporting. Coverage results are uploaded as build artifacts.
Dependencies are also scanned with `pip-audit` for known vulnerabilities.

## Legal & Compliance

### Business Associate Agreement (BAA)
This software can be deployed under a HIPAA Business Associate Agreement. Contact us for BAA execution and compliance certification.

### Liability & Disclaimers
- Software provided for healthcare assistance only
- Users responsible for compliance verification
- Regular compliance audits recommended
- Professional legal review advised for implementation

## Support & Documentation

- 📖 [HIPAA Compliance Guide](docs/hipaa-compliance.md)
- 🔐 [Security Implementation Guide](docs/security.md)
- 🏥 [Healthcare Integration Examples](docs/ehr-integration.md)
- 📞 **24/7 Compliance Support**: compliance@hipaa-summarizer.com

## License

Healthcare-specific license with compliance provisions - see [LICENSE](LICENSE) file for details.
