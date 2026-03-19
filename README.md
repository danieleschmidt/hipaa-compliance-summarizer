# HIPAA Compliance Summarizer

> **⚠️ Legal Disclaimer:** This tool is provided for educational and informational
> purposes only. It is **not legal advice** and does not guarantee HIPAA compliance.
> Consult qualified legal counsel and HIPAA compliance officers before deploying in
> any production healthcare environment.

A pure-Python, zero-dependency toolkit for automated PHI detection, redaction, audit
logging, and compliance reporting — covering all **18 HIPAA Safe Harbor** identifiers
as defined in 45 CFR § 164.514(b)(2).

---

## Features

- **PHIDetector** — regex-based detection of all 18 PHI categories
- **PHIRedactor** — replaces PHI spans with `[PHI_TYPE]` placeholders
- **ComplianceAuditor** — timestamped audit log, exportable to JSON
- **ComplianceSummarizer** — risk-level classification + redaction completeness score

No ML models, no spaCy, no internet required. Pure Python 3.8+.

---

## The 18 HIPAA Safe Harbor Identifiers

| # | Category | Examples |
|---|----------|---------|
| 1 | **Names** | John Smith, Dr. Jane Doe |
| 2 | **Geographic data** | Street addresses, ZIP codes, state codes |
| 3 | **Dates** (except year) | 03/15/1965, February 14 2023 |
| 4 | **Phone numbers** | (555) 867-5309 |
| 5 | **Fax numbers** | Fax: (555) 867-5310 |
| 6 | **Email addresses** | jdoe@hospital.org |
| 7 | **Social Security Numbers** | 123-45-6789 |
| 8 | **Medical record numbers** | MRN: 987654 |
| 9 | **Health plan beneficiary numbers** | Beneficiary ID: BEN987654 |
| 10 | **Account numbers** | Account #: ACC-123456 |
| 11 | **Certificate/license numbers** | License: MD123456 |
| 12 | **Vehicle identifiers** | VIN: 1HGBH41JXMN109186, Plate: ABC1234 |
| 13 | **Device identifiers / serial numbers** | S/N: DEV12345 |
| 14 | **Web URLs** | https://patient-portal.hospital.org |
| 15 | **IP addresses** | 192.168.1.100 |
| 16 | **Biometric identifiers** | Fingerprint ID: FP-7823 |
| 17 | **Full-face photographs** | photo: headshot.jpg |
| 18 | **Ages over 89** | 95 year old, age: 102 |

---

## Installation

```bash
git clone https://github.com/danieleschmidt/hipaa-compliance-summarizer
cd hipaa-compliance-summarizer
# No additional dependencies required — uses Python stdlib only
```

---

## Quick Start

```python
from hipaa_compliance import (
    PHIDetector,
    PHIRedactor,
    ComplianceAuditor,
    ComplianceSummarizer,
)

text = "Patient John Smith, DOB 03/15/1965, SSN 123-45-6789, MRN: 987654."

detector  = PHIDetector()
redactor  = PHIRedactor()
auditor   = ComplianceAuditor()
summarizer = ComplianceSummarizer()

# 1. Detect PHI
findings = detector.detect(text)

# 2. Redact
redacted = redactor.redact(text, findings)
print(redacted)
# → "[NAME] Smith, DOB [DATE], SSN [SSN], [MRN]."

# 3. Audit
auditor.record("doc-001", findings, redacted=True)

# 4. Report
report = summarizer.summarize(auditor.log, text, redacted)
print(report["risk_level"])    # HIGH
print(report["redaction_score"])  # 100.0
```

---

## Demo

```bash
python demo_hipaa_pipeline.py
```

### Example Output

```
========================================================================
DOCUMENT: sample_1
========================================================================
Original:
  Patient John Smith, DOB 03/15/1965, SSN 123-45-6789, MRN: 987654, admitted 01/10/2024.

Redacted:
  [NAME] Smith, DOB [DATE], SSN [SSN], [MRN], admitted [DATE].

Compliance Report:
  Risk Level: HIGH
  PHI Categories Found: DATE, MRN, NAME, SSN
  Total PHI Instances: 5
  Redacted: 5/5 (100.0% completeness)
  ✅ All detected PHI has been redacted.
```

---

## Running Tests

```bash
python -m pytest tests/test_hipaa.py -v
```

37 tests covering every PHI category, redactor edge cases, auditor, summarizer, and
a full end-to-end pipeline test.

---

## Module Reference

### `PHIDetector`

```python
detector = PHIDetector()
findings = detector.detect(text)
# Returns: list of {type, value, start, end}
```

### `PHIRedactor`

```python
redactor = PHIRedactor()
redacted_text = redactor.redact(original_text, findings)
```

### `ComplianceAuditor`

```python
auditor = ComplianceAuditor()
auditor.record(document_id, findings, redacted=True, metadata={})
print(auditor.to_json())      # full JSON audit log
auditor.summary_by_type()     # {PHI_TYPE: count}
```

### `ComplianceSummarizer`

```python
report = summarizer.summarize(audit_log, original_text, redacted_text)
# Returns:
# {
#   risk_level: "HIGH" | "MEDIUM" | "LOW" | "NONE",
#   phi_categories: [...],
#   phi_count: int,
#   redaction_score: float (0-100),
#   summary: str (human-readable),
# }
```

**Risk level logic:**
- **HIGH**: SSN, MRN, DATE (DOB proxy), or health plan beneficiary number found
- **MEDIUM**: Name co-occurs with geographic, phone, fax, email, or age-over-89
- **LOW**: Other PHI categories only
- **NONE**: No PHI detected

---

## Compliance Notes

- This tool implements the **Safe Harbor method** for de-identification as described
  in 45 CFR § 164.514(b)(2).
- Regex-based detection has inherent limitations (false positives and false negatives).
  For clinical production use, supplement with expert determination per
  § 164.514(b)(1).
- Audit logs generated by `ComplianceAuditor` can support HIPAA audit requirements
  but must be stored securely and access-controlled.
- This is a **tool**, not a compliance programme. HIPAA compliance requires
  administrative, physical, and technical safeguards beyond PHI redaction.

---

## License

MIT — see `LICENSE`.
