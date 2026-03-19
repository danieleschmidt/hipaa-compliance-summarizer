#!/usr/bin/env python3
"""
demo_hipaa_pipeline.py
======================
Demonstrates the HIPAA Compliance Pipeline on three sample medical texts.

Usage:
    python demo_hipaa_pipeline.py
"""

import json
from hipaa_compliance import (
    PHIDetector,
    PHIRedactor,
    ComplianceAuditor,
    ComplianceSummarizer,
)

# ── Sample medical texts ──────────────────────────────────────────────────────
SAMPLE_TEXTS = [
    (
        "sample_1",
        (
            "Patient John Smith, DOB 03/15/1965, SSN 123-45-6789, "
            "MRN: 987654, admitted 01/10/2024. "
            "Contact: (555) 234-5678."
        ),
    ),
    (
        "sample_2",
        (
            "Dr. Jane Doe at 123 Main St, Springfield, IL 62701. "
            "Phone: (555) 867-5309, fax: (555) 867-5310, "
            "email: jdoe@hospital.org"
        ),
    ),
    (
        "sample_3",
        (
            "The 95 year old patient with IP 192.168.1.100 visited on "
            "February 14, 2023. See photo at headshot.jpg. "
            "Device serial S/N: ABC12345."
        ),
    ),
]

# ── Pipeline ──────────────────────────────────────────────────────────────────
detector = PHIDetector()
redactor = PHIRedactor()
auditor = ComplianceAuditor()
summarizer = ComplianceSummarizer()

SEP = "=" * 72

for doc_id, text in SAMPLE_TEXTS:
    print(f"\n{SEP}")
    print(f"DOCUMENT: {doc_id}")
    print(SEP)
    print(f"Original:\n  {text}\n")

    # 1. Detect
    findings = detector.detect(text)

    # 2. Redact
    redacted = redactor.redact(text, findings)
    print(f"Redacted:\n  {redacted}\n")

    # 3. Audit
    auditor.record(doc_id, findings, redacted=True)

    # 4. Per-document report
    doc_log = [e for e in auditor.log if e["document_id"] == doc_id]
    report = summarizer.summarize(doc_log, text, redacted)
    print("Compliance Report:")
    print(f"  {report['summary'].replace(chr(10), chr(10)+'  ')}")
    print(f"\n  PHI Count   : {report['phi_count']}")
    print(f"  Risk Level  : {report['risk_level']}")
    print(f"  Redact Score: {report['redaction_score']}%")

print(f"\n{SEP}")
print("FULL AUDIT LOG (JSON excerpt — first 3 entries):")
print(SEP)
all_entries = json.loads(auditor.to_json())
print(json.dumps(all_entries[:3], indent=2))
print(f"\n... ({len(all_entries)} total audit entries)")
print(f"\n{SEP}")
print("ALL DONE ✅")
print(SEP)
